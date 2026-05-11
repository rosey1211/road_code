// road_centerline_node.cpp
// ROS2 node: subscribes to a camera image, runs the road centerline CNN,
// publishes CenterlineResult + a sensor_msgs/Image with visual overlays.

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>

#include <torch/script.h>
#include <torch/torch.h>

#include <opencv2/opencv.hpp>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/point32.hpp>
#include <cv_bridge/cv_bridge/cv_bridge.hpp>

#include "road_centerline/msg/centerline_result.hpp"

// ── Colours (BGR for OpenCV) ──────────────────────────────────────────────────
static const cv::Scalar _PRED  (255, 160,  50);   // blue
static const cv::Scalar _GUIDE ( 80, 210,   0);   // green
static const cv::Scalar _RED   ( 30,  30, 220);

// ── Model geometry ─────────────────────────────────────────────────────────────
struct ModelConfig {
    int    image_width  = 160;
    int    image_height = 120;
    int    n_buckets    = 40;
    int    n_rows       = 3;
    std::vector<double> row_fractions;
    std::vector<double> norm_mean;
    std::vector<double> norm_std;
    double crop_top     = 0.0;
    double crop_bottom  = 0.0;
    std::vector<std::vector<double>> bucket_edges; // [n_rows][n_buckets+1]
};

static ModelConfig loadConfig(const std::string & path)
{
    YAML::Node y = YAML::LoadFile(path);
    ModelConfig c;
    c.image_width  = y["image_width"].as<int>();
    c.image_height = y["image_height"].as<int>();
    c.n_buckets    = y["n_buckets"].as<int>();
    c.n_rows       = y["n_rows"].as<int>();
    for (auto v : y["row_fractions"]) c.row_fractions.push_back(v.as<double>());
    for (auto v : y["norm_mean"])     c.norm_mean.push_back(v.as<double>());
    for (auto v : y["norm_std"])      c.norm_std.push_back(v.as<double>());
    c.crop_top    = y["crop_top"].as<double>();
    c.crop_bottom = y["crop_bottom"].as<double>();
    for (auto row : y["bucket_edges"]) {
        std::vector<double> edges;
        for (auto e : row) edges.push_back(e.as<double>());
        c.bucket_edges.push_back(edges);
    }
    return c;
}

// ── Peak detection ─────────────────────────────────────────────────────────────
struct Peak {
    double cx_frac;
    float  conf;
};

static Peak findPrimaryPeak(const std::vector<float>       & probs,
                             const std::vector<double>      & edges,
                             float                            cluster_thresh_frac)
{
    int   n         = static_cast<int>(probs.size());
    float peak_conf = *std::max_element(probs.begin(), probs.end());
    float thresh    = std::max(0.05f, peak_conf * cluster_thresh_frac);

    struct Cluster { int lo, hi; float peak_conf; };
    std::vector<Cluster> clusters;
    int start = -1;
    for (int b = 0; b < n; ++b) {
        if (probs[b] >= thresh) {
            if (start < 0) start = b;
        } else if (start >= 0) {
            float pc = *std::max_element(probs.begin()+start, probs.begin()+b);
            clusters.push_back({start, b-1, pc});
            start = -1;
        }
    }
    if (start >= 0) {
        float pc = *std::max_element(probs.begin()+start, probs.end());
        clusters.push_back({start, n-1, pc});
    }

    auto computeCx = [&](int lo, int hi) -> double {
        double tw = 0, wx = 0;
        for (int b = lo; b <= hi; ++b) {
            double cx = (edges[b] + edges[b+1]) * 0.5;
            tw += probs[b]; wx += cx * probs[b];
        }
        return tw > 0 ? wx/tw : (edges[lo]+edges[hi+1])*0.5;
    };

    if (clusters.empty()) {
        int pb = static_cast<int>(std::max_element(probs.begin(),probs.end())-probs.begin());
        return {(edges[pb]+edges[pb+1])*0.5, peak_conf};
    }
    auto best = std::max_element(clusters.begin(), clusters.end(),
        [](const Cluster & a, const Cluster & b){ return a.peak_conf < b.peak_conf; });
    return {computeCx(best->lo, best->hi), best->peak_conf};
}

// ── Drawing helpers ────────────────────────────────────────────────────────────

static void dashedHLine(cv::Mat & img, int y, cv::Scalar color, int dash, int thickness)
{
    int w = img.cols;
    if (y < 0 || y >= img.rows) return;
    for (int x = 0; x < w; x += dash * 2)
        cv::line(img, {x, y}, {std::min(x+dash-1, w-1), y}, color, thickness, cv::LINE_AA);
}

static void drawBranch(cv::Mat & canvas,
                       cv::Point p_near, cv::Point p_mid, cv::Point * p_far,
                       int x_lo, int x_hi, int lw)
{
    cv::line(canvas, p_near, p_mid, _PRED, lw, cv::LINE_AA);
    if (!p_far) return;

    cv::Point2d _p2(p_near), _p1(p_mid), _p0(*p_far);
    cv::Point2d seg   = _p1 - _p2;
    double seg_len    = cv::norm(seg);
    cv::Point2d tdir_s = seg_len > 0 ? seg / seg_len : cv::Point2d(0, -1);
    cv::Point2d chord  = _p0 - _p1;
    double chord_len   = cv::norm(chord);
    cv::Point2d chord_hat = chord_len > 0 ? chord / chord_len : tdir_s;
    double dot    = tdir_s.x*chord_hat.x + tdir_s.y*chord_hat.y;
    cv::Point2d tdir_e = 2.0*dot*chord_hat - tdir_s;
    double alpha  = chord_len / 3.0;

    // Swing budget: constrain lateral excursion to 0.5 × vertical span
    double swing = 0.5 * (std::abs(_p1.y - _p2.y) + std::abs(_p0.y - _p1.y));
    double pts_x_min = std::min({_p2.x, _p1.x, _p0.x});
    double pts_x_max = std::max({_p2.x, _p1.x, _p0.x});
    x_lo = static_cast<int>(std::max((double)x_lo, pts_x_min - swing));
    x_hi = static_cast<int>(std::min((double)x_hi, pts_x_max + swing));

    int n_pts = std::max(20, static_cast<int>(chord_len));
    std::vector<cv::Point> pts;
    for (int i = 0; i <= n_pts; ++i) {
        double t = static_cast<double>(i) / n_pts;
        double mt = 1.0 - t;
        cv::Point2d p = mt*mt*mt*_p1
                      + 3*mt*mt*t*(_p1 + alpha*tdir_s)
                      + 3*mt*t*t*(_p0 - alpha*tdir_e)
                      + t*t*t*_p0;
        int x = std::clamp(static_cast<int>(std::round(p.x)), x_lo, x_hi);
        int y = std::clamp(static_cast<int>(std::round(p.y)), 0, canvas.rows-1);
        pts.push_back({x, y});
    }
    std::vector<std::vector<cv::Point>> contours{pts};
    cv::polylines(canvas, contours, false, _PRED, lw, cv::LINE_AA);
}

// ── ROS2 node ──────────────────────────────────────────────────────────────────
class RoadCenterlineNode : public rclcpp::Node
{
public:
    explicit RoadCenterlineNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
        : Node("road_centerline_node", options)
    {
        declare_parameter("model_path",          "road_model.pt");
        declare_parameter("config_path",         "road_model_config.yaml");
        declare_parameter("image_topic",         "/camera/image_raw");
        declare_parameter("output_topic",        "road_centerline");
        declare_parameter("visual_topic",        "road_centerline/visual");
        declare_parameter("road_threshold",      0.5);
        declare_parameter("min_peak_conf",       0.2);
        declare_parameter("cluster_thresh",      0.45);
        declare_parameter("debug_mode",          false);
        declare_parameter("min_display_height",  480);

        road_threshold_     = get_parameter("road_threshold").as_double();
        min_peak_conf_      = get_parameter("min_peak_conf").as_double();
        cluster_thresh_     = get_parameter("cluster_thresh").as_double();
        debug_mode_         = get_parameter("debug_mode").as_bool();
        min_display_height_ = get_parameter("min_display_height").as_int();

        cfg_ = loadConfig(get_parameter("config_path").as_string());
        RCLCPP_INFO(get_logger(), "Model config: %dx%d  buckets=%d  rows=%d",
            cfg_.image_width, cfg_.image_height, cfg_.n_buckets, cfg_.n_rows);

        model_ = torch::jit::load(get_parameter("model_path").as_string(), torch::kCPU);
        model_.eval();

        norm_mean_ = torch::tensor(cfg_.norm_mean).to(torch::kFloat32).reshape({1,3,1,1});
        norm_std_  = torch::tensor(cfg_.norm_std).to(torch::kFloat32).reshape({1,3,1,1});

        auto image_topic  = get_parameter("image_topic").as_string();
        auto output_topic = get_parameter("output_topic").as_string();
        auto visual_topic = get_parameter("visual_topic").as_string();

        image_sub_ = create_subscription<sensor_msgs::msg::Image>(
            image_topic, 10,
            std::bind(&RoadCenterlineNode::imageCallback, this, std::placeholders::_1));
        result_pub_ = create_publisher<road_centerline::msg::CenterlineResult>(output_topic, 10);
        visual_pub_ = create_publisher<sensor_msgs::msg::Image>(visual_topic, 10);

        RCLCPP_INFO(get_logger(), "Listening on '%s'", image_topic.c_str());
        RCLCPP_INFO(get_logger(), "Publishing results on '%s'", output_topic.c_str());
        RCLCPP_INFO(get_logger(), "Publishing visuals on '%s'", visual_topic.c_str());
    }

private:
    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        // Convert to BGR OpenCV mat
        cv::Mat bgr;
        try {
            bgr = cv_bridge::toCvCopy(msg, "bgr8")->image;
        } catch (const cv_bridge::Exception & e) {
            RCLCPP_WARN(get_logger(), "cv_bridge: %s", e.what());
            return;
        }

        // Resize to model input
        cv::Mat resized;
        cv::resize(bgr, resized,
                   cv::Size(cfg_.image_width, cfg_.image_height), 0, 0, cv::INTER_LINEAR);

        // BGR uint8 → RGB float [0,1] → tensor [1,3,H,W]
        cv::Mat rgb_f;
        cv::cvtColor(resized, rgb_f, cv::COLOR_BGR2RGB);
        rgb_f.convertTo(rgb_f, CV_32F, 1.0/255.0);

        std::vector<cv::Mat> ch(3);
        cv::split(rgb_f, ch);
        torch::Tensor tensor = torch::zeros({1,3,cfg_.image_height,cfg_.image_width}, torch::kFloat32);
        for (int c = 0; c < 3; ++c)
            tensor[0][c] = torch::from_blob(ch[c].data, {cfg_.image_height,cfg_.image_width}, torch::kFloat32).clone();
        tensor = (tensor - norm_mean_) / norm_std_;

        // Forward pass
        torch::jit::IValue output;
        {
            torch::NoGradGuard ng;
            output = model_.forward({tensor});
        }
        auto elems      = output.toTuple()->elements();
        auto cls_logits = elems[0].toTensor().squeeze(0);
        std::vector<torch::Tensor> row_logits(3);
        for (int r = 0; r < 3; ++r)
            row_logits[r] = elems[r+1].toTensor().squeeze(0);

        float road_prob   = torch::softmax(cls_logits, 0)[1].item<float>();
        bool  road_present = road_prob > road_threshold_;

        // Per-row peak detection
        std::vector<Peak>              peaks(3);
        std::vector<std::vector<float>> row_probs(3);
        float conf_sum = 0.0f;
        for (int r = 0; r < 3; ++r) {
            auto sig = torch::sigmoid(row_logits[r]);
            row_probs[r].assign(sig.data_ptr<float>(),
                                 sig.data_ptr<float>() + cfg_.n_buckets);
            peaks[r] = findPrimaryPeak(row_probs[r], cfg_.bucket_edges[r],
                                        static_cast<float>(cluster_thresh_));
            conf_sum += peaks[r].conf;
        }
        float mean_conf = conf_sum / 3.0f;
        if (mean_conf < min_peak_conf_) road_present = false;

        // ── Publish CenterlineResult ──────────────────────────────────────────
        road_centerline::msg::CenterlineResult result;
        result.header          = msg->header;
        result.road_present    = road_present;
        result.road_confidence = road_prob;
        result.peak_confidence = mean_conf;
        for (int r = 0; r < 3; ++r) result.row_confidences[r] = peaks[r].conf;
        if (road_present) {
            for (int r = 2; r >= 0; --r) {
                geometry_msgs::msg::Point32 pt;
                pt.x = static_cast<float>(peaks[r].cx_frac);
                pt.y = static_cast<float>(cfg_.row_fractions[r]);
                pt.z = peaks[r].conf;
                result.points.push_back(pt);
            }
        }
        result_pub_->publish(result);

        // ── Build and publish visual ──────────────────────────────────────────
        if (visual_pub_->get_subscription_count() > 0) {
            auto vis = buildVisual(resized, row_logits, row_probs, peaks,
                                   road_present, road_prob, mean_conf,
                                   std::vector<float>{peaks[0].conf, peaks[1].conf, peaks[2].conf});
            std_msgs::msg::Header hdr = msg->header;
            auto vis_msg = cv_bridge::CvImage(hdr, "bgr8", vis).toImageMsg();
            visual_pub_->publish(*vis_msg);
        }
    }

    // ── Visual overlay builder ────────────────────────────────────────────────
    cv::Mat buildVisual(const cv::Mat                  & src_bgr,
                        const std::vector<torch::Tensor> & row_logits,
                        const std::vector<std::vector<float>> & row_probs,
                        const std::vector<Peak>          & peaks,
                        bool   road_present,
                        float  road_prob,
                        float  mean_conf,
                        const std::vector<float>         & row_confs)
    {
        // Upscale small images to min_display_height
        cv::Mat canvas = src_bgr.clone();
        if (canvas.rows < min_display_height_) {
            double up = static_cast<double>(min_display_height_) / canvas.rows;
            cv::resize(canvas, canvas,
                       cv::Size(static_cast<int>(canvas.cols*up), min_display_height_),
                       0, 0, cv::INTER_LINEAR);
        }
        int h = canvas.rows, w = canvas.cols;

        // Scale factors (reference = 480px tall)
        double s    = h / 480.0;
        int lw      = std::max(1, static_cast<int>(std::round(2*s)));
        int cr      = std::max(2, static_cast<int>(std::round(6*s)));
        int dot_r   = std::max(1, static_cast<int>(std::round(2*s)));
        double fs   = std::max(0.4, 0.7*s);
        double fs_l = std::max(0.4, 0.8*s);
        int lbl_h   = std::max(18, static_cast<int>(std::round(36*s)));
        int lbl_y   = std::max(12, static_cast<int>(std::round(26*s)));
        int border  = std::max(3,  static_cast<int>(std::round(8*s)));
        int dash    = std::max(4,  static_cast<int>(std::round(8*s)));

        // Bucket edge → pixel x
        auto bucketCx = [&](int r, int b) -> int {
            return static_cast<int>(
                (cfg_.bucket_edges[r][b] + cfg_.bucket_edges[r][b+1]) * 0.5 * w);
        };
        auto edgePx = [&](int r, int b) -> int {
            return static_cast<int>(cfg_.bucket_edges[r][b] * w);
        };

        // Guide dashed lines
        for (int r = 0; r < cfg_.n_rows; ++r) {
            int row_y = static_cast<int>(cfg_.row_fractions[r] * h);
            dashedHLine(canvas, row_y, _GUIDE, dash, lw);
        }

        // Debug sigmoid bars
        if (debug_mode_) {
            cv::Mat dbg = canvas.clone();
            int bar_max_h = std::max(2, h/8);
            for (int r = 0; r < cfg_.n_rows; ++r) {
                int row_y = static_cast<int>(cfg_.row_fractions[r] * h);
                for (int b = 0; b < cfg_.n_buckets; ++b) {
                    float p = row_probs[r][b];
                    if (p < 0.01f) continue;
                    int bar_h = std::max(1, static_cast<int>(p * bar_max_h));
                    int x0 = edgePx(r, b);
                    int x1 = edgePx(r, b+1);
                    int y0 = std::max(0, row_y - bar_h);
                    cv::rectangle(dbg, {x0, y0}, {x1, row_y}, cv::Scalar(80,220,0), cv::FILLED);
                }
            }
            cv::addWeighted(dbg, 0.45, canvas, 0.55, 0, canvas);
        }

        if (road_present) {
            // Translucent peak rectangles
            cv::Mat overlay = canvas.clone();
            int rect_hh = std::max(2, h/30);
            for (int r = 0; r < cfg_.n_rows; ++r) {
                int cx  = static_cast<int>(peaks[r].cx_frac * w);
                int row_y = static_cast<int>(cfg_.row_fractions[r] * h);
                // find nearest bucket
                int best_b = 0;
                double best_d = 1e9;
                for (int b = 0; b < cfg_.n_buckets; ++b) {
                    double d = std::abs(bucketCx(r,b) - cx);
                    if (d < best_d) { best_d = d; best_b = b; }
                }
                int bx0 = edgePx(r, best_b);
                int bx1 = edgePx(r, best_b+1);
                cv::rectangle(overlay, {bx0, row_y-rect_hh}, {bx1, row_y+rect_hh},
                              _PRED, cv::FILLED);
            }
            cv::addWeighted(overlay, 0.35, canvas, 0.65, 0, canvas);

            // Convert peak fractions to pixel coords
            cv::Point p_near(static_cast<int>(peaks[2].cx_frac * w),
                              static_cast<int>(cfg_.row_fractions[2] * h));
            cv::Point p_mid (static_cast<int>(peaks[1].cx_frac * w),
                              static_cast<int>(cfg_.row_fractions[1] * h));
            cv::Point p_far (static_cast<int>(peaks[0].cx_frac * w),
                              static_cast<int>(cfg_.row_fractions[0] * h));
            drawBranch(canvas, p_near, p_mid, &p_far, 0, w-1, lw);

            // Peak circles + confidence labels
            for (int r = 0; r < cfg_.n_rows; ++r) {
                int cx    = static_cast<int>(peaks[r].cx_frac * w);
                int row_y = static_cast<int>(cfg_.row_fractions[r] * h);
                cv::circle(canvas, {cx, row_y}, cr,    _PRED,              cv::FILLED, cv::LINE_AA);
                cv::circle(canvas, {cx, row_y}, cr,    cv::Scalar(80,40,10), 1,         cv::LINE_AA);
                cv::circle(canvas, {cx, row_y}, dot_r, cv::Scalar(255,255,255), cv::FILLED, cv::LINE_AA);

                std::string txt = std::to_string(static_cast<int>(peaks[r].conf * 100)) + "%";
                int tx  = std::min(cx + std::max(2, static_cast<int>(4*s)), w - 40);
                int ty  = row_y - std::max(2, static_cast<int>(4*s));
                cv::putText(canvas, txt, {tx,ty}, cv::FONT_HERSHEY_PLAIN, fs,
                            cv::Scalar(0,0,0), 2, cv::LINE_AA);
                cv::putText(canvas, txt, {tx,ty}, cv::FONT_HERSHEY_PLAIN, fs,
                            _PRED,             1, cv::LINE_AA);
            }
        }

        // No-road red border
        if (!road_present) {
            canvas.rowRange(0, border)     = _RED;
            canvas.rowRange(h-border, h)   = _RED;
            canvas.colRange(0, border)     = _RED;
            canvas.colRange(w-border, w)   = _RED;
        }

        // Status bar
        cv::Scalar bar_col = road_present ? cv::Scalar(30,140,30) : cv::Scalar(40,40,160);
        cv::Mat label_bar(lbl_h, w, CV_8UC3, bar_col);
        std::string label;
        if (road_present)
            label = "ROAD " + std::to_string(static_cast<int>(road_prob*100)) + "%"
                  + "  conf:" + std::to_string(static_cast<int>(mean_conf*100)) + "%"
                  + "  [" + std::to_string(static_cast<int>(row_confs[0]*100))
                  + "/" + std::to_string(static_cast<int>(row_confs[1]*100))
                  + "/" + std::to_string(static_cast<int>(row_confs[2]*100)) + "%]";
        else
            label = "NO ROAD " + std::to_string(static_cast<int>((1-road_prob)*100)) + "%";

        cv::putText(label_bar, label, {std::max(2,static_cast<int>(4*s)), lbl_y},
                    cv::FONT_HERSHEY_PLAIN, fs_l, cv::Scalar(255,255,255),
                    std::max(1,static_cast<int>(std::round(1.5*s))), cv::LINE_AA);

        cv::Mat out;
        cv::vconcat(label_bar, canvas, out);
        return out;
    }

    // ── Members ───────────────────────────────────────────────────────────────
    ModelConfig cfg_;
    torch::jit::script::Module model_;
    torch::Tensor norm_mean_, norm_std_;

    double road_threshold_;
    double min_peak_conf_;
    double cluster_thresh_;
    bool   debug_mode_;
    int    min_display_height_;

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Publisher<road_centerline::msg::CenterlineResult>::SharedPtr result_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr visual_pub_;
};

static const std::string DEFAULT_PARAMS_FILE =
    "/home/rosey1211/code/road_code/v2/runtime_version/config/road_centerline_params.yaml";

int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);

    // If the user didn't supply --params-file on the command line, load the
    // default config so the node always has sensible values out of the box.
    bool has_params_file = false;
    for (int i = 0; i < argc; ++i) {
        if (std::string(argv[i]) == "--params-file") {
            has_params_file = true;
            break;
        }
    }

    rclcpp::NodeOptions options;
    if (!has_params_file && std::filesystem::exists(DEFAULT_PARAMS_FILE)) {
        options.arguments({"--ros-args", "--params-file", DEFAULT_PARAMS_FILE});
    }

    rclcpp::spin(std::make_shared<RoadCenterlineNode>(options));
    rclcpp::shutdown();
    return 0;
}
