import os
from launch import LaunchDescription
from launch_ros.actions import Node

TORCH_LIB = "/home/rosey1211/pytorch_env/venv/lib/python3.12/site-packages/torch/lib"

def generate_launch_description():
    env = os.environ.copy()
    existing = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{TORCH_LIB}:{existing}" if existing else TORCH_LIB

    return LaunchDescription([

        Node(
            package='usb_cam',
            executable='usb_cam_node_exe',
            name='usb_cam',
            remappings=[('/image_raw', '/camera/image_raw')],
            parameters=[{
                'video_device': '/dev/video0',
            }],
        ),

        Node(
            package='road_centerline',
            executable='road_centerline_node',
            name='road_centerline_node',
            additional_env={"LD_LIBRARY_PATH": env["LD_LIBRARY_PATH"]},
        ),

        Node(
            package='rqt_image_view',
            executable='rqt_image_view',
            name='rqt_image_view',
            arguments=['/road_centerline/visual'],
        ),

    ])
