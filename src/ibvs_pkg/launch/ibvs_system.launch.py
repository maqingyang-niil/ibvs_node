import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("realsense2_camera"),
                "launch",
                "rs_launch.py",
            )
        ),
        launch_arguments={
            "align_depth.enable": "true",
        }.items(),
    )

    ibvs_node = Node(
        package="ibvs_pkg",
        executable="ibvs_node",
        name="ibvs_node",
        output="screen",
        emulate_tty=True,
    )

    return LaunchDescription([
        realsense_launch,
        ibvs_node,
    ])
