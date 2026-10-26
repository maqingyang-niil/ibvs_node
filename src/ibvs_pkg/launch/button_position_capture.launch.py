import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    yolo_engine_path = LaunchConfiguration("yolo_engine_path")
    yolo_confidence_threshold = LaunchConfiguration("yolo_confidence_threshold")
    current_pose_topic = LaunchConfiguration("current_pose_topic")
    output_topic = LaunchConfiguration("output_topic")
    output_pose_topic = LaunchConfiguration("output_pose_topic")
    base_frame_id = LaunchConfiguration("base_frame_id")
    required_stable_detections = LaunchConfiguration("required_stable_detections")

    launch_arguments = [
        DeclareLaunchArgument(
            "yolo_engine_path",
            default_value=os.path.expanduser("~/best_fp16.engine"),
            description="YOLO红色按钮TensorRT engine路径",
        ),
        DeclareLaunchArgument(
            "yolo_confidence_threshold",
            default_value="0.5",
            description="YOLO红色按钮最小置信度",
        ),
        DeclareLaunchArgument(
            "current_pose_topic",
            default_value="/rm_driver/udp_arm_position",
            description="机械臂末端在基座坐标系中的当前位姿话题",
        ),
        DeclareLaunchArgument(
            "output_topic",
            default_value="/ibvs/button_position",
            description="一次性按钮基座坐标输出话题",
        ),
        DeclareLaunchArgument(
            "output_pose_topic",
            default_value="/ibvs/button_pose",
            description="一次性按钮基座6D位姿输出话题",
        ),
        DeclareLaunchArgument(
            "base_frame_id",
            default_value="base_link",
            description="按钮位置输出坐标系",
        ),
        DeclareLaunchArgument(
            "required_stable_detections",
            default_value="5",
            description="锁定按钮位置所需的连续稳定检测次数",
        ),
    ]

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

    capture_node = Node(
        package="ibvs_pkg",
        executable="button_position_capture_node",
        name="button_position_capture_node",
        output="screen",
        emulate_tty=True,
        parameters=[{
            "yolo_engine_path": ParameterValue(yolo_engine_path, value_type=str),
            "yolo_confidence_threshold": ParameterValue(yolo_confidence_threshold, value_type=float),
            "current_pose_topic": ParameterValue(current_pose_topic, value_type=str),
            "output_topic": ParameterValue(output_topic, value_type=str),
            "output_pose_topic": ParameterValue(output_pose_topic, value_type=str),
            "base_frame_id": ParameterValue(base_frame_id, value_type=str),
            "required_stable_detections": ParameterValue(required_stable_detections, value_type=int),
        }],
    )

    return LaunchDescription(launch_arguments + [realsense_launch, capture_node])
