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
    output_pose_topic = LaunchConfiguration("output_pose_topic")
    output_pose_link6_topic = LaunchConfiguration("output_pose_link6_topic")
    output_pose_camera_topic = LaunchConfiguration("output_pose_camera_topic")
    base_frame_id = LaunchConfiguration("base_frame_id")
    link6_frame_id = LaunchConfiguration("link6_frame_id")
    camera_frame_id = LaunchConfiguration("camera_frame_id")
    required_stable_detections = LaunchConfiguration("required_stable_detections")
    trigger_topic = LaunchConfiguration("trigger_topic")
    status_topic = LaunchConfiguration("status_topic")
    trigger_settle_duration = LaunchConfiguration("trigger_settle_duration")
    detection_timeout = LaunchConfiguration("detection_timeout")

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
            "output_pose_topic",
            default_value="/ibvs/button_pose",
            description="一次性按钮基座6D位姿输出话题",
        ),
        DeclareLaunchArgument(
            "output_pose_link6_topic",
            default_value="/ibvs/button_pose_link6",
            description="一次性按钮link6坐标系6D位姿输出话题",
        ),
        DeclareLaunchArgument(
            "output_pose_camera_topic",
            default_value="/ibvs/button_pose_camera",
            description="一次性按钮相机坐标系6D位姿输出话题",
        ),
        DeclareLaunchArgument(
            "base_frame_id",
            default_value="base_link",
            description="按钮位置输出坐标系",
        ),
        DeclareLaunchArgument(
            "link6_frame_id",
            default_value="link6",
            description="按钮末端相对位姿输出坐标系",
        ),
        DeclareLaunchArgument(
            "camera_frame_id",
            default_value="camera_color_optical_frame",
            description="按钮相机相对位姿输出坐标系",
        ),
        DeclareLaunchArgument(
            "required_stable_detections",
            default_value="5",
            description="锁定按钮位置所需的连续稳定检测次数",
        ),
        DeclareLaunchArgument(
            "trigger_topic",
            default_value="/ibvs/capture_trigger",
            description="收到std_msgs/Bool(true)后执行一次按钮检测",
        ),
        DeclareLaunchArgument(
            "status_topic",
            default_value="/ibvs/capture_success",
            description="单次检测结果（true=成功，false=超时失败）",
        ),
        DeclareLaunchArgument(
            "trigger_settle_duration",
            default_value="0.5",
            description="触发后等待机械臂停稳的秒数，期间不推理",
        ),
        DeclareLaunchArgument(
            "detection_timeout",
            default_value="10.0",
            description="触发后未锁定按钮位姿的超时秒数",
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
            "output_pose_topic": ParameterValue(output_pose_topic, value_type=str),
            "output_pose_link6_topic": ParameterValue(output_pose_link6_topic, value_type=str),
            "output_pose_camera_topic": ParameterValue(output_pose_camera_topic, value_type=str),
            "base_frame_id": ParameterValue(base_frame_id, value_type=str),
            "link6_frame_id": ParameterValue(link6_frame_id, value_type=str),
            "camera_frame_id": ParameterValue(camera_frame_id, value_type=str),
            "required_stable_detections": ParameterValue(required_stable_detections, value_type=int),
            "trigger_topic": ParameterValue(trigger_topic, value_type=str),
            "status_topic": ParameterValue(status_topic, value_type=str),
            "trigger_settle_duration": ParameterValue(trigger_settle_duration, value_type=float),
            "detection_timeout": ParameterValue(detection_timeout, value_type=float),
        }],
    )

    return LaunchDescription(launch_arguments + [realsense_launch, capture_node])
