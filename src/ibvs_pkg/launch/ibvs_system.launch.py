import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    EnvironmentVariable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    workspace_root = LaunchConfiguration("workspace_root")
    yolo_model_path = LaunchConfiguration("yolo_model_path")
    depth_scale = LaunchConfiguration("depth_scale")
    yolo_confidence_threshold = LaunchConfiguration(
        "yolo_confidence_threshold"
    )
    yolo_nms_threshold = LaunchConfiguration("yolo_nms_threshold")
    yolo_input_size = LaunchConfiguration("yolo_input_size")
    banana_class_id = LaunchConfiguration("banana_class_id")
    desired_center_u_ratio = LaunchConfiguration("desired_center_u_ratio")
    desired_center_v_ratio = LaunchConfiguration("desired_center_v_ratio")
    desired_box_width_ratio = LaunchConfiguration("desired_box_width_ratio")
    desired_box_height_ratio = LaunchConfiguration("desired_box_height_ratio")

    launch_arguments = [
        DeclareLaunchArgument(
            "workspace_root",
            default_value=PathJoinSubstitution([
                EnvironmentVariable("HOME"),
                "ibvs_node",
            ]),
            description="ibvs_node工作区根目录",
        ),
        DeclareLaunchArgument(
            "yolo_model_path",
            default_value=PathJoinSubstitution([
                workspace_root,
                "models",
                "yolov8n.onnx",
            ]),
            description="YOLO ONNX模型路径",
        ),
        DeclareLaunchArgument(
            "depth_scale",
            default_value="0.001",
            description="16UC1深度值转换为米的比例",
        ),
        DeclareLaunchArgument(
            "yolo_confidence_threshold",
            default_value="0.5",
            description="YOLO香蕉检测置信度阈值",
        ),
        DeclareLaunchArgument(
            "yolo_nms_threshold",
            default_value="0.45",
            description="YOLO非极大值抑制阈值",
        ),
        DeclareLaunchArgument(
            "yolo_input_size",
            default_value="640",
            description="YOLO方形输入图像尺寸",
        ),
        DeclareLaunchArgument(
            "banana_class_id",
            default_value="46",
            description="COCO数据集中banana类别编号",
        ),
        DeclareLaunchArgument(
            "desired_center_u_ratio",
            default_value="0.5",
            description="期望检测框中心横坐标相对图像宽度的比例",
        ),
        DeclareLaunchArgument(
            "desired_center_v_ratio",
            default_value="0.5",
            description="期望检测框中心纵坐标相对图像高度的比例",
        ),
        DeclareLaunchArgument(
            "desired_box_width_ratio",
            default_value="0.4",
            description="期望检测框宽度相对图像宽度的比例",
        ),
        DeclareLaunchArgument(
            "desired_box_height_ratio",
            default_value="0.4",
            description="期望检测框高度相对图像高度的比例",
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

    ibvs_node = Node(
        package="ibvs_pkg",
        executable="ibvs_node",
        name="ibvs_node",
        output="screen",
        emulate_tty=True,
        parameters=[{
            "yolo_model_path": yolo_model_path,
            "depth_scale": ParameterValue(
                depth_scale,
                value_type=float,
            ),
            "yolo_confidence_threshold": ParameterValue(
                yolo_confidence_threshold,
                value_type=float,
            ),
            "yolo_nms_threshold": ParameterValue(
                yolo_nms_threshold,
                value_type=float,
            ),
            "yolo_input_size": ParameterValue(
                yolo_input_size,
                value_type=int,
            ),
            "banana_class_id": ParameterValue(
                banana_class_id,
                value_type=int,
            ),
            "desired_center_u_ratio": ParameterValue(
                desired_center_u_ratio,
                value_type=float,
            ),
            "desired_center_v_ratio": ParameterValue(
                desired_center_v_ratio,
                value_type=float,
            ),
            "desired_box_width_ratio": ParameterValue(
                desired_box_width_ratio,
                value_type=float,
            ),
            "desired_box_height_ratio": ParameterValue(
                desired_box_height_ratio,
                value_type=float,
            ),
        }],
    )

    return LaunchDescription(launch_arguments + [
        realsense_launch,
        ibvs_node,
    ])
