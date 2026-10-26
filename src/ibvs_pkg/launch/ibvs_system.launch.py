import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    depth_scale = LaunchConfiguration("depth_scale")
    yolo_engine_path = LaunchConfiguration("yolo_engine_path")
    yolo_confidence_threshold = LaunchConfiguration("yolo_confidence_threshold")
    yolo_nms_threshold = LaunchConfiguration("yolo_nms_threshold")
    yolo_input_size = LaunchConfiguration("yolo_input_size")
    yolo_button_class_id = LaunchConfiguration("yolo_button_class_id")
    button_maximum_center_offset_ratio = LaunchConfiguration("button_maximum_center_offset_ratio")
    desired_center_u_ratio = LaunchConfiguration("desired_center_u_ratio")
    desired_center_v_ratio = LaunchConfiguration("desired_center_v_ratio")
    desired_button_radius_ratio = LaunchConfiguration("desired_button_radius_ratio")
    button_feature_offset_ratio = LaunchConfiguration("button_feature_offset_ratio")
    enable_velocity_transform = LaunchConfiguration("enable_velocity_transform")

    launch_arguments = [
        DeclareLaunchArgument(
            "depth_scale",
            default_value="0.001",
            description="16UC1深度值转换为米的比例",
        ),
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
            "yolo_nms_threshold",
            default_value="0.45",
            description="YOLO非极大值抑制阈值",
        ),
        DeclareLaunchArgument(
            "yolo_input_size",
            default_value="640",
            description="YOLO模型输入尺寸",
        ),
        DeclareLaunchArgument(
            "yolo_button_class_id",
            default_value="0",
            description="YOLO红色按钮类别编号",
        ),
        DeclareLaunchArgument(
            "button_maximum_center_offset_ratio",
            default_value="0.25",
            description="候选按钮相对期望中心的最大归一化偏移",
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
            "desired_button_radius_ratio",
            default_value="0.08",
            description="期望按钮半径相对图像短边的比例",
        ),
        DeclareLaunchArgument(
            "button_feature_offset_ratio",
            default_value="0.35",
            description="四个IBVS特征点相对按钮半径的偏移比例",
        ),
        DeclareLaunchArgument(
            "enable_velocity_transform",
            default_value="false",
            description="是否启动相机速度到工具速度的转换节点",
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
            "depth_scale": ParameterValue(
                depth_scale,
                value_type=float,
            ),
            "yolo_engine_path": ParameterValue(
                yolo_engine_path,
                value_type=str,
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
            "yolo_button_class_id": ParameterValue(
                yolo_button_class_id,
                value_type=int,
            ),
            "button_maximum_center_offset_ratio": ParameterValue(
                button_maximum_center_offset_ratio,
                value_type=float,
            ),
            "desired_center_u_ratio": ParameterValue(
                desired_center_u_ratio,
                value_type=float,
            ),
            "desired_center_v_ratio": ParameterValue(
                desired_center_v_ratio,
                value_type=float,
            ),
            "desired_button_radius_ratio": ParameterValue(
                desired_button_radius_ratio,
                value_type=float,
            ),
            "button_feature_offset_ratio": ParameterValue(
                button_feature_offset_ratio,
                value_type=float,
            ),
        }],
    )

    velocity_transform_node = Node(
        package="ibvs_pkg",
        executable="hand_velocity_node",
        name="hand_velocity_node",
        output="screen",
        emulate_tty=True,
        condition=IfCondition(enable_velocity_transform),
    )

    hand_delta_pose_node = Node(
        package="ibvs_pkg",
        executable="hand_delta_pose_node",
        name="hand_delta_pose_node",
        output="screen",
        emulate_tty=True,
        condition=IfCondition(enable_velocity_transform),
    )

    hand_new_pose_node = Node(
        package="ibvs_pkg",
        executable="hand_new_pose_node",
        name="hand_new_pose_node",
        output="screen",
        emulate_tty=True,
        condition=IfCondition(enable_velocity_transform),
    )

    return LaunchDescription(launch_arguments + [
        realsense_launch,
        ibvs_node,
        velocity_transform_node,
        hand_delta_pose_node,
        hand_new_pose_node,
    ])
