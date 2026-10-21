import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    depth_scale = LaunchConfiguration("depth_scale")
    button_minimum_saturation = LaunchConfiguration("button_minimum_saturation")
    button_minimum_value = LaunchConfiguration("button_minimum_value")
    button_minimum_area_ratio = LaunchConfiguration("button_minimum_area_ratio")
    button_maximum_area_ratio = LaunchConfiguration("button_maximum_area_ratio")
    button_minimum_circularity = LaunchConfiguration("button_minimum_circularity")
    button_maximum_center_offset_ratio = LaunchConfiguration("button_maximum_center_offset_ratio")
    button_minimum_detection_score = LaunchConfiguration("button_minimum_detection_score")
    desired_center_u_ratio = LaunchConfiguration("desired_center_u_ratio")
    desired_center_v_ratio = LaunchConfiguration("desired_center_v_ratio")
    desired_button_radius_ratio = LaunchConfiguration("desired_button_radius_ratio")
    button_feature_offset_ratio = LaunchConfiguration("button_feature_offset_ratio")

    launch_arguments = [
        DeclareLaunchArgument(
            "depth_scale",
            default_value="0.001",
            description="16UC1深度值转换为米的比例",
        ),
        DeclareLaunchArgument(
            "button_minimum_saturation",
            default_value="90",
            description="红色按钮最小HSV饱和度",
        ),
        DeclareLaunchArgument(
            "button_minimum_value",
            default_value="50",
            description="红色按钮最小HSV亮度",
        ),
        DeclareLaunchArgument(
            "button_minimum_area_ratio",
            default_value="0.0005",
            description="红色按钮轮廓最小面积比例",
        ),
        DeclareLaunchArgument(
            "button_maximum_area_ratio",
            default_value="0.08",
            description="红色按钮轮廓最大面积比例",
        ),
        DeclareLaunchArgument(
            "button_minimum_circularity",
            default_value="0.65",
            description="红色按钮轮廓最小圆度",
        ),
        DeclareLaunchArgument(
            "button_maximum_center_offset_ratio",
            default_value="0.25",
            description="候选按钮相对期望中心的最大归一化偏移",
        ),
        DeclareLaunchArgument(
            "button_minimum_detection_score",
            default_value="0.65",
            description="红色按钮最小综合检测得分",
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
            "button_minimum_saturation": ParameterValue(
                button_minimum_saturation,
                value_type=int,
            ),
            "button_minimum_value": ParameterValue(
                button_minimum_value,
                value_type=int,
            ),
            "button_minimum_area_ratio": ParameterValue(
                button_minimum_area_ratio,
                value_type=float,
            ),
            "button_maximum_area_ratio": ParameterValue(
                button_maximum_area_ratio,
                value_type=float,
            ),
            "button_minimum_circularity": ParameterValue(
                button_minimum_circularity,
                value_type=float,
            ),
            "button_maximum_center_offset_ratio": ParameterValue(
                button_maximum_center_offset_ratio,
                value_type=float,
            ),
            "button_minimum_detection_score": ParameterValue(
                button_minimum_detection_score,
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

    return LaunchDescription(launch_arguments + [
        realsense_launch,
        ibvs_node,
    ])
