#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/bool.hpp>

#include "yolo_red_button_detector.hpp"

class ButtonPositionCaptureNode : public rclcpp::Node {
  public:
    using Image = sensor_msgs::msg::Image;
    using Pose = geometry_msgs::msg::Pose;
    using PoseStamped = geometry_msgs::msg::PoseStamped;
    using SyncPolicy = message_filters::sync_policies::ApproximateTime<Image, Image>;

    ButtonPositionCaptureNode() : Node("button_position_capture_node") {
        const std::string color_topic = this->declare_parameter<std::string>("color_topic", "/camera/camera/color/image_raw");
        const std::string depth_topic =
            this->declare_parameter<std::string>("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw");
        const std::string camera_info_topic =
            this->declare_parameter<std::string>("camera_info_topic", "/camera/camera/color/camera_info");
        const std::string engine_path = this->declare_parameter<std::string>("yolo_engine_path", "");
        current_pose_topic_ = this->declare_parameter<std::string>("current_pose_topic", "/rm_driver/udp_arm_position");
        output_pose_topic_ = this->declare_parameter<std::string>("output_pose_topic", "/ibvs/button_pose");
        output_pose_link6_topic_ =
            this->declare_parameter<std::string>("output_pose_link6_topic", "/ibvs/button_pose_link6");
        output_pose_camera_topic_ =
            this->declare_parameter<std::string>("output_pose_camera_topic", "/ibvs/button_pose_camera");
        base_frame_id_ = this->declare_parameter<std::string>("base_frame_id", "base_link");
        link6_frame_id_ = this->declare_parameter<std::string>("link6_frame_id", "link6");
        camera_frame_id_ = this->declare_parameter<std::string>("camera_frame_id", "camera_color_optical_frame");

        depth_scale_ = this->declare_parameter<double>("depth_scale", 0.001);
        const float confidence_threshold = static_cast<float>(this->declare_parameter<double>("yolo_confidence_threshold", 0.5));
        const float nms_threshold = static_cast<float>(this->declare_parameter<double>("yolo_nms_threshold", 0.45));
        const int input_size = this->declare_parameter<int>("yolo_input_size", 640);
        const int button_class_id = this->declare_parameter<int>("yolo_button_class_id", 0);
        const auto rotation_values = this->declare_parameter<std::vector<double>>(
            "camera_to_tool_rotation",
            {0.99994390, -0.00776698, 0.00720273, 0.00782760, 0.99993386, -0.00842698, -0.00713680, 0.00848289, 0.99993855});
        const auto translation_values =
            this->declare_parameter<std::vector<double>>("camera_to_tool_translation", {-0.03669205, -0.09531331, 0.02498297});
        current_pose_timeout_ = this->declare_parameter<double>("current_pose_timeout", 0.5);
        required_stable_detections_ = this->declare_parameter<int>("required_stable_detections", 5);
        trigger_topic_ = this->declare_parameter<std::string>("trigger_topic", "/ibvs/capture_trigger");
        status_topic_ = this->declare_parameter<std::string>("status_topic", "/ibvs/capture_success");
        trigger_settle_duration_ = this->declare_parameter<double>("trigger_settle_duration", 0.5);
        detection_timeout_ = this->declare_parameter<double>("detection_timeout", 10.0);
        maximum_pixel_shift_ = this->declare_parameter<double>("maximum_pixel_shift", 3.0);
        maximum_depth_change_ = this->declare_parameter<double>("maximum_depth_change", 0.005);
        maximum_position_change_ = this->declare_parameter<double>("maximum_position_change", 0.01);
        constexpr double radians_per_degree = 0.017453292519943295;
        maximum_orientation_change_ =
            this->declare_parameter<double>("maximum_orientation_change_degrees", 5.0) * radians_per_degree;
        orientation_sample_radius_ratio_ = this->declare_parameter<double>("orientation_sample_radius_ratio", 0.35);
        orientation_depth_tolerance_ = this->declare_parameter<double>("orientation_depth_tolerance", 0.01);
        maximum_plane_residual_ = this->declare_parameter<double>("maximum_plane_residual", 0.003);
        minimum_orientation_samples_ = this->declare_parameter<int>("minimum_orientation_samples", 30);

        if (output_pose_topic_.empty() || output_pose_link6_topic_.empty() || output_pose_camera_topic_.empty() || base_frame_id_.empty() ||
            link6_frame_id_.empty() || camera_frame_id_.empty() || trigger_topic_.empty() || status_topic_.empty()) {
            throw std::invalid_argument("输出话题、触发话题、状态话题和坐标系名称不能为空");
        }
        if (rotation_values.size() != 9 || translation_values.size() != 3) {
            throw std::invalid_argument("手眼外参必须包含9个旋转元素和3个平移元素");
        }
        if (!std::isfinite(current_pose_timeout_) || current_pose_timeout_ <= 0.0 || required_stable_detections_ <= 0 ||
            !std::isfinite(maximum_pixel_shift_) || maximum_pixel_shift_ < 0.0 || !std::isfinite(maximum_depth_change_) ||
            maximum_depth_change_ < 0.0 || !std::isfinite(maximum_position_change_) || maximum_position_change_ < 0.0 ||
            !std::isfinite(maximum_orientation_change_) || maximum_orientation_change_ <= 0.0 ||
            !std::isfinite(orientation_sample_radius_ratio_) || orientation_sample_radius_ratio_ <= 0.0 ||
            orientation_sample_radius_ratio_ > 0.5 || !std::isfinite(orientation_depth_tolerance_) || orientation_depth_tolerance_ <= 0.0 ||
            !std::isfinite(maximum_plane_residual_) || maximum_plane_residual_ <= 0.0 || minimum_orientation_samples_ < 3 ||
            !std::isfinite(depth_scale_) || depth_scale_ <= 0.0 ||
            !std::isfinite(trigger_settle_duration_) || trigger_settle_duration_ < 0.0 ||
            !std::isfinite(detection_timeout_) || detection_timeout_ <= 0.0) {
            throw std::invalid_argument("一次性按钮位置确认参数无效");
        }

        camera_to_tool_rotation_ << rotation_values[0], rotation_values[1], rotation_values[2], rotation_values[3], rotation_values[4],
            rotation_values[5], rotation_values[6], rotation_values[7], rotation_values[8];
        camera_to_tool_translation_ << translation_values[0], translation_values[1], translation_values[2];
        const double orthogonality_error =
            (camera_to_tool_rotation_.transpose() * camera_to_tool_rotation_ - Eigen::Matrix3d::Identity()).norm();
        if (!camera_to_tool_rotation_.allFinite() || !camera_to_tool_translation_.allFinite() || orthogonality_error > 1.0e-3 ||
            std::abs(camera_to_tool_rotation_.determinant() - 1.0) > 1.0e-3) {
            throw std::invalid_argument("camera_to_tool_rotation不是有效的旋转矩阵");
        }

        detector_ = std::make_unique<YoloRedButtonDetector>(engine_path, depth_scale_, confidence_threshold, nms_threshold, input_size,
                                                             button_class_id);

        auto output_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();
        pose_pub_ = this->create_publisher<PoseStamped>(output_pose_topic_, output_qos);
        link6_pose_pub_ = this->create_publisher<PoseStamped>(output_pose_link6_topic_, output_qos);
        camera_pose_pub_ = this->create_publisher<PoseStamped>(output_pose_camera_topic_, output_qos);
        status_pub_ = this->create_publisher<std_msgs::msg::Bool>(status_topic_, output_qos);
        trigger_sub_ = this->create_subscription<std_msgs::msg::Bool>(
            trigger_topic_, rclcpp::QoS(10).reliable(),
            std::bind(&ButtonPositionCaptureNode::triggerCallback, this, std::placeholders::_1));
        timeout_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(100), std::bind(&ButtonPositionCaptureNode::timeoutCallback, this));
        current_pose_sub_ = this->create_subscription<Pose>(
            current_pose_topic_, rclcpp::SensorDataQoS(), std::bind(&ButtonPositionCaptureNode::currentPoseCallback, this, std::placeholders::_1));
        camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
            camera_info_topic, rclcpp::SensorDataQoS(), std::bind(&ButtonPositionCaptureNode::cameraInfoCallback, this, std::placeholders::_1));

        color_sub_.subscribe(this, color_topic, rmw_qos_profile_sensor_data);
        depth_sub_.subscribe(this, depth_topic, rmw_qos_profile_sensor_data);
        image_sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(SyncPolicy(10), color_sub_, depth_sub_);
        image_sync_->registerCallback(&ButtonPositionCaptureNode::imageCallback, this);

        RCLCPP_INFO(this->get_logger(),
                    "触发式按钮检测就绪：等待%s上的触发，收到后连续%d帧稳定即发布%s、%s和%s",
                    trigger_topic_.c_str(), required_stable_detections_, output_pose_topic_.c_str(),
                    output_pose_link6_topic_.c_str(), output_pose_camera_topic_.c_str());
        RCLCPP_INFO(this->get_logger(), "触发后先等待%.2fs让机械臂停稳，之后%.2fs内未锁定则发布%s=false并回到空闲",
                    trigger_settle_duration_, detection_timeout_, status_topic_.c_str());
        RCLCPP_WARN(this->get_logger(), "圆形按钮绕表面法线的旋转不可观测；按钮X轴按相机X轴在按钮平面上的投影定义");
    }

  private:
    enum class CaptureState { Idle, Settling, Detecting };

    struct PositionSample {
        cv::Point2f pixel;
        double depth{0.0};
        Eigen::Vector3d camera_position{Eigen::Vector3d::Zero()};
        Eigen::Quaterniond camera_orientation{Eigen::Quaterniond::Identity()};
        Eigen::Vector3d link6_position{Eigen::Vector3d::Zero()};
        Eigen::Quaterniond link6_orientation{Eigen::Quaterniond::Identity()};
        Eigen::Vector3d base_position{Eigen::Vector3d::Zero()};
        Eigen::Quaterniond base_orientation{Eigen::Quaterniond::Identity()};
    };

    static bool finitePose(const Pose &pose) {
        return std::isfinite(pose.position.x) && std::isfinite(pose.position.y) && std::isfinite(pose.position.z) &&
               std::isfinite(pose.orientation.x) && std::isfinite(pose.orientation.y) && std::isfinite(pose.orientation.z) &&
               std::isfinite(pose.orientation.w);
    }

    static double median(std::vector<double> values) {
        const std::size_t middle = values.size() / 2;
        std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(middle), values.end());
        if (values.size() % 2 == 1) { return values[middle]; }
        const double upper_middle = values[middle];
        const double lower_middle = *std::max_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(middle));
        return 0.5 * (lower_middle + upper_middle);
    }

    static Eigen::Vector3d medianPosition(const std::vector<Eigen::Vector3d> &positions) {
        std::vector<double> x_values;
        std::vector<double> y_values;
        std::vector<double> z_values;
        x_values.reserve(positions.size());
        y_values.reserve(positions.size());
        z_values.reserve(positions.size());
        for (const Eigen::Vector3d &position : positions) {
            x_values.push_back(position.x());
            y_values.push_back(position.y());
            z_values.push_back(position.z());
        }
        return Eigen::Vector3d(median(x_values), median(y_values), median(z_values));
    }

    static Eigen::Quaterniond averageOrientation(const std::vector<Eigen::Quaterniond> &orientations) {
        Eigen::Vector4d orientation_sum = Eigen::Vector4d::Zero();
        const Eigen::Quaterniond &orientation_reference = orientations.front();
        for (const Eigen::Quaterniond &orientation : orientations) {
            Eigen::Vector4d coefficients = orientation.coeffs();
            if (orientation_reference.dot(orientation) < 0.0) { coefficients = -coefficients; }
            orientation_sum += coefficients;
        }
        if (orientation_sum.norm() < 1.0e-9) { throw std::runtime_error("稳定姿态四元数平均失败"); }
        Eigen::Quaterniond average;
        average.coeffs() = orientation_sum.normalized();
        if (average.w() < 0.0) { average.coeffs() = -average.coeffs(); }
        return average;
    }

    static PoseStamped makePoseMessage(const builtin_interfaces::msg::Time &stamp, const std::string &frame_id,
                                       const Eigen::Vector3d &position, const Eigen::Quaterniond &orientation) {
        PoseStamped message;
        message.header.stamp = stamp;
        message.header.frame_id = frame_id;
        message.pose.position.x = position.x();
        message.pose.position.y = position.y();
        message.pose.position.z = position.z();
        message.pose.orientation.x = orientation.x();
        message.pose.orientation.y = orientation.y();
        message.pose.orientation.z = orientation.z();
        message.pose.orientation.w = orientation.w();
        return message;
    }

    double readDepthMeters(const cv::Mat &depth_image, int u, int v, const std::string &depth_encoding) const {
        if (depth_encoding == sensor_msgs::image_encodings::TYPE_16UC1) {
            if (depth_image.type() != CV_16UC1) { throw std::runtime_error("16UC1编码与cv::Mat类型不匹配"); }
            const std::uint16_t raw_depth = depth_image.at<std::uint16_t>(v, u);
            return raw_depth == 0 ? std::numeric_limits<double>::quiet_NaN() : static_cast<double>(raw_depth) * depth_scale_;
        }
        if (depth_encoding == sensor_msgs::image_encodings::TYPE_32FC1) {
            if (depth_image.type() != CV_32FC1) { throw std::runtime_error("32FC1编码与cv::Mat类型不匹配"); }
            return static_cast<double>(depth_image.at<float>(v, u));
        }
        throw std::runtime_error("不支持的深度图编码: " + depth_encoding);
    }

    std::optional<Eigen::Quaterniond> estimateCameraButtonOrientation(const cv::Mat &depth_image, const std::string &depth_encoding,
                                                                      const DetectionResult &detection,
                                                                      const Eigen::Vector3d &camera_position) const {
        const cv::Rect2f &box = detection.bounding_box;
        if (!std::isfinite(box.x) || !std::isfinite(box.y) || !std::isfinite(box.width) || !std::isfinite(box.height) || box.width <= 0.0F ||
            box.height <= 0.0F) {
            return std::nullopt;
        }

        const int center_u = static_cast<int>(std::lround(detection.center_pixel.x));
        const int center_v = static_cast<int>(std::lround(detection.center_pixel.y));
        const int sample_radius =
            std::max(2, static_cast<int>(std::lround(orientation_sample_radius_ratio_ * std::min(box.width, box.height))));
        const int minimum_u = std::max(0, center_u - sample_radius);
        const int maximum_u = std::min(depth_image.cols - 1, center_u + sample_radius);
        const int minimum_v = std::max(0, center_v - sample_radius);
        const int maximum_v = std::min(depth_image.rows - 1, center_v + sample_radius);

        std::vector<Eigen::Vector3d> points;
        points.reserve(static_cast<std::size_t>((2 * sample_radius + 1) * (2 * sample_radius + 1)));
        for (int v = minimum_v; v <= maximum_v; ++v) {
            for (int u = minimum_u; u <= maximum_u; ++u) {
                const int delta_u = u - center_u;
                const int delta_v = v - center_v;
                if (delta_u * delta_u + delta_v * delta_v > sample_radius * sample_radius) { continue; }

                const double depth = readDepthMeters(depth_image, u, v, depth_encoding);
                if (!std::isfinite(depth) || depth <= 0.0 || std::abs(depth - detection.depth) > orientation_depth_tolerance_) { continue; }
                points.emplace_back((static_cast<double>(u) - cx_) * depth / fx_, (static_cast<double>(v) - cy_) * depth / fy_, depth);
            }
        }
        if (points.size() < static_cast<std::size_t>(minimum_orientation_samples_)) { return std::nullopt; }

        Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
        for (const Eigen::Vector3d &point : points) { centroid += point; }
        centroid /= static_cast<double>(points.size());

        Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
        for (const Eigen::Vector3d &point : points) {
            const Eigen::Vector3d offset = point - centroid;
            covariance.noalias() += offset * offset.transpose();
        }
        covariance /= static_cast<double>(points.size());

        const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen_solver(covariance);
        if (eigen_solver.info() != Eigen::Success || !eigen_solver.eigenvalues().allFinite() ||
            eigen_solver.eigenvalues()[1] <= 1.0e-10 || std::sqrt(std::max(0.0, eigen_solver.eigenvalues()[0])) > maximum_plane_residual_) {
            return std::nullopt;
        }

        Eigen::Vector3d z_axis = eigen_solver.eigenvectors().col(0).normalized();
        if (z_axis.dot(camera_position) > 0.0) { z_axis = -z_axis; }
        Eigen::Vector3d x_axis = Eigen::Vector3d::UnitX() - z_axis * z_axis.x();
        if (x_axis.norm() < 1.0e-6) { x_axis = Eigen::Vector3d::UnitY() - z_axis * z_axis.y(); }
        x_axis.normalize();
        Eigen::Vector3d y_axis = z_axis.cross(x_axis).normalized();
        x_axis = y_axis.cross(z_axis).normalized();

        Eigen::Matrix3d camera_to_button_rotation;
        camera_to_button_rotation.col(0) = x_axis;
        camera_to_button_rotation.col(1) = y_axis;
        camera_to_button_rotation.col(2) = z_axis;
        Eigen::Quaterniond orientation(camera_to_button_rotation);
        orientation.normalize();
        if (!orientation.coeffs().allFinite()) { return std::nullopt; }
        return orientation;
    }

    void cameraInfoCallback(const sensor_msgs::msg::CameraInfo::ConstSharedPtr message) {
        if (camera_info_available_) { return; }
        fx_ = message->k[0];
        fy_ = message->k[4];
        cx_ = message->k[2];
        cy_ = message->k[5];
        if (!std::isfinite(fx_) || !std::isfinite(fy_) || !std::isfinite(cx_) || !std::isfinite(cy_) || fx_ <= 0.0 || fy_ <= 0.0) {
            RCLCPP_ERROR(this->get_logger(), "收到无效相机内参");
            return;
        }
        camera_info_available_ = true;
        RCLCPP_INFO(this->get_logger(), "相机内参已就绪：fx=%.3f fy=%.3f cx=%.3f cy=%.3f", fx_, fy_, cx_, cy_);
    }

    void currentPoseCallback(const Pose::ConstSharedPtr message) {
        if (!finitePose(*message)) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "收到无效机械臂当前位姿");
            return;
        }
        Eigen::Quaterniond orientation(message->orientation.w, message->orientation.x, message->orientation.y, message->orientation.z);
        const double norm = orientation.norm();
        if (!std::isfinite(norm) || norm < 1.0e-9) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "机械臂当前位姿的四元数无效");
            return;
        }
        orientation.normalize();
        base_to_tool_rotation_ = orientation.toRotationMatrix();
        base_to_tool_translation_ = Eigen::Vector3d(message->position.x, message->position.y, message->position.z);
        current_pose_update_time_ = std::chrono::steady_clock::now();
        current_pose_available_ = true;
    }

    void imageCallback(const Image::ConstSharedPtr &color_message, const Image::ConstSharedPtr &depth_message) {
        if (state_ != CaptureState::Detecting) { return; }
        if (!camera_info_available_) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "等待彩色相机内参");
            stable_samples_.clear();
            return;
        }
        if (!current_pose_available_ ||
            std::chrono::steady_clock::now() - current_pose_update_time_ > std::chrono::duration<double>(current_pose_timeout_)) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "等待新鲜的机械臂当前位姿：%s",
                                 current_pose_topic_.c_str());
            stable_samples_.clear();
            return;
        }

        try {
            const cv_bridge::CvImageConstPtr color_cv =
                cv_bridge::toCvShare(color_message, sensor_msgs::image_encodings::BGR8);
            const cv_bridge::CvImageConstPtr depth_cv = cv_bridge::toCvShare(depth_message);
            const DetectionResult detection = detector_->detect(color_cv->image, depth_cv->image, depth_message->encoding);
            if (!detection.success) {
                stable_samples_.clear();
                if (detection.detected) {
                    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "YOLO已检测到按钮，但按钮区域没有有效深度");
                } else {
                    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "YOLO尚未检测到目标按钮");
                }
                return;
            }

            const cv::Point2f center = detection.center_pixel;
            const double depth = detection.depth;
            if (!std::isfinite(center.x) || !std::isfinite(center.y) || !std::isfinite(depth) || depth <= 0.0) {
                stable_samples_.clear();
                return;
            }

            const Eigen::Vector3d camera_position((static_cast<double>(center.x) - cx_) * depth / fx_,
                                                  (static_cast<double>(center.y) - cy_) * depth / fy_, depth);
            const auto camera_button_orientation =
                estimateCameraButtonOrientation(depth_cv->image, depth_message->encoding, detection, camera_position);
            if (!camera_button_orientation.has_value()) {
                stable_samples_.clear();
                RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                     "YOLO和中心深度有效，但按钮表面深度点不足或平面拟合不稳定");
                return;
            }
            const Eigen::Vector3d link6_position = camera_to_tool_rotation_ * camera_position + camera_to_tool_translation_;
            Eigen::Quaterniond link6_orientation(camera_to_tool_rotation_ * camera_button_orientation->toRotationMatrix());
            link6_orientation.normalize();
            const Eigen::Vector3d base_position = base_to_tool_rotation_ * link6_position + base_to_tool_translation_;
            Eigen::Quaterniond base_orientation(base_to_tool_rotation_ * link6_orientation.toRotationMatrix());
            base_orientation.normalize();
            if (!camera_position.allFinite() || !link6_position.allFinite() || !base_position.allFinite() ||
                !camera_button_orientation->coeffs().allFinite() || !link6_orientation.coeffs().allFinite() ||
                !base_orientation.coeffs().allFinite()) {
                stable_samples_.clear();
                return;
            }

            const PositionSample sample{center,          depth,            camera_position, *camera_button_orientation,
                                        link6_position, link6_orientation, base_position,   base_orientation};
            if (!stable_samples_.empty()) {
                const PositionSample &previous = stable_samples_.back();
                const double pixel_shift = cv::norm(sample.pixel - previous.pixel);
                const double depth_change = std::abs(sample.depth - previous.depth);
                const double position_change = (sample.base_position - previous.base_position).norm();
                const double orientation_change = sample.base_orientation.angularDistance(previous.base_orientation);
                if (pixel_shift > maximum_pixel_shift_ || depth_change > maximum_depth_change_ || position_change > maximum_position_change_ ||
                    orientation_change > maximum_orientation_change_) {
                    stable_samples_.clear();
                }
            }
            stable_samples_.push_back(sample);
            RCLCPP_INFO(this->get_logger(), "按钮6D位姿稳定确认：%zu/%d，base=(%.4f, %.4f, %.4f)m", stable_samples_.size(),
                        required_stable_detections_, base_position.x(), base_position.y(), base_position.z());
            if (stable_samples_.size() < static_cast<std::size_t>(required_stable_detections_)) { return; }

            publishLockedPose(color_message->header.stamp);
        } catch (const cv_bridge::Exception &exception) {
            stable_samples_.clear();
            RCLCPP_ERROR(this->get_logger(), "cv_bridge转换失败：%s", exception.what());
        } catch (const std::exception &exception) {
            stable_samples_.clear();
            RCLCPP_ERROR(this->get_logger(), "按钮位置捕获失败：%s", exception.what());
        }
    }

    void publishLockedPose(const builtin_interfaces::msg::Time &stamp) {
        std::vector<Eigen::Vector3d> camera_positions;
        std::vector<Eigen::Vector3d> link6_positions;
        std::vector<Eigen::Vector3d> base_positions;
        std::vector<Eigen::Quaterniond> camera_orientations;
        std::vector<Eigen::Quaterniond> link6_orientations;
        std::vector<Eigen::Quaterniond> base_orientations;
        camera_positions.reserve(stable_samples_.size());
        link6_positions.reserve(stable_samples_.size());
        base_positions.reserve(stable_samples_.size());
        camera_orientations.reserve(stable_samples_.size());
        link6_orientations.reserve(stable_samples_.size());
        base_orientations.reserve(stable_samples_.size());
        for (const PositionSample &sample : stable_samples_) {
            camera_positions.push_back(sample.camera_position);
            link6_positions.push_back(sample.link6_position);
            base_positions.push_back(sample.base_position);
            camera_orientations.push_back(sample.camera_orientation);
            link6_orientations.push_back(sample.link6_orientation);
            base_orientations.push_back(sample.base_orientation);
        }

        const Eigen::Vector3d camera_position = medianPosition(camera_positions);
        const Eigen::Vector3d link6_position = medianPosition(link6_positions);
        const Eigen::Vector3d base_position = medianPosition(base_positions);
        const Eigen::Quaterniond camera_orientation = averageOrientation(camera_orientations);
        const Eigen::Quaterniond link6_orientation = averageOrientation(link6_orientations);
        const Eigen::Quaterniond base_orientation = averageOrientation(base_orientations);
        const Eigen::Matrix3d rotation = base_orientation.toRotationMatrix();
        constexpr double degrees_per_radian = 57.29577951308232;
        const double roll = std::atan2(rotation(2, 1), rotation(2, 2)) * degrees_per_radian;
        const double pitch = std::atan2(-rotation(2, 0), std::hypot(rotation(0, 0), rotation(1, 0))) * degrees_per_radian;
        const double yaw = std::atan2(rotation(1, 0), rotation(0, 0)) * degrees_per_radian;

        const PoseStamped pose_message = makePoseMessage(stamp, base_frame_id_, base_position, base_orientation);
        const PoseStamped link6_pose_message = makePoseMessage(stamp, link6_frame_id_, link6_position, link6_orientation);
        const PoseStamped camera_pose_message = makePoseMessage(stamp, camera_frame_id_, camera_position, camera_orientation);

        pose_pub_->publish(pose_message);
        link6_pose_pub_->publish(link6_pose_message);
        camera_pose_pub_->publish(camera_pose_message);
        publishStatus(true);
        state_ = CaptureState::Idle;
        stable_samples_.clear();
        RCLCPP_INFO(this->get_logger(),
                    "按钮6D位姿已发布，回到空闲等待下一次触发：%s position=(%.6f, %.6f, %.6f)m quaternion=(%.6f, %.6f, %.6f, %.6f) "
                    "RPY=(%.2f, %.2f, %.2f)deg",
                    base_frame_id_.c_str(), pose_message.pose.position.x, pose_message.pose.position.y, pose_message.pose.position.z,
                    base_orientation.x(), base_orientation.y(), base_orientation.z(), base_orientation.w(), roll, pitch, yaw);
        RCLCPP_INFO(this->get_logger(), "按钮link6位姿：%s position=(%.6f, %.6f, %.6f)m quaternion=(%.6f, %.6f, %.6f, %.6f)",
                    link6_frame_id_.c_str(), link6_position.x(), link6_position.y(), link6_position.z(), link6_orientation.x(),
                    link6_orientation.y(), link6_orientation.z(), link6_orientation.w());
        RCLCPP_INFO(this->get_logger(), "按钮相机位姿：%s position=(%.6f, %.6f, %.6f)m quaternion=(%.6f, %.6f, %.6f, %.6f)",
                    camera_frame_id_.c_str(), camera_position.x(), camera_position.y(), camera_position.z(), camera_orientation.x(),
                    camera_orientation.y(), camera_orientation.z(), camera_orientation.w());
    }

    void triggerCallback(const std_msgs::msg::Bool::ConstSharedPtr message) {
        if (!message->data) { return; }
        if (state_ == CaptureState::Detecting) {
            RCLCPP_WARN(this->get_logger(), "检测进行中，忽略本次触发");
            return;
        }
        state_ = CaptureState::Settling;
        settle_start_time_ = std::chrono::steady_clock::now();
        stable_samples_.clear();
        RCLCPP_INFO(this->get_logger(), "收到检测触发，等待%.2fs让机械臂停稳后开始推理", trigger_settle_duration_);
    }

    void timeoutCallback() {
        const auto now = std::chrono::steady_clock::now();
        if (state_ == CaptureState::Settling) {
            if (now - settle_start_time_ >= std::chrono::duration<double>(trigger_settle_duration_)) {
                state_ = CaptureState::Detecting;
                detection_start_time_ = now;
                stable_samples_.clear();
                RCLCPP_INFO(this->get_logger(), "机械臂稳定等待结束，开始按钮检测");
            }
            return;
        }
        if (state_ == CaptureState::Detecting &&
            now - detection_start_time_ >= std::chrono::duration<double>(detection_timeout_)) {
            publishStatus(false);
            state_ = CaptureState::Idle;
            stable_samples_.clear();
            RCLCPP_ERROR(this->get_logger(), "检测超时：%.2fs内未获得稳定的按钮6D位姿，回到空闲等待下一次触发",
                         detection_timeout_);
        }
    }

    void publishStatus(bool success) {
        std_msgs::msg::Bool message;
        message.data = success;
        status_pub_->publish(message);
        RCLCPP_INFO(this->get_logger(), "发布%s=%s", status_topic_.c_str(), success ? "true" : "false");
    }

    std::string trigger_topic_;
    std::string status_topic_;
    std::string current_pose_topic_;
    std::string output_pose_topic_;
    std::string output_pose_link6_topic_;
    std::string output_pose_camera_topic_;
    std::string base_frame_id_;
    std::string link6_frame_id_;
    std::string camera_frame_id_;
    double depth_scale_{0.001};
    double current_pose_timeout_{0.5};
    double trigger_settle_duration_{0.5};
    double detection_timeout_{10.0};
    int required_stable_detections_{5};
    double maximum_pixel_shift_{3.0};
    double maximum_depth_change_{0.005};
    double maximum_position_change_{0.01};
    double maximum_orientation_change_{0.08726646259971647};
    double orientation_sample_radius_ratio_{0.35};
    double orientation_depth_tolerance_{0.01};
    double maximum_plane_residual_{0.003};
    int minimum_orientation_samples_{30};
    double fx_{0.0};
    double fy_{0.0};
    double cx_{0.0};
    double cy_{0.0};
    bool camera_info_available_{false};
    bool current_pose_available_{false};
    CaptureState state_{CaptureState::Idle};
    std::chrono::steady_clock::time_point settle_start_time_{};
    std::chrono::steady_clock::time_point detection_start_time_{};
    std::chrono::steady_clock::time_point current_pose_update_time_{};
    Eigen::Matrix3d camera_to_tool_rotation_{Eigen::Matrix3d::Identity()};
    Eigen::Vector3d camera_to_tool_translation_{Eigen::Vector3d::Zero()};
    Eigen::Matrix3d base_to_tool_rotation_{Eigen::Matrix3d::Identity()};
    Eigen::Vector3d base_to_tool_translation_{Eigen::Vector3d::Zero()};
    std::vector<PositionSample> stable_samples_;
    std::unique_ptr<YoloRedButtonDetector> detector_;
    message_filters::Subscriber<Image> color_sub_;
    message_filters::Subscriber<Image> depth_sub_;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> image_sync_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
    rclcpp::Subscription<Pose>::SharedPtr current_pose_sub_;
    rclcpp::Publisher<PoseStamped>::SharedPtr pose_pub_;
    rclcpp::Publisher<PoseStamped>::SharedPtr link6_pose_pub_;
    rclcpp::Publisher<PoseStamped>::SharedPtr camera_pose_pub_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr status_pub_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr trigger_sub_;
    rclcpp::TimerBase::SharedPtr timeout_timer_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    try {
        rclcpp::spin(std::make_shared<ButtonPositionCaptureNode>());
    } catch (const std::exception &exception) {
        RCLCPP_FATAL(rclcpp::get_logger("button_position_capture_node"), "节点启动失败：%s", exception.what());
    }
    rclcpp::shutdown();
    return 0;
}
