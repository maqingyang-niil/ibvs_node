#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>

#include <Eigen/Geometry>
#include <builtin_interfaces/msg/time.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <rclcpp/rclcpp.hpp>

class HandNewPoseNode : public rclcpp::Node
{
public:
    using Pose = geometry_msgs::msg::Pose;
    using PoseStamped = geometry_msgs::msg::PoseStamped;

    HandNewPoseNode()
        : Node("hand_new_pose_node")
    {
        current_pose_topic_=this->declare_parameter<std::string>("current_pose_topic","/rm_driver/udp_arm_position");
        current_pose_frame_id_=this->declare_parameter<std::string>("current_pose_frame_id","base_link");
        delta_pose_topic_=this->declare_parameter<std::string>("delta_pose_topic","/ibvs/hand_delta_pose");
        output_topic_=this->declare_parameter<std::string>("output_topic","/ibvs/hand_new_pose");
        expected_delta_frame_=this->declare_parameter<std::string>("expected_delta_frame","tool");
        current_pose_timeout_=this->declare_parameter<double>("current_pose_timeout",0.5);
        max_delta_translation_=this->declare_parameter<double>("max_delta_translation",0.01);
        max_delta_rotation_=this->declare_parameter<double>("max_delta_rotation",0.05);
        if (!std::isfinite(current_pose_timeout_)||current_pose_timeout_<=0.0){throw std::invalid_argument("current_pose_timeout必须是有限正数");}
        if (!std::isfinite(max_delta_translation_)||max_delta_translation_<=0.0){throw std::invalid_argument("max_delta_translation必须是有限正数");}
        if (!std::isfinite(max_delta_rotation_)||max_delta_rotation_<=0.0){throw std::invalid_argument("max_delta_rotation必须是有限正数");}

        target_pose_pub_=this->create_publisher<PoseStamped>(output_topic_,10);
        if (current_pose_frame_id_.empty()){throw std::invalid_argument("current_pose_frame_id不能为空");}

        current_pose_sub_=this->create_subscription<Pose>(
            current_pose_topic_,
            rclcpp::SensorDataQoS(),
            std::bind(&HandNewPoseNode::currentPoseCallback,this,std::placeholders::_1));
        delta_pose_sub_=this->create_subscription<PoseStamped>(
            delta_pose_topic_,
            10,
            std::bind(&HandNewPoseNode::deltaPoseCallback,this,std::placeholders::_1));

        RCLCPP_INFO(
            this->get_logger(),
            "订阅当前位姿%s（坐标系%s）和增量位姿%s，发布新目标位姿%s",
            current_pose_topic_.c_str(),
            current_pose_frame_id_.c_str(),
            delta_pose_topic_.c_str(),
            output_topic_.c_str());
    }

private:
    static bool finitePose(const Pose& pose)
    {
        return std::isfinite(pose.position.x)&&std::isfinite(pose.position.y)&&
               std::isfinite(pose.position.z)&&std::isfinite(pose.orientation.x)&&
               std::isfinite(pose.orientation.y)&&std::isfinite(pose.orientation.z)&&
               std::isfinite(pose.orientation.w);
    }

    static std::int64_t stampNanoseconds(const builtin_interfaces::msg::Time& stamp)
    {
        return static_cast<std::int64_t>(stamp.sec)*1000000000LL+static_cast<std::int64_t>(stamp.nanosec);
    }

    static Eigen::Vector3d positionFromPose(const Pose& pose)
    {
        return Eigen::Vector3d(
            pose.position.x,
            pose.position.y,
            pose.position.z);
    }

    static bool orientationFromPose(const Pose& pose,Eigen::Quaterniond& orientation)
    {
        orientation=Eigen::Quaterniond(
            pose.orientation.w,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z);
        const double norm=orientation.norm();
        if (!std::isfinite(norm)||norm<1.0e-9){return false;}
        orientation.normalize();
        return true;
    }

    void currentPoseCallback(const Pose::ConstSharedPtr message)
    {
        Eigen::Quaterniond orientation;
        if (!finitePose(*message)||!orientationFromPose(*message,orientation)){
            RCLCPP_WARN_THROTTLE(this->get_logger(),*this->get_clock(),2000,"当前位姿无效，无法用于目标位姿计算");
            return;
        }

        current_position_=positionFromPose(*message);
        current_orientation_=orientation;
        last_current_pose_time_=std::chrono::steady_clock::now();
        has_current_pose_=true;
    }

    void deltaPoseCallback(const PoseStamped::ConstSharedPtr message)
    {
        const auto now=std::chrono::steady_clock::now();
        if (!has_current_pose_||now-last_current_pose_time_>std::chrono::duration<double>(current_pose_timeout_)){
            RCLCPP_WARN_THROTTLE(this->get_logger(),*this->get_clock(),2000,"当前机械臂位姿不可用或已超时，拒绝合成增量位姿");
            return;
        }
        if (!expected_delta_frame_.empty()&&message->header.frame_id!=expected_delta_frame_){
            RCLCPP_WARN_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                2000,
                "增量位姿坐标系为%s，期望%s，拒绝合成",
                message->header.frame_id.c_str(),
                expected_delta_frame_.c_str());
            return;
        }

        Eigen::Quaterniond delta_orientation;
        if (!finitePose(message->pose)||!orientationFromPose(message->pose,delta_orientation)){
            RCLCPP_WARN_THROTTLE(this->get_logger(),*this->get_clock(),2000,"增量位姿无效，拒绝合成");
            return;
        }
        const std::int64_t delta_stamp_ns=stampNanoseconds(message->header.stamp);
        if (delta_stamp_ns<=0||delta_stamp_ns<=last_delta_stamp_ns_){
            RCLCPP_WARN_THROTTLE(this->get_logger(),*this->get_clock(),2000,"增量位姿时间戳无效、重复或倒退，拒绝重复合成");
            return;
        }

        const Eigen::Vector3d delta_translation=positionFromPose(message->pose);
        const double delta_translation_norm=delta_translation.norm();
        const double delta_rotation_angle=2.0*std::acos(std::clamp(std::abs(delta_orientation.w()),0.0,1.0));
        if (delta_translation_norm>max_delta_translation_||delta_rotation_angle>max_delta_rotation_){
            RCLCPP_ERROR(
                this->get_logger(),
                "增量位姿超过限制：平移%.6fm/%.6fm，旋转%.6frad/%.6frad，拒绝合成",
                delta_translation_norm,
                max_delta_translation_,
                delta_rotation_angle,
                max_delta_rotation_);
            return;
        }

        target_position_=current_position_+current_orientation_*delta_translation;
        target_orientation_=(current_orientation_*delta_orientation).normalized();
        target_frame_id_=current_pose_frame_id_;
        last_delta_stamp_ns_=delta_stamp_ns;
        publishTargetPose(message->header.stamp);
    }

    void publishTargetPose(const builtin_interfaces::msg::Time& stamp)
    {
        PoseStamped message;
        message.header.stamp=stamp;
        message.header.frame_id=target_frame_id_;
        message.pose.position.x=target_position_.x();
        message.pose.position.y=target_position_.y();
        message.pose.position.z=target_position_.z();
        message.pose.orientation.x=target_orientation_.x();
        message.pose.orientation.y=target_orientation_.y();
        message.pose.orientation.z=target_orientation_.z();
        message.pose.orientation.w=target_orientation_.w();
        target_pose_pub_->publish(message);
    }

    std::string current_pose_topic_;
    std::string current_pose_frame_id_;
    std::string delta_pose_topic_;
    std::string output_topic_;
    std::string expected_delta_frame_;
    double current_pose_timeout_=0.5;
    double max_delta_translation_=0.01;
    double max_delta_rotation_=0.05;
    Eigen::Vector3d current_position_=Eigen::Vector3d::Zero();
    Eigen::Quaterniond current_orientation_=Eigen::Quaterniond::Identity();
    Eigen::Vector3d target_position_=Eigen::Vector3d::Zero();
    Eigen::Quaterniond target_orientation_=Eigen::Quaterniond::Identity();
    std::string target_frame_id_;
    std::chrono::steady_clock::time_point last_current_pose_time_{};
    std::int64_t last_delta_stamp_ns_=0;
    bool has_current_pose_=false;
    rclcpp::Subscription<Pose>::SharedPtr current_pose_sub_;
    rclcpp::Subscription<PoseStamped>::SharedPtr delta_pose_sub_;
    rclcpp::Publisher<PoseStamped>::SharedPtr target_pose_pub_;
};

int main(int argc,char* argv[])
{
    rclcpp::init(argc,argv);
    try{
        rclcpp::spin(std::make_shared<HandNewPoseNode>());
    }catch(const std::exception& exception){
        RCLCPP_FATAL(rclcpp::get_logger("hand_new_pose_node"),"节点启动失败：%s",exception.what());
    }
    rclcpp::shutdown();
    return 0;
}
