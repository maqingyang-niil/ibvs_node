#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <builtin_interfaces/msg/time.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <rclcpp/rclcpp.hpp>

class HandDeltaPoseNode : public rclcpp::Node
{
public:
    using PoseStamped = geometry_msgs::msg::PoseStamped;
    using TwistStamped = geometry_msgs::msg::TwistStamped;

    HandDeltaPoseNode()
        : Node("hand_delta_pose_node")
    {
        input_topic_=this->declare_parameter<std::string>("input_topic","/ibvs/hand_velocity");
        output_topic_=this->declare_parameter<std::string>("output_topic","/ibvs/hand_delta_pose");
        expected_input_frame_=this->declare_parameter<std::string>("expected_input_frame","tool");
        output_frame_id_=this->declare_parameter<std::string>("output_frame_id","");
        max_dt_=this->declare_parameter<double>("max_dt",0.2);
        if (!std::isfinite(max_dt_)||max_dt_<=0.0){throw std::invalid_argument("max_dt必须是有限正数");}

        delta_pose_pub_=this->create_publisher<PoseStamped>(output_topic_,10);
        velocity_sub_=this->create_subscription<TwistStamped>(
            input_topic_,
            10,
            std::bind(&HandDeltaPoseNode::velocityCallback,this,std::placeholders::_1));

        RCLCPP_INFO(
            this->get_logger(),
            "订阅%s并发布%s，最大积分间隔%.3fs",
            input_topic_.c_str(),
            output_topic_.c_str(),
            max_dt_);
    }

private:
    static bool finiteTwist(const TwistStamped& message)
    {
        return std::isfinite(message.twist.linear.x)&&std::isfinite(message.twist.linear.y)&&
               std::isfinite(message.twist.linear.z)&&std::isfinite(message.twist.angular.x)&&
               std::isfinite(message.twist.angular.y)&&std::isfinite(message.twist.angular.z);
    }

    static Eigen::Matrix3d skew(const Eigen::Vector3d& vector)
    {
        Eigen::Matrix3d matrix;
        matrix<<0.0,-vector.z(),vector.y(),
                vector.z(),0.0,-vector.x(),
                -vector.y(),vector.x(),0.0;
        return matrix;
    }

    static std::int64_t stampNanoseconds(const builtin_interfaces::msg::Time& stamp)
    {
        return static_cast<std::int64_t>(stamp.sec)*1000000000LL+static_cast<std::int64_t>(stamp.nanosec);
    }

    void velocityCallback(const TwistStamped::ConstSharedPtr message)
    {
        const std::int64_t stamp_ns=stampNanoseconds(message->header.stamp);
        if (stamp_ns<=0){
            RCLCPP_WARN_THROTTLE(this->get_logger(),*this->get_clock(),2000,"速度消息时间戳无效，跳过积分");
            return;
        }
        if (!expected_input_frame_.empty()&&message->header.frame_id!=expected_input_frame_){
            RCLCPP_WARN_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                2000,
                "输入坐标系为%s，期望%s，跳过积分",
                message->header.frame_id.c_str(),
                expected_input_frame_.c_str());
            resetTimestamp(stamp_ns);
            return;
        }
        if (!finiteTwist(*message)){
            RCLCPP_WARN_THROTTLE(this->get_logger(),*this->get_clock(),2000,"收到非有限速度，跳过积分");
            resetTimestamp(stamp_ns);
            return;
        }
        if (!has_previous_stamp_){
            resetTimestamp(stamp_ns);
            return;
        }

        const double dt=static_cast<double>(stamp_ns-previous_stamp_ns_)*1.0e-9;
        previous_stamp_ns_=stamp_ns;
        if (dt<=0.0){
            RCLCPP_WARN_THROTTLE(this->get_logger(),*this->get_clock(),2000,"速度消息时间戳未递增，跳过积分");
            return;
        }
        if (dt>max_dt_){
            RCLCPP_WARN_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                2000,
                "速度消息间隔%.3fs超过max_dt=%.3fs，跳过本次积分",
                dt,
                max_dt_);
            return;
        }

        const Eigen::Vector3d linear_velocity(
            message->twist.linear.x,
            message->twist.linear.y,
            message->twist.linear.z);
        const Eigen::Vector3d angular_velocity(
            message->twist.angular.x,
            message->twist.angular.y,
            message->twist.angular.z);
        const Eigen::Vector3d rotation_vector=angular_velocity*dt;
        const double rotation_angle=rotation_vector.norm();
        const Eigen::Matrix3d rotation_skew=skew(rotation_vector);

        Eigen::Matrix3d delta_rotation=Eigen::Matrix3d::Identity();
        Eigen::Matrix3d translation_jacobian=Eigen::Matrix3d::Identity();
        if (rotation_angle<1.0e-8){
            delta_rotation+=rotation_skew+0.5*rotation_skew*rotation_skew;
            translation_jacobian+=0.5*rotation_skew+(1.0/6.0)*rotation_skew*rotation_skew;
        }else{
            const double angle_squared=rotation_angle*rotation_angle;
            const double angle_cubed=angle_squared*rotation_angle;
            delta_rotation=Eigen::AngleAxisd(rotation_angle,rotation_vector/rotation_angle).toRotationMatrix();
            translation_jacobian+=
                ((1.0-std::cos(rotation_angle))/angle_squared)*rotation_skew+
                ((rotation_angle-std::sin(rotation_angle))/angle_cubed)*rotation_skew*rotation_skew;
        }

        const Eigen::Vector3d delta_translation=translation_jacobian*(linear_velocity*dt);
        Eigen::Quaterniond delta_orientation(delta_rotation);
        delta_orientation.normalize();
        publishDeltaPose(*message,delta_translation,delta_orientation);
    }

    void resetTimestamp(std::int64_t stamp_ns)
    {
        previous_stamp_ns_=stamp_ns;
        has_previous_stamp_=true;
    }

    void publishDeltaPose(
        const TwistStamped& velocity,
        const Eigen::Vector3d& delta_translation,
        const Eigen::Quaterniond& delta_orientation)
    {
        PoseStamped message;
        message.header.stamp=velocity.header.stamp;
        message.header.frame_id=output_frame_id_.empty()?velocity.header.frame_id:output_frame_id_;
        message.pose.position.x=delta_translation.x();
        message.pose.position.y=delta_translation.y();
        message.pose.position.z=delta_translation.z();
        message.pose.orientation.x=delta_orientation.x();
        message.pose.orientation.y=delta_orientation.y();
        message.pose.orientation.z=delta_orientation.z();
        message.pose.orientation.w=delta_orientation.w();
        delta_pose_pub_->publish(message);
    }

    std::string input_topic_;
    std::string output_topic_;
    std::string expected_input_frame_;
    std::string output_frame_id_;
    double max_dt_=0.2;
    std::int64_t previous_stamp_ns_=0;
    bool has_previous_stamp_=false;
    rclcpp::Subscription<TwistStamped>::SharedPtr velocity_sub_;
    rclcpp::Publisher<PoseStamped>::SharedPtr delta_pose_pub_;
};

int main(int argc,char* argv[])
{
    rclcpp::init(argc,argv);
    try{
        rclcpp::spin(std::make_shared<HandDeltaPoseNode>());
    }catch(const std::exception& exception){
        RCLCPP_FATAL(rclcpp::get_logger("hand_delta_pose_node"),"节点启动失败：%s",exception.what());
    }
    rclcpp::shutdown();
    return 0;
}
