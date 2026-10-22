#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <rclcpp/rclcpp.hpp>

class HandVelocityNode : public rclcpp::Node
{
public:
    using TwistStamped = geometry_msgs::msg::TwistStamped;
    using Vector6d = Eigen::Matrix<double,6,1>;

    HandVelocityNode()
        : Node("hand_velocity_node")
    {
        input_topic_=this->declare_parameter<std::string>("input_topic","/ibvs/velocity");
        output_topic_=this->declare_parameter<std::string>("output_topic","/ibvs/tool_velocity");
        expected_input_frame_=this->declare_parameter<std::string>("expected_input_frame","camera_color_optical_frame");
        output_frame_id_=this->declare_parameter<std::string>("output_frame_id","tool");
        const auto rotation_values=this->declare_parameter<std::vector<double>>(
            "camera_to_tool_rotation",
            {0.99994390,-0.00776698,0.00720273,
             0.00782760,0.99993386,-0.00842698,
             -0.00713680,0.00848289,0.99993855});
        const auto translation_values=this->declare_parameter<std::vector<double>>(
            "camera_to_tool_translation",
            {-0.03669205,-0.09531331,0.02498297});
        max_linear_speed_=this->declare_parameter<double>("max_linear_speed",0.03);
        max_angular_speed_=this->declare_parameter<double>("max_angular_speed",0.10);
        const int publish_period_ms=this->declare_parameter<int>("publish_period_ms",20);
        const int command_timeout_ms=this->declare_parameter<int>("command_timeout_ms",200);

        if (rotation_values.size()!=9){throw std::invalid_argument("camera_to_tool_rotation必须包含9个元素");}
        if (translation_values.size()!=3){throw std::invalid_argument("camera_to_tool_translation必须包含3个元素");}
        if (!std::isfinite(max_linear_speed_)||!std::isfinite(max_angular_speed_)||max_linear_speed_<=0.0||max_angular_speed_<=0.0){throw std::invalid_argument("速度限制必须是有限正数");}
        if (publish_period_ms<=0||command_timeout_ms<=0){throw std::invalid_argument("发布周期和超时时间必须大于0");}

        camera_to_tool_rotation_<<
            rotation_values[0],rotation_values[1],rotation_values[2],
            rotation_values[3],rotation_values[4],rotation_values[5],
            rotation_values[6],rotation_values[7],rotation_values[8];
        camera_to_tool_translation_<<translation_values[0],translation_values[1],translation_values[2];
        const double orthogonality_error=(camera_to_tool_rotation_.transpose()*camera_to_tool_rotation_-Eigen::Matrix3d::Identity()).norm();
        const double determinant=camera_to_tool_rotation_.determinant();
        if (!camera_to_tool_rotation_.allFinite()||!camera_to_tool_translation_.allFinite()||orthogonality_error>1.0e-3||std::abs(determinant-1.0)>1.0e-3){
            throw std::invalid_argument("camera_to_tool_rotation不是有效的旋转矩阵");
        }

        command_timeout_=std::chrono::milliseconds(command_timeout_ms);
        velocity_pub_=this->create_publisher<TwistStamped>(output_topic_,10);
        velocity_sub_=this->create_subscription<TwistStamped>(
            input_topic_,
            10,
            std::bind(&HandVelocityNode::velocityCallback,this,std::placeholders::_1));
        velocity_timer_=this->create_wall_timer(
            std::chrono::milliseconds(publish_period_ms),
            std::bind(&HandVelocityNode::publishLatestVelocity,this));

        RCLCPP_INFO(
            this->get_logger(),
            "订阅%s并发布%s，周期%dms，超时%dms",
            input_topic_.c_str(),
            output_topic_.c_str(),
            publish_period_ms,
            command_timeout_ms);
    }

private:
    struct LatestVelocity
    {
        Vector6d value=Vector6d::Zero();
        std::chrono::steady_clock::time_point update_time{};
        bool available=false;
    };

    static bool finiteTwist(const TwistStamped& message)
    {
        return std::isfinite(message.twist.linear.x)&&std::isfinite(message.twist.linear.y)&&
               std::isfinite(message.twist.linear.z)&&std::isfinite(message.twist.angular.x)&&
               std::isfinite(message.twist.angular.y)&&std::isfinite(message.twist.angular.z);
    }

    Vector6d transformVelocity(const TwistStamped& message)const
    {
        const Eigen::Vector3d linear_camera(
            message.twist.linear.x,
            message.twist.linear.y,
            message.twist.linear.z);
        const Eigen::Vector3d angular_camera(
            message.twist.angular.x,
            message.twist.angular.y,
            message.twist.angular.z);
        Eigen::Vector3d angular_tool=camera_to_tool_rotation_*angular_camera;
        Eigen::Vector3d linear_tool=camera_to_tool_rotation_*linear_camera+camera_to_tool_translation_.cross(angular_tool);
        if (!linear_tool.allFinite()||!angular_tool.allFinite()){return Vector6d::Zero();}
        const double linear_norm=linear_tool.norm();
        const double angular_norm=angular_tool.norm();
        double scale=1.0;
        if (linear_norm>max_linear_speed_){scale=std::min(scale,max_linear_speed_/linear_norm);}
        if (angular_norm>max_angular_speed_){scale=std::min(scale,max_angular_speed_/angular_norm);}
        linear_tool*=scale;
        angular_tool*=scale;
        Vector6d velocity;
        velocity<<linear_tool,angular_tool;
        return velocity;
    }

    void velocityCallback(const TwistStamped::ConstSharedPtr message)
    {
        Vector6d velocity=Vector6d::Zero();
        if (!expected_input_frame_.empty()&&message->header.frame_id!=expected_input_frame_){
            if (!frame_mismatch_reported_){
                RCLCPP_ERROR(
                    this->get_logger(),
                    "输入坐标系为%s，期望%s；输出零速度",
                    message->header.frame_id.c_str(),
                    expected_input_frame_.c_str());
                frame_mismatch_reported_=true;
            }
        }else if (!finiteTwist(*message)){
            RCLCPP_ERROR(this->get_logger(),"收到非有限速度；输出零速度");
        }else{
            velocity=transformVelocity(*message);
        }
        std::lock_guard<std::mutex> lock(command_mutex_);
        latest_velocity_.value=velocity;
        latest_velocity_.update_time=std::chrono::steady_clock::now();
        latest_velocity_.available=true;
    }

    void publishLatestVelocity()
    {
        LatestVelocity velocity;
        {
            std::lock_guard<std::mutex> lock(command_mutex_);
            velocity=latest_velocity_;
        }
        if (!velocity.available||std::chrono::steady_clock::now()-velocity.update_time>command_timeout_){
            publishVelocity(Vector6d::Zero());
            return;
        }
        publishVelocity(velocity.value);
    }

    void publishVelocity(const Vector6d& velocity)
    {
        TwistStamped message;
        message.header.stamp=this->now();
        message.header.frame_id=output_frame_id_;
        message.twist.linear.x=velocity(0);
        message.twist.linear.y=velocity(1);
        message.twist.linear.z=velocity(2);
        message.twist.angular.x=velocity(3);
        message.twist.angular.y=velocity(4);
        message.twist.angular.z=velocity(5);
        velocity_pub_->publish(message);
    }

    std::string input_topic_;
    std::string output_topic_;
    std::string expected_input_frame_;
    std::string output_frame_id_;
    Eigen::Matrix3d camera_to_tool_rotation_=Eigen::Matrix3d::Identity();
    Eigen::Vector3d camera_to_tool_translation_=Eigen::Vector3d::Zero();
    double max_linear_speed_=0.03;
    double max_angular_speed_=0.10;
    std::chrono::steady_clock::duration command_timeout_{};
    std::mutex command_mutex_;
    LatestVelocity latest_velocity_;
    bool frame_mismatch_reported_=false;
    rclcpp::Subscription<TwistStamped>::SharedPtr velocity_sub_;
    rclcpp::Publisher<TwistStamped>::SharedPtr velocity_pub_;
    rclcpp::TimerBase::SharedPtr velocity_timer_;
};

int main(int argc,char* argv[])
{
    rclcpp::init(argc,argv);
    try{
        rclcpp::spin(std::make_shared<HandVelocityNode>());
    }catch(const std::exception& exception){
        RCLCPP_FATAL(rclcpp::get_logger("hand_velocity_node"),"节点启动失败：%s",exception.what());
    }
    rclcpp::shutdown();
    return 0;
}
