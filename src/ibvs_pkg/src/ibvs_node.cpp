#include "rclcpp/rclcpp.hpp"
#include "Eigen/Core"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include <message_filters/subscriber.hpp>
#include <message_filters/synchronizer.hpp>
#include <message_filters/sync_policies/approximate_time.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <memory>
#include "cv_bridge/cv_bridge.hpp"
#include <sensor_msgs/image_encodings.hpp>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "ibvs_controller.hpp"
#include "planar_obj_detector.hpp"

class IbvsNode:public rclcpp::Node{
public:
    static constexpr int roi_x=235;
    static constexpr int roi_y=95;
    static constexpr int roi_width=1060;
    static constexpr int roi_height=815;

    IbvsNode():Node("ibvs_node"){
        //彩色图
        color_sub_.subscribe(
            this,
            "/camera/camera/color/image_raw",
            rmw_qos_profile_sensor_data
        );
        //深度图
        depth_sub_.subscribe(
            this,
            "/camera/camera/aligned_depth_to_color/image_raw",
            rmw_qos_profile_sensor_data
        );
        //图像信息同步器
        image_sync_=std::make_shared<message_filters::Synchronizer<SyncPolicy>>(SyncPolicy(10),color_sub_,depth_sub_);
        image_sync_->registerCallback(
            &IbvsNode::imageCallback,
            this
        );
        //相机参数
        camera_info_sub_=this->create_subscription<sensor_msgs::msg::CameraInfo>(
            "/camera/camera/color/camera_info",
            rclcpp::SensorDataQoS(),
            [this](const sensor_msgs::msg::CameraInfo::ConstSharedPtr msg){
                this->cameraInfoCallback(msg);
            }
        );
        //发布速度
        velocity_pub_=this->create_publisher<geometry_msgs::msg::TwistStamped>(
            "/ibvs/velocity",
            10
        );
        const cv::Mat desired_image=cv::imread("/path",cv::IMREAD_COLOR);
        const cv::Rect target_roi(roi_x,roi_y,roi_width,roi_height);
        detector_=std::make_unique<PlanarObjectFeatureDetector>(desired_image,target_roi,0.001);
        RCLCPP_INFO(this->get_logger(),"IBVS节点初始化完成");

    }


private:
    struct LatestCommand{
        IbvsController::CameraVelocity velocity{IbvsController::CameraVelocity::Zero()};
        rclcpp::Time update_time;
        std::string frame_id;
        bool available{false};
    };
    LatestCommand latest_command_;
    using ImageMsg=sensor_msgs::msg::Image;
    using SyncPolicy=message_filters::sync_policies::ApproximateTime<ImageMsg,ImageMsg>;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> image_sync_;
    message_filters::Subscriber<sensor_msgs::msg::Image> color_sub_;
    message_filters::Subscriber<sensor_msgs::msg::Image> depth_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
    rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr velocity_pub_;
    std::vector<double> dist_coeffs_;
    Eigen::Matrix3d K_=Eigen::Matrix3d::Identity();
    std::unique_ptr<IbvsController> controller_;
    std::unique_ptr<PlanarObjectFeatureDetector> detector_;
    void publishVelocity(
        const IbvsController::CameraVelocity& velocity,
        const std::string& frame_id)
    {
        if (!velocity.allFinite()) {
            publishZeroVelocity(frame_id);
            return;
        }
        geometry_msgs::msg::TwistStamped message;
        message.header.stamp =this->now();
        message.header.frame_id =frame_id;
        message.twist.linear.x =velocity(0);
        message.twist.linear.y =velocity(1);
        message.twist.linear.z =velocity(2);
        message.twist.angular.x =velocity(3);
        message.twist.angular.y =velocity(4);
        message.twist.angular.z =velocity(5);
        velocity_pub_->publish(message);
    }

    void publishZeroVelocity(
        const std::string& frame_id)
    {
        geometry_msgs::msg::TwistStamped message;
        message.header.stamp =this->now();
        message.header.frame_id =frame_id;

        message.twist.linear.x = 0.0;
        message.twist.linear.y = 0.0;
        message.twist.linear.z = 0.0;

        message.twist.angular.x = 0.0;
        message.twist.angular.y = 0.0;
        message.twist.angular.z = 0.0;

        velocity_pub_->publish(message);
    }
    void cameraInfoCallback(const sensor_msgs::msg::CameraInfo::ConstSharedPtr msg){
        if (controller_){return;}
        K_<<msg->k[0],msg->k[1],msg->k[2],
            msg->k[3],msg->k[4],msg->k[5],
            msg->k[6],msg->k[7],msg->k[8];
        dist_coeffs_=msg->d;
        controller_=std::make_unique<IbvsController>(
            K_,
            dist_coeffs_,
            1e-3,
            0.02,
            0.05,
            0.2
        );
        RCLCPP_INFO(this->get_logger(),"已受到相机内参");
    }

    void imageCallback(
        const sensor_msgs::msg::Image::ConstSharedPtr& color_msg,
        const sensor_msgs::msg::Image::ConstSharedPtr& depth_msg
    ){
        if (!controller_){
            RCLCPP_WARN_THROTTLE(this->get_logger(),*get_clock(),2000,"等待camera_info");
            return;
        }
        RCLCPP_INFO_THROTTLE(this->get_logger(),*get_clock(),2000,"a收到同步彩色图和深度图");
        try{
            cv_bridge::CvImageConstPtr color_cv=cv_bridge::toCvShare(
                color_msg,
                sensor_msgs::image_encodings::BGR8
            );
            cv_bridge::CvImageConstPtr depth_cv=cv_bridge::toCvShare(depth_msg);

            const cv::Mat& color_image=color_cv->image;
            const cv::Mat& depth_image=depth_cv->image;

            if (color_image.empty()||depth_image.empty()){
                publishZeroVelocity(color_msg->header.frame_id);
                return;
            }
            if (color_image.rows != depth_image.rows||color_image.cols != depth_image.cols) {
                publishZeroVelocity(color_msg->header.frame_id);
                return;
            }
            const DetectionResult detection=detector_->detect(color_image,depth_image,depth_msg->encoding);
            if (!detection.success||detection.confidence<0.5){
                publishZeroVelocity(color_msg->header.frame_id);
                return;
            }
            const IbvsController::CameraVelocity velocity=controller_->computeCommand(
                detection.pixel_points,
                detector_->desiredPixelPoints(),
                detection.depths,
                IbvsController::InteractionMode::Current
            );
            publishVelocity(velocity,color_msg->header.frame_id);
        }catch(const cv_bridge::Exception& exception){
            RCLCPP_ERROR(
                get_logger(),
                "cv_bridge转换失败: %s",
                exception.what()
            );
        }
    }
};

int main(int argc,char** argv){
    rclcpp::init(argc,argv);
    rclcpp::spin(std::make_shared<IbvsNode>());
    rclcpp::shutdown();
    return 0;
}
