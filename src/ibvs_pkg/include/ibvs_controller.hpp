#pragma once
#include "Eigen/Core"
#include "Eigen/Dense"
#include "Eigen/SVD"
#include <array>
#include <vector>
#include <stdexcept>
#include <opencv2/core.hpp>

class IbvsController{
public:
    using PixelPoints=std::array<cv::Point2f,4>;
    using Features=Eigen::Matrix<double,4,2>;
    using Depths=Eigen::Matrix<double,4,1>;
    using InteractionMatrix=Eigen::Matrix<double,2,6>;
    using MultiInteractionMatrices=Eigen::Matrix<double,8,6>;
    using FeatureError=Eigen::Matrix<double,8,1>;
    using CameraVelocity=Eigen::Matrix<double,6,1>;

    enum class InteractionMode{
        Current,
        Desired,
        Average
    };
    IbvsController(
        const Eigen::Matrix3d& k,
        const std::vector<double> dist_coeffs,
        double gain,
        double tolerance,
        double max_linear,
        double max_angular
    ):K_(k),dist_coeffs_(dist_coeffs),gain_(gain),tolerance_(tolerance),max_linear_(max_linear),max_angular_(max_angular){
        camera_matrix_cv_=cv::Mat::zeros(3, 3, CV_64F);
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                camera_matrix_cv_.at<double>(row,col)=K_(row, col);
            }
        }
        if (!dist_coeffs_.empty()) {
            dist_coeffs_cv_ = cv::Mat::zeros(
            1,
                static_cast<int>(
                dist_coeffs_.size()
                ),
                CV_64F
            );

            for (
                std::size_t i = 0;
                i < dist_coeffs_.size();
                ++i
            ) {
                dist_coeffs_cv_.at<double>(
                    0,
                    static_cast<int>(i)
                ) = dist_coeffs_[i];
            }
        }

    }

    CameraVelocity computeCommand(
        const PixelPoints& current_pixel_points,
        const PixelPoints& desired_pixel_points,
        const Depths& current_depths,
        InteractionMode mode =
        InteractionMode::Current,
        const MultiInteractionMatrices* desired_L=nullptr
    )const;

private:
    Eigen::Matrix3d K_;
    std::vector<double> dist_coeffs_;
    double gain_;
    double tolerance_;
    double max_linear_;
    double max_angular_;
    cv::Mat camera_matrix_cv_;
    cv::Mat dist_coeffs_cv_;

    InteractionMatrix interactionMatrix(double x,double y, double depth)const;
    MultiInteractionMatrices multiInteractionMatrices(const Features& features,const Depths& depth)const;
    CameraVelocity computeCameraVelocity(const MultiInteractionMatrices& L,const FeatureError& error)const;
    CameraVelocity limitVelocity(const CameraVelocity& velocity)const;
    Features pixelsToNormalizedPoints(const PixelPoints& pixel_points)const;
};