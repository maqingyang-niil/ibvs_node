#pragma once

#include "ibvs_controller.hpp"

#include <string>
#include <vector>

#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

struct DetectionResult{
    bool success{false};
    IbvsController::PixelPoints pixel_points{};
    IbvsController::Depths depths=IbvsController::Depths::Zero();
    double confidence{0.0};
};

class PlanarObjectFeatureDetector{
public:
    PlanarObjectFeatureDetector(
        const cv::Mat& desired_image,
        const cv::Rect& roi,
        double depth_scale
    );
    DetectionResult detect(
        const cv::Mat& color_image,
        const cv::Mat& depth_image,
        const std::string& depth_encoding
    );

    const IbvsController::PixelPoints&
    desiredPixelPoints() const
    {
        return desired_pixel_points_;
    }
private:
    double readDepthMedian(
        const cv::Mat& depth_image,
        const cv::Point2f& point,
        const std::string& depth_encoding
    ) const;

    cv::Ptr<cv::SIFT> sift_;

    cv::Ptr<cv::DescriptorMatcher> matcher_;

    std::vector<cv::KeyPoint>
        reference_keypoints_;

    cv::Mat reference_descriptors_;

    IbvsController::PixelPoints
        desired_pixel_points_{};

    double depth_scale_;
};
