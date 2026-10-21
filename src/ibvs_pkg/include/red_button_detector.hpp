#pragma once

#include <opencv2/core.hpp>
#include <optional>
#include <string>

#include "detection_result.hpp"

class RedButtonDetector {
  public:
    RedButtonDetector(double depth_scale, int minimum_saturation = 90, int minimum_value = 50, double minimum_area_ratio = 0.0005,
                      double maximum_area_ratio = 0.08, double minimum_circularity = 0.65, double maximum_center_offset_ratio = 0.25,
                      double minimum_detection_score = 0.65, float desired_center_u_ratio = 0.5F, float desired_center_v_ratio = 0.5F,
                      float desired_radius_ratio = 0.08F, float feature_offset_ratio = 0.35F);

    DetectionResult detect(const cv::Mat &color_image, const cv::Mat &depth_image, const std::string &depth_encoding) const;

    IbvsController::PixelPoints desiredPixelPoints(const cv::Size &image_size) const;

  private:
    struct ButtonCircle {
        cv::Point2f center;
        float radius{0.0F};
        double score{0.0};
    };

    std::optional<ButtonCircle> detectBestButton(const cv::Mat &image) const;

    IbvsController::PixelPoints circleToPoints(const cv::Point2f &center, float radius) const;

    double readDepthMedian(const cv::Mat &depth_image, const cv::Point2f &center, float radius, const std::string &depth_encoding) const;

    double depth_scale_;
    int minimum_saturation_;
    int minimum_value_;
    double minimum_area_ratio_;
    double maximum_area_ratio_;
    double minimum_circularity_;
    double maximum_center_offset_ratio_;
    double minimum_detection_score_;
    float desired_center_u_ratio_;
    float desired_center_v_ratio_;
    float desired_radius_ratio_;
    float feature_offset_ratio_;
};
