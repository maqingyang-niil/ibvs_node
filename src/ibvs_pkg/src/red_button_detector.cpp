#include "red_button_detector.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <opencv2/imgproc.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <stdexcept>
#include <vector>

namespace {
constexpr double pi = 3.14159265358979323846;
}

RedButtonDetector::RedButtonDetector(double depth_scale, int minimum_saturation, int minimum_value, double minimum_area_ratio, double maximum_area_ratio,
                                     double minimum_circularity, double maximum_center_offset_ratio, double minimum_detection_score,
                                     float desired_center_u_ratio, float desired_center_v_ratio, float desired_radius_ratio, float feature_offset_ratio)
    : depth_scale_(depth_scale), minimum_saturation_(minimum_saturation), minimum_value_(minimum_value), minimum_area_ratio_(minimum_area_ratio),
      maximum_area_ratio_(maximum_area_ratio), minimum_circularity_(minimum_circularity), maximum_center_offset_ratio_(maximum_center_offset_ratio),
      minimum_detection_score_(minimum_detection_score), desired_center_u_ratio_(desired_center_u_ratio), desired_center_v_ratio_(desired_center_v_ratio),
      desired_radius_ratio_(desired_radius_ratio), feature_offset_ratio_(feature_offset_ratio) {
    if (!std::isfinite(depth_scale_) || depth_scale_ <= 0.0) { throw std::invalid_argument("深度比例必须是有效正数"); }
    if (minimum_saturation_ < 0 || minimum_saturation_ > 255 || minimum_value_ < 0 || minimum_value_ > 255) {
        throw std::invalid_argument("红色按钮HSV阈值必须在[0,255]范围内");
    }
    if (!std::isfinite(minimum_area_ratio_) || !std::isfinite(maximum_area_ratio_) || minimum_area_ratio_ <= 0.0 ||
        maximum_area_ratio_ <= minimum_area_ratio_ || maximum_area_ratio_ > 1.0) {
        throw std::invalid_argument("红色按钮面积比例范围无效");
    }
    if (!std::isfinite(minimum_circularity_) || minimum_circularity_ <= 0.0 || minimum_circularity_ > 1.0) {
        throw std::invalid_argument("红色按钮最小圆度必须在(0,1]范围内");
    }
    if (!std::isfinite(maximum_center_offset_ratio_) || maximum_center_offset_ratio_ <= 0.0 || maximum_center_offset_ratio_ > 1.0) {
        throw std::invalid_argument("红色按钮最大中心偏移比例必须在(0,1]范围内");
    }
    if (!std::isfinite(minimum_detection_score_) || minimum_detection_score_ < 0.0 || minimum_detection_score_ > 1.0) {
        throw std::invalid_argument("红色按钮最小检测分数必须在[0,1]范围内");
    }
    if (!std::isfinite(desired_center_u_ratio_) || !std::isfinite(desired_center_v_ratio_) || desired_center_u_ratio_ < 0.0F ||
        desired_center_u_ratio_ > 1.0F || desired_center_v_ratio_ < 0.0F || desired_center_v_ratio_ > 1.0F) {
        throw std::invalid_argument("期望按钮中心比例必须在[0,1]范围内");
    }
    if (!std::isfinite(desired_radius_ratio_) || desired_radius_ratio_ <= 0.0F || desired_radius_ratio_ > 0.5F) {
        throw std::invalid_argument("期望按钮半径比例必须在(0,0.5]范围内");
    }
    if (!std::isfinite(feature_offset_ratio_) || feature_offset_ratio_ <= 0.0F || feature_offset_ratio_ >= 0.70F) {
        throw std::invalid_argument("按钮特征点偏移比例必须在(0,0.70)范围内");
    }
}

std::optional<RedButtonDetector::ButtonCircle> RedButtonDetector::detectBestButton(const cv::Mat &image) const {
    if (image.empty()) { return std::nullopt; }

    cv::Mat hsv_image;
    cv::cvtColor(image, hsv_image, cv::COLOR_BGR2HSV);

    cv::Mat lower_red_mask;
    cv::Mat upper_red_mask;
    cv::inRange(hsv_image, cv::Scalar(0, minimum_saturation_, minimum_value_), cv::Scalar(10, 255, 255), lower_red_mask);
    cv::inRange(hsv_image, cv::Scalar(170, minimum_saturation_, minimum_value_), cv::Scalar(179, 255, 255), upper_red_mask);

    cv::Mat red_mask;
    cv::bitwise_or(lower_red_mask, upper_red_mask, red_mask);
    const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(red_mask, red_mask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(red_mask, red_mask, cv::MORPH_CLOSE, kernel);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(red_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    const double image_area = static_cast<double>(image.cols) * static_cast<double>(image.rows);
    const cv::Point2f desired_center(desired_center_u_ratio_ * static_cast<float>(image.cols), desired_center_v_ratio_ * static_cast<float>(image.rows));
    std::optional<ButtonCircle> best_button;

    for (const auto &contour : contours) {
        const double area = cv::contourArea(contour);
        const double area_ratio = area / image_area;
        if (area_ratio < minimum_area_ratio_ || area_ratio > maximum_area_ratio_) { continue; }

        const double perimeter = cv::arcLength(contour, true);
        if (perimeter <= 0.0) { continue; }

        const double circularity = std::clamp(4.0 * pi * area / (perimeter * perimeter), 0.0, 1.0);
        if (circularity < minimum_circularity_) { continue; }

        cv::Point2f center;
        float radius = 0.0F;
        cv::minEnclosingCircle(contour, center, radius);
        if (!std::isfinite(radius) || radius < 2.0F) { continue; }

        const double fill_ratio = std::clamp(area / (pi * static_cast<double>(radius) * static_cast<double>(radius)), 0.0, 1.0);
        const double normalized_u_offset = static_cast<double>(center.x - desired_center.x) / static_cast<double>(image.cols);
        const double normalized_v_offset = static_cast<double>(center.y - desired_center.y) / static_cast<double>(image.rows);
        const double center_offset = std::hypot(normalized_u_offset, normalized_v_offset);
        if (center_offset > maximum_center_offset_ratio_) { continue; }

        const double center_score = 1.0 - std::clamp(center_offset / maximum_center_offset_ratio_, 0.0, 1.0);
        const double score = 0.45 * circularity + 0.25 * fill_ratio + 0.30 * center_score;
        if (score < minimum_detection_score_) { continue; }

        if (!best_button.has_value() || score > best_button->score) { best_button = ButtonCircle{center, radius, score}; }
    }

    return best_button;
}

IbvsController::PixelPoints RedButtonDetector::circleToPoints(const cv::Point2f &center, float radius) const {
    const float offset = feature_offset_ratio_ * radius;
    return {cv::Point2f(center.x - offset, center.y - offset), cv::Point2f(center.x + offset, center.y - offset),
            cv::Point2f(center.x + offset, center.y + offset), cv::Point2f(center.x - offset, center.y + offset)};
}

IbvsController::PixelPoints RedButtonDetector::desiredPixelPoints(const cv::Size &image_size) const {
    if (image_size.width <= 0 || image_size.height <= 0) { throw std::invalid_argument("实时图像尺寸无效"); }

    const cv::Point2f center(desired_center_u_ratio_ * static_cast<float>(image_size.width), desired_center_v_ratio_ * static_cast<float>(image_size.height));
    const float radius = desired_radius_ratio_ * static_cast<float>(std::min(image_size.width, image_size.height));
    return circleToPoints(center, radius);
}

DetectionResult RedButtonDetector::detect(const cv::Mat &color_image, const cv::Mat &depth_image, const std::string &depth_encoding) const {
    DetectionResult result;
    if (color_image.empty() || depth_image.empty() || color_image.size() != depth_image.size()) { return result; }

    const auto button = detectBestButton(color_image);
    if (!button.has_value()) { return result; }

    result.confidence = button->score;
    result.pixel_points = circleToPoints(button->center, button->radius);

    const double depth = readDepthMedian(depth_image, button->center, button->radius, depth_encoding);
    if (!std::isfinite(depth) || depth <= 0.0) {
        result.invalid_depth_index = 0;
        return result;
    }

    result.depths.setConstant(depth);
    result.success = true;
    return result;
}

double RedButtonDetector::readDepthMedian(const cv::Mat &depth_image, const cv::Point2f &center, float radius, const std::string &depth_encoding) const {
    if (depth_image.empty() || !std::isfinite(center.x) || !std::isfinite(center.y) || !std::isfinite(radius) || radius <= 0.0F) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const int center_u = static_cast<int>(std::lround(center.x));
    const int center_v = static_cast<int>(std::lround(center.y));
    const int sample_radius = std::max(2, static_cast<int>(std::lround(0.35F * radius)));
    const int minimum_u = std::max(0, center_u - sample_radius);
    const int maximum_u = std::min(depth_image.cols - 1, center_u + sample_radius);
    const int minimum_v = std::max(0, center_v - sample_radius);
    const int maximum_v = std::min(depth_image.rows - 1, center_v + sample_radius);
    std::vector<double> valid_depths;

    for (int v = minimum_v; v <= maximum_v; ++v) {
        for (int u = minimum_u; u <= maximum_u; ++u) {
            const int delta_u = u - center_u;
            const int delta_v = v - center_v;
            if (delta_u * delta_u + delta_v * delta_v > sample_radius * sample_radius) { continue; }

            double depth = std::numeric_limits<double>::quiet_NaN();
            if (depth_encoding == sensor_msgs::image_encodings::TYPE_16UC1) {
                if (depth_image.type() != CV_16UC1) { throw std::runtime_error("16UC1编码与cv::Mat类型不匹配"); }
                const std::uint16_t raw_depth = depth_image.at<std::uint16_t>(v, u);
                if (raw_depth != 0) { depth = static_cast<double>(raw_depth) * depth_scale_; }
            } else if (depth_encoding == sensor_msgs::image_encodings::TYPE_32FC1) {
                if (depth_image.type() != CV_32FC1) { throw std::runtime_error("32FC1编码与cv::Mat类型不匹配"); }
                depth = static_cast<double>(depth_image.at<float>(v, u));
            } else {
                throw std::runtime_error("不支持的深度图编码: " + depth_encoding);
            }

            if (std::isfinite(depth) && depth > 0.0) { valid_depths.push_back(depth); }
        }
    }

    if (valid_depths.empty()) { return std::numeric_limits<double>::quiet_NaN(); }
    const std::size_t middle = valid_depths.size() / 2;
    std::nth_element(valid_depths.begin(), valid_depths.begin() + static_cast<std::ptrdiff_t>(middle), valid_depths.end());
    if (valid_depths.size() % 2 == 1) { return valid_depths[middle]; }

    const double upper_middle = valid_depths[middle];
    const double lower_middle = *std::max_element(valid_depths.begin(), valid_depths.begin() + static_cast<std::ptrdiff_t>(middle));
    return 0.5 * (lower_middle + upper_middle);
}
