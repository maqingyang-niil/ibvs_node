#include "planar_obj_detector.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <sensor_msgs/image_encodings.hpp>

namespace {

cv::Mat convertToGray(const cv::Mat& image) {
    if (image.empty()) {
        throw std::invalid_argument("输入图像为空");
    }

    if (image.channels() == 1) {
        return image;
    }

    cv::Mat gray_image;

    if (image.channels() == 3) {
        cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);

        return gray_image;
    }

    if (image.channels() == 4) {
        cv::cvtColor(image, gray_image, cv::COLOR_BGRA2GRAY);

        return gray_image;
    }

    throw std::invalid_argument("不支持的彩色图像通道数");
}

} // namespace

PlanarObjectFeatureDetector::PlanarObjectFeatureDetector(const cv::Mat& desired_image, const cv::Rect& roi, double depth_scale)
    : sift_(cv::SIFT::create(1500)), matcher_(cv::BFMatcher::create(cv::NORM_L2, false)), depth_scale_(depth_scale) {
    if (desired_image.empty()) {
        throw std::invalid_argument("期望图像为空");
    }
    if (!std::isfinite(depth_scale_) || depth_scale_ <= 0.0) {
        throw std::invalid_argument("深度比例必须是有效正数");
    }

    if (roi.x < 0 || roi.y < 0 || roi.width <= 0 || roi.height <= 0 || roi.x + roi.width > desired_image.cols || roi.y + roi.height > desired_image.rows) {
        throw std::invalid_argument("ROI超出期望图像范围");
    }

    /*
     * 将控制点从ROI边界向内部移动，
     * 避免深度落到目标边缘或背景。
     */
    constexpr float margin = 0.30F;
    const float left = static_cast<float>(roi.x) + margin * static_cast<float>(roi.width);
    const float right = static_cast<float>(roi.x) + (1.0F - margin) * static_cast<float>(roi.width);
    const float top = static_cast<float>(roi.y) + margin * static_cast<float>(roi.height);
    const float bottom = static_cast<float>(roi.y) + (1.0F - margin) * static_cast<float>(roi.height);
    desired_pixel_points_ = {cv::Point2f(left, top), cv::Point2f(right, top), cv::Point2f(right, bottom), cv::Point2f(left, bottom)};

    /*
     * 创建ROI掩膜，只提取目标区域内的SIFT特征。
     */
    cv::Mat reference_mask = cv::Mat::zeros(desired_image.rows, desired_image.cols, CV_8UC1);

    reference_mask(roi).setTo(255);

    const cv::Mat desired_gray = convertToGray(desired_image);

    sift_->detectAndCompute(desired_gray, reference_mask, reference_keypoints_, reference_descriptors_);

    if (reference_descriptors_.empty() || reference_keypoints_.size() < 12) {
        throw std::runtime_error("期望图像ROI中的SIFT特征不足");
    }
}

DetectionResult PlanarObjectFeatureDetector::detect(const cv::Mat& color_image, const cv::Mat& depth_image, const std::string& depth_encoding) {
    DetectionResult result;

    if (color_image.empty() || depth_image.empty()) {
        return result;
    }

    if (color_image.rows != depth_image.rows || color_image.cols != depth_image.cols) {
        return result;
    }

    const cv::Mat current_gray = convertToGray(color_image);

    std::vector<cv::KeyPoint> current_keypoints;

    cv::Mat current_descriptors;

    sift_->detectAndCompute(current_gray, cv::noArray(), current_keypoints, current_descriptors);
    // 当前检测到的SIFT特征点数量
    std::cout << "[detect] current_keypoints=" << current_keypoints.size() << std::endl;

    if (current_descriptors.empty() || current_keypoints.empty()) {
        return result;
    }

    /*
     * KNN匹配：
     * queryIdx对应期望图像，
     * trainIdx对应当前图像。
     */
    std::vector<std::vector<cv::DMatch>> matches;

    matcher_->knnMatch(reference_descriptors_, current_descriptors, matches, 2);

    std::vector<cv::DMatch> good_matches;

    good_matches.reserve(matches.size());

    constexpr float ratio_threshold = 0.75F;

    for (const auto& pair : matches) {
        if (pair.size() != 2) {
            continue;
        }

        const cv::DMatch& first = pair[0];

        const cv::DMatch& second = pair[1];

        if (first.distance < ratio_threshold * second.distance) {
            good_matches.push_back(first);
        }
    }
    // 参考图中的SIFT特征点数量
    std::cout << "[detect] reference_keypoints=" << reference_keypoints_.size() << ", good_matches=" << good_matches.size() << std::endl;

    if (good_matches.size() < 12) {
        return result;
    }

    std::vector<cv::Point2f> reference_points;

    std::vector<cv::Point2f> current_points;

    reference_points.reserve(good_matches.size());

    current_points.reserve(good_matches.size());

    for (const auto& match : good_matches) {
        reference_points.push_back(reference_keypoints_.at(static_cast<std::size_t>(match.queryIdx)).pt);

        current_points.push_back(current_keypoints.at(static_cast<std::size_t>(match.trainIdx)).pt);
    }

    cv::Mat inlier_mask;

    const cv::Mat homography = cv::findHomography(reference_points, current_points, cv::RANSAC, 3.0, inlier_mask);

    if (homography.empty() || inlier_mask.empty()) {
        return result;
    }

    const int inlier_count = cv::countNonZero(inlier_mask);
    // 参考图和实时图匹配数量
    std::cout << "[detect] inliers=" << inlier_count << "/" << good_matches.size() << std::endl;

    if (inlier_count < 8) {
        return result;
    }

    /*
     * 用单应矩阵把期望控制点映射到当前图像。
     */
    const std::vector<cv::Point2f> desired_points(desired_pixel_points_.begin(), desired_pixel_points_.end());

    std::vector<cv::Point2f> current_control_points;

    cv::perspectiveTransform(desired_points, current_control_points, homography);

    if (current_control_points.size() != 4) {
        return result;
    }

    for (std::size_t i = 0; i < 4; ++i) {
        const cv::Point2f& point = current_control_points[i];

        if (!std::isfinite(point.x) || !std::isfinite(point.y) || point.x < 0.0F || point.y < 0.0F || point.x >= static_cast<float>(color_image.cols) ||
            point.y >= static_cast<float>(color_image.rows)) {
            // 越界情况
            std::cout << "[detect] point[" << i << "] out of image: " << point.x << "," << point.y << std::endl;
            return DetectionResult{};
        }

        result.pixel_points[i] = point;

        const double depth = readDepthMedian(depth_image, point, depth_encoding);
        // 每个控制点的深度
        std::cout << "[detect] point[" << i << "]=" << point.x << "," << point.y << ", depth=" << depth << ", encoding=" << depth_encoding << std::endl;

        if (!std::isfinite(depth) || depth <= 0.0) {
            return DetectionResult{};
        }

        result.depths(static_cast<Eigen::Index>(i)) = depth;
    }
    result.confidence = static_cast<double>(inlier_count) / static_cast<double>(good_matches.size());

    result.success = true;

    return result;
}

double PlanarObjectFeatureDetector::readDepthMedian(const cv::Mat& depth_image, const cv::Point2f& point, const std::string& depth_encoding) const {
    if (depth_image.empty() || !std::isfinite(point.x) || !std::isfinite(point.y)) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const int center_u = static_cast<int>(std::lround(point.x));

    const int center_v = static_cast<int>(std::lround(point.y));

    if (center_u < 0 || center_u >= depth_image.cols || center_v < 0 || center_v >= depth_image.rows) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    constexpr int radius = 2;

    const int minimum_u = std::max(0, center_u - radius);

    const int maximum_u = std::min(depth_image.cols - 1, center_u + radius);

    const int minimum_v = std::max(0, center_v - radius);

    const int maximum_v = std::min(depth_image.rows - 1, center_v + radius);

    std::vector<double> valid_depths;

    valid_depths.reserve((2 * radius + 1) * (2 * radius + 1));

    if (depth_encoding == sensor_msgs::image_encodings::TYPE_16UC1) {
        if (depth_image.type() != CV_16UC1) {
            throw std::runtime_error("深度编码是16UC1，但cv::Mat类型不匹配");
        }

        for (int v = minimum_v; v <= maximum_v; ++v) {
            for (int u = minimum_u; u <= maximum_u; ++u) {
                const std::uint16_t raw_depth = depth_image.at<std::uint16_t>(v, u);

                if (raw_depth == 0) {
                    continue;
                }

                const double depth = static_cast<double>(raw_depth) * depth_scale_;

                if (std::isfinite(depth) && depth > 0.0) {
                    valid_depths.push_back(depth);
                }
            }
        }
    } else if (depth_encoding == sensor_msgs::image_encodings::TYPE_32FC1) {
        if (depth_image.type() != CV_32FC1) {
            throw std::runtime_error("深度编码是32FC1，但cv::Mat类型不匹配");
        }

        for (int v = minimum_v; v <= maximum_v; ++v) {
            for (int u = minimum_u; u <= maximum_u; ++u) {
                const float raw_depth = depth_image.at<float>(v, u);

                if (!std::isfinite(raw_depth) || raw_depth <= 0.0F) {
                    continue;
                }

                // 32FC1通常已经以米为单位
                valid_depths.push_back(static_cast<double>(raw_depth));
            }
        }
    } else {
        throw std::runtime_error("不支持的深度图编码: " + depth_encoding);
    }

    if (valid_depths.size() < 3) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    std::sort(valid_depths.begin(), valid_depths.end());

    const std::size_t middle = valid_depths.size() / 2;

    if (valid_depths.size() % 2 == 1) {
        return valid_depths[middle];
    }

    return 0.5 * (valid_depths[middle - 1] + valid_depths[middle]);
}
