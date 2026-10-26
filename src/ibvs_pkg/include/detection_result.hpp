#pragma once

#include <opencv2/core.hpp>

struct DetectionResult {
    bool success{false};
    bool detected{false};
    cv::Point2f center_pixel{};
    cv::Rect2f bounding_box{};
    double depth{0.0};
    double confidence{0.0};
};
