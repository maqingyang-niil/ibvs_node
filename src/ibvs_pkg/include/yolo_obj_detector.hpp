#pragma once

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <optional>
#include <string>
#include <vector>

#include "detection_result.hpp"

class YoloBananaDetector {
public:
    YoloBananaDetector(const std::string &model_path, double depth_scale, float confidence_threshold = 0.5F, float nms_threshold = 0.45F, int input_size = 640,
                       int banana_class_id = 46, float desired_center_u_ratio = 0.5F, float desired_center_v_ratio = 0.5F, float desired_box_width_ratio = 0.4F,
                       float desired_box_height_ratio = 0.4F);

    DetectionResult detect(const cv::Mat &color_image, const cv::Mat &depth_image, const std::string &depth_encoding);

    IbvsController::PixelPoints desiredPixelPoints(const cv::Size &image_size) const;

private:
    struct BananaBox {
        cv::Rect2f box;
        float confidence{0.0F};
    };

    struct LetterboxResult {
        cv::Mat image;
        float scale{1.0F};
        int pad_x{0};
        int pad_y{0};
    };

    LetterboxResult letterbox(const cv::Mat &image) const;

    std::optional<BananaBox> detectBestBanana(const cv::Mat &image);

    IbvsController::PixelPoints boxToPoints(const cv::Rect2f &box) const;

    double readDepthMedian(const cv::Mat &depth_image, const cv::Point2f &point, const std::string &depth_encoding) const;

    cv::dnn::Net net_;

    double depth_scale_;
    float confidence_threshold_;
    float nms_threshold_;
    int input_size_;
    int banana_class_id_;
    float desired_center_u_ratio_;
    float desired_center_v_ratio_;
    float desired_box_width_ratio_;
    float desired_box_height_ratio_;
};
