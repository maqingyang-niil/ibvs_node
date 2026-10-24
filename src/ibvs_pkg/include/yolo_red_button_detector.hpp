#pragma once

#include <memory>
#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <optional>
#include <string>

#include "detection_result.hpp"

class YoloRedButtonDetector {
  public:
    YoloRedButtonDetector(const std::string &model_path, double depth_scale, float confidence_threshold = 0.5F, float nms_threshold = 0.45F,
                          int input_size = 640, int button_class_id = 0, double maximum_center_offset_ratio = 0.25,
                          float desired_center_u_ratio = 0.5F, float desired_center_v_ratio = 0.5F, float desired_radius_ratio = 0.08F,
                          float feature_offset_ratio = 0.35F);

    DetectionResult detect(const cv::Mat &color_image, const cv::Mat &depth_image, const std::string &depth_encoding);

    IbvsController::PixelPoints desiredPixelPoints(const cv::Size &image_size) const;

  private:
    struct ButtonBox {
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

    std::optional<ButtonBox> detectBestButton(const cv::Mat &image);

    IbvsController::PixelPoints circleToPoints(const cv::Point2f &center, float radius) const;

    double readDepthMedian(const cv::Mat &depth_image, const cv::Point2f &center, float radius, const std::string &depth_encoding) const;

    Ort::Env ort_env_;
    Ort::SessionOptions ort_session_options_;
    std::unique_ptr<Ort::Session> ort_session_;
    std::string input_name_;
    std::string output_name_;
    double depth_scale_;
    float confidence_threshold_;
    float nms_threshold_;
    int input_size_;
    int button_class_id_;
    double maximum_center_offset_ratio_;
    float desired_center_u_ratio_;
    float desired_center_v_ratio_;
    float desired_radius_ratio_;
    float feature_offset_ratio_;
};
