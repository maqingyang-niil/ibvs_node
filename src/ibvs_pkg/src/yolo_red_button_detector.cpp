#include "yolo_red_button_detector.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <stdexcept>
#include <vector>

YoloRedButtonDetector::YoloRedButtonDetector(const std::string &model_path, double depth_scale, float confidence_threshold, float nms_threshold,
                                             int input_size, int button_class_id, double maximum_center_offset_ratio,
                                             float desired_center_u_ratio, float desired_center_v_ratio, float desired_radius_ratio,
                                             float feature_offset_ratio)
    : ort_env_(ORT_LOGGING_LEVEL_WARNING, "ibvs_yolo"), depth_scale_(depth_scale), confidence_threshold_(confidence_threshold),
      nms_threshold_(nms_threshold), input_size_(input_size), button_class_id_(button_class_id),
      maximum_center_offset_ratio_(maximum_center_offset_ratio), desired_center_u_ratio_(desired_center_u_ratio),
      desired_center_v_ratio_(desired_center_v_ratio), desired_radius_ratio_(desired_radius_ratio), feature_offset_ratio_(feature_offset_ratio) {
    if (model_path.empty()) { throw std::invalid_argument("YOLO模型路径为空"); }
    if (!std::isfinite(depth_scale_) || depth_scale_ <= 0.0) { throw std::invalid_argument("深度比例必须是有效正数"); }
    if (!std::isfinite(confidence_threshold_) || confidence_threshold_ < 0.0F || confidence_threshold_ > 1.0F) {
        throw std::invalid_argument("YOLO置信度阈值必须在[0,1]范围内");
    }
    if (!std::isfinite(nms_threshold_) || nms_threshold_ < 0.0F || nms_threshold_ > 1.0F) {
        throw std::invalid_argument("YOLO NMS阈值必须在[0,1]范围内");
    }
    if (input_size_ <= 0) { throw std::invalid_argument("YOLO输入尺寸必须为正数"); }
    if (button_class_id_ < 0) { throw std::invalid_argument("YOLO按钮类别编号不能为负数"); }
    if (!std::isfinite(maximum_center_offset_ratio_) || maximum_center_offset_ratio_ <= 0.0 || maximum_center_offset_ratio_ > 1.0) {
        throw std::invalid_argument("按钮最大中心偏移比例必须在(0,1]范围内");
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

    ort_session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    ort_session_options_.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    ort_session_ = std::make_unique<Ort::Session>(ort_env_, model_path.c_str(), ort_session_options_);
    if (ort_session_->GetInputCount() != 1 || ort_session_->GetOutputCount() < 1) {
        throw std::runtime_error("YOLO模型输入输出数量不受支持");
    }

    Ort::AllocatorWithDefaultOptions allocator;
    const auto input_name = ort_session_->GetInputNameAllocated(0, allocator);
    const auto output_name = ort_session_->GetOutputNameAllocated(0, allocator);
    input_name_ = input_name.get();
    output_name_ = output_name.get();

    const std::vector<int64_t> input_shape = ort_session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    if (input_shape.size() != 4 || (input_shape[1] != 3 && input_shape[1] != -1) ||
        (input_shape[2] != input_size_ && input_shape[2] != -1) || (input_shape[3] != input_size_ && input_shape[3] != -1)) {
        throw std::runtime_error("YOLO模型输入形状与yolo_input_size配置不匹配");
    }
}

YoloRedButtonDetector::LetterboxResult YoloRedButtonDetector::letterbox(const cv::Mat &image) const {
    if (image.empty()) { throw std::invalid_argument("YOLO输入图像为空"); }

    const float scale =
        std::min(static_cast<float>(input_size_) / static_cast<float>(image.cols), static_cast<float>(input_size_) / static_cast<float>(image.rows));
    const int resized_width = static_cast<int>(std::lround(static_cast<float>(image.cols) * scale));
    const int resized_height = static_cast<int>(std::lround(static_cast<float>(image.rows) * scale));
    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(resized_width, resized_height));

    const int remaining_width = input_size_ - resized_width;
    const int remaining_height = input_size_ - resized_height;
    const int pad_left = remaining_width / 2;
    const int pad_right = remaining_width - pad_left;
    const int pad_top = remaining_height / 2;
    const int pad_bottom = remaining_height - pad_top;

    LetterboxResult result;
    result.scale = scale;
    result.pad_x = pad_left;
    result.pad_y = pad_top;
    cv::copyMakeBorder(resized_image, result.image, pad_top, pad_bottom, pad_left, pad_right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    return result;
}

std::optional<YoloRedButtonDetector::ButtonBox> YoloRedButtonDetector::detectBestButton(const cv::Mat &image) {
    if (image.empty()) { return std::nullopt; }

    const LetterboxResult prepared = letterbox(image);
    cv::Mat rgb_float;
    cv::cvtColor(prepared.image, rgb_float, cv::COLOR_BGR2RGB);
    rgb_float.convertTo(rgb_float, CV_32FC3, 1.0 / 255.0);

    const std::size_t plane_size = static_cast<std::size_t>(input_size_) * static_cast<std::size_t>(input_size_);
    std::vector<float> input_values(3 * plane_size);
    std::vector<cv::Mat> input_channels;
    input_channels.reserve(3);
    for (std::size_t channel = 0; channel < 3; ++channel) {
        input_channels.emplace_back(input_size_, input_size_, CV_32F, input_values.data() + channel * plane_size);
    }
    cv::split(rgb_float, input_channels);

    const std::array<int64_t, 4> input_shape{1, 3, input_size_, input_size_};
    const Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor =
        Ort::Value::CreateTensor<float>(memory_info, input_values.data(), input_values.size(), input_shape.data(), input_shape.size());
    const char *input_names[] = {input_name_.c_str()};
    const char *output_names[] = {output_name_.c_str()};
    std::vector<Ort::Value> outputs = ort_session_->Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
    if (outputs.empty() || !outputs.front().IsTensor()) { return std::nullopt; }

    const auto output_info = outputs.front().GetTensorTypeAndShapeInfo();
    const std::vector<int64_t> output_shape = output_info.GetShape();
    if (output_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) { throw std::runtime_error("YOLO输出类型不是float32"); }
    if (output_shape.size() != 3 || output_shape[0] != 1 || output_shape[1] <= 0 || output_shape[2] <= 0) {
        throw std::runtime_error("YOLO输出应为(1,属性数,预测数)或(1,预测数,属性数)");
    }

    const int dimension_one = static_cast<int>(output_shape[1]);
    const int dimension_two = static_cast<int>(output_shape[2]);
    const bool attributes_first = dimension_one < dimension_two;
    const int attribute_count = attributes_first ? dimension_one : dimension_two;
    const int prediction_count = attributes_first ? dimension_two : dimension_one;
    if (attribute_count <= 4 + button_class_id_) { throw std::runtime_error("YOLO输出类别数量不足，请检查yolo_button_class_id"); }

    cv::Mat predictions;
    if (attributes_first) {
        const cv::Mat raw(attribute_count, prediction_count, CV_32F, outputs.front().GetTensorMutableData<float>());
        cv::transpose(raw, predictions);
    } else {
        const cv::Mat raw(prediction_count, attribute_count, CV_32F, outputs.front().GetTensorMutableData<float>());
        predictions = raw.clone();
    }

    const cv::Point2f desired_center(desired_center_u_ratio_ * static_cast<float>(image.cols),
                                     desired_center_v_ratio_ * static_cast<float>(image.rows));
    std::vector<cv::Rect> candidate_boxes;
    std::vector<float> candidate_scores;
    for (int row_index = 0; row_index < predictions.rows; ++row_index) {
        const float *row = predictions.ptr<float>(row_index);
        const float score = row[4 + button_class_id_];
        if (!std::isfinite(score) || score < confidence_threshold_) { continue; }

        float left = row[0] - 0.5F * row[2];
        float top = row[1] - 0.5F * row[3];
        float right = row[0] + 0.5F * row[2];
        float bottom = row[1] + 0.5F * row[3];
        left = (left - static_cast<float>(prepared.pad_x)) / prepared.scale;
        right = (right - static_cast<float>(prepared.pad_x)) / prepared.scale;
        top = (top - static_cast<float>(prepared.pad_y)) / prepared.scale;
        bottom = (bottom - static_cast<float>(prepared.pad_y)) / prepared.scale;
        left = std::clamp(left, 0.0F, static_cast<float>(image.cols - 1));
        right = std::clamp(right, 0.0F, static_cast<float>(image.cols - 1));
        top = std::clamp(top, 0.0F, static_cast<float>(image.rows - 1));
        bottom = std::clamp(bottom, 0.0F, static_cast<float>(image.rows - 1));

        const int integer_left = static_cast<int>(std::floor(left));
        const int integer_top = static_cast<int>(std::floor(top));
        const int integer_right = static_cast<int>(std::ceil(right));
        const int integer_bottom = static_cast<int>(std::ceil(bottom));
        const int width = integer_right - integer_left;
        const int height = integer_bottom - integer_top;
        if (width < 2 || height < 2) { continue; }

        const cv::Point2f center(0.5F * static_cast<float>(integer_left + integer_right),
                                 0.5F * static_cast<float>(integer_top + integer_bottom));
        const double normalized_u_offset = static_cast<double>(center.x - desired_center.x) / static_cast<double>(image.cols);
        const double normalized_v_offset = static_cast<double>(center.y - desired_center.y) / static_cast<double>(image.rows);
        if (std::hypot(normalized_u_offset, normalized_v_offset) > maximum_center_offset_ratio_) { continue; }

        candidate_boxes.emplace_back(integer_left, integer_top, width, height);
        candidate_scores.push_back(score);
    }

    if (candidate_boxes.empty()) { return std::nullopt; }
    std::vector<int> selected_indices;
    cv::dnn::NMSBoxes(candidate_boxes, candidate_scores, confidence_threshold_, nms_threshold_, selected_indices);
    if (selected_indices.empty()) { return std::nullopt; }

    const auto best_iterator = std::max_element(selected_indices.begin(), selected_indices.end(),
                                                [&candidate_scores](int left, int right) { return candidate_scores[left] < candidate_scores[right]; });
    const int best_index = *best_iterator;
    return ButtonBox{cv::Rect2f(candidate_boxes[best_index]), candidate_scores[best_index]};
}

IbvsController::PixelPoints YoloRedButtonDetector::circleToPoints(const cv::Point2f &center, float radius) const {
    const float offset = feature_offset_ratio_ * radius;
    return {cv::Point2f(center.x - offset, center.y - offset), cv::Point2f(center.x + offset, center.y - offset),
            cv::Point2f(center.x + offset, center.y + offset), cv::Point2f(center.x - offset, center.y + offset)};
}

IbvsController::PixelPoints YoloRedButtonDetector::desiredPixelPoints(const cv::Size &image_size) const {
    if (image_size.width <= 0 || image_size.height <= 0) { throw std::invalid_argument("实时图像尺寸无效"); }
    const cv::Point2f center(desired_center_u_ratio_ * static_cast<float>(image_size.width),
                             desired_center_v_ratio_ * static_cast<float>(image_size.height));
    const float radius = desired_radius_ratio_ * static_cast<float>(std::min(image_size.width, image_size.height));
    return circleToPoints(center, radius);
}

DetectionResult YoloRedButtonDetector::detect(const cv::Mat &color_image, const cv::Mat &depth_image, const std::string &depth_encoding) {
    DetectionResult result;
    if (color_image.empty() || depth_image.empty() || color_image.size() != depth_image.size()) { return result; }

    const auto button = detectBestButton(color_image);
    if (!button.has_value()) { return result; }

    const cv::Point2f center(button->box.x + 0.5F * button->box.width, button->box.y + 0.5F * button->box.height);
    const float radius = 0.5F * std::min(button->box.width, button->box.height);
    if (!std::isfinite(radius) || radius < 2.0F) { return result; }

    result.confidence = button->confidence;
    result.pixel_points = circleToPoints(center, radius);
    const double depth = readDepthMedian(depth_image, center, radius, depth_encoding);
    if (!std::isfinite(depth) || depth <= 0.0) {
        result.invalid_depth_index = 0;
        return result;
    }

    result.depths.setConstant(depth);
    result.success = true;
    return result;
}

double YoloRedButtonDetector::readDepthMedian(const cv::Mat &depth_image, const cv::Point2f &center, float radius,
                                              const std::string &depth_encoding) const {
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
