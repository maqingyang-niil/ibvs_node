#include "yolo_red_button_detector.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <stdexcept>
#include <vector>

namespace {
void checkCuda(cudaError_t status, const char *operation) {
    if (status != cudaSuccess) { throw std::runtime_error(std::string(operation) + "失败: " + cudaGetErrorString(status)); }
}
}

void YoloRedButtonDetector::TensorRtLogger::log(Severity severity, const char *message) noexcept {
    if (severity <= Severity::kWARNING) { std::cerr << "[TensorRT] " << message << '\n'; }
}

YoloRedButtonDetector::YoloRedButtonDetector(const std::string &engine_path, double depth_scale, float confidence_threshold, float nms_threshold,
                                             int input_size, int button_class_id, double maximum_center_offset_ratio,
                                             float desired_center_u_ratio, float desired_center_v_ratio, float desired_radius_ratio,
                                             float feature_offset_ratio)
    : depth_scale_(depth_scale), confidence_threshold_(confidence_threshold), nms_threshold_(nms_threshold), input_size_(input_size),
      button_class_id_(button_class_id),
      maximum_center_offset_ratio_(maximum_center_offset_ratio), desired_center_u_ratio_(desired_center_u_ratio),
      desired_center_v_ratio_(desired_center_v_ratio), desired_radius_ratio_(desired_radius_ratio), feature_offset_ratio_(feature_offset_ratio) {
    if (engine_path.empty()) { throw std::invalid_argument("TensorRT engine路径为空"); }
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

    try {
        initializeTensorRt(engine_path);
    } catch (...) {
        releaseCudaResources();
        throw;
    }
}

YoloRedButtonDetector::~YoloRedButtonDetector() { releaseCudaResources(); }

std::size_t YoloRedButtonDetector::tensorElementCount(const nvinfer1::Dims &dimensions) {
    if (dimensions.nbDims <= 0) { throw std::runtime_error("TensorRT张量维度为空"); }
    std::size_t count = 1;
    for (int index = 0; index < dimensions.nbDims; ++index) {
        if (dimensions.d[index] <= 0) { throw std::runtime_error("TensorRT engine包含动态或无效张量维度"); }
        count *= static_cast<std::size_t>(dimensions.d[index]);
    }
    return count;
}

void YoloRedButtonDetector::initializeTensorRt(const std::string &engine_path) {
    std::ifstream engine_file(engine_path, std::ios::binary | std::ios::ate);
    if (!engine_file) { throw std::runtime_error("无法打开TensorRT engine: " + engine_path); }

    const std::streamsize engine_size = engine_file.tellg();
    if (engine_size <= 0) { throw std::runtime_error("TensorRT engine文件为空: " + engine_path); }
    engine_file.seekg(0, std::ios::beg);
    std::vector<char> engine_data(static_cast<std::size_t>(engine_size));
    if (!engine_file.read(engine_data.data(), engine_size)) { throw std::runtime_error("读取TensorRT engine失败: " + engine_path); }

    tensor_rt_runtime_.reset(nvinfer1::createInferRuntime(tensor_rt_logger_));
    if (!tensor_rt_runtime_) { throw std::runtime_error("创建TensorRT runtime失败"); }
    tensor_rt_engine_.reset(tensor_rt_runtime_->deserializeCudaEngine(engine_data.data(), engine_data.size()));
    if (!tensor_rt_engine_) { throw std::runtime_error("反序列化TensorRT engine失败，请检查设备和TensorRT版本"); }
    tensor_rt_context_.reset(tensor_rt_engine_->createExecutionContext());
    if (!tensor_rt_context_) { throw std::runtime_error("创建TensorRT执行上下文失败"); }

    int input_count = 0;
    int output_count = 0;
    const int io_tensor_count = tensor_rt_engine_->getNbIOTensors();
    for (int index = 0; index < io_tensor_count; ++index) {
        const char *tensor_name = tensor_rt_engine_->getIOTensorName(index);
        if (tensor_name == nullptr) { throw std::runtime_error("TensorRT engine包含无效I/O名称"); }
        const nvinfer1::TensorIOMode mode = tensor_rt_engine_->getTensorIOMode(tensor_name);
        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            input_name_ = tensor_name;
            ++input_count;
        } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
            output_name_ = tensor_name;
            ++output_count;
        }
    }
    if (input_count != 1 || output_count != 1) { throw std::runtime_error("TensorRT engine必须恰好包含一个输入和一个输出"); }
    if (tensor_rt_engine_->getTensorDataType(input_name_.c_str()) != nvinfer1::DataType::kFLOAT ||
        tensor_rt_engine_->getTensorDataType(output_name_.c_str()) != nvinfer1::DataType::kFLOAT) {
        throw std::runtime_error("TensorRT engine输入输出必须是float32");
    }

    const nvinfer1::Dims input_dimensions = tensor_rt_engine_->getTensorShape(input_name_.c_str());
    if (input_dimensions.nbDims != 4 || input_dimensions.d[0] != 1 || input_dimensions.d[1] != 3 ||
        input_dimensions.d[2] != input_size_ || input_dimensions.d[3] != input_size_) {
        throw std::runtime_error("TensorRT engine输入形状与yolo_input_size配置不匹配");
    }

    const nvinfer1::Dims output_dimensions = tensor_rt_engine_->getTensorShape(output_name_.c_str());
    if (output_dimensions.nbDims != 3 || output_dimensions.d[0] != 1 || output_dimensions.d[1] <= 0 || output_dimensions.d[2] <= 0) {
        throw std::runtime_error("TensorRT engine输出应为(1,属性数,预测数)或(1,预测数,属性数)");
    }
    output_attributes_first_ = output_dimensions.d[1] < output_dimensions.d[2];
    output_attribute_count_ = static_cast<int>(output_attributes_first_ ? output_dimensions.d[1] : output_dimensions.d[2]);
    output_prediction_count_ = static_cast<int>(output_attributes_first_ ? output_dimensions.d[2] : output_dimensions.d[1]);
    if (output_attribute_count_ <= 4 + button_class_id_) { throw std::runtime_error("TensorRT engine输出类别数量不足"); }

    input_element_count_ = tensorElementCount(input_dimensions);
    output_element_count_ = tensorElementCount(output_dimensions);
    checkCuda(cudaStreamCreate(&cuda_stream_), "创建CUDA stream");
    checkCuda(cudaMalloc(&device_input_, input_element_count_ * sizeof(float)), "分配TensorRT输入显存");
    checkCuda(cudaMalloc(&device_output_, output_element_count_ * sizeof(float)), "分配TensorRT输出显存");
    if (!tensor_rt_context_->setTensorAddress(input_name_.c_str(), device_input_) ||
        !tensor_rt_context_->setTensorAddress(output_name_.c_str(), device_output_)) {
        throw std::runtime_error("设置TensorRT输入输出地址失败");
    }
}

void YoloRedButtonDetector::releaseCudaResources() noexcept {
    if (cuda_stream_ != nullptr) { cudaStreamSynchronize(cuda_stream_); }
    if (device_output_ != nullptr) { cudaFree(device_output_); }
    if (device_input_ != nullptr) { cudaFree(device_input_); }
    if (cuda_stream_ != nullptr) { cudaStreamDestroy(cuda_stream_); }
    device_output_ = nullptr;
    device_input_ = nullptr;
    cuda_stream_ = nullptr;
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

    if (input_values.size() != input_element_count_) { throw std::runtime_error("TensorRT输入元素数量与engine不匹配"); }
    std::vector<float> output_values(output_element_count_);
    checkCuda(cudaMemcpyAsync(device_input_, input_values.data(), input_element_count_ * sizeof(float), cudaMemcpyHostToDevice, cuda_stream_),
              "复制TensorRT输入到GPU");
    if (!tensor_rt_context_->enqueueV3(cuda_stream_)) { throw std::runtime_error("TensorRT enqueueV3推理失败"); }
    checkCuda(cudaMemcpyAsync(output_values.data(), device_output_, output_element_count_ * sizeof(float), cudaMemcpyDeviceToHost, cuda_stream_),
              "复制TensorRT输出到CPU");
    checkCuda(cudaStreamSynchronize(cuda_stream_), "等待TensorRT推理完成");

    cv::Mat predictions;
    if (output_attributes_first_) {
        const cv::Mat raw(output_attribute_count_, output_prediction_count_, CV_32F, output_values.data());
        cv::transpose(raw, predictions);
    } else {
        const cv::Mat raw(output_prediction_count_, output_attribute_count_, CV_32F, output_values.data());
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
