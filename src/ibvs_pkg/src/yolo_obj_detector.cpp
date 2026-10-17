#include "yolo_obj_detector.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <opencv2/imgproc.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <stdexcept>

YoloBananaDetector::YoloBananaDetector(const std::string &model_path, double depth_scale, float confidence_threshold, float nms_threshold, int input_size,
                                       int banana_class_id, float desired_center_u_ratio, float desired_center_v_ratio, float desired_box_width_ratio,
                                       float desired_box_height_ratio)
    : depth_scale_(depth_scale), confidence_threshold_(confidence_threshold), nms_threshold_(nms_threshold), input_size_(input_size),
      banana_class_id_(banana_class_id), desired_center_u_ratio_(desired_center_u_ratio), desired_center_v_ratio_(desired_center_v_ratio),
      desired_box_width_ratio_(desired_box_width_ratio), desired_box_height_ratio_(desired_box_height_ratio) {
    if (model_path.empty()) { throw std::invalid_argument("YOLO模型路径为空"); }

    if (!std::isfinite(depth_scale_) || depth_scale_ <= 0.0) { throw std::invalid_argument("深度比例必须是有效正数"); }

    if (input_size_ <= 0) { throw std::invalid_argument("YOLO输入尺寸必须为正数"); }

    if (!std::isfinite(desired_center_u_ratio_) || !std::isfinite(desired_center_v_ratio_) || !std::isfinite(desired_box_width_ratio_) ||
        !std::isfinite(desired_box_height_ratio_)) {
        throw std::invalid_argument("期望框比例必须是有限数值");
    }

    if (desired_box_width_ratio_ <= 0.0F || desired_box_width_ratio_ > 1.0F || desired_box_height_ratio_ <= 0.0F || desired_box_height_ratio_ > 1.0F) {
        throw std::invalid_argument("期望框宽高比例必须在(0,1]范围内");
    }

    const float half_width = 0.5F * desired_box_width_ratio_;
    const float half_height = 0.5F * desired_box_height_ratio_;
    if (desired_center_u_ratio_ - half_width < 0.0F || desired_center_u_ratio_ + half_width > 1.0F || desired_center_v_ratio_ - half_height < 0.0F ||
        desired_center_v_ratio_ + half_height > 1.0F) {
        throw std::invalid_argument("期望框超出归一化图像范围");
    }

    net_ = cv::dnn::readNetFromONNX(model_path);

    if (net_.empty()) { throw std::runtime_error("YOLO模型加载失败: " + model_path); }

    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
}

YoloBananaDetector::LetterboxResult YoloBananaDetector::letterbox(const cv::Mat &image) const {
    if (image.empty()) { throw std::invalid_argument("YOLO输入图像为空"); }

    const float scale =
        std::min(static_cast<float>(input_size_) / static_cast<float>(image.cols), static_cast<float>(input_size_) / static_cast<float>(image.rows));

    const int resized_width = static_cast<int>(std::lround(image.cols * scale));

    const int resized_height = static_cast<int>(std::lround(image.rows * scale));

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

std::optional<YoloBananaDetector::BananaBox> YoloBananaDetector::detectBestBanana(const cv::Mat &image) {
    if (image.empty()) { return std::nullopt; }

    const LetterboxResult prepared = letterbox(image);

    const cv::Mat blob = cv::dnn::blobFromImage(prepared.image, 1.0 / 255.0, cv::Size(input_size_, input_size_), cv::Scalar(), true, false);

    net_.setInput(blob);

    std::vector<cv::Mat> outputs;

    net_.forward(outputs, net_.getUnconnectedOutLayersNames());

    if (outputs.empty()) { return std::nullopt; }

    cv::Mat output = outputs.front();

    if (output.type() != CV_32F) { throw std::runtime_error("YOLO输出类型不是CV_32F"); }

    if (output.dims != 3 || output.size[0] != 1) { throw std::runtime_error("不支持的YOLO输出维度"); }

    if (!output.isContinuous()) { output = output.clone(); }

    const int dimension_one = output.size[1];
    const int dimension_two = output.size[2];

    /*
     * 当前模型输出为：
     * (1, 84, 8400)
     *
     * 同时兼容：
     * (1, 8400, 84)
     */
    const bool attributes_first = dimension_one < dimension_two;

    const int attribute_count = attributes_first ? dimension_one : dimension_two;

    const int prediction_count = attributes_first ? dimension_two : dimension_one;

    if (attribute_count <= 4 + banana_class_id_) { throw std::runtime_error("YOLO输出类别数量不足"); }

    cv::Mat predictions;

    if (attributes_first) {
        const cv::Mat raw(attribute_count, prediction_count, CV_32F, output.ptr<float>());

        cv::transpose(raw, predictions);
    } else {
        predictions = cv::Mat(prediction_count, attribute_count, CV_32F, output.ptr<float>());
    }

    std::vector<cv::Rect> candidate_boxes;
    std::vector<float> candidate_scores;

    for (int row_index = 0; row_index < predictions.rows; ++row_index) {
        const float *row = predictions.ptr<float>(row_index);

        const cv::Mat class_scores(1, attribute_count - 4, CV_32F, const_cast<float *>(row + 4));

        double best_score = 0.0;
        cv::Point best_class_location;

        cv::minMaxLoc(class_scores, nullptr, &best_score, nullptr, &best_class_location);

        const int class_id = best_class_location.x;

        if (class_id != banana_class_id_) { continue; }

        if (best_score < confidence_threshold_) { continue; }

        const float center_x = row[0];
        const float center_y = row[1];
        const float box_width = row[2];
        const float box_height = row[3];

        float left = center_x - 0.5F * box_width;

        float top = center_y - 0.5F * box_height;

        float right = center_x + 0.5F * box_width;

        float bottom = center_y + 0.5F * box_height;

        /*
         * 从640×640 letterbox坐标恢复到原图坐标。
         */
        left = (left - prepared.pad_x) / prepared.scale;

        right = (right - prepared.pad_x) / prepared.scale;

        top = (top - prepared.pad_y) / prepared.scale;

        bottom = (bottom - prepared.pad_y) / prepared.scale;

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

        candidate_boxes.emplace_back(integer_left, integer_top, width, height);

        candidate_scores.push_back(static_cast<float>(best_score));
    }

    if (candidate_boxes.empty()) { return std::nullopt; }

    std::vector<int> selected_indices;

    cv::dnn::NMSBoxes(candidate_boxes, candidate_scores, confidence_threshold_, nms_threshold_, selected_indices);

    if (selected_indices.empty()) { return std::nullopt; }

    const auto best_iterator = std::max_element(selected_indices.begin(), selected_indices.end(),
                                                [&candidate_scores](int left, int right) { return candidate_scores[left] < candidate_scores[right]; });

    const int best_index = *best_iterator;

    BananaBox result;

    result.box = cv::Rect2f(candidate_boxes[best_index]);

    result.confidence = candidate_scores[best_index];

    return result;
}

IbvsController::PixelPoints YoloBananaDetector::boxToPoints(const cv::Rect2f &box) const {
    constexpr float inset = 0.30F;

    const float left = box.x + inset * box.width;

    const float right = box.x + (1.0F - inset) * box.width;

    const float top = box.y + inset * box.height;

    const float bottom = box.y + (1.0F - inset) * box.height;

    return {cv::Point2f(left, top), cv::Point2f(right, top), cv::Point2f(right, bottom), cv::Point2f(left, bottom)};
}

IbvsController::PixelPoints YoloBananaDetector::desiredPixelPoints(const cv::Size &image_size) const {
    if (image_size.width <= 0 || image_size.height <= 0) { throw std::invalid_argument("实时图像尺寸无效"); }

    const float image_width = static_cast<float>(image_size.width);
    const float image_height = static_cast<float>(image_size.height);
    const float box_width = desired_box_width_ratio_ * image_width;
    const float box_height = desired_box_height_ratio_ * image_height;
    const float center_u = desired_center_u_ratio_ * image_width;
    const float center_v = desired_center_v_ratio_ * image_height;
    const cv::Rect2f desired_box(center_u - 0.5F * box_width, center_v - 0.5F * box_height, box_width, box_height);
    return boxToPoints(desired_box);
}

DetectionResult YoloBananaDetector::detect(const cv::Mat &color_image, const cv::Mat &depth_image, const std::string &depth_encoding) {
    DetectionResult result;

    if (color_image.empty() || depth_image.empty()) { return result; }

    if (color_image.size() != depth_image.size()) { return result; }

    const auto banana = detectBestBanana(color_image);

    if (!banana.has_value()) { return result; }

    result.pixel_points = boxToPoints(banana->box);

    for (std::size_t index = 0; index < result.pixel_points.size(); ++index) {
        const double depth = readDepthMedian(depth_image, result.pixel_points[index], depth_encoding);

        if (!std::isfinite(depth) || depth <= 0.0) { return DetectionResult{}; }

        result.depths(static_cast<Eigen::Index>(index)) = depth;
    }

    result.confidence = banana->confidence;

    result.success = true;

    return result;
}

double YoloBananaDetector::readDepthMedian(const cv::Mat &depth_image, const cv::Point2f &point, const std::string &depth_encoding) const {
    if (depth_image.empty() || !std::isfinite(point.x) || !std::isfinite(point.y)) { return std::numeric_limits<double>::quiet_NaN(); }

    const int center_u = static_cast<int>(std::lround(point.x));

    const int center_v = static_cast<int>(std::lround(point.y));

    if (center_u < 0 || center_u >= depth_image.cols || center_v < 0 || center_v >= depth_image.rows) { return std::numeric_limits<double>::quiet_NaN(); }

    constexpr int radius = 2;

    const int minimum_u = std::max(0, center_u - radius);

    const int maximum_u = std::min(depth_image.cols - 1, center_u + radius);

    const int minimum_v = std::max(0, center_v - radius);

    const int maximum_v = std::min(depth_image.rows - 1, center_v + radius);

    std::vector<double> valid_depths;

    if (depth_encoding == sensor_msgs::image_encodings::TYPE_16UC1) {
        if (depth_image.type() != CV_16UC1) { throw std::runtime_error("16UC1编码与cv::Mat类型不匹配"); }

        for (int v = minimum_v; v <= maximum_v; ++v) {
            for (int u = minimum_u; u <= maximum_u; ++u) {
                const std::uint16_t raw_depth = depth_image.at<std::uint16_t>(v, u);

                if (raw_depth == 0) { continue; }

                const double depth = raw_depth * depth_scale_;

                if (std::isfinite(depth) && depth > 0.0) { valid_depths.push_back(depth); }
            }
        }
    } else if (depth_encoding == sensor_msgs::image_encodings::TYPE_32FC1) {
        if (depth_image.type() != CV_32FC1) { throw std::runtime_error("32FC1编码与cv::Mat类型不匹配"); }

        for (int v = minimum_v; v <= maximum_v; ++v) {
            for (int u = minimum_u; u <= maximum_u; ++u) {
                const float depth = depth_image.at<float>(v, u);

                if (std::isfinite(depth) && depth > 0.0F) { valid_depths.push_back(depth); }
            }
        }
    } else {
        throw std::runtime_error("不支持的深度编码: " + depth_encoding);
    }

    if (valid_depths.size() < 3) { return std::numeric_limits<double>::quiet_NaN(); }

    std::sort(valid_depths.begin(), valid_depths.end());

    const std::size_t middle = valid_depths.size() / 2;

    if (valid_depths.size() % 2 == 1) { return valid_depths[middle]; }

    return 0.5 * (valid_depths[middle - 1] + valid_depths[middle]);
}
