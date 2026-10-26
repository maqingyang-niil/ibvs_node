#pragma once

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <opencv2/core.hpp>
#include <optional>
#include <string>
#include <vector>

#include "detection_result.hpp"

class YoloRedButtonDetector {
  public:
    YoloRedButtonDetector(const std::string &engine_path, double depth_scale, float confidence_threshold = 0.5F, float nms_threshold = 0.45F,
                          int input_size = 640, int button_class_id = 0);

    ~YoloRedButtonDetector();

    DetectionResult detect(const cv::Mat &color_image, const cv::Mat &depth_image, const std::string &depth_encoding);

  private:
    class TensorRtLogger : public nvinfer1::ILogger {
      public:
        void log(Severity severity, const char *message) noexcept override;
    };

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

    double readDepthMedian(const cv::Mat &depth_image, const cv::Point2f &center, float radius, const std::string &depth_encoding) const;

    void initializeTensorRt(const std::string &engine_path);

    void releaseCudaResources() noexcept;

    static std::size_t tensorElementCount(const nvinfer1::Dims &dimensions);

    TensorRtLogger tensor_rt_logger_;
    std::unique_ptr<nvinfer1::IRuntime> tensor_rt_runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> tensor_rt_engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> tensor_rt_context_;
    cudaStream_t cuda_stream_{nullptr};
    void *device_input_{nullptr};
    void *device_output_{nullptr};
    std::string input_name_;
    std::string output_name_;
    std::size_t input_element_count_{0};
    std::size_t output_element_count_{0};
    int output_attribute_count_{0};
    int output_prediction_count_{0};
    bool output_attributes_first_{false};
    double depth_scale_;
    float confidence_threshold_;
    float nms_threshold_;
    int input_size_;
    int button_class_id_;
};
