#ifndef _YOLOV5_DETECTOR_HPP_
#define _YOLOV5_DETECTOR_HPP_
#pragma once

#include "yolov5_detector_internal.h"

namespace yolov5 {
class Detector {
 public:
  Detector() noexcept;
  ~Detector() noexcept;

 private:
  Detector(const Detector& src) noexcept;

 public:
  Result init(int flags = 0) noexcept;

  bool isInitialized() const noexcept;

  Result loadEngine(const std::string& filepath) noexcept;

  Result loadEngine(const std::vector<char>& data) noexcept;

  bool isEngineLoaded() const noexcept;

  int numClasses() const noexcept;

  Result setClasses(const Classes& classes) noexcept;

  Result detect(const cv::Mat& img, std::vector<Detection>* out,
                int flags = 0) noexcept;

  Result detect(const cv::cuda::GpuMat& img, std::vector<Detection>* out,
                int flags = 0) noexcept;

  Result detectBatch(const std::vector<cv::Mat>& images,
                     std::vector<std::vector<Detection>>* out,
                     int flags = 0) noexcept;

  Result detectBatch(const std::vector<cv::cuda::GpuMat>& images,
                     std::vector<std::vector<Detection>>* out,
                     int flags = 0) noexcept;

  double scoreThreshold() const noexcept;

  Result setScoreThreshold(const double& v) noexcept;

  double nmsThreshold() const noexcept;

  Result setNmsThreshold(const double& v) noexcept;

  int batchSize() const noexcept;

  cv::Size inferenceSize() const noexcept;

  Result setLogger(std::shared_ptr<Logger> logger) noexcept;

  std::shared_ptr<Logger> logger() const noexcept;

 private:
  Detector& operator=(const Detector& rhs);

  Result _loadEngine(const std::vector<char>& data) noexcept;

  void _printBindings(
      const std::unique_ptr<nvinfer1::ICudaEngine>& engine) const noexcept;

  int _batchSize() const noexcept;

  int _numClasses() const noexcept;

  Result _detect(std::vector<Detection>* out);

  Result _detectBatch(const int& nrImages,
                      std::vector<std::vector<Detection>>* out);

  Result _inference(const char* logid);

  Result _decodeOutput(const char* logid, const int& index,
                       std::vector<Detection>* out);

 private:
  bool _initialized;

  std::shared_ptr<Logger> _logger;

  Classes _classes;
  double _scoreThreshold;
  double _nmsThreshold;

  /*  TensorRT    */
  std::unique_ptr<TensorRT_Logger> _trtLogger;
  std::unique_ptr<nvinfer1::IRuntime> _trtRuntime;

  /*  note: execution context depends on the engine, and should be destroyed
          _before_ the engine is destroyed */
  std::unique_ptr<nvinfer1::ICudaEngine> _trtEngine;
  std::unique_ptr<nvinfer1::IExecutionContext> _trtExecutionContext;

  /*  I/O  */
  internal::EngineBinding _inputBinding;
  internal::EngineBinding _outputBinding;

  std::unique_ptr<internal::Preprocessor> _preprocessor;

  internal::DeviceMemory _deviceMemory;

  std::vector<float> _outputHostMemory;
};

} /*  namespace yolov5    */

#endif /*  include guard   */