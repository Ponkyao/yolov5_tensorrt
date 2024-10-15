#ifndef _YOLOV5_BUILDER_HPP_
#define _YOLOV5_BUILDER_HPP_
#pragma once

#include <cstring>
#include <memory>
#include <vector>

#include "yolov5_logging.h"

namespace yolov5 {
class Builder {
 public:
  Builder() noexcept;
  ~Builder();

 public:
  Result init() noexcept;

  Result buildEngine(const std::string& inputFilePath,
                     const std::string& outputFilePath,
                     Precision precision = PRECISION_FP32) const noexcept;

  Result setLogger(std::shared_ptr<Logger> logger) noexcept;

  std::shared_ptr<Logger> logger() const noexcept;

 private:
  Result _buildEngine(const std::string& inputFilePath,
                      std::shared_ptr<nvinfer1::IHostMemory>* output,
                      Precision precision) const noexcept;

 private:
  bool _initialized;

  std::shared_ptr<Logger> _logger;

  std::unique_ptr<TensorRT_Logger> _trtLogger;
};

} /*  namespace yolov5    */

#endif /*  include guard   */