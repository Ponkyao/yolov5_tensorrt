#ifndef _YOLOV5_LOGGING_HPP_
#define _YOLOV5_LOGGING_HPP_
#pragma once
#include <NvInfer.h>

#include <memory>

#include "yolov5_common.h"

namespace yolov5 {

enum LogLevel {
  LOGGING_DEBUG = 0, /**<  verbose, low-level details   */

  LOGGING_INFO = 1, /**<  informational messages  */

  LOGGING_WARNING = 2, /**<  warning messages    */

  LOGGING_ERROR = 3 /**<  error messages  */
};

const char* loglevel_to_string(const LogLevel& l) noexcept;

bool loglevel_to_string(const LogLevel& l, std::string* out) noexcept;

class Logger {
 public:
  Logger() noexcept;

  virtual ~Logger();

 public:
  virtual void print(const LogLevel& level, const char* msg);

  void log(const LogLevel& level, const char* msg) noexcept;

  void logf(const LogLevel& level, const char* fmt, ...) noexcept
      __attribute__((format(printf, 3, 4)));

 private:
};

class TensorRT_Logger : public nvinfer1::ILogger {
 public:
  TensorRT_Logger() noexcept;

  TensorRT_Logger(std::shared_ptr<Logger> logger) noexcept;

  ~TensorRT_Logger();

 public:
  void setLogger(std::shared_ptr<Logger> logger) noexcept;

  virtual void log(nvinfer1::ILogger::Severity severity,
                   const char* msg) noexcept override;

 private:
  std::shared_ptr<Logger> _logger;
};

} /*  namespace yolov5    */

#endif /*  include guard   */