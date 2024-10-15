#ifndef _YOLOV5_DETECTION_HPP_
#define _YOLOV5_DETECTION_HPP_

#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

#include "yolov5_logging.h"

namespace yolov5 {

/**
 * Represents an object detected in an image by the YoloV5 model
 */
class Detection {
 public:
  Detection() noexcept;

  Detection(const int& classId, const cv::Rect& boundingBox,
            const double& score) noexcept;

  ~Detection() noexcept;

 public:
  const int32_t& classId() const noexcept;

  const cv::Rect& boundingBox() const noexcept;

  const double& score() const noexcept;

  const std::string& className() const noexcept;

  bool setClassName(const std::string& name) noexcept;

 private:
  int32_t _classId;
  std::string _className;

  cv::Rect _boundingBox;
  double _score;
};

Result visualizeDetection(const std::vector<yolov5::Detection>& detections,
                          cv::Mat* image, int fps) noexcept;

/**
 * Represents the classes of your model
 *
 * This can be used to map classIds to actual understandable names,
 * such as "human" or "suitcase".
 */
class Classes {
 public:
  Classes() noexcept;

  ~Classes() noexcept;

 public:
  Result load(const std::vector<std::string>& names) noexcept;

  bool isLoaded() const noexcept;

  Result getName(const int& classId, std::string* out) const noexcept;

  void setLogger(std::shared_ptr<Logger> logger) noexcept;

 private:
  std::shared_ptr<Logger> _logger;

  std::vector<std::string> _names;
};

} /*  namespace yolov5    */

#endif /*  include guard   */