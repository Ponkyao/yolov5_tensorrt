#include "yolov5_detection.h"

#include <fstream>

namespace yolov5 {
const std::vector<std::string> CLASS_NAMES = {
    "person",        "bicycle",      "car",
    "motorcycle",    "airplane",     "bus",
    "train",         "truck",        "boat",
    "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench",        "bird",
    "cat",           "dog",          "horse",
    "sheep",         "cow",          "elephant",
    "bear",          "zebra",        "giraffe",
    "backpack",      "umbrella",     "handbag",
    "tie",           "suitcase",     "frisbee",
    "skis",          "snowboard",    "sports ball",
    "kite",          "baseball bat", "baseball glove",
    "skateboard",    "surfboard",    "tennis racket",
    "bottle",        "wine glass",   "cup",
    "fork",          "knife",        "spoon",
    "bowl",          "banana",       "apple",
    "sandwich",      "orange",       "broccoli",
    "carrot",        "hot dog",      "pizza",
    "donut",         "cake",         "chair",
    "couch",         "potted plant", "bed",
    "dining table",  "toilet",       "tv",
    "laptop",        "mouse",        "remote",
    "keyboard",      "cell phone",   "microwave",
    "oven",          "toaster",      "sink",
    "refrigerator",  "book",         "clock",
    "vase",          "scissors",     "teddy bear",
    "hair drier",    "toothbrush"};

Detection::Detection() noexcept : _classId(-1), _score(0) {}

Detection::Detection(const int& classId, const cv::Rect& boundingBox,
                     const double& score) noexcept
    : _classId(classId), _boundingBox(boundingBox), _score(score) {}

Detection::~Detection() noexcept {}

const int32_t& Detection::classId() const noexcept { return _classId; }

const cv::Rect& Detection::boundingBox() const noexcept { return _boundingBox; }

const double& Detection::score() const noexcept { return _score; }

const std::string& Detection::className() const noexcept { return _className; }

bool Detection::setClassName(const std::string& name) noexcept {
  try {
    _className = name;
  } catch (const std::exception& e) {
    return false;
  }
  return true;
}

Result visualizeDetection(const std::vector<yolov5::Detection>& detections,
                          cv::Mat* image, int fps) noexcept {
  if (image == nullptr) {
    return RESULT_SUCCESS;
  }

  try {
    /*  Draw bounding box around the detection  */
    const int bboxThickness = 4;
    const int textThickness = 2;
    int baseline = 0;
    const double fontScale = 1.0;
    const cv::Scalar color(255, 51, 153);

    cv::putText(*image, "FPS: " + std::to_string(fps), cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);

    std::cout << "Get a new frame successfully [ Object detected:  ";
    for (const auto& it : detections) {
      const cv::Rect& bbox = it.boundingBox();
      cv::rectangle(*image, bbox, color, bboxThickness);

      if (!it.className().empty()) {
        std::cout << it.className() << "  ";
      }

      std::stringstream ss;
      ss << std::fixed << std::setprecision(2) << it.score();
      const std::string label = it.className() + ": " + ss.str();

      const cv::Size textSize = cv::getTextSize(
          label, cv::FONT_HERSHEY_PLAIN, fontScale, textThickness, &baseline);
      const cv::Point tl(bbox.x - bboxThickness / 2.0,
                         bbox.y - textSize.height);
      const cv::Rect labelRect(tl, textSize);
      cv::rectangle(*image, labelRect, color, -1);

      const cv::Point bl(tl.x, bbox.y - bboxThickness / 2.0);
      cv::putText(*image, label, bl, cv::FONT_HERSHEY_PLAIN, fontScale,
                  cv::Scalar(255, 255, 255), textThickness);
    }
    std::cout << "]" << std::endl;

  } catch (const std::exception& e) {
    return RESULT_FAILURE_OPENCV_ERROR;
  }
  return RESULT_SUCCESS;
}

Classes::Classes() noexcept : _names(CLASS_NAMES) {}

Classes::~Classes() noexcept {}

Result Classes::load(const std::vector<std::string>& names) noexcept {
  if (names.size() == 0 && _logger) {
    if (_logger) {
      _logger->log(LOGGING_ERROR,
                   "[Classes] load() warning: specified "
                   "list of class names is empty!");
    }
    return RESULT_FAILURE_INVALID_INPUT;
  }

  try {
    _names = names;
  } catch (const std::exception& e) {
    if (_logger) {
      _logger->logf(LOGGING_ERROR,
                    "[Classes] load() failure: got "
                    "exception trying to copy names: %s",
                    e.what());
    }
    return RESULT_FAILURE_ALLOC;
  }

  if (_logger) {
    _logger->logf(LOGGING_INFO, "[Classes] Loaded %d classes",
                  (unsigned int)names.size());
  }
  return RESULT_SUCCESS;
}

bool Classes::isLoaded() const noexcept { return (_names.size() > 0); }

Result Classes::getName(const int& classId, std::string* out) const noexcept {
  if ((unsigned int)classId >= _names.size() || classId < 0) {
    if (_logger) {
      _logger->logf(LOGGING_ERROR,
                    "[Classes] getName() failure: no "
                    "info about specified classId '%i'",
                    classId);
    }
    return RESULT_FAILURE_INVALID_INPUT;
  }

  if (out != nullptr) {
    try {
      *out = _names[classId];
    } catch (const std::exception& e) {
      if (_logger) {
        _logger->logf(LOGGING_ERROR,
                      "[Classes] getName() failure: got"
                      " exception when setting output: %s",
                      e.what());
      }
      return RESULT_FAILURE_ALLOC;
    }
  }
  return RESULT_SUCCESS;
}

void Classes::setLogger(std::shared_ptr<Logger> logger) noexcept {
  _logger = logger;
}

} /*  namespace yolov5    */