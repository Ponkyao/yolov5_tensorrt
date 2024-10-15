#include "yolov5_common.h"

#include <cstring>

namespace yolov5 {

const char* result_to_string(yolov5::Result r) noexcept {
  if (r == RESULT_FAILURE_INVALID_INPUT) {
    return "invalid input";
  } else if (r == RESULT_FAILURE_NOT_INITIALIZED) {
    return "not initialized";
  } else if (r == RESULT_FAILURE_NOT_LOADED) {
    return "not loaded";
  } else if (r == RESULT_FAILURE_MODEL_ERROR) {
    return "model error";
  } else if (r == RESULT_FAILURE_OPENCV_NO_CUDA) {
    return "opencv lacks cuda";
  } else if (r == RESULT_FAILURE_FILESYSTEM_ERROR) {
    return "filesystem error";
  } else if (r == RESULT_FAILURE_CUDA_ERROR) {
    return "cuda error";
  } else if (r == RESULT_FAILURE_TENSORRT_ERROR) {
    return "tensorrt error";
  } else if (r == RESULT_FAILURE_OPENCV_ERROR) {
    return "opencv error";
  } else if (r == RESULT_FAILURE_ALLOC) {
    return "alloc error";
  } else if (r == RESULT_FAILURE_OTHER) {
    return "other error";
  } else if (r == RESULT_SUCCESS) {
    return "success";
  } else {
    return "";
  }
}

bool result_to_string(Result r, std::string* out) noexcept {
  const char* str = result_to_string(r);
  if (std::strlen(str) == 0) {
    return false;
  }

  if (out != nullptr) {
    try {
      *out = str;
    } catch (const std::exception& e) {
    }
  }
  return true;
}

const char* precision_to_string(Precision p) noexcept {
  if (p == PRECISION_FP32) {
    return "fp32";
  } else if (p == PRECISION_FP16) {
    return "fp16";
  } else {
    return "";
  }
}

bool precision_to_string(Precision p, std::string* out) noexcept {
  const char* str = precision_to_string(p);
  if (std::strlen(str) == 0) {
    return false;
  }

  if (out != nullptr) {
    try {
      *out = str;
    } catch (const std::exception& e) {
    }
  }
  return true;
}

} /*  namespace yolov5    */