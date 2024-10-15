#ifndef _YOLOV5_COMMON_HPP_
#define _YOLOV5_COMMON_HPP_
#pragma once
#include <string>

#define YOLOV5_UNUSED(x) (void)x;

namespace yolov5 {

enum Result {
  /*      Invalid input specified. This typically indicates a programming
          error in your software (i.e. a bug in your software).
   */
  RESULT_FAILURE_INVALID_INPUT = -100,

  /*      Not initialized yet */
  RESULT_FAILURE_NOT_INITIALIZED = -90,

  /*      Not loaded yet (e.g. no engine loaded yet)    */
  RESULT_FAILURE_NOT_LOADED = -80,

  /*      Issue with the loaded model (e.g. input binding is missing) */
  RESULT_FAILURE_MODEL_ERROR = -70,

  /*      Indicates that you are trying to use a function that OpenCV-CUDA,
          but your OpenCV has no support for this. This typically indicates a
          programming error in your software  */
  RESULT_FAILURE_OPENCV_NO_CUDA = -21,

  /*      Error related to filesystem (e.g. could not open file)    */
  RESULT_FAILURE_FILESYSTEM_ERROR = -50,

  /*      Internal cuda error (e.g. could not allocate memory)    */
  RESULT_FAILURE_CUDA_ERROR = -40,

  /*      Internal TensorRT error (e.g. could not setup execution context  */
  RESULT_FAILURE_TENSORRT_ERROR = -30,

  /*      Internal OpenCV error   */
  RESULT_FAILURE_OPENCV_ERROR = -20,

  /*      Memory-related error    */
  RESULT_FAILURE_ALLOC = -11,

  /*      Other error */
  RESULT_FAILURE_OTHER = -10,

  /*      Successfull execution   */
  RESULT_SUCCESS = 0
};

const char* result_to_string(Result r) noexcept;

bool result_to_string(Result r, std::string* out) noexcept;

enum Precision {
  PRECISION_FP32 = 0, /**<    32-bit floating point mode  */

  PRECISION_FP16 = 1, /**<    16-bit floating point mode  */
};

const char* precision_to_string(Precision p) noexcept;

bool precision_to_string(Precision p, std::string* out) noexcept;
/**
 * Additional flags that can be passed to the Detector
 */
enum DetectorFlag {
  INPUT_BGR = 1,
  /**<    input image is in BGR colorspace(opencv default) */

  INPUT_RGB = 2,
  /**<    input image is in RGB colorspace */

  PREPROCESSOR_CVCUDA = 4,
  /**<    OpenCV-CUDA pre-processing should be used */

  PREPROCESSOR_CVCPU = 8
  /**<    OpenCV-CPU pre-processing should be used */
};

} /*  namespace yolov5    */

#endif /*  include guard   */