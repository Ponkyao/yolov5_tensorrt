#include <chrono>
#include <filesystem>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#include "yolov5_builder.h"
#include "yolov5_detector.h"

char* getCmdOption(char** begin, char** end, const std::string& option) {
  char** itr = std::find(begin, end, option);
  if (itr != end && ++itr != end) {
    return *itr;
  }
  return 0;
}

bool cmdOptionExists(char** begin, char** end, const std::string& option,
                     bool value = false) {
  char** itr = std::find(begin, end, option);
  if (itr == end) {
    return false;
  }
  if (value && itr == end - 1) {
    std::cout << "Warning: option '" << option << "' requires a value"
              << std::endl;
    return false;
  }
  return true;
}

void printHelp() {
  std::cout << "Options:\n"
               "-h --help :       show this help menu\n"
               "--model :         [mandatory] specify the ONNX model file\n"
               "--video :         [optional] specify the video file path\n"
               "--camera :        [optional] camera index\n"
               "Example usage:\n"
               "./yolov5_detect --onnx ../yolov5s.onnx --video ../video.mp4\n"
               "or\n"
               "./yolov5_detect --onnx ../yolov5s.onnx --camera 0"
            << std::endl;
}

bool buildEngineFile(const std::string& modelFile) {
  const std::string outputFile =
      modelFile.substr(0, modelFile.find_last_of(".")) + ".engine";
  const yolov5::Precision precision = yolov5::PRECISION_FP16;

  yolov5::Builder builder;
  yolov5::Result r = builder.init();
  if (r != yolov5::RESULT_SUCCESS) {
    std::cout << "init() failed: " << yolov5::result_to_string(r) << std::endl;
    return false;
  }

  r = builder.buildEngine(modelFile, outputFile, precision);
  if (r != yolov5::RESULT_SUCCESS) {
    std::cout << "buildEngine() failed: " << yolov5::result_to_string(r)
              << std::endl;
    return false;
  }
  return true;
}

int main(int argc, char* argv[]) {
  if (cmdOptionExists(argv, argv + argc, "--help") ||
      cmdOptionExists(argv, argv + argc, "-h")) {
    printHelp();
    return 0;
  }

  if (!cmdOptionExists(argv, argv + argc, "--onnx", true)) {
    std::cout << "Missing mandatory argument: --onnx" << std::endl;
    printHelp();
    return 1;
  }

  const std::string onnxFile(getCmdOption(argv, argv + argc, "--onnx"));
  const std::string videoFile =
      cmdOptionExists(argv, argv + argc, "--video", true)
          ? getCmdOption(argv, argv + argc, "--video")
          : "";
  const std::string cameraIndexOption =
      cmdOptionExists(argv, argv + argc, "--camera", true)
          ? getCmdOption(argv, argv + argc, "--camera")
          : "";
  int cameraIndex =
      cameraIndexOption.empty() ? -1 : std::atoi(cameraIndexOption.c_str());

  const std::string engineFile =
      onnxFile.substr(0, onnxFile.find_last_of(".")) + ".engine";

  if (!std::filesystem::exists(engineFile)) {
    if (!buildEngineFile(onnxFile)) {
      return 1;
    }
    std::cout << "Successfully built engine file!" << std::endl;
  }

  yolov5::Detector detector;
  yolov5::Result r = detector.init();
  if (r != yolov5::RESULT_SUCCESS) {
    std::cout << "init() failed: " << yolov5::result_to_string(r) << std::endl;
    return 1;
  }

  r = detector.loadEngine(engineFile);
  if (r != yolov5::RESULT_SUCCESS) {
    std::cout << "loadEngine() failed: " << yolov5::result_to_string(r)
              << std::endl;
    return 1;
  }

  yolov5::Classes classes;
  classes.setLogger(detector.logger());
  detector.setClasses(classes);

  cv::namedWindow("yolov5_tensorrt", cv::WINDOW_NORMAL);
  cv::resizeWindow("yolov5_tensorrt", 1280, 1080);

  cv::VideoCapture capture;
  if (!videoFile.empty()) {
    // Open video file
    if (!capture.open(videoFile)) {
      std::cout << "Failure: could not open video file" << std::endl;
      return 1;
    }
  } else if (cameraIndex >= 0) {
    // Open camera
    if (!capture.open(cameraIndex, cv::CAP_ANY)) {
      std::cout << "Failure: could not open capture device" << std::endl;
      return 1;
    }
  } else {
    std::cout << "Must specify either --video or --camera" << std::endl;
    return 1;
  }

  cv::Mat image;
  std::vector<yolov5::Detection> detections;
  while (true) {
    auto startTime = std::chrono::high_resolution_clock::now();
    if (!capture.read(image)) {
      std::cout << "Failure: could not read new frames" << std::endl;
      break;
    }

    r = detector.detect(image, &detections, yolov5::INPUT_BGR);
    if (r != yolov5::RESULT_SUCCESS) {
      std::cout << "detect() failed: " << yolov5::result_to_string(r)
                << std::endl;
      return 1;
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                        endTime - startTime)
                        .count();
    int fps = 1000 / duration;

    yolov5::visualizeDetection(detections, &image, fps);
    cv::imshow("yolov5_tensorrt", image);
    if (cv::waitKey(1) >= 0) break;  // Exit on key press
  }
  capture.release();
  cv::destroyAllWindows();
  return 0;
}