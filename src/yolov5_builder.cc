#include "yolov5_builder.h"

#include <NvOnnxParser.h>

#include <fstream>

namespace yolov5 {

Builder::Builder() noexcept : _initialized(false) {}

Builder::~Builder() {}

Result Builder::init() noexcept {
  /*  Initialize Logger */
  if (!_logger) {
    try {
      _logger = std::make_shared<Logger>();
    } catch (const std::exception& e) {
      /*  logging not available  */
      return RESULT_FAILURE_ALLOC;
    }
  }

  /*  Initialize TensorRT logger  */
  if (!_trtLogger) {
    try {
      _trtLogger = std::make_unique<TensorRT_Logger>(_logger);
    } catch (const std::exception& e) {
      _logger->logf(
          LOGGING_ERROR,
          "[Builder] init() failure: could not create TensorRT logger: %s",
          e.what());
      return RESULT_FAILURE_ALLOC;
    }
  }

  _initialized = true;
  return RESULT_SUCCESS;
}

Result Builder::buildEngine(const std::string& inputFilePath,
                            const std::string& outputFilePath,
                            Precision precision) const noexcept {
  if (!_initialized) {
    if (_logger) {
      _logger->log(
          LOGGING_ERROR,
          "[Builder] buildEngine() failure: builder is not initialized yet");
    }
    return RESULT_FAILURE_NOT_INITIALIZED;
  }

  std::shared_ptr<nvinfer1::IHostMemory> engineOutput;
  Result r = _buildEngine(inputFilePath, &engineOutput, precision);
  if (r != RESULT_SUCCESS) {
    return r;
  }

  /*  Write to disk   */
  _logger->logf(
      LOGGING_INFO,
      "[Builder] buildEngine(): writing serialized engine to file: %s",
      outputFilePath.c_str());

  std::ofstream outputFile;
  outputFile.open(outputFilePath, std::ios::out | std::ios::binary);
  outputFile.write((char*)engineOutput->data(), engineOutput->size());
  if (!outputFile.good()) {
    _logger->log(LOGGING_ERROR,
                 "[Builder] buildEngine() failure: error encountered writing "
                 "to output file");
    return RESULT_FAILURE_FILESYSTEM_ERROR;
  }
  outputFile.close();

  return RESULT_SUCCESS;
}

Result Builder::setLogger(std::shared_ptr<Logger> logger) noexcept {
  if (!logger) {
    if (_logger) {
      _logger->log(LOGGING_ERROR,
                   "[Builder] setLogger() failure: provided logger is nullptr");
    }
    return RESULT_FAILURE_INVALID_INPUT;
  }
  _logger = logger;

  return RESULT_SUCCESS;
}

std::shared_ptr<Logger> Builder::logger() const noexcept { return _logger; }

Result Builder::_buildEngine(const std::string& inputFilePath,
                             std::shared_ptr<nvinfer1::IHostMemory>* output,
                             Precision precision) const noexcept {
  const char* precisionStr = precision_to_string(precision);
  if (std::strlen(precisionStr) == 0) {
    _logger->log(
        LOGGING_ERROR,
        "[Builder] buildEngine() failure: invalid precision specified");
    return RESULT_FAILURE_INVALID_INPUT;
  }

  try {
    std::unique_ptr<nvinfer1::IBuilder> builder(
        nvinfer1::createInferBuilder(*_trtLogger));

    const auto explicitBatch =
        1U << static_cast<uint32_t>(
            nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    std::unique_ptr<nvinfer1::INetworkDefinition> network(
        builder->createNetworkV2(explicitBatch));

    builder->setMaxBatchSize(1);

    std::unique_ptr<nvonnxparser::IParser> parser(
        nvonnxparser::createParser(*network, *_trtLogger));
    if (!parser->parseFromFile(inputFilePath.c_str(),
                               (int)nvinfer1::ILogger::Severity::kWARNING)) {
      _logger->log(LOGGING_ERROR,
                   "[Builder] buildEngine() failure: could not parse ONNX "
                   "model from file");
      return RESULT_FAILURE_MODEL_ERROR;
    }

    std::unique_ptr<nvinfer1::IBuilderConfig> config(
        builder->createBuilderConfig());
    config->setMaxWorkspaceSize(1 << 20);

    if (precision == PRECISION_FP32) {
      /*  this is the default */
    } else if (precision == PRECISION_FP16) {
      if (!builder->platformHasFastFp16()) {
        _logger->log(LOGGING_ERROR,
                     "[Builder] buildEngine() failure: fp16 precision "
                     "specified, but not supported by current platform");
        return RESULT_FAILURE_INVALID_INPUT;
      }
      config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }

    _logger->logf(LOGGING_INFO,
                  "[Builder] buildEngine(): building and serializing engine at "
                  "%s precision. This may take a while",
                  precisionStr);

    std::shared_ptr<nvinfer1::IHostMemory> serialized(
        builder->buildSerializedNetwork(*network, *config));
    if (!serialized) {
      _logger->log(
          LOGGING_ERROR,
          "[Builder] buildEngine() failure: could not build serialized engine");
      return RESULT_FAILURE_TENSORRT_ERROR;
    }
    *output = serialized;
  } catch (const std::exception& e) {
    _logger->logf(LOGGING_ERROR,
                  "[Builder] buildEngine() failure: got exception: %s",
                  e.what());
    return RESULT_FAILURE_OTHER;
  }
  return RESULT_SUCCESS;
}

} /*  namespace yolov5    */