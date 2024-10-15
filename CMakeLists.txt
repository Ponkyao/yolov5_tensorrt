cmake_minimum_required(VERSION 3.10)
project(yolov5-tensorrt VERSION 0.1 DESCRIPTION "Real-time object detection with YOLOv5 and TensorRT")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -g -Wall -Wextra -Wno-deprecated -fPIC")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -g -pthread")

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

include_directories(include)

file(GLOB SOURCES "src/*.cc")

add_executable(yolov5_detect 
    main.cc
    ${SOURCES}
)

target_include_directories(yolov5_detect PUBLIC 
    ${OpenCV_INCLUDE_DIRS} 
    ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(yolov5_detect 
    nvinfer
    nvonnxparser
    ${CUDA_LIBRARIES} 
    ${OpenCV_LIBRARIES}
)
