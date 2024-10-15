mkdir build && cd build 
camke .. && make
./yolov5_detect --onnx ../yolov5s.onnx --video ../video.mp4
 or
./yolov5_detect --onnx ../yolov5s.onnx --camera 0
