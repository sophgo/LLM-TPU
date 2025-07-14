## 编译方法 (没有自带sophon-opencv)

``` shell
# 安装opencv依赖
sudo apt update
sudo apt install libopencv-dev

# 编译
mkdir build && cd build 
cmake .. && make
```


## 编译方法 (自带/opt/sophon/sophon-opencv-latest)

需要修改CMakeLists.txt中的这几行如下：
```cmake
include_directories(/opt/sophon/sophon-opencv-latest/include/opencv4)
include_directories(/opt/sophon/sophon-ffmpeg-latest/include)
link_directories(/opt/sophon/sophon-opencv-latest/lib)
# find_package(OpenCV REQUIRED)
# include_directories(${OpenCV_INCLUDE_DIRS})
# link_directories(${OpenCV_LIBRARY_DIRS})
# message(STATUS "OpenCV version: ${OpenCV_VERSION}")
```

然后编译
``` shell
mkdir build && cd build 
cmake .. && make
```

## 运行
./pipeline -m bmodel_path -c config

