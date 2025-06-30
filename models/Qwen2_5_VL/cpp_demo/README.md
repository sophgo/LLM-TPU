## 编译方法 (没有自带opencv)

``` shell
# 安装opencv依赖
sudo apt update
sudo apt install libopencv-dev

# 编译
mkdir build && cd build 
cmake .. && make
```


## 编译方法 (自带/opt/sophon/sophon-opencv-latest)

删除CMakeLists.txt中的这几行
```cmake
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
```
然后编译
``` shell
mkdir build && cd build 
cmake .. && make
```

## 运行
./pipeline -m bmodel_path -c config

