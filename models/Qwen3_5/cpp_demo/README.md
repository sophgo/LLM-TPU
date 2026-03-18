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

需要修改CMakeLists.txt中的这一行如下：
```cmake
set(SOPHON_OPENCV TRUE)
```

然后编译
``` shell
mkdir build && cd build 
cmake .. && make
```

## 运行
./pipeline -m bmodel_path -c config

