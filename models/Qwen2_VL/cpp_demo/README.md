``` shell
# 安装opencv依赖
sudo apt update
sudo apt install libopencv-dev

# 编译
mkdir build && cd build 
cmake .. && make

# 运行
./qwen2_vl -m bmodel_path -c config
```
