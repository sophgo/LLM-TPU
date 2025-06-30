## LLM调试代码

当LLM输出不正确时可以使用这里的代码做调试

## 方法

将本目录的代码拷贝到demo代码中，CMakeLists.txt中链接库需要加上libz.so，如下：

```CMake
pybind11_add_module(chat chat.cpp cnpy.cpp)
target_link_libraries(chat PUBLIC bmrt bmlib z)
```

dump_net_input_to_file 可以把网络的输入导出到npz文件

dump_net_output_to_file 可以把网络的输出导出到npz文件