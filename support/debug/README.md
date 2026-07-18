## LLM debugging code

When the LLM output is incorrect, you can use the code here for debugging

## Method

Copy the code in this directory into the demo code. The linked libraries in CMakeLists.txt need to include libz.so, as follows:

```CMake
pybind11_add_module(chat chat.cpp cnpy.cpp)
target_link_libraries(chat PUBLIC bmrt bmlib z)
```

dump_net_input_to_file can export the network inputs to an npz file

dump_net_output_to_file can export the network outputs to an npz file

dump_net_to_file can export both the network inputs and outputs to an npz file