## python demo

### Install dependent
```bash
sudo apt-get update
pip3 install pybind11[global]
```

### Compile chat.cpp

可以直接下载编译好的模型，不用自己编译
```bash
pip3 install dfss
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/minicpm3-4b_int4_seq512_1dev.bmodel
```

```
mkdir build
cd build && cmake .. && make && cp *cpython* .. && cd ..
```

### CLI demo
```bash
python pipeline.py --model_path ./drivemm_w4bf16_seq2048/drivemm_w4bf16_seq2048_1dev_20250405_141520.bmodel --tokenizer_path ./drivemm_w4bf16_seq2048/exported_tokenizer/ --devid 0
```

按照以下方式输入
```
Question: There is an image of traffic captured from the front view of the ego vehicle. Focus on objects influencing the ego car's driving behavior: vehicles (cars, trucks, buses, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), traffic signs (no parking, warning, directional, etc.), traffic lights (red, green, yellow), traffic cones, barriers, miscellaneous(debris, dustbin, animals, etc.). You must not discuss any objects beyond the seven categories above. Please provide driving suggestions for the ego car based on the current scene.

Image or Video Path: codalm3.png

Image or Video Type: image
```
