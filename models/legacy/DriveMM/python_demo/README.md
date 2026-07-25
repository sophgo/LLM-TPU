## python demo

### Install dependent
```bash
sudo apt-get update
pip3 install pybind11[global]
```

### Compile chat.cpp

You can directly download the pre-compiled model instead of compiling it yourself
```bash
pip3 install dfss
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/drivemm_w4bf16_seq2048.tar.gz
tar -zxvf drivemm_w4bf16_seq2048.tar.gz
```

```
mkdir build
cd build && cmake .. && make && cp *cpython* .. && cd ..
```

### CLI demo
```bash
python pipeline.py --model_path ./drivemm_w4bf16_seq2048/drivemm_w4bf16_seq2048_1dev_20250405_141520.bmodel --tokenizer_path ./drivemm_w4bf16_seq2048/exported_tokenizer/ --devid 0
```

Provide the input in the following way
```
Question: There is an image of traffic captured from the front view of the ego vehicle. Focus on objects influencing the ego car's driving behavior: vehicles (cars, trucks, buses, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), traffic signs (no parking, warning, directional, etc.), traffic lights (red, green, yellow), traffic cones, barriers, miscellaneous(debris, dustbin, animals, etc.). You must not discuss any objects beyond the seven categories above. Please provide driving suggestions for the ego car based on the current scene.

Image or Video Path: codalm3.png

Image or Video Type: image
```
