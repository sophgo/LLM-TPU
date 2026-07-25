## python demo

### Install dependent
```bash
sudo apt-get update
pip3 install pybind11[global]
```

### Compile chat.cpp

You can also directly download the compiled model instead of compiling it yourself
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
python3 pipeline.py --model_path your_bmodel_path --tokenizer_path ../support/token_config/ --devid 0 --generation_mode greedy
```
