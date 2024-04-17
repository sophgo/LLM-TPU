### python demo
```shell
cd python_demo
sudo pip3 install pybind11[global] sentencepiece
```

### 下载迁移好的模型
也可以直接下载编译好的模型，不用自己编译
```shell
pip3 install dfss
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/chatglm3-6b_int4_1dev_2048.bmodel
```

```shell
cd /workspace/LLM-TPU/models/ChatGLM3/python_demo
mkdir build
cd build && cmake .. && make && cp *cpython* .. && cd ..


python3 pipeline.py --model_path ../compile/chatglm3-6b_int4_1dev.bmodel --tokenizer_path ../support/token_config/ --devid 0 --generation_mode greedy
```
