### python demo

对于python demo，一定要在LLM-TPU里面source envsetup.sh（与tpu-mlir里面的envsetup.sh有区别）
```shell
cd /workspace/LLM-TPU
source envsetup.sh
```

```
sudo apt-get install pybind11[global]
pip3 install pybind11[global] transformers_stream_generator einops tiktoken accelerate transformers==4.32.0
```

```
cd /workspace/LLM-TPU/models/CodeFuse/python_demo
mkdir build && cd build
cmake .. && make
cp *cpython* ..
cd ..


python3 pipeline.py --model_path codefuse-7b_int4_1dev.bmodel --tokenizer_path ../token_config/ --devid 0 --generation_mode greedy
```
