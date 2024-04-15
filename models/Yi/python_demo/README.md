### python demo
```
sudo apt-get update
sudo apt-get install pybind11-dev
pip3 install transformers_stream_generator einops tiktoken accelerate transformers==4.39.0
```

```
cd /workspace/LLM-TPU/models/Yi/python_demo
mkdir build && cd build
cmake .. && make
cp *cpython* ..
cd ..


python3 pipeline.py --model_path your_bmodel_path --tokenizer_path ../token_config/ --devid 0 --generation_mode greedy
```

