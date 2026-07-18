# Environment Setup
> (This must be executed before running either the python demo or the web demo)
```
sudo apt-get update
sudo apt-get install pybind11-dev
pip3 install sentencepiece transformers==4.30.2
pip3 install gradio==3.39.0 mdtex2html==1.2.0 dfss
```

If you do not plan to compile the model yourself, you can directly use the downloaded model
```
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/chatglm3-6b_int4_1dev_16k.bmodel
```

Compile the library files
```
mkdir build
cd build && cmake .. && make && cp *cpython* .. && cd ..
```

# python demo
```
python3 pipeline.py --model_path chatglm3-6b_int4_1dev_16k.bmodel --tokenizer_path ../support/token_config/ --devid 0 --generation_mode greedy
```

## FAQ
### Could not create share link. Missing file: /usr/local/lib/python3.10/dist-packages/gradio/frpc_linux_amd64_v0.2.

AMD / PCIE
```
1. Download this file: https://cdn-media.huggingface.co/frpc-gradio-0.2/frpc_linux_amd64
2. Rename the downloaded file to: frpc_linux_amd64_v0.2
3. Move the file to this location: /usr/local/lib/python3.10/dist-packages/gradio
```

ARM / SOC

### ImportError:/home/linaro/.local/lib/python3.8/site-packages/torch/libs/libgomp-6e1a1d1b.so.1.0.0: cannot allocate memory in static TLs block

```
export LD_PRELOAD=/home/linaro/.local/lib/python3.8/site-packages/torch/lib/libgomp-d22c30c5.so.1
```

### OSError: /home/linaro/.local/lib/python3.8/site-packages/torch/lib/libgomp-d22c30c5.so.1: cannotallocate memoryin staticTLS block

```
export LD_PRELOAD=/home/linaro/.local/lib/python3.8/site-packages/torch/lib/libgomp-d22c30c5.so.1
```
