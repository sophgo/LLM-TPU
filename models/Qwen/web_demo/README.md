```
pip3 install pybind11[global] gradio==3.39.0 mdtex2html==1.2.0 dfss
pip3 install transformers_stream_generator einops tiktoken

cd /path_to/web_demo
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen-7b_int4_1dev.bmodel

pushd ../../../ && source envsetup.sh && popd

pushd ../python_demo && mkdir build
cd build/ && cmake .. && make -j && cp *cpython* .. && cd ..
popd

python3 web_demo.py --model_path qwen-7b_int4_1dev.bmodel --tokenizer_path ../support/token_config/ --devid 0
```

## Could not create share link. Missing file: /usr/local/lib/python3.10/dist-packages/gradio/frpc_linux_amd64_v0.2.

AMD / PCIE
```
1. Download this file: https://cdn-media.huggingface.co/frpc-gradio-0.2/frpc_linux_amd64
2. Rename the downloaded file to: frpc_linux_amd64_v0.2
3. Move the file to this location: /usr/local/lib/python3.10/dist-packages/gradio
```

ARM / SOC

