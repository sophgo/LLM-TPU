# Environment Setup
Download the required dependencies
```
sudo apt-get update
sudo apt-get install pybind11-dev
```

Compile the library files
```
mkdir build
cd build && cmake .. && make && cp *cpython* .. && cd ..
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
