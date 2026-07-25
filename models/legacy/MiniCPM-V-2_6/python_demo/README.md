# Environment Setup
> (This must be executed before running the python demo)
```
sudo apt-get update
sudo apt-get install pybind11-dev
pip3 install sentencepiece transformers==4.40.0
pip3 install gradio==3.39.0 mdtex2html==1.2.0 dfss
```

If you do not plan to compile the model yourself, you can directly use the downloaded model
```
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/minicpmv26_bm1684x_int4_seq1024_imsize448.bmodel
```

Compile the library files
```
mkdir build
cd build && cmake .. && make && cp *cpython* .. && cd ..
```

# python demo
```
python3 pipeline.py --model_path minicpmv26_bm1684x_int4_seq1024_imsize448.bmodel --processor_path ../support/processor_config/ --devid 0
```
