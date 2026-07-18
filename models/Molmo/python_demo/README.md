# Environment setup
> (This must be done before running the python demo)
```
sudo apt-get update
sudo apt-get install pybind11-dev
```

If you don't plan to compile the model yourself, you can use the pre-downloaded model directly
```
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/molmo-7b_int4_seq1024_384x384.bmodel
```

Build the library files
```
mkdir build
cd build && cmake .. && make && cp *cpython* .. && cd ..
```

# python demo
```
python3 pipeline.py -m molmo-7b_int4_seq1024_384x384.bmodel -i ./test.jpg -s image_size -t ../processor_config --devid 0
```
