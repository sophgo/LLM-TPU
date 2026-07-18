# Compilation

``` shell
cd compile

llm_convert.py -m $model_path -s $seq_length -q w4f16

python export_onnx.py -m $model_path -i $image_path

./compile.sh
```
* llm_convert.py currently does not support directly exporting the vit part

First use llm_convert.py to directly compile the llm part of the bmodel, then use export_onnx.py to export the onnx of the vit part, and finally use compile.sh to compile the vit part of the bmodel and combine them together.


Adding the -i parameter to export_onnx.py to specify the image allows compiling a static bmodel for that image shape, which can improve inference performance at the same shape


# Run
``` shell
# Download the pre-compiled model
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/nvila-8b_seq4096_w4f16_bm1684x.bmodel


cd python_demo
mkdir build && cd build 
cmake .. && make && mv chat.*so ..

python pipeline.py -m $bmodel_path -d $device_id
```
