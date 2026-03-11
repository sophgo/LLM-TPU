## Command

### Export onnx

```
pip install transformers==4.46.2 torch==2.4.1+cu121 torchvision==0.19.0+cu121
cp files/modeling_moss.py moss-moon-003-sft/
```

your_torch_model是你模型的路径

```
python3 export_onnx.py --model_path your_torch_model --seq_length 512
./simple.sh
```

### Compile bmodel

```
./gen_bmodel.sh --target bm1684x --mode int4 --name moss --addr_mode io_alone --seq_length 512
```

