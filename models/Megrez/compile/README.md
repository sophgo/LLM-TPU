# Command

## Export onnx

```shell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install sentencepiece transformers==4.44.1
cp file/Megrez-3B-Instruct/modeling_llama.py /your/path/to/transformers/path
```

```shell
python3 export_onnx.py --model_path your_torch_model --seq_length 512 --device cpu
```

## Compile bmodel
使用io_alone
```
./compile.sh --mode int4 --name megrez --seq_length 512
```