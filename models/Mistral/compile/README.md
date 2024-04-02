```
pip3 install transformers==4.36.0
cp files/Mistral-7B-Instruct-v0.2/modeling_mistral.py /usr/local/lib/python3.10/dist-packages/transformers/models/mistral/modeling_mistral.py 

python export_onnx.py --model_path Mistral-7B-Instruct-v0.2/
```