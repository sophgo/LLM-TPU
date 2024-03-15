
## Command

```
mkdir ceval-exam 
cd ceval-exam
wget https://huggingface.co/datasets/ceval/ceval-exam/resolve/main/ceval-exam.zip
unzip ceval-exam

python evaluate_chatglm3.py --devid 10 --model_path ../../models/ChatGLM3/compile/chatglm3-6b_int4_1dev.bmodel --tokenizer_path ../../models/ChatGLM3/support/tokenizer.model
```
