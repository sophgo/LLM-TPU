
# 运行指南
## 项目编译
请参考LLM-TPU/models/ChatGLM3/eval-demo/README.md进行项目编译

## 搭建数据环境
下载并准备数据
```
mkdir ceval-exam 
cd ceval-exam
wget https://huggingface.co/datasets/ceval/ceval-exam/resolve/main/ceval-exam.zip
unzip ceval-exam
```

## 运行评测例程
```
export PYTHONPATH=../../
python evaluate_chatglm3.py --devid 10 --model_path ../../models/ChatGLM3/compile/chatglm3-6b_int4_1dev.bmodel --tokenizer_path $PATH_TO_TOKENIZER --eval_mode fast
```
