# 功能
本demo用于在C-Eval数据集上测试ChatGLM模型性能

# 运行指南
## 1. 项目编译
请参考 [ChatGLM3模型编译](../../models/ChatGLM3/eval_demo/README.md) 将模型编译为bmodel类型文件  
🚗 如果已有可测试的模型文件，可跳过此步  

## 2. 搭建数据环境

数据集：C-Eval  
🌐[官网](https://cevalbenchmark.com/) • 🤗[Hugging Face](https://huggingface.co/datasets/ceval/ceval-exam") • 💻[GitHub](https://github.com/hkust-nlp/ceval/tree/main)

#### 创建数据集文件夹
```
mkdir ceval-exam 
cd ceval-exam
```

#### 下载C-Eval数据集
此处采用wget下载方式，其他下载方式可参考 [GitHub](https://github.com/hkust-nlp/ceval/tree/main)
```
wget https://huggingface.co/datasets/ceval/ceval-exam/resolve/main/ceval-exam.zip
unzip ceval-exam
```


## 3. 运行评测例程
### 运行命令
```
python evaluate_chatglm3.py --devid [DEVICE ID] --model_path [PATH_TO_MODEL] --tokenizer_path [PATH_TO_TOKENIZER] --eval_mode fast
```
 ### 参数说明

| 参数           | 说明                       |
|:--------------:|:---------------------------:|
| `--devid`      | 可用设备 ID                    |
| `--model_path` | 模型路径，即步骤1中编译的模型文件                   |
| `--tokenizer_path` | 分词器路径               |
| `--eval_mode`  | 评估模式，有`fast`和`default`两种类型   |

📌 本项目提供分词器，路径为`LLM-TPU/models/ChatGLM3/support/token_config/ `

### 运行结果

运行结束后将得到一个测试结果文件，命名方式为`submission_{}.json`

### 示例
当文件路径如下所示时
```
LLM-TPU
|_ harness
  |_ C-Eval
    |_ evaluate_chatglm3.py
    |_ ceval-exam
    |_ subject_mapping.json
|_ models
  |_ ChatGLM3
    |_ compile
        |_ chatglm3-6b_int4_1dev_1024.bmodel
    |_ support
        |_ token_config
```
运行命令为
```
export PYTHONPATH=../../
python evaluate_chatglm3.py --devid 10 --model_path ../../models/ChatGLM3/compile/chatglm3-6b_int4_1dev_1024.bmodel --tokenizer_path ../../models/ChatGLM3/support/token_config/ --eval_mode fast
```

## 效果验证

C-Eval数据集不提供测试集标签。为了验证效果，需要将结果文件提交到官方网站⬇

[📎结果提交页](https://cevalbenchmark.com/static/user_interface.html)