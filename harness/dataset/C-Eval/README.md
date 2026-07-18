# Features
This demo is used to test ChatGLM model performance on the C-Eval dataset

# Usage Guide
## 1. Project Compilation
Please refer to [ChatGLM3 Model Compilation](../../models/ChatGLM3/eval_demo/README.md) to compile the model into a bmodel file  
🚗 If you already have a testable model file, you can skip this step  

## 2. Set Up the Data Environment

Dataset: C-Eval  
🌐[Official Website](https://cevalbenchmark.com/) • 🤗[Hugging Face](https://huggingface.co/datasets/ceval/ceval-exam") • 💻[GitHub](https://github.com/hkust-nlp/ceval/tree/main)

#### Create the dataset folder
```
mkdir ceval-exam 
cd ceval-exam
```

#### Download the C-Eval dataset
The wget download method is used here; for other download methods, refer to [GitHub](https://github.com/hkust-nlp/ceval/tree/main)
```
wget https://huggingface.co/datasets/ceval/ceval-exam/resolve/main/ceval-exam.zip
unzip ceval-exam
```


## 3. Run the Evaluation Demo
### Run Command
```
python evaluate_chatglm3.py --devid [DEVICE ID] --model_path [PATH_TO_MODEL] --tokenizer_path [PATH_TO_TOKENIZER] --eval_mode fast
```
 ### Parameter Description

| Parameter           | Description                       |
|:--------------:|:---------------------------:|
| `--devid`      | Available device ID                    |
| `--model_path` | Model path, i.e. the model file compiled in step 1                   |
| `--tokenizer_path` | Tokenizer path               |
| `--eval_mode`  | Evaluation mode; two types available: `fast` and `default`   |

📌 This project provides a tokenizer at the path `LLM-TPU/models/ChatGLM3/support/token_config/ `

### Run Results

After the run finishes, you will get a test result file named `submission_{}.json`

### Example
When the file paths are as shown below
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
The run command is
```
export PYTHONPATH=../../
python evaluate_chatglm3.py --devid 10 --model_path ../../models/ChatGLM3/compile/chatglm3-6b_int4_1dev_1024.bmodel --tokenizer_path ../../models/ChatGLM3/support/token_config/ --eval_mode fast
```

## Result Verification

The C-Eval dataset does not provide test set labels. To verify the results, you need to submit the result file to the official website ⬇

[📎Result Submission Page](https://cevalbenchmark.com/static/user_interface.html)