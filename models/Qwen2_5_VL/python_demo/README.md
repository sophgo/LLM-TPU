## 环境说明

(尤其SoC环境，python需要用3.10版本，transformers需要更新到4.49或以上)

sudo apt-get install python3.10-dev
sudo apt-get install pybind11-dev
pip3 install torchvision pillow qwen_vl_utils transformers --upgrade


## 编译方法

将`python_demo`拷贝到环境后，执行如下：
``` shell
mkdir build
cd build
cmake ..
make
mv chat*.so ../
```

## 下载模型
```shell
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen2.5-vl-3b_w4bf16_seq2048.bmodel
```

## 运行
``` shell
python3 pipeline.py --model_path ../qwen2.5-vl-3b_w4bf16_seq2048.bmodel --config_path config/
```