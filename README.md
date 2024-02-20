# LLM-TPU
# LLM-TPU

### Quick Start

#### 1. install docker & enter docker
```
docker pull sophgo/tpuc_dev:latest
docker run --privileged --name mlir -v /dev:/dev -v $PWD:/workspace -it sophgo/tpuc_dev:latest bash
docker exec -it mlir bash
```

#### 2. clone LLM-TPU & run
```
git clone https://github.com/sophgo/LLM-TPU.git
cd LLM-TPU/models/Llama2
./run_demo.sh --download --compile
```

### Table

|Model             |Command                                                                  |INT4                |INT8                |F16                 |
|:-                |:-                                                                       |:-                  |:-                  |:-                  |
|ChatGLM3          |./run_demo.sh --download --compile                                       |:white\_check\_mark:|                    |                    |
|Llama2            |./run_demo.sh --download --compile                                       |:white\_check\_mark:|                    |                    |
|Qwen              |./run_demo.sh --download --compile --arch pcie                           |:white\_check\_mark:|                    |                    |
