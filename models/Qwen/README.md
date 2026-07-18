# Qwen

This project implements the deployment of the large language model [Qwen-7B-Chat](https://huggingface.co/Qwen/Qwen-7B-Chat) on BM1684X. The model is converted into a bmodel using the [TPU-MLIR](https://github.com/sophgo/tpu-mlir) compiler, and deployed with C++ code to the BM1684X PCIE environment or SoC environment.

* This project also supports [Qwen-14-Chat](https://huggingface.co/Qwen/Qwen-14B-Chat); the operation method is the same as for `Qwen-7B-Chat`.
* This project also supports [Qwen-1_8-Chat](https://huggingface.co/Qwen/Qwen-1_8B-Chat); the operation method is the same as for `Qwen-7B-Chat`.

## Development Environment Setup

### 1. Download docker and start the container

``` shell
docker pull sophgo/tpuc_dev:latest

# myname1234 is just an example, you can set your own name
docker run --privileged --name myname1234 -v $PWD:/workspace -it sophgo/tpuc_dev:latest bash

docker exec -it myname1234 bash
```
The following assumes the environment is in the `/workspace` directory of the docker container.

### 2. Download the `TPU-MLIR` code and compile it

(You can also directly download and extract the prebuilt release package.)

``` shell
cd /workspace
git clone git@github.com:sophgo/tpu-mlir.git
cd tpu-mlir
source ./envsetup.sh  # activate environment variables
./build.sh # compile mlir
```

### 3. Export the ONNX model

Refer to the README.md under [compile](./compile) and [compile](./compile).

### 4. Compile the model

Refer to the README.md under [compile](./compile) and [compile](./compile).

### 5. Compile and run the program

Refer to the README.md under [python_demo](./python_demo) and [demo](./demo).


# Adjustments to the `modeling_qwen.py` file code

1) The first modification is as follows (this is because TORCH2 operators fail to convert to ONNX):

    ``` python
    # SUPPORT_TORCH2 = hasattr(torch, '__version__') and int(torch.__version__.split(".")[0]) >= 2
    SUPPORT_TORCH2 = False
    ```

2) The second modification is as follows (this is because converting to ONNX reports a shape inference failure):

    ```python
    # attn_weights = attn_weights / torch.full(
    #     [],
    #     size_temp ** 0.5,
    #     dtype=attn_weights.dtype,
    #     device=attn_weights.device,
    # )
    attn_weights = attn_weights / (size_temp ** 0.5)
    ```

3) The third modification is as follows (this entire code block is commented out because `attention_mask` can be used directly, avoiding complex logic and improving performance):

    ```python
    # if self.use_cache_quantization:
    #     query_length, key_length = query.size(-2), key[0].size(-2)
    # else:
    #     query_length, key_length = query.size(-2), key.size(-2)
    # causal_mask = registered_causal_mask[
    #     :, :, key_length - query_length : key_length, :key_length
    # ]
    # mask_value = torch.finfo(attn_weights.dtype).min
    # mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(
    #     attn_weights.device
    # )
    # attn_weights = torch.where(
    #     causal_mask, attn_weights.to(attn_weights.dtype), mask_value
    # )
    ```

4) The fourth modification is as follows (same reason as above):

    ``` python
    # query_length, key_length = query.size(-2), key.size(-2)
    # causal_mask = registered_causal_mask[
    #     :, :, key_length - query_length : key_length, :key_length
    # ]
    # mask_value = torch.finfo(attn_weights.dtype).min
    # mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(
    #     attn_weights.device
    # )
    # attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    ```

5) The fifth modification: move the following code before `if layer_past is not None:`:

    ``` python
    if use_cache:
        present = (key, value)
    else:
        present = None
    ```

    This is because the KV Cache only needs to output 1 unit instead of all outputs, which improves efficiency.
