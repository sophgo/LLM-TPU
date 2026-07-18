# LocateAnything-3B

This project demonstrates deploying the visual grounding large model [LocateAnything-3B](https://huggingface.co/NVIDIA/LocateAnything-3B) on BM1684X/BM1688. The model is converted into a bmodel using the [TPU-MLIR](https://github.com/sophgo/tpu-mlir) compiler and deployed to a PCIE environment or an SoC environment.

This model supports visual grounding in images: it takes a natural language description plus an image as input, and outputs the bounding box coordinates or point coordinates of the target.

This document covers how to compile the bmodel and how to run the bmodel in BM1684X/BM1688 environments. The bmodel compilation step can be skipped by directly downloading the precompiled bmodel from the following links (W4BF16, seq2048, max_input_length 1280, static text + dynamic ViT):

``` shell
# BM1684X (PCIe / SoC, 1 dev)
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/locateanything-3b-autoround-w4a16_w4bf16_seq2048_bm1684x_1dev_static_20260709_154503.bmodel
# BM1688 (PCIe / SoC, 2 core)
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/locateanything-3b-autoround-w4a16_w4bf16_seq2048_bm1688_2core_static_20260709_162252.bmodel
```

## Compile the LLM model

This section describes how to compile the LLM into a bmodel.

#### 1. Download the model from HuggingFace

``` shell
git lfs install
git clone https://huggingface.co/groxaxo/LocateAnything-3B-AutoRound-W4A16
```

#### 2. Download docker and start the container

``` shell
docker pull sophgo/tpuc_dev:latest

docker run --privileged --name myname1234 -v $PWD:/workspace -it sophgo/tpuc_dev:latest
```
The following assumes that the environment is in the docker `/workspace` directory.

#### 3. Download the `TPU-MLIR` code and compile it

``` shell
cd /workspace
git clone git@github.com:sophgo/tpu-mlir.git
cd tpu-mlir
source ./envsetup.sh
./build.sh
```

#### 4. Compile the model to generate the bmodel

``` shell
llm_convert.py -m /workspace/LocateAnything-3B-AutoRound-W4A16 \
  -q w4bf16 -s 2048 --max_input_length 1280 \
  -c bm1684x --max_pixels 896,896 --num_device 1 \
  -o /workspace/locateanything_bmodel
```

Compilation parameter description:
- `-q w4bf16`: the text LLM uses W4A16 quantization (AutoRound), and the ViT stays in BF16
- `-s 2048`: maximum sequence length (total KV Cache length, input + output)
- `--max_input_length 1280`: maximum input length (image tokens + text prompt); with static compilation, prefill is padded to this value
- `--max_pixels 896,896`: maximum image resolution (corresponding to a 64×64 patch grid, 1024 image tokens)
- `-c bm1684x` / `-c bm1688`: target chip. bm1688 uses 2 cores by default
- The ViT is always dynamically compiled, supporting images of any size; the text model is static by default (prefill padded to `max_input_length`). For dynamic text (prefill based on the actual token count, faster for small images), add `--dynamic` when compiling

## Compile and run the program (python)

Both PCIE and SoC environments are supported. In the SoC environment, compile the library files directly on the device (linking against the device's built-in libsophon).

### 1. Environment preparation

A python3.10 environment is required. If not available, refer to [this document](https://github.com/sophgo/sophon-demo/blob/release/docs/FAQ.md#13-se7%E5%AE%89%E8%A3%85python310) to install it.

``` shell
sudo apt-get update
sudo apt-get install pybind11-dev

pip3 install torch==2.6.0 torchvision==0.21.0 transformers==5.7.0 \
            pillow numpy lmdb opencv-python-headless
```

> The model's processor (trust_remote_code) imports `decord`/`lmdb`/`cv2` at the top level, and the `check_imports` of transformers 5.7.0 requires all of them to be installed. `lmdb` and `opencv-python-headless` have wheels for aarch64; `decord` has **no PyPI wheel** for aarch64 (SoC):
> - PCIe (x86): just `pip install decord`.
> - For image-only deployment on SoC (aarch64), you can create a stub module so the import passes (the video path will not be executed):
>   ``` shell
>   SP=$(python3.10 -c "import site;print(site.getusersitepackages())")
>   mkdir -p "$SP" && cat > "$SP/decord.py" <<'EOF'
>   class VideoReader:
>       def __init__(self,*a,**k):
>           raise ImportError("decord not installed on aarch64; video unsupported")
>   EOF
>   ```
>   If video is needed, compile decord from source (depends on ffmpeg-dev).

### 2. Compile the library files

Compile the C++ library files to generate `chat.cpython*.so`:

``` shell
cd python_demo
mkdir build
cd build && cmake .. && make && cp *cpython* .. && cd ..
```

> If cmake selects a non-3.10 default python on SoC (e.g. python3.8), specify it explicitly:
> `cmake -DPython_EXECUTABLE=$(which python3.10) ..`

### 3. Run the demo

``` shell
python3 pipeline.py -m /path/to/model.bmodel -c ../config
```

In interactive mode, enter a description text and an image path to get the bounding box coordinates. You can also use the `-p` single-inference mode (see the examples below).

### Running examples

The following tasks are supported (the prompt templates are consistent with the source model):

**Single-object detection / phrase grounding**

``` shell
python3 pipeline.py -m model.bmodel -c ../config -p "detect bed" --media_path test.jpg
# token output: <ref>bed</ref><box><0><585><627><998></box>
# coordinates are normalized values in [0,1000], corresponding to (x1,y1,x2,y2)
# parsed output (pixel coordinates): [bed] box (0,283)-(401,482)
```

**Multi-object detection / dense detection**

Multiple `<box>` entries can be output for one `<ref>`, and multiple categories are supported (separated by `</c>`):

``` shell
python3 pipeline.py -m model.bmodel -c ../config \
  -p "Locate all the instances that matches the following description: bed</c>window</c>pillow" \
  --media_path test.jpg
# parsed output (keeping ref associations: 1 bed box, 1 window box, 16 pillow boxes):
#   [bed] box (0,283)-(401,482)
#   [window] box (259,36)-(394,226)
#   [pillow] box (0,283)-(57,317)
#   ... (16 pillow boxes in total)
```

**Pointing**

Outputs 2 coordinates (x, y):

``` shell
python3 pipeline.py -m model.bmodel -c ../config -p "Point to: bed" --media_path test.jpg
# token output: <ref>bed</ref><box><333><811></box>
# parsed output (pixel coordinates): [bed] point (213,392)
```

> The pipeline automatically parses normalized coordinates in [0,1000] into pixel coordinates (`parse_boxes`/`parse_points`/`parse_result`) and prints them grouped by `<ref>`. The raw tokens are also printed in real time.

## Technical notes

- **Quantization scheme**: the text LLM (Qwen2.5-3B) uses AutoRound W4A16 symmetric quantization; the ViT (MoonViT-SO-400M) and the MLP1 projector stay in BF16
- **Inference mode**: currently only slow mode (pure autoregressive) is supported; MTP multi-token prediction is to be implemented later
- **Batch inference**: batch inference is not supported yet (the source model provides `batch_infer.py` + the `la_flash` backend); currently only single-image serial inference is supported
- **ViT dynamic compilation**: supports images of any size; the pipeline dynamically computes pos_emb (bicubic interpolation), 2D RoPE cos/sin, and merger_index, and passes them into the ViT bmodel

### Image token count calculation

Images are first split into 14×14 patches, then spatially merged 2×2 (merge), so each 28×28 pixel block corresponds to 1 image token. The processor resizes the image to satisfy `max_pixels` with each dimension aligned to 28 pixels (`merge_kernel_size × patch_size = 2 × 14`), then computes the grid.

- Formula: `image_tokens = (grid_h × grid_w) / 4`, where `grid_h = resized_H / 14`, `grid_w = resized_W / 14`
- Equivalently: `image_tokens ≈ number of image pixels / 784`
- There are also 2 wrapper tokens (`<img>`, `</img>`) enclosing the image tokens
- With `max_pixels 896,896`, the upper limit is **1024 image tokens**

| Image size | grid (h×w) | image token count |
|---------|-----------|---------------|
| 640×483 | 36×46 | 414 |
| 896×896 | 64×64 | 1024 |

The sequence length `-s 2048` (total KV Cache length) must accommodate: image tokens + text prompt + generated answer. `--max_input_length 1280` limits the input upper bound (1024 image tokens + text); the output budget = 2048 − 1280 = 768 tokens. Larger images consume more of the input budget, leaving less room for the generated answer.
