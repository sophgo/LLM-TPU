# Step3_VL

This project demonstrates deploying the multimodal large model [Step3_VL](https://huggingface.co/stepfun/Step3-VL-10B-AWQ) on BM1684X. The model is converted into a bmodel using the [TPU-MLIR](https://github.com/sophgo/tpu-mlir) compiler and deployed to a PCIE environment using C++ code.

This model supports image understanding (image-text-to-text). A Python demo is provided.

## Model Architecture

- **Text**: Qwen3-8B (36 layers, hidden=4096, GQA 32:8, head_dim=128, SiLU, RMSNorm, QK-norm, RoPE θ=1e6)
- **Vision**: PE-lang ViT (47 layers, width=1536, 16 heads, MHA, QuickGELU, LayerScale, absolute positional embedding + 2D RoPE)
- **Projector**: 2× stride-2 Conv2d downsampler + Linear(6144→4096, no bias)
- **Quantization**: compressed-tensors 4-bit symmetric weights, BF16 activations (w4bf16, group_size=32)
- **Multi-crop**: Global 728×728 view (169 tokens) + up to 4 local 504×504 crops (81 tokens each)

## Download pre-compiled bmodel

Two pre-compiled bmodels are available. Choose based on your sequence length needs:

```shell
# seq_length=2048, max_patches=4 
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/step3-vl-10b-awq_w4bf16_seq2048_bm1684x_1dev_static_20260803_111848.bmodel

# seq_length=512 (lower memory, no vit_patch)
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/step3-vl-10b-awq_w4bf16_seq512_bm1684x_1dev_static_20260803_185639.bmodel
```

## Compile the bmodel

#### 1. Download model weights from ModelScope

```shell
git clone https://www.modelscope.cn/cyankiwi/Step3-VL-10B-AWQ-4bit.git
```

This checkpoint uses compressed-tensors symmetric quantization (4-bit weights, BF16 activations, group_size=32).

#### 2. Download TPU-MLIR and build

```shell
git clone git@github.com:sophgo/tpu-mlir.git
cd tpu-mlir
source ./envsetup.sh
./build.sh
```

#### 3. Compile

```shell
cd /workspace  # or your working directory
llm_convert.py -m /workspace/Step3-VL-10B-AWQ-4bit \
    -s 2048 --max_input_length 1024 \
    -q w4bf16 -c bm1684x \
    --max_pixels 728,728 \
    -o step3vl_w4bf16
```

Parameters:
| Parameter | Value | Description |
|---|---|---|
| `-m` | model path | Path to the downloaded HF model weights |
| `-s` | 2048 | Max sequence length (text + image tokens) |
| `--max_input_length` | 1024 | Max input token length for prefill |
| `-q` | w4bf16 | AWQ 4-bit weights with BF16 activations |
| `-c` | bm1684x | Target chip |
| `--max_pixels` | 728,728 | Controls `max_patches` (the number of local crop patches). |

> **Note**: The `-q w4bf16` flag is required. Using `w4f16` will cause FP16 activation overflow.
>
> **Note**: `--max_pixels` determines `max_patches` (the batch size for local crop processing). The default `728,728` is square, which produces 0 patches in the HF processor, so `vit_patch` is skipped to save ~3.5 GB. To enable patch support, specify a non-square resolution. The global view resolution is always 728×728 regardless of this setting.
>
> **Important**: The number of patches an image produces depends on its aspect ratio, not just its pixel count. If an image produces more patches than the compiled `max_patches`, the excess patches are silently skipped. Set `--max_pixels` to the resolution of images you will actually process (e.g. `1920,1080` for HD camera feed, `1024,1024` for square images).

## Run Inference (Python)

#### 1. Environment preparation

```shell
pip3 install pillow transformers>=4.57.0
```

#### 2. Build the C++ library

```shell
cd python_demo
mkdir build && cd build
cmake .. && make -j$(nproc)
cp chat.*.so ..
cd ..
```

#### 3. Run

```shell
# Interactive mode
python3 pipeline.py -m step3-vl-10b-awq_w4bf16_seq2048_bm1684x.bmodel -c ../config

# Single prompt (text only)
python3 pipeline.py -m step3-vl-10b-awq_w4bf16_seq2048_bm1684x.bmodel \
    -c ../config -p "Hello, what is 1+1?"

# Single prompt (with image)
python3 pipeline.py -m step3-vl-10b-awq_w4bf16_seq2048_bm1684x.bmodel \
    -c ../config -p "Describe this image. @test.jpg"

# Disable thinking mode (faster, less tokens)
python3 pipeline.py -m step3-vl-10b-awq_w4bf16_seq2048_bm1684x.bmodel \
    -c ../config --no_think -p "What is in this image? @test.jpg"
```

In interactive mode:
- Use `@<path>` to attach an image: `what is in this photo? @./photo.jpg`
- Type `/clear` or `/new` to start a new session
- Type `/exit` or `/q` to quit

## CLI Parameters

| Parameter | Default | Description |
|---|---|---|
| `-m, --model_path` | (required) | Path to the bmodel file |
| `-c, --config_path` | `../config` | Path to the tokenizer/processor config |
| `-d, --devid` | 0 | TPU device ID |
| `-p, --prompt` | None | Single prompt mode; use `@<path>` for image |
| `--no_think` | off | Disable thinking mode |


## FAQ

### Image tokens and resolutions

The model uses a multi-crop strategy with fixed resolutions determined by the architecture:

- **Global view (728×728)**: Every input image is resized to 728×728. This produces a 52×52 patch grid, which after 2× downsampling yields **169 vision tokens**.
- **Local patches (504×504)**: Depending on the image's aspect ratio, 0 to 4 additional crops of 504×504 are extracted via sliding window. Each crop produces a 36×36 patch grid → **81 vision tokens**.

These resolutions are fixed by the model's vision encoder (positional embeddings and downsampler are designed for these exact sizes) and cannot be changed.

**How different input images are handled:**

| Input image | Global view | Patches | Total image tokens |
|---|---|---|---|
| Long side ≤ 728 and aspect ratio ≤ 1.5 (e.g. 640×483) | Resize to 728×728 | None | 171 |
| Long side ≤ 728 and aspect ratio > 1.5 (e.g. 728×364) | Resize to 728×728 | Sliding-window crops (size = short side) | 171 + N×83 |
| Long side > 728 (e.g. 1920×1080) | Resize to 728×728 | 504×504 sliding-window crops | 171 + N×83 |
| Very small image (< 32px on short side) | Pad to square, resize to 728×728 | None | 171 |

> **Note**: The HF processor may produce more than 4 patches for wide/high images (e.g. 1920×1080 → 8 patches), but the bmodel is compiled with `max_patches=4`. Images that produce more than 4 patches will be rejected by the pipeline. To handle such images, recompile with a larger `max_patches` in the Converter.

**Token structure per image:**

- Global view: `<im_start>` (1) + `<im_patch>` (169) + `<im_end>` (1) = **171 tokens**
- Each local crop: `<patch_start>` (1) + `<im_patch>` (81) + `<patch_end>` (1) + optional `<patch_newline>` (1) = **83-84 tokens**
- Maximum total with 4 patches: 171 + 4×84 = **507 tokens**

### Memory requirements

The bmodel requires ~10.8 GB device memory. BM1684X with 16 GB VRAM is sufficient.

### Known limitations

- Static compilation only (fixed input shapes); dynamic shape not yet supported
- Local patches (`vit_patch`) not compiled by default; specify `--max_pixels` with a non-square resolution to enable (e.g. `--max_pixels 1920,1080`). Images that produce more patches than compiled `max_patches` will have patches silently skipped.
- No video support in the current demo
- No multi-turn conversation history (static mode without `--use_history_kv`)
