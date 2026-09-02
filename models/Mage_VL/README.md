# Mage-VL

This project deploys [Mage-VL](https://huggingface.co/microsoft/Mage-VL-AWQ) (Microsoft, ~5B params) on SOPHGO BM1684X / BM1688 / CV186X TPU chips. The weights are converted to a quantized bmodel with the [TPU-MLIR](https://github.com/sophgo/tpu-mlir) toolchain and served by a Python demo backed by the TPU runtime.

Mage-VL is a multimodal model composed of three stages:

| Stage | Component | Notes |
| :--- | :--- | :--- |
| Vision encoder | **Mage-ViT** | 24-layer ViT, embed_dim 1024, 16 heads (head_dim 64), GELU (erf), **3D RoPE** (t:h:w = 4:6:6, interleaved `rotate_half`), 2×2 spatial merge → 2560-dim visual tokens. |
| Language model | **Qwen3-4B** | Plain Qwen3 backbone (hidden 2560, 36 layers, head_dim 128, 32 q-heads / 8 kv-heads, **1D RoPE**, θ=5e6). The vision tokens are injected into the LM context at the `<|image_pad|>` / `<|video_pad|>` span. |
| Streaming Gate | **StreamMind Gate** | PreNet + VideoMamba (1-layer, T=4 static) + PostNet → ClsNet (4-layer Qwen3 binary classifier). Compiled into the combined bmodel for real-time streaming video decisions. |

This demo covers both **offline (non-streaming)** and **streaming** text / image / video modalities. The vision tower (Mage-ViT) is a static 392-patch bmodel net; an image is pre-resized to a 392-patch grid, and a video is sampled into `N` frames each processed as an independent single-frame ViT pass (no cross-frame attention inside the ViT — temporal reasoning happens in the LM). In streaming mode, the video is divided into consecutive segments of `T=4` frames; each segment gets a Gate decision ("speak" or "silent"), and the LLM generates only when "speak" is triggered.

## Compile the bmodel

> Skip this section if you already have a pre-compiled bmodel.

#### 1. Download the weights

```shell
git lfs install
git clone https://www.modelscope.cn/sahilchachra/Mage-VL-AWQ.git   # AWQ W4A16, compressed-tensors format
```

#### 2. Set up the TPU-MLIR environment

```shell
docker pull sophgo/tpuc_dev:latest
docker run --privileged --name magevl -v $PWD:/workspace -it sophgo/tpuc_dev:latest
# inside the container:
cd /workspace
git clone https://github.com/sophgo/tpu-mlir.git
cd tpu-mlir && source ./envsetup.sh
./build.sh
```

#### 3. Compile the bmodel

```shell
llm_convert.py -m /workspace/Mage-VL-AWQ -s 2048 --max_input_length 1024 \
  -c bm1684x --max_pixels 224,448 -o out
```

| Flag | Value | Meaning |
| :--- | :--- | :--- |
| `-m` | weights dir | HuggingFace source model. |
| `-s` | `2048` | Total sequence length (prefill + generated tokens). |
| `--max_input_length` | `1024` | Max prefill length. |
| `-c` | `bm1684x` | Target chip (`bm1684x` / `bm1688` / `cv186x`). |
| `--max_pixels` | `224,448` | Vision grid cap → 224×448 = 100352 px = **392 patches** (the ViT net's static size). Keep this exact value; the demo's resize logic assumes 392 patches. |
| `-q` | *(omit)* | Follow the source AWQ quantization (W4A16). Do not pass `-q` for AWQ/GPTQ sources. |

The converter emits a single-chip bmodel plus the `config/` directory (tokenizer, chat template, processor code). **Never mix a bmodel with a `config/` from a different model or revision.**

## Run the demo (Python)

#### 1. Environment

Requires Python 3.10 + pybind11. If Python 3.10 is not installed on the device, refer to [this document](https://github.com/sophgo/sophon-demo/blob/release/docs/FAQ.md#13-se7%E5%AE%89%E8%A3%85python310) to install it.

```shell
sudo apt-get update && sudo apt-get install pybind11-dev
pip3 install torch==2.6.0 transformers==5.7.0 pillow numpy av
```

- `torch` is CPU-only on the host (the TPU does the heavy lifting); it is needed because the processor returns PyTorch tensors.
- `av` ([PyAV](https://github.com/PyAV-Org/PyAV)) decodes video frames; required for any video input (offline or streaming).
- `transformers` ≥ 5.6 is required; the custom `MageVLProcessor` is loaded via `trust_remote_code=True` from the `config/` directory.

#### 2. Build the pybind11 module

```shell
cd python_demo
rm -rf build && mkdir build && cd build && cmake .. && make
cp *cpython* ..
```

`CMakeLists.txt` links against `bmrt` / `bmlib` from `/opt/sophon/libsophon-current` and requires `pybind11`. If cmake selects a default Python other than 3.10 (common on SoC with multiple Python versions), specify it explicitly:

```shell
cmake -DPython_EXECUTABLE=$(which python3.10) ..
```

> **Note**: If using a virtual environment (e.g. `/data2/xin/py310`), activate it before running cmake so the correct Python and pybind11 are found automatically.

#### 3. Run

```shell
cd python_demo
# interactive chat
python3 pipeline.py -m <path/to.bmodel> -c ../config --devid 0

# single-shot (non-interactive): @<path> attaches an image or video
python3 pipeline.py -m <path/to.bmodel> -c ../config --devid 0 \
  -p "Describe this image. @test.jpg"

# streaming mode: video is divided into 4-frame segments, each gets a
# Gate decision ("speak"/"silent"), LLM generates only on "speak"
python3 pipeline.py -m <path/to_streaming.bmodel> -c ../config --devid 0 \
  --stream -p "Describe what you see. @video.mp4"

# interactive mode: /stream toggles streaming, /threshold adjusts sensitivity
python3 pipeline.py -m <path/to_streaming.bmodel> -c ../config --devid 0
# then type: /stream
# then: Describe what you see. @video.mp4

# RTSP real-time streaming (separate script)
python3 run_rtsp.py --rtsp rtsp://<host>:<port>/<stream> \
  -m <path/to_streaming.bmodel> -c ../config \
  -p "Describe what you see." --threshold -0.5
```

The `@<path>` syntax auto-detects the media type by extension (`.jpg/.png/...` → image; `.mp4/.avi/.mov/.mkv/.webm/...` → video) and inserts the right placeholder token for the chat template.

### CLI arguments

| Argument | Default | Description |
| :--- | :--- | :--- |
| `-m, --model_path` | *(required)* | Path to the compiled bmodel. |
| `-c, --config_path` | `../config` | Processor config directory (must match the bmodel). |
| `-d, --devid` | `0` | TPU device id. |
| `-p, --prompt` | `None` | Non-interactive single-shot mode. Use `@<path>` to attach an image/video. Omit for the interactive REPL. |
| `--num_video_frames` | `4` | Number of evenly-spaced frames sampled per video. Each frame is one ViT call; `N` frames add `N × 392` vision tokens to the LM context (keep total within `MAX_INPUT_LENGTH`). |
| `--stream` | `false` | Enable streaming mode for video input. The video is divided into consecutive segments of `GATE_FRAMES` (4) frames; each segment gets a Gate decision, and the LLM generates only on "speak". Requires a bmodel compiled with Gate + ClsNet. |
| `--gate_threshold` | `0.0` | Speak-margin threshold for streaming gate decisions. Default 0 = argmax (mean speak score > mean silent score). Higher values require more confidence to speak; negative values lower the bar. |

### How video works

The HF processor's `frames` backend samples `N` evenly-spaced frames, resizes each to the 392-patch grid, and emits `N` per-frame `<|vision_start|>…<|vision_end|>` blocks. Because each frame's temporal grid is `t=1` (≤ `frame_window_size`), the ViT runs each frame as an independent bidirectional-attention chunk — so the existing static 392-patch ViT bmodel is simply called **once per frame** with that frame's pixel values and per-frame `(t, h, w)` patch positions. No bmodel recompilation is needed for video.

Frame decoding (PyAV) is shared between this demo and the HF reference via `magevl_video.py`, so the two sides decode identical frames.

### How streaming works

Streaming mode (`--stream`) divides a video into consecutive segments of `T=4` frames and runs a **Gate decision** per segment to determine whether the LLM should generate a description ("speak") or skip ("silent").

```
Video → [frame 1..4] → ViT(×4) → mean pool → Gate → ClsNet → logits[4,2]
                                                              ↓
                                              speak? → LLM generates description
                                              silent? → skip, next segment
        [frame 5..8] → ViT(×4) → mean pool → Gate → ClsNet → ...
        ...
```

For each segment:

1. **ViT**: 4 frames are processed individually (same as offline video — 4 × `forward_vit` calls).
2. **Mean pool**: Per-frame merged embeddings `[98, 2560]` are averaged into `[2560]` vectors.
3. **Gate** (bmodel net): `[1, 4, 2560]` bf16 → PreNet + VideoMamba (1-layer, T=4 static selective scan) + PostNet → `[1, 4, 2560]` bf16.
4. **ClsNet** (bmodel net): `[1, 4, 2560]` bf16 → 4-layer Qwen3 binary classifier (vocab=2: silent/speak) → `[1, 4, 2]` bf16 logits.
5. **Decision**: Mean logits across frames → argmax (or threshold comparison via `--gate_threshold`).
6. **Generation**: On "speak", KV cache is cleared and the LLM generates independently from the same segment's video context.

Frames are read lazily from the video (via PyAV), so only `T=4` frames are held in memory at any time — this keeps system RAM usage low even for long videos on memory-constrained SoC devices.

## <a id="limitations"></a>Limitations & roadmap

- **Streaming bmodel required for `--stream`**: Streaming mode requires a bmodel compiled with Gate + ClsNet nets included. The standard offline bmodel (without gate) works for text/image/video but not streaming. To compile a streaming-enabled bmodel, use the same `llm_convert.py` command — the converter automatically includes Gate + ClsNet when the source model has `streammind_gate.safetensors`.
- **Codec (info-density) video backend**: *not supported out-of-the-box on this SoC.* The source model's codec backend (`video_backend="codec"`) samples patches by per-frame bit-cost. It improves long-video accuracy / token efficiency but is **not** faster.

  The SoC *does* have a video toolchain (VPU/VPP hardware acceleration, sophon-ffmpeg, sophon-opencv) — the blocker is a **generation mismatch**, not a missing toolchain. Verified on the target SoC (aarch64, glibc 2.31, Python 3.10):

  | Codec engine | Blocker |
  | :--- | :--- |
  | `hevc` | `codec-video-prep` (PyPI) ships a self-contained aarch64 cp310 wheel, but its bundled **ffmpeg 5.x** libs (`libavformat.so.59` / `libavcodec.so.59`) require **GLIBC_2.34/2.35**; the SoC has glibc 2.31 → `ImportError: version 'GLIBC_2.35' not found`. The SoC's own sophon-ffmpeg is **4.1.3** (`libavcodec.so.58`, different ABI major) so it cannot substitute. There is also **no Python-3 `cv2`** on the SoC (sophon-opencv ships a `cv2.so` for Python 2 only; `opencv-python` has no aarch64 cp310 wheel), and the codec import chain needs it. |
  | `dcvc-rt` | requires CUDA, which the SoC does not have. |

  Feasible remediation (none is a demo-side tweak; all are future-phase work): (a) run `cv-preinfer` info-density sampling on a companion **x86 host** and feed the selected frames/positions to the SoC — cleanest, and matches the codec backend's design as a preprocessing step; (b) rebuild `codec-video-prep` from source against sophon-ffmpeg 4.1.3 / glibc 2.31 and provide a Python-3 `cv2` binding; (c) upgrade the SoC OS to glibc ≥ 2.35 (risky on embedded hardware). The static 392-patch ViT also caps per-frame resolution, so the codec backend's resolution-flexibility advantage is limited here anyway.
- **No multi-chip / history-KV variants** in this demo.

## FAQ

**Q: The image looks right but the answer is gibberish.**
Check that `--config_path` points at the `config/` shipped with the bmodel. A mismatched tokenizer/chat-template produces valid-looking but wrong tokens.

**Q: Video answer is wrong / first frame looks "black".**
Confirm the target machine has `av` (PyAV) installed and it can open the file. The first sampled frame of a black-start video legitimately encodes near-zero content (low merger cosine vs. a differently-decoded reference) — this is a decoder YUV→RGB difference, not a bmodel bug; generated tokens still match the HF reference exactly on identical inputs.

**Q: `input length N exceeds MAX_INPUT_LENGTH`.**
Either shorten the prompt or recompile with a larger `--max_input_length`. Long videos add `N × 392` vision tokens; lower `--num_video_frames` if needed.
