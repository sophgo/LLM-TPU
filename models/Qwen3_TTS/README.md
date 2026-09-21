# Qwen3-TTS

This project deploys [Qwen3-TTS-12Hz-0.6B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) (Qwen Team, ~0.6B params) on SOPHGO BM1684X / BM1688 / CV84X6 TPU chips. The weights are converted to a BF16 bmodel with the [TPU-MLIR](https://github.com/sophgo/tpu-mlir) toolchain and served by a Python demo backed by the TPU runtime. It supports 3-second voice cloning across 10 languages (zh / en / ja / ko / de / fr / ru / pt / es / it).

Qwen3-TTS is a **text-to-speech** model built on a discrete multi-codebook language-model architecture. Each audio frame produces 16 codec codes, and the whole pipeline — speaker encoder, multi-codebook LM, and neural codec decoder — runs on the TPU as a single bmodel:

| Stage | Component | Notes |
| :--- | :--- | :--- |
| Speaker encoder | **ECAPA-TDNN** | Extracts a speaker embedding from the reference audio (x-vector clone). |
| Multi-codebook LM | **Talker + CodePredictor** | Talker (28-layer) emits codec code 0; CodePredictor (5-layer) emits codec codes 1..15. |
| Codec decoder | **Mimi** | Decodes the 16 codec codes into a 24 kHz waveform. |


## Download pre-compiled bmodel

```shell
# BM1684X
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen3-tts-12hz-0.6b-base_bf16_seq2048_bm1684x_1dev_static_20260916_111859.bmodel

# BM1688
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen3-tts-12hz-0.6b-base_bf16_seq2048_bm1688_2core_static_20260916_112020.bmodel

# CV84X6
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen3-tts-12hz-0.6b-base_bf16_seq2048_bm1684x2_4core_static_20260920_202140.bmodel
```

This bmodel (~2.0 GB, BF16) includes the full pipeline: speaker encoder + Talker + CodePredictor + Mimi decoder + sampling heads.

## Compile the bmodel

> Skip this section if you already have a pre-compiled bmodel. Compiling requires an x86 host (no TPU hardware needed).

#### 1. Download the weights

```shell
git lfs install
git clone https://www.modelscope.cn/Qwen/Qwen3-TTS-12Hz-0.6B-Base.git
```

#### 2. Set up the TPU-MLIR environment

```shell
docker pull sophgo/tpuc_dev:latest
docker run --privileged --name qwen3tts -v $PWD:/workspace -it sophgo/tpuc_dev:latest
# inside the container:
cd /workspace
git clone https://github.com/sophgo/tpu-mlir.git
cd tpu-mlir && source ./envsetup.sh
./build.sh
```

#### 3. Compile the bmodel

```shell
llm_convert.py -m /workspace/Qwen3-TTS-12Hz-0.6B-Base -s 2048 --max_input_length 1024 \
  -c bm1684x -q bf16 --do_sample -o out
```

For BM1688 or CV84X6, switch `-c` to `bm1688` or `bm1684x2` respectively.

| Flag | Value | Meaning |
| :--- | :--- | :--- |
| `-m` | weights dir | HuggingFace source model. |
| `-s` | `2048` | Total sequence length (prefill + generated codec frames). |
| `--max_input_length` | `1024` | Max prefill length (text tokens). |
| `-c` | `bm1684x` | Target chip (`bm1684x` / `bm1688` / `cv186x` / `bm1684x2`). |
| `-q` | `bf16` | Quantization — the source is BF16, kept as-is. |
| `--do_sample` | *(flag)* | Deploy `greedy_head` + `sample_head` nets so the Talker `code0` is sampled on-device with repetition penalty; the `lm_head` (codec_head) emits raw `[1,3072]` logits instead of an argmax. |

**Reference audio length** (optional): the speaker encoder compiles with a static input of 750 mel frames (~8 s). To match a different reference length, pass `--audio_length <mel_frames>` (1 s = 93.75 frames, e.g. `--audio_length 563` for a 6 s reference). Recompile to apply it.

**In-context-learning (ICL) clone mode is not supported.** ICL `--ref_text` voice cloning has unaligned accuracy on this toolchain and is not delivered; only x-vector-only cloning (`--ref_wav`) is supported.


## Run inference (Python)

#### 1. Environment

* Environment preparation
> (This must be done before running python_demo.)

```shell
pip3 install transformers librosa soundfile torch numpy
```

The demo loads the tokenizer from the `config/` dir via `transformers` (the HuggingFace Qwen3-TTS weights are not needed at runtime); the only host-side preprocessing is mel STFT.


#### 2. Build the C++ extension

```shell
cd models/Qwen3_TTS/python_demo
mkdir build && cd build && cmake .. && make
cp *cpython* ..
```

#### 3. Run

Single-shot synthesis:

```shell
cd models/Qwen3_TTS/python_demo
python3 pipeline.py --model_path ../qwen3-tts.bmodel --config_path ../config \
  --text "Hello, how are you today?" --ref_wav clone.wav --out_wav out.wav
```

Interactive mode (omit `--text` / `--ref_wav`):

```shell
python3 pipeline.py --model_path ../qwen3-tts.bmodel --config_path ../config
# then type text / ref_wav / out_wav at the prompts (Ctrl-D to exit)
```

The reference audio is resampled to 24 kHz, converted to a 128-mel spectrogram, and zero-padded to the speaker encoder's static input length (default 750 mel frames ≈ 8 s) so the ECAPA speaker encoder sees the whole clip. Generation stops at the codec EOS token.

## CLI parameters

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `-m, --model_path` | *(required)* | Path to the `.bmodel`. |
| `-c, --config_path` | *(required)* | Path to the `config/` dir. |
| `-d, --devid` | `0` | TPU device id. |
| `--text` | `""` | Text to synthesize (omit for interactive mode). |
| `--ref_wav` | `""` | Reference audio for voice cloning (omit for interactive mode). |
| `--out_wav` | `out.wav` | Output wav path. |
| `--language` | `Auto` | Spoken language hint. |
| `--max_new_tokens` | `2048` | Max codec frames to generate before forcing a stop. |
| `--do_sample` / `--greedy` | `--do_sample` | Sample (HF default) vs greedy argmax. |
| `--temperature` | `0.9` | Talker `code0` sampling temperature. |
| `--top_k` | `50` | Talker `code0` top-k. |
| `--top_p` | `1.0` | Talker `code0` nucleus. |
| `--repetition_penalty` | `1.05` | Talker `code0` repetition penalty (on-device, sliding window). |
| `--sub_temperature` | `0.9` | CodePredictor `code1..15` sampling temperature. |
| `--sub_top_k` | `50` | CodePredictor top-k. |
| `--sub_top_p` | `1.0` | CodePredictor nucleus. |
| `--repetition_window` | `1024` | Sliding window length for Talker repetition penalty. |
| `--seed` | `1234` | RNG seed for the multinomial draws. |

## How it works

Each audio frame is one Talker decode step (28 layers) + 16 CodePredictor decode steps (5 layers) → 16 codec codes (code0 from Talker, code1..15 from CodePredictor). The 16 codec embeddings are summed with the trailing text hidden state, fed back into the Talker for the next frame, and accumulated into a `[T, 16]` code tensor. The Mimi decoder then turns the codes into a 24 kHz waveform (`12.5 Hz` frame rate → `1920` audio samples per codec frame, chunked at 256 frames per call).

## FAQ

- **Token / frame accounting**: `SEQLEN = 2048` covers prefill (9 prompt tokens) + up to ~2036 generated codec frames (~163 s of audio at 12.5 Hz). The demo stops early on the codec EOS token.
- **Reference audio length**: the speaker encoder has a static input length of 750 mel frames (~8 s at 24 kHz / hop 256). **Use a reference clip of 8–9 s** — clips longer than 8 s are truncated, and clips much shorter are zero-padded, which dilutes the speaker embedding and may lower output quality. To match a shorter reference, [recompile](#compile-the-bmodel) with `--audio_length <mel_frames>` (1 s = 93.75 frames) so the static length matches your reference.
- **Memory**: the bmodel is ~2.0 GB on device; peak TPU memory also holds the per-layer KV caches (28 Talker + 5 CodePredictor layers).
- **Sampling vs greedy**: `--greedy` reproduces the HF greedy trajectory (useful for debugging); `--do_sample` (default) gives natural, non-degenerate speech. The Talker needs repetition penalty to avoid falling into code loops on long generations.
