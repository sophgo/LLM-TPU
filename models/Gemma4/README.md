# Gemma4

Gemma4 is a multimodal large language model released by Google, supporting text, image, video, and audio inputs.

- Gemma4-E2B-it: 2.3B effective parameters (5.1B including the embedding layer)
- Gemma4-E4B-it: 4.5B effective parameters (8B including the embedding layer)

The following pre-compiled models are available for direct download:
```bash
pip3 install dfss
# E2B BM1684X 2k context
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/gemma-4-e2b-it-awq-4bit_w4f16_seq2048_bm1684x_1dev_static_20260618_152146.bmodel
# E2B BM1684X 4k context
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/gemma-4-e2b-it-awq-4bit_w4f16_seq4096_bm1684x_1dev_static_20260618_184549.bmodel
# E2B BM1688 2k context
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/gemma-4-e2b-it-awq-4bit_w4f16_seq2048_bm1688_2core_static_20260618_143212.bmodel
# E2B BM1688 4k context
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/gemma-4-e2b-it-awq-4bit_w4f16_seq4096_bm1688_2core_static_20260617_212224.bmodel
# E2B embedding layer weights, required for both BM1684X and BM1688
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/weight/per_layer_token_embd_e2b.bin

# E4B BM1684X
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/gemma-4-e4b-it-awq-4bit_w4f16_seq4096_bm1684x_1dev_static_20260521_154219.bmodel
```

## 1. Compile the bmodel

### 1.1 Download the source model

The model is quite large and will take a long time to download. Please download the AWQ or GPTQ quantized version, or quantize it yourself.
```bash
# E2B
git clone https://huggingface.co/Chunity/gemma-4-E2B-it-AWQ-4bit
# E4B
git clone https://huggingface.co/Chunity/gemma-4-E4B-it-AWQ-4bit
```

You can also compile the model yourself.

### 1.2 Set up the MLIR environment

```bash
docker pull sophgo/tpuc_dev:latest

# myname1234 is just an example, you can set your own name
docker run --privileged --name myname1234 -v $PWD:/workspace -it sophgo/tpuc_dev:latest

cd /workspace
# Download the pre-compiled mlir package
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/mlir_club/tpu-mlir_v1.0.0.dev-ed1fc7c49-20260618.tar.gz
tar zxf tpu-mlir_v1.0.0.dev-ed1fc7c49-20260618.tar.gz
cd tpu-mlir_v1.0.0.dev-ed1fc7c49-20260618
source ./envsetup.sh  # activate the environment variables
```

### 1.3 Compile the model

```bash
# E2B 
llm_convert.py -m gemma-4-E2B-it-AWQ-4bit -s 4096 --max_input_length 2560 -q w4f16 -c bm1684x -o bmodel/ --audio_length 750

# E4B
llm_convert.py -m gemma-4-E4B-it-AWQ-4bit -s 2048 --max_input_length 1024 -q w4f16 -c bm1684x -o bmodel/ --audio_length 750
```

Parameter description:
- `-m`: source model path
- `-s`: sequence length (seq_length)
- `--max_input_length`: maximum input length (number of valid tokens in the prefill stage)
- `-q`: quantization type (bf16 / w4f16, etc.)
- `-c`: target chip (bm1684x / bm1688, etc.)
- `-o`: output directory
- `--audio_length`: maximum number of audio tokens, default 750

## 2. Set up the environment

```bash
sudo apt-get update
sudo apt-get install pybind11-dev

pip3 install torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0 transformers==5.5.4 Jinja2==3.1.2 librosa ffmpeg-python av
```

## 3. Compile the demo

```bash
cd python_demo
mkdir build
cd build && cmake .. && make && cp *cpython* .. && cd ..
```

## 4. Run the demo

### CLI demo (interactive)

```bash
python3 pipeline.py -m your_bmodel_path --embed_path per_layer_token_embd_e2b.bin -c ../config -d 0
```

After launching, enter questions interactively; you may optionally attach an image/video/audio path.

### Supported input types

| Input type | Supported file formats |
|---------|-------------|
| Text | Enter the question directly |
| Image | jpg, jpeg, png, gif, bmp, webp |
| Video | mp4, avi, mov, mkv, flv, wmv |
| Audio | wav, mp3, flac, ogg, m4a, aac, wma |

### Notes

Token usage per modality:
- **Image**: token count = `(resize后H // 16) × (resize后W // 16) // 9`. Images of any resolution are resized while preserving the aspect ratio to at most 645120 pixels, i.e. 280 tokens. For example, a square image is resized to 768×768, corresponding to 256 tokens.
- **Video**: fixed at 32 frames, with at most 70 tokens per frame (161280 pixels), totaling 2240 tokens. Videos longer than 32 frames are uniformly sampled down to 32 frames; videos shorter than 32 frames are padded by repeating the last frame up to 32 frames. Models with `max_input_length` below 2240 do not support video input.
- **Audio**: 25 tokens per second, supporting up to 30 seconds of audio (750 tokens); the `--audio_length` parameter at compile time controls the upper limit.

| Parameter | Description | Default |
|------|------|--------|
| `-m, --model_path` | Path to the bmodel file | Required |
| `-c, --config_path` | Path to the tokenizer/config directory | `../config` |
| `-d, --devid` | Device ID to use | `0` |

### Examples

**Text Q&A:**
```
Question: Write a short joke about saving RAM.
Image, Video, or Audio Path: (press Enter to skip)
```

**Image understanding:**
```
Question: What is shown in this image?
Image, Video, or Audio Path: test.jpg
```

**Video understanding:**
```
Question: Describe this video.
Image, Video, or Audio Path: test.mp4
```

**Audio understanding:**
```
Question: Transcribe the following speech segment in its original language. Follow these specific instructions for formatting the answer:\n* Only output the transcription, with no newlines.\n* When transcribing numbers, write the digits, i.e. write 1.7 and not one point seven, and write 3 instead of three.
Image, Video, or Audio Path: asr_zh.wav
```
