---
license: other
license_name: nvidia-license-agreement
license_link: https://huggingface.co/nvidia/LocateAnything-3B
tags:
  - locateanything
  - vision-grounding
  - quant
  - int4
  - auto-round
  - w4a16
  - text-generation
  - multimodal
base_model: nvidia/LocateAnything-3B
pipeline_tag: image-text-to-text
---

# LocateAnything-3B-AutoRound-W4A16

**W4A16 (4-bit weight, 16-bit activation) INT4 quantization of [nvidia/LocateAnything-3B](https://huggingface.co/nvidia/LocateAnything-3B)** using
Intel's [AutoRound](https://github.com/intel/auto-round) v0.13.0 (per-channel symmetric, group_size=128).

This checkpoint is a **drop-in replacement** for the original BF16 model that delivers a 51% size reduction and 54% VRAM reduction with **no measurable accuracy loss** on a 50-image single-object grounding benchmark. The vision encoder (MoonViT), multimodal projector (MLP1), embedding, and `lm_head` are preserved in BF16; only the Qwen2.5-3B text decoder linears are quantized.

## TL;DR

| | BF16 (base) | INT4 (this) | Δ |
|---|---|---|---|
| Disk size | 7.3 GB | **3.55 GB** | **−51%** |
| Runtime VRAM | 7.64 GB | **3.54 GB** | **−54%** |
| Mean IoU (n=50) | 0.753 | 0.754 | +0.002 |
| IoU@0.5 | 92% | **96%** | **+4 pts** |
| Output validity | 100% | 100% | 0 |
| Latency / call | 2.75 s | 2.61 s | −5% |

**No accuracy drop.** INT4 is statistically tied with BF16 on mean IoU and is +4 points on IoU@0.5 (the "did the model find the object" metric). On letters specifically, BF16 occasionally hallucinates oversized bounding boxes due to MTP-speculation drift; INT4's quantization suppresses this drift and produces tighter, more accurate boxes on the hardest examples.

## How to use

The model uses custom code from the [NVIDIA/Eagle](https://github.com/NVlabs/EAGLE) `LocateAnything` repo, so `trust_remote_code=True` is required. **Use SDPA attention (not magi)** unless you are on Hopper+ — the magi backend is not available on Ampere (RTX 3090) GPUs.

```python
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoProcessor
from auto_round.inference import convert_hf_model

MODEL = "groxaxo/LocateAnything-3B-AutoRound-W4A16"

tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModel.from_pretrained(
    MODEL,
    dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="sdpa",
    device_map={"": "cuda"},
).eval()

# CRITICAL: swap the standard linears for AutoRound QuantLinear runtime layers.
convert_hf_model(model, target_device=str(model.device))

image = Image.open("your_image.jpg").convert("RGB")
question = "Please provide the bounding box of the <ref>red car</ref>."

messages = [{"role": "user", "content": [
    {"type": "image", "image": image},
    {"type": "text", "text": question},
]}]
text = processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
images, videos = processor.process_vision_info(messages)
inputs = processor(text=[text], images=images, videos=videos, return_tensors="pt").to(model.device)
inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=2048,
        do_sample=False,
        custom_generate="locateanything",   # or omit if you use worker.predict()
    )
answer = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=False)
print(answer)
```

Or use the official worker directly:

```python
import sys
sys.path.append("Eagle/Embodied")
from locateanything_worker import LocateAnythingWorker
worker = LocateAnythingWorker(MODEL, device="cuda", dtype=torch.bfloat16)
convert_hf_model(worker.model, target_device=str(worker.device))
result = worker.predict(image, question, generation_mode="hybrid")
print(result["answer"])
```

> **Note** if you call `model.generate(...)` directly: the custom `generate` in `modeling_locateanything.py` requires both `use_cache=True` and a `tokenizer=` positional argument (it reads `tokenizer.model_max_length`). Easiest path is to use `LocateAnythingWorker.predict(...)` which sets both for you.

## Quantization recipe

```bash
# Extract Qwen2.5-3B text decoder weights from the LocateAnything checkpoint
python scripts/extract_text_decoder.py

# Quantize the text decoder (252 linears → QuantLinear, 200 iters on Pile/CC)
python scripts/quantize_text_decoder.py --profile final

# Repack quantized Qwen2 + BF16 vision + BF16 projector into a full
# LocateAnythingForConditionalGeneration checkpoint
python scripts/repack_locateanything.py
```

- **Method**: AutoRound 0.13.0, LLM path (the MLLM path is broken for `LocateAnythingForConditionalGeneration` because the custom processor wraps the image list in a way that doesn't compose with AutoRound's hf processor)
- **Recipe**: W4A16, symmetric, group_size=128, batch_size=8, nsamples=128, seqlen=2048, 200 iters
- **What stays BF16**: `vision_model.*` (MoonViT, 326 tensors), `mlp1.*` (multimodal projector, 6 tensors), `language_model.lm_head`, `language_model.model.embed_tokens`, `language_model.model.norm`
- **What gets quantized**: 36 decoder layers × 7 linears (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) = **252 linears → 756 tensors** in the AutoRound packed format
- **Triton fix**: downgraded `triton` from 3.3.0 to 3.2.0 to dodge `PY_SSIZE_T_CLEAN` kernel-build failures on Python 3.11

### Why split-and-repack?

AutoRound 0.13.0's MLLM quantization path requires the model class to be pre-registered with `AutoModelForCausalLM` and uses the standard `Processor.__call__` flow. `LocateAnythingForConditionalGeneration`'s custom processor wraps images in a way that breaks `make_list_of_images` (it receives `[[path]]` instead of `[path]`). Working around this in the calibration pipeline is fragile. The robust path is to extract the inner `Qwen2ForCausalLM` text decoder, quantize it as a plain LLM (which AutoRound handles cleanly), then merge the quantized Qwen2 weights back into a full LocateAnything checkpoint alongside the unmodified BF16 vision/projector.

## Evaluation

50-image synthetic benchmark (1024×1024, 25 colored solid geometric shapes + 25 black capital letters). Ground-truth bounding boxes are the model's prompt-aligned object bbox. **Same prompts sent to BF16 and INT4 in parallel across two RTX 3090s (2.76 s/iter avg, 138 s total wall)**.

Per-class breakdown:

| Class | n | BF16 mIoU | INT4 mIoU | BF16 IoU@0.5 | INT4 IoU@0.5 |
|---|---|---|---|---|---|
| Shapes | 25 | 0.955 | 0.955 | 100% | 100% |
| Letters | 25 | 0.550 | 0.554 | 84% | **92%** |

The hard cases are letters: the model often returns a box slightly larger than the tight text bbox because of MTP-speculation drift. INT4 quantization suppresses that drift and yields tighter boxes on the worst examples. See `benchmarks/results/viz/` for 12 side-by-side annotated comparisons (green = ground truth, red/blue = prediction).

## Known caveats

- **vLLM 0.19.1 is not supported.** `LocateAnythingForConditionalGeneration` is not in the vLLM architecture matrix and `--model-impl auto` does not pick it up. The custom mask/MagI generation path is not implemented in vLLM. Use the official Transformers worker (above) or write a vLLM plugin (out of scope here).
- **Magi attention is not available** on Ampere GPUs (RTX 3090, A100). Use `attn_implementation="sdpa"`. On Hopper/Blackwell you can keep the original magi path for max speed.
- The synthetic 50-image benchmark measures single-object box grounding. For multi-object detect, point queries, and hybrid generation, run your own evaluation on a real distribution (e.g. RefCOCO, D3, ReasonSeg, etc.). Smoke tests pass cleanly on all modes (`hybrid`, `slow`, `fast`, `detect`, `point`).

## Credits

- **Base model**: [nvidia/LocateAnything-3B](https://huggingface.co/nvidia/LocateAnything-3B) — Wang et al., *"Locate Anything, Serve Anything"* ([paper](https://research.nvidia.com/labs/dvl/projects/locate_anything/))
- **Quantization**: [Intel AutoRound 0.13.0](https://github.com/intel/auto-round)
- **Reference implementation**: [NVlabs/EAGLE](https://github.com/NVlabs/EAGLE) (the `Eagle/Embodied/locateanything_worker.py` is the canonical inference path)

## License

Inherits the [NVIDIA LocateAnything license](https://huggingface.co/nvidia/LocateAnything-3B). Read the upstream terms before commercial use.
