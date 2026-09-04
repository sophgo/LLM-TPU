"""Real-time streaming demo: RTSP → Gate decision → LLM generation.

Reads frames live from an RTSP stream, buffers segments of GATE_FRAMES,
runs ViT + Gate + ClsNet per segment, and triggers LLM generation on
"speak" decisions.
"""
import argparse
import sys
import time
import numpy as np

sys.path.insert(0, ".")
from pipeline import Mage_VL, torch_where, extract_media


def run_rtsp_stream(model: Mage_VL,
                    rtsp_url: str,
                    prompt: str,
                    threshold: float,
                    max_segments: int,
                    fps=None):
    import av
    from PIL import Image

    T = model.model.GATE_FRAMES
    frame_interval = 1.0 / fps if fps and fps > 0 else 0.0

    print(f"Opening RTSP: {rtsp_url}")
    if frame_interval:
        print(f"Sampling at {fps} fps (1 frame every {frame_interval:.2f}s)")
    container = av.open(rtsp_url, "r", format="rtsp")

    speak_count = 0
    seg_idx = 0
    frame_buf = []
    frame_count = 0
    skipped = 0
    last_sampled_time = -1e9

    try:
        for frame in container.decode(video=0):
            if frame_interval:
                ft = frame.time
                if ft is not None and ft - last_sampled_time < frame_interval:
                    skipped += 1
                    continue
                last_sampled_time = ft if ft is not None else 0

            img = frame.to_image().convert("RGB")
            img = model.resize_to_fixed_pixels(img)
            frame_buf.append(img)
            frame_count += 1

            if len(frame_buf) < T:
                continue

            # --- We have T frames, process one segment ---
            seg_idx += 1
            if max_segments and seg_idx > max_segments:
                break

            frames = frame_buf
            frame_buf = []

            # Build processor inputs for T frames
            model.input_str = prompt
            model.media_path = ""  # not used by processor for video
            messages = model.video_message("")
            text = model.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            inputs = model.processor(text=[text],
                                     videos=[frames],
                                     video_backend="frames",
                                     return_tensors="pt")

            # ViT + read embeddings
            vit_t0 = time.time()
            model.model.forward_embed(inputs.input_ids.numpy().reshape(-1))
            model.vit_process_image(inputs)
            vit_t1 = time.time()

            vit_token_list = torch_where(
                inputs.input_ids == model.ID_VISION_START)
            frame_embs = []
            for idx, vit_offset in enumerate(vit_token_list):
                grid_thw = inputs.image_grid_thw[idx]
                num_patches = (int(grid_thw[0]) * int(grid_thw[1]) *
                               int(grid_thw[2]))
                num_merged = num_patches // (model.merge_size**2)
                embs = model.model.read_vit_embeddings(vit_offset + 1,
                                                       num_merged)
                frame_embs.append(np.array(embs).mean(axis=0))

            averaged = np.array(frame_embs, dtype=np.float32)

            # Gate decision
            gate_t0 = time.time()
            logits = model.model.forward_gate(averaged)
            speak, margin = model.should_speak(logits, threshold)
            gate_t1 = time.time()

            status = "SPEAK" if speak else "SILENT"
            print(f"[Seg {seg_idx:>3}] frame #{frame_count}  "
                  f"ViT {vit_t1 - vit_t0:.3f}s  "
                  f"Gate {gate_t1 - gate_t0:.3f}s  "
                  f"→ {status} (margin {margin:+.4f})")

            if not speak:
                continue

            # LLM generation — reuse gate path embeddings.
            speak_count += 1
            model.model.clear_history()

            token_len = inputs.input_ids.numel()
            position_ids = np.arange(token_len, dtype=np.int32)
            token = model.model.forward_first(position_ids)

            tok_num = 0
            full_word_tokens = []
            text_out = ""
            gen_t0 = time.time()
            while (token not in [model.ID_IM_END, model.ID_END]
                   and model.model.history_length < model.model.SEQLEN):
                full_word_tokens.append(token)
                word = model.tokenizer.decode(full_word_tokens,
                                              skip_special_tokens=True)
                if "�" not in word:
                    if len(full_word_tokens) == 1:
                        pre_word = word
                        word = model.tokenizer.decode(
                            [token, token],
                            skip_special_tokens=True)[len(pre_word):]
                    text_out += word
                    print(word, flush=True, end="")
                    full_word_tokens = []
                token = model.model.forward_next()
                tok_num += 1
                if tok_num > 200:  # cap for live demo
                    break
            gen_t1 = time.time()

            tps = tok_num / (gen_t1 - gen_t0) if (gen_t1 - gen_t0) > 0 else 0
            print(f"\n  [{speak_count}] {tok_num} tok, {tps:.1f} tok/s\n")

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        container.close()

    print(f"\nDone. {frame_count} frames sampled, {seg_idx} segments, "
          f"{speak_count} spoke" + (f", {skipped} skipped (target {fps} fps)."
                                    if frame_interval else "."))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rtsp",
                        type=str,
                        required=True,
                        help="RTSP stream URL")
    parser.add_argument("-m", "--model_path", type=str, required=True)
    parser.add_argument("-c", "--config_path", type=str, default="../config")
    parser.add_argument("-d", "--devid", type=int, default=0)
    parser.add_argument("-p",
                        "--prompt",
                        type=str,
                        default="Describe what you see in the video.")
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Target frame sampling rate (e.g. 1.0 for 1 fps). "
        "Default: keep all frames.")
    parser.add_argument("--max_segments",
                        type=int,
                        default=0,
                        help="0 = run until Ctrl-C")
    args = parser.parse_args()

    model_args = argparse.Namespace(
        devid=args.devid,
        model_path=args.model_path,
        config_path=args.config_path,
        num_video_frames=4,
        stream=False,
        gate_threshold=0.0,
        prompt=None,
    )
    model = Mage_VL(model_args)
    run_rtsp_stream(model, args.rtsp, args.prompt, args.threshold,
                    args.max_segments, args.fps)


if __name__ == "__main__":
    main()
