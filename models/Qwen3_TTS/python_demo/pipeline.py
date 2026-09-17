#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Qwen3-TTS-12Hz-0.6B-Base TPU pipeline.

End-to-end voice clone running entirely on the SOPHGO TPU via the bmodel produced
in step 2. Features:

  * Multi-language support (10 languages: en, zh, ja, ko, de, fr, ru, es, it, pt)
  * Think / no-think mode control
  * Streaming audio output (writes wav chunks incrementally during generation)
  * x-vector-only voice clone (reference wav -> speaker embedding)

The pipeline only depends on:
  * the bmodel (qwen3-tts.bmodel) + config/ dir (tokenizer + configs)
  * librosa / soundfile / torch (host-side mel STFT + audio I/O)

It does NOT load the HuggingFace Qwen3-TTS weights: text/codec embeddings and
the speaker encoder run as nets inside the bmodel; mel STFT is the only host
preprocessing (matching HF's mel_spectrogram exactly).

Flow (mirrors qwen_tts Qwen3TTSForConditionalGeneration.generate):
  1. mel = STFT(ref_wav@24kHz) -> [2048,128] (zero-padded); spk_embed = speaker_encoder(mel)
  2. build talker_input_embed (9-10 tokens) from text/codec embeds + spk_embed
     - depends on language (+ think/nothink) parameters
  3. build trailing_text_hidden (text body + tts_eos, streaming mode)
  4. forward_first -> code0; loop forward_next until codec_eos -> codes [T,16]
  5. streaming mimi_decoder: every 256 frames -> wav chunk -> write incrementally

Voice clone is x-vector-only: the speaker embedding (ECAPA-TDNN over the
reference mel) carries timbre. In-context-learning (ref_code) clone mode is
not supported -- the mimi_encoder ref_code precision could not be aligned on
this toolchain (see plan §12.15). Passing --ref_text raises.
"""

import argparse
import os
import sys
import time
import warnings

# librosa.filters.mel -> pooch -> paramiko emits a TripleDES deprecation warning
# at import time (paramiko still ships a legacy cipher). It is harmless (we never
# use paramiko; pooch is only pulled in for librosa's optional data fetching) and
# fires before any of our code runs, so silence it early.
warnings.filterwarnings("ignore", message="TripleDES has been moved")

import numpy as np
import torch

import chat

# ----------------------------------------------------------------------------
# Constants (from config.json talker_config; verified in step 2).
# ----------------------------------------------------------------------------
TTS_BOS = 151672
TTS_EOS = 151673
TTS_PAD = 151671
IM_START = 151644
IM_END = 151645
CODEC_PAD = 2148
CODEC_BOS = 2149
CODEC_EOS = 2150
CODEC_THINK = 2154
CODEC_NOTHINK = 2155
CODEC_THINK_BOS = 2156
CODEC_THINK_EOS = 2157

# Language ID mapping from config (talker_config.codec_language_id)
# 10 languages: chinese, english, german, italian, portuguese, spanish,
# japanese, korean, french, russian
LANGUAGE_TO_ID = {
    "chinese": 2055,
    "english": 2050,
    "german": 2053,
    "italian": 2070,
    "portuguese": 2071,
    "spanish": 2054,
    "japanese": 2058,
    "korean": 2064,
    "french": 2061,
    "russian": 2069,
    # aliases
    "zh": 2055,
    "en": 2050,
    "de": 2053,
    "it": 2070,
    "pt": 2071,
    "es": 2054,
    "ja": 2058,
    "ko": 2064,
    "fr": 2061,
    "ru": 2069,
}

SPEAKER_SR = 24000          # speaker encoder expects 24 kHz
MEL_HOP = 256
MEL_WIN = 1024
MEL_NFFT = 1024
MEL_NMELS = 128
MEL_FMIN = 0
MEL_FMAX = 12000
# speaker_encoder is STATIC [1, SPK_MEL_MAX, 128] (~21.8s). The pipeline
# zero-pads the full-length reference mel to SPK_MEL_MAX (no 300-frame
# truncation) so the embedding sees the whole reference audio.
SPK_MEL_MAX = 2048


def load_audio(path: str):
    """Load a wav as float32 mono numpy at its native sr."""
    import soundfile as sf
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=-1)
    return wav.astype(np.float32), int(sr)


def resample(wav: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return wav
    import librosa
    return librosa.resample(wav.astype(np.float32), orig_sr=orig_sr,
                            target_sr=target_sr)


# Mel filterbank (slaney norm), cached.
_MEL_BASIS = None


def _mel_basis():
    global _MEL_BASIS
    if _MEL_BASIS is None:
        import librosa
        mel = librosa.filters.mel(sr=SPEAKER_SR, n_fft=MEL_NFFT, n_mels=MEL_NMELS,
                                  fmin=MEL_FMIN, fmax=MEL_FMAX)
        _MEL_BASIS = torch.from_numpy(mel).float()
    return _MEL_BASIS


def mel_spectrogram(wav: np.ndarray) -> np.ndarray:
    """Replicate HF mel_spectrogram (center=False, reflect pad). Returns
    [num_mels, T_frames] float32."""
    y = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0)
    mel_basis = _mel_basis()
    hann_window = torch.hann_window(MEL_WIN)
    padding = (MEL_NFFT - MEL_HOP) // 2  # 384
    y = torch.nn.functional.pad(y.unsqueeze(1), (padding, padding),
                                mode="reflect").squeeze(1)
    spec = torch.stft(y, MEL_NFFT, hop_length=MEL_HOP, win_length=MEL_WIN,
                      window=hann_window, center=False, pad_mode="reflect",
                      normalized=False, onesided=True, return_complex=True)
    spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)
    mel_spec = mel_basis @ spec
    mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))  # dynamic range comp
    return mel_spec.squeeze(0).numpy()  # [num_mels, T_frames]


def compute_speaker_mel(ref_wav_path: str, target_len: int = None) -> np.ndarray:
    """Compute mel for the speaker encoder. If target_len is given (the bmodel's
    static SPK_MEL_MAX), fit the mel to exactly that length. When the bmodel is
    compiled at the ref's actual mel length (QWEN3_TTS_SPK_MEL_FRAMES), no
    zero-padding is needed and the embedding matches HF."""
    wav, sr = load_audio(ref_wav_path)
    wav = resample(wav, sr, SPEAKER_SR)
    mel = mel_spectrogram(wav)  # [128, T]
    t = mel.shape[1]
    tgt = target_len if target_len else SPK_MEL_MAX
    if t > tgt:
        print(f"[warn] ref mel {t} frames > target {tgt} "
              f"({t * MEL_HOP / SPEAKER_SR:.1f}s); truncating")
        mel = mel[:, :tgt]
    elif t < tgt:
        # Zero-pad ONLY when target > actual (static bmodel larger than the ref).
        # When the bmodel is compiled at the ref's actual length (no padding),
        # this branch is not taken and the embedding matches HF. The padded
        # silence frames dilute the attentive-stats-pooled embedding, hence warn.
        print(f"[warn] ref mel {t} frames < target {tgt} "
              f"({t * MEL_HOP / SPEAKER_SR:.1f}s); zero-padding to {tgt} "
              f"(embedding will be diluted; recompile with "
              f"QWEN3_TTS_SPK_MEL_FRAMES={t} for exact match)")
        pad = np.zeros((mel.shape[0], tgt - t), dtype=mel.dtype)
        mel = np.concatenate([mel, pad], axis=1)
    return mel.T.astype(np.float32)  # [tgt, 128]


class Qwen3TTS:
    def __init__(self, args):
        self.model = chat.Qwen3TTS()
        self.model.init(args.devid, args.model_path)
        self.SEQLEN = self.model.SEQLEN
        self.MAX_INPUT_LENGTH = self.model.MAX_INPUT_LENGTH
        self.HIDDEN = self.model.HIDDEN_SIZE
        self.MIMI_HOP = self.model.MIMI_HOP
        # tokenizer (text) from config dir. The tokenizer_config.json declares
        # tokenizer_class=Qwen2Tokenizer (a standard Qwen2 BPE tokenizer, no
        # auto_map). Load it directly instead of AutoTokenizer: AutoTokenizer
        # builds an AutoConfig under the hood, and since config.json's
        # model_type "qwen3_tts" is not registered in this transformers build it
        # prints a spurious "instantiate a model of type ''" advisory every run.
        from transformers import Qwen2Tokenizer
        self.tokenizer = Qwen2Tokenizer.from_pretrained(args.config_path)
        self.max_new_tokens = args.max_new_tokens
        # sampling config (HF generation_config defaults). code0 uses the TPU
        # sample_head net (RP+top_k/top_p/temp on device); CP code1..15 use
        # host-side top_k/top_p/temp sampling (no RP, matching HF).
        self.model.set_sampling(
            args.temperature, args.top_k, args.top_p, args.repetition_penalty,
            args.do_sample, args.sub_temperature, args.sub_top_k,
            args.sub_top_p, args.repetition_window, args.seed)

    # ---- embed helpers (call bmodel nets, return host float) ----
    def text_embed(self, ids):
        ids = np.asarray(ids, dtype=np.int32)
        return np.asarray(self.model.text_embedding(ids))

    def codec_embed(self, ids):
        ids = np.asarray(ids, dtype=np.int32)
        return np.asarray(self.model.codec_embedding(ids))

    # ---- prompt assembly (HF generate, x-vector-only, streaming trailing) ----
    def build_prompt(self, text, spk_embed, language="Auto", think=True):
        # ---- tokenize and embed text part first (language-independent) ----
        assistant_text = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        ids = self.tokenizer(assistant_text, return_tensors="pt")["input_ids"][0].tolist()
        # Structure: [im_start, assistant, \n, text..., im_end, \n, im_start, assistant, \n]
        assert ids[0] == IM_START and ids[1] != IM_END, "unexpected tokenization (role prefix)"
        # slices
        role_ids = ids[:3]                      # [im_start, assistant, \n]
        first_text_id = ids[3:4]                # [t0]
        text_body_ids = ids[4:-5]               # [t1 .. t_{m-1}]
        # embed text tokens via the embedding net (text_embedding + text_projection)
        role_e = self.text_embed(role_ids)              # [3, H]
        first_text_e = self.text_embed(first_text_id)   # [1, H]
        text_body_e = self.text_embed(text_body_ids)    # [m-1, H]
        tts_e = self.text_embed([TTS_BOS, TTS_EOS, TTS_PAD])  # [3, H]
        tts_bos_e, tts_eos_e, tts_pad_e = tts_e[0], tts_e[1], tts_e[2]

        # ---- resolve language_id ----
        # Resolve language_id: "Auto" or None → no language_id (nothink/think mode only)
        # Otherwise look up in LANGUAGE_TO_ID (case-insensitive)
        language_id = None
        if language and language.lower() != "auto":
            lang_key = language.lower()
            if lang_key in LANGUAGE_TO_ID:
                language_id = LANGUAGE_TO_ID[lang_key]
            else:
                print(f"[warn] unknown language '{language}', treating as 'Auto' (no language_id)")

        # ---- build codec_prefill based on language_id + think flag ----
        # - language_id is None: [nothink, think_bos, think_eos] OR [think, think_bos, think_eos]
        # - language_id is not None: [think, think_bos, language_id, think_eos]
        if language_id is None:
            if think:
                codec_prefill = [CODEC_THINK, CODEC_THINK_BOS, CODEC_THINK_EOS]  # [think, think_bos, think_eos]
            else:
                codec_prefill = [CODEC_NOTHINK, CODEC_THINK_BOS, CODEC_THINK_EOS]  # [nothink, think_bos, think_eos]
        else:
            # with language: [think, think_bos, language_id, think_eos]
            codec_prefill = [CODEC_THINK, CODEC_THINK_BOS, language_id, CODEC_THINK_EOS]

        codec_tail = [CODEC_PAD, CODEC_BOS]
        ce0 = self.codec_embed(codec_prefill)   # [3 or 4, H]
        ce1 = self.codec_embed(codec_tail)      # [2, H]
        spk_e = spk_embed.astype(np.float32)    # [H]
        codec_input = np.concatenate(
            [ce0, spk_e[None, :], ce1], axis=0)  # [6 or 7, H]

        # _talker_input_embed = cat([tts_pad * (len(codec_input)-2), tts_bos]) + codec_input[:-1]
        # len(codec_input) = 6 (no language) or 7 (with language)
        n_pad = len(codec_input) - 2
        pad_n = np.tile(tts_pad_e[None, :], (n_pad, 1))   # [4 or 5, H]
        pre = np.concatenate([pad_n, tts_bos_e[None, :]], axis=0)  # [5 or 6, H]
        pre = pre + codec_input[:-1]                       # [5 or 6, H]

        # last: first_text_e + codec_input[-1:] (codec_bos is always last)
        last = first_text_e + codec_input[-1:]             # [1, H]

        talker_embed = np.concatenate([role_e, pre, last], axis=0)  # [9 or 10, H]

        # trailing_text_hidden = cat([text_body_e, tts_eos]) padded with tts_pad
        # (streaming trailing: text_body + tts_eos, not non-streaming tts_pad only)
        trailing = np.concatenate([text_body_e, tts_eos_e[None, :]], axis=0)  # [m, H]
        trailing_full = np.tile(tts_pad_e[None, :], (self.SEQLEN, 1))
        n = min(len(trailing), self.SEQLEN)
        trailing_full[:n] = trailing[:n]
        return talker_embed.astype(np.float32), trailing_full.astype(np.float32)

    # ---- ICL prompt assembly (HF generate_icl_prompt, streaming mode) ----
    def build_icl_segment(self, text, ref_text, ref_code):
        """Build the in-context example segment appended after the x-vector
        prompt. Mirrors generate_icl_prompt (modeling:1968-2013, streaming):

          text_embed  = text_proj(text_emb(cat([ref_id, text_id]))) + tts_eos
          codec_embed = codec_bos + sum_i(embed_i(ref_code[:, i]))
          if text_lens > codec_lens: icl = text_embed[:codec_lens] + codec_embed
                                        trailing = text_embed[codec_lens:]
          else:                       icl = pad(text_embed) + codec_embed
                                        trailing = tts_pad

        Returns (icl_embed [L,H], trailing [SEQLEN,H]).
        """
        # --- ref_text + target text tokenization (same template as HF) ---
        ref_assistant = f"<|im_start|>assistant\n{ref_text}<|im_end|>\n"
        ref_ids = self.tokenizer(ref_assistant, return_tensors="pt")["input_ids"][0].tolist()
        assistant_text = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        ids = self.tokenizer(assistant_text, return_tensors="pt")["input_ids"][0].tolist()
        # ref_ids: [im_start, assistant, \n, r0 .. r_{m-1}, im_end, \n]; text part
        # is ids[3:-5] ([t0 .. t_{m-1}] excluding the second role prefix).
        ref_body_ids = ref_ids[3:-2]            # [r0 .. r_{m-1}] (drop role+im_end+\n)
        text_body_ids = ids[3:-5]               # [t0 .. t_{m-1}]
        text_embed = self.text_embed(ref_body_ids + text_body_ids)  # [r+t, H]
        tts_e = self.text_embed([TTS_EOS, TTS_PAD])
        tts_eos_e, tts_pad_e = tts_e[0], tts_e[1]
        text_embed = np.concatenate(
            [text_embed, tts_eos_e[None, :]], axis=0)  # [r+t+1, H]

        # --- codec embed: codec_bos + sum of 16 per-codebook embeds ---
        # ref_code: [16, T_ref] int -> per frame sum of embeds.
        # code group 0 uses the Talker codec_embedding (3072 vocab, same net as
        # codec_prefill); groups 1..15 use the CP embed nets (2048 vocab).
        ref_code = np.asarray(ref_code, dtype=np.int64)      # [16, T_ref]
        T_ref = ref_code.shape[1]
        e0 = self.codec_embed(ref_code[0].tolist())          # [T_ref, H]
        cp = [np.asarray(self.model.cp_embed_group(
                  ref_code[q].tolist(), q - 1)) for q in range(1, 16)]
        codec_body = e0
        for q in range(15):
            codec_body = codec_body + cp[q]                  # [T_ref, H]
        bos = self.codec_embed([CODEC_BOS])                   # [1, H]
        codec_embed = np.concatenate([bos, codec_body], axis=0)  # [1+T_ref, H]

        # --- streaming overlap (HF: text_lens vs codec_lens) ---
        text_lens = text_embed.shape[0]
        codec_lens = codec_embed.shape[0]
        if text_lens > codec_lens:
            icl = text_embed[:codec_lens] + codec_embed
            trail = text_embed[codec_lens:]
        else:
            pad = np.tile(tts_pad_e[None, :], (codec_lens - text_lens, 1))
            icl = np.concatenate([text_embed, pad], axis=0) + codec_embed
            trail = tts_pad_e[None, :]
        # HF generate adds trailing_text_hidden[step] ONLY for step < trailing_len;
        # beyond that it adds NOTHING. chat.cpp adds trailing[step] every step, so
        # pad the remainder with ZEROS (not tts_pad) to match "add nothing".
        trailing_full = np.zeros((self.SEQLEN, trail.shape[1]), dtype=np.float32)
        n = min(len(trail), self.SEQLEN)
        trailing_full[:n] = trail[:n]
        return icl.astype(np.float32), trailing_full.astype(np.float32)

    def synthesize(self, text, ref_wav_path, out_wav_path, language="Auto",
                   think=True, ref_text=None, ref_code_override=None):
        # In-context-learning (ref_code) voice clone is not supported: the
        # mimi_encoder ref_code precision could not be aligned on this toolchain
        # (plan §12.15). Only x-vector-only clone is delivered. The ICL prompt
        # assembly below is preserved for reference but is unreachable.
        if ref_text or ref_code_override is not None:
            raise NotImplementedError(
                "ICL (ref_text / ref_code) voice clone is not supported. "
                "Use x-vector-only clone (--ref_wav without --ref_text).")
        ref_code = None  # ICL disabled; kept for the unreachable prepend logic below
        # ---- perf timing: t_start covers speaker_encoder + prompt assembly
        # + prefill + decode loop + mimi_decoder. RTF = (t_end - t_start) /
        # audio_duration. FTL = time to first audio chunk. ----
        t_start = time.perf_counter()
        # 1. speaker embedding (fit mel to the bmodel's static SPK_MEL_MAX;
        # if compiled at the ref's actual length, no zero-padding happens)
        mel = compute_speaker_mel(ref_wav_path, int(self.model.SPK_MEL_MAX))
        spk_embed = np.asarray(self.model.forward_speaker_encoder(mel))  # [1024]

        # 2. prompt assembly (with language + think mode)
        talker_embed, trailing = self.build_prompt(text, spk_embed, language, think)
        P = talker_embed.shape[0]
        assert P <= self.MAX_INPUT_LENGTH, f"prefill {P} > MAX_INPUT_LENGTH"
        self.model.set_talker_prefill(talker_embed)
        self.model.set_trailing(trailing)

        # 3. prefill -> first code0 (frame 0's code0; code1..15 computed in the
        #    first forward_next)
        self.model.forward_first()
        t_prefill = time.perf_counter()
        codes = []

        # 4. decode loop with streaming: each forward_next completes one frame,
        #    collect chunks, write wav at the end (streaming write has issues with soundfile)
        import soundfile as sf
        MIMI_FRAME = self.model.MIMI_FRAME
        MIMI_HOP = self.model.MIMI_HOP
        wav_chunks = []  # collect wav chunks
        code0 = None
        t_first_chunk = None  # set when first audio chunk is decoded (FTL)
        for step in range(self.max_new_tokens):
            code0 = self.model.forward_next()
            codes.append(list(self.model.get_frame_codes()))  # frame just completed

            # Every MIMI_FRAME frames, decode to wav chunk. In ICL mode EACH chunk
            # is decoded with ref_code prepended (causal conv warm-up on the
            # reference; HF: codes_for_decode = cat([ref_code, generated])).
            # forward_mimi_decoder zero-pads to MIMI_FRAME internally and chunks
            # independently; the ref portion is dropped at the end.
            if len(codes) % MIMI_FRAME == 0:
                chunk_codes = codes[-MIMI_FRAME:]
                chunk_arr = np.array(chunk_codes, dtype=np.int32).T  # [16, MIMI_FRAME]
                if ref_code is not None:
                    chunk_arr = np.concatenate(
                        [ref_code.astype(np.int32), chunk_arr], axis=1)
                chunk_wav = np.asarray(self.model.forward_mimi_decoder(chunk_arr))
                chunk_wav = chunk_wav.astype(np.float32)
                wav_chunks.append(chunk_wav)
                if t_first_chunk is None:
                    t_first_chunk = time.perf_counter()
                print(f"  [streaming] decoded chunk at frame {len(codes)}")

            if code0 == CODEC_EOS:
                print(f"  [streaming] EOS at frame {len(codes)}, finishing...")
                break
            if self.model.token_length >= self.SEQLEN - 2:
                print(f"[warn] SEQLEN limit reached at step {step}")
                break

        # Flush remaining frames (may be < MIMI_FRAME)
        remaining = len(codes) % MIMI_FRAME
        if remaining > 0:
            chunk_codes = codes[-remaining:]
            chunk_arr = np.array(chunk_codes, dtype=np.int32).T  # [16, remaining]
            if ref_code is not None:
                chunk_arr = np.concatenate(
                    [ref_code.astype(np.int32), chunk_arr], axis=1)
            chunk_wav = np.asarray(self.model.forward_mimi_decoder(chunk_arr))
            chunk_wav = chunk_wav.astype(np.float32)
            # only take the valid portion (remaining * MIMI_HOP samples) AFTER the ref
            if ref_code is not None:
                ref_samples = ref_code.shape[1] * MIMI_HOP
                chunk_wav = chunk_wav[ref_samples:ref_samples + remaining * MIMI_HOP]
            else:
                chunk_wav = chunk_wav[:remaining * MIMI_HOP]
            wav_chunks.append(chunk_wav)
            print(f"  [streaming] decoded final {remaining} frames")
        t_end = time.perf_counter()

        # Write all chunks to wav file.
        wav = np.concatenate(wav_chunks)
        sf.write(out_wav_path, wav, SPEAKER_SR)
        audio_dur = len(wav) / SPEAKER_SR
        print(f"generated {len(codes)} frames (last code0={code0}), wrote {out_wav_path} ({audio_dur:.2f}s)")
        # ---- perf report (mirrors Qwen3_5 FTL/TPS; TTS adds RTF) ----
        gen_total = t_end - t_start
        prefill_dur = t_prefill - t_start
        decode_dur = t_end - t_prefill
        rtf = gen_total / audio_dur if audio_dur > 0 else 0.0
        fps = len(codes) / decode_dur if decode_dur > 0 else 0.0
        # FTL = time to first code0 (prefill), analogous to LLM first-token
        # latency. First-audio-chunk latency is only meaningful when streaming
        # kicks in (audio > MIMI_FRAME frames); reported separately if so.
        print(f"\nFTL: {prefill_dur:.3f} s  (time to first code0)")
        if t_first_chunk is not None and t_first_chunk < t_end - 1e-6:
            print(f"first-audio-chunk: {t_first_chunk - t_start:.3f} s "
                  f"(streaming, MIMI_FRAME={MIMI_FRAME})")
        print(f"RTF: {rtf:.3f}  (gen {gen_total:.3f}s / audio {audio_dur:.3f}s; "
              f"{'<1 REAL-TIME' if rtf < 1 else '>=1 slower than real-time'})")
        print(f"FPS: {fps:.3f} frames/s  (decode {decode_dur:.3f}s for {len(codes)} frames)")

    def chat(self):
        print("Qwen3-TTS interactive. Ctrl-D to exit.")
        while True:
            try:
                text = input("text > ").strip()
                if not text:
                    continue
                ref = input("ref_wav > ").strip()
                out = input("out_wav > ").strip()
                rt = input("ref_text (empty = x-vector only) > ").strip()
                self.synthesize(text, ref, out, ref_text=rt or None)
            except EOFError:
                break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", type=str, required=True)
    parser.add_argument("-c", "--config_path", type=str, required=True)
    parser.add_argument("-d", "--devid", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--text", type=str, default="")
    parser.add_argument("--ref_wav", type=str, default="")
    parser.add_argument("--ref_text", type=str, default="",
                        help="NOT SUPPORTED: ICL (ref_code) voice clone could not "
                             "be aligned on this toolchain. Use x-vector-only clone "
                             "(--ref_wav without --ref_text).")
    parser.add_argument("--out_wav", type=str, default="out.wav")
    parser.add_argument("--language", type=str, default="Auto",
                        help="language for TTS (Auto, English, Chinese, Japanese, etc.)")
    parser.add_argument("--think", action="store_true", default=True,
                        help="use think mode (default on). Use --no-think for nothink mode.")
    parser.add_argument("--no-think", dest="think", action="store_false",
                        help="use nothink mode (no thinking, direct TTS)")
    # sampling (HF generation_config defaults)
    parser.add_argument("--do_sample", action="store_true", default=True,
                        help="sample code0/code1..15 (default on)")
    parser.add_argument("--greedy", dest="do_sample", action="store_false",
                        help="greedy argmax (disables sampling)")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--sub_temperature", type=float, default=0.9)
    parser.add_argument("--sub_top_k", type=int, default=50)
    parser.add_argument("--sub_top_p", type=float, default=1.0)
    parser.add_argument("--repetition_window", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    tts = Qwen3TTS(args)
    if args.text and args.ref_wav:
        tts.synthesize(args.text, args.ref_wav, args.out_wav, args.language,
                       args.think, ref_text=args.ref_text or None)
    else:
        tts.chat()
    tts.model.deinit()


if __name__ == "__main__":
    main()
