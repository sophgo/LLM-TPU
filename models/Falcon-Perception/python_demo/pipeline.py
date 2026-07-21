# ==============================================================================
# Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
#
# Falcon-Perception (early-fusion segmentation VLM) inference pipeline.
#
# Implements the backbone prefill + decode loop with the mixed attention mask,
# 3D golden RoPE cos/sin precompute, host-side image-patch scatter,
# coord/size/seg/mask head dispatch, Fourier 回灌 (coord_encoder/size_encoder),
# AnyUp (once after prefill) + mask_head per seg token, coord dedup, and full
# mask postprocess (crop -> resize -> sigmoid -> NMS) aligned with HF.

import time
import argparse
import os
import sys
import numpy as np
import torch
import einops as E
from PIL import Image
import chat

MASK_VAL = -1e9  # additive attention mask for masked positions (f32)


class FalconPerception():

    def __init__(self, args):
        self.device = args.devid
        self.model = chat.FalconPerception()
        self.model.init(self.device, args.model_path)

        from transformers import AutoTokenizer, AutoConfig
        self.config = AutoConfig.from_pretrained(
            args.config_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.config_path, trust_remote_code=True)

        # process_batch is a standalone function in the custom processor module
        # shipped alongside config.json in config_path (no AutoProcessor registered).
        if args.config_path not in sys.path:
            sys.path.insert(0, args.config_path)
        from processing_falcon_perception import process_batch
        self.process_batch = process_batch

        # special token ids (from config)
        c = self.config
        self.ID_IMG = c.img_id                    # 227
        self.ID_EOS = c.eos_id                    # 11
        self.ID_SOI = c.image_cls_token_id        # 244
        self.ID_EOI = c.img_end_id                # 230
        self.ID_COORD = c.coord_token_id          # 240
        self.ID_SIZE = c.size_token_id            # 241
        self.ID_SEG = c.seg_token_id              # 262
        self.ID_PAD = self.tokenizer.convert_tokens_to_ids("<|pad|>")  # 0
        self.ID_END_OF_QUERY = self.tokenizer.convert_tokens_to_ids(
            "<|end_of_query|>")                   # 263
        self.stop_ids = [self.ID_EOS, self.ID_END_OF_QUERY]

        self.patch_size = c.spatial_patch_size    # 16
        self.merge_size = 1
        self.hidden = c.dim                       # 1024
        self.n_heads = c.n_heads                  # 16
        self.head_dim = c.head_dim                # 128
        self.rope_quart = (self.head_dim // 2) // 2   # 32

        # head / segmentation dims
        self.num_bins = c.coord_out_dim // 2      # 1024 bins per coord/size dim
        self.segm_out_dim = getattr(c, "segm_out_dim", 256)
        self.max_segm_tokens = 16                 # mask_head batch (K) dim
        self.size_min = float(np.log2(1.0 / self.num_bins))   # log2(1/1024) = -10
        # coord dedup
        self.coord_repeat_threshold = 0.01
        self.max_coord_attempts = 100

        self.MAX_INPUT_LENGTH = self.model.MAX_INPUT_LENGTH  # 512
        self.SEQLEN = self.model.SEQLEN                       # 4096

        # img_projector + golden rope freqs are exported into config_path as a
        # small npz (falcon_extra_weights.npz) so the pipeline need not touch the
        # original multi-GB safetensors at runtime.
        self._load_weights(args.config_path)

        self.max_posid = 0

    def _load_weights(self, config_path):
        data = np.load(os.path.join(config_path, "falcon_extra_weights.npz"))
        self.img_projector = data["img_projector_weight"].astype(np.float32)  # [1024,768] numpy
        self.freqs_cis_golden = torch.from_numpy(
            data["freqs_cis_golden"].astype(np.float32))         # [16,32,2]
        # AnyUp window attention mask is geometry-only (static 256x256), so it
        # is baked into the anyup bmodel as a const weight — not needed here.

    # ------------------------------------------------------------------ process
    def build_prompt(self, query):
        # No newline after the image-token placeholder: it tokenizes to an extra
        # token between the image block and "Segment", making prefill 1 token
        # longer than HF and shifting every position / decode pos_id by 1.
        image_token = self.tokenizer.convert_ids_to_tokens(self.ID_IMG)
        return f"{image_token}Segment these expressions in the image:<|start_of_query|>{query}<|REF_SEG|>"

    def process(self, image_path, query):
        prompt = self.build_prompt(query)
        batch = self.process_batch(
            self.tokenizer, self.config, [(image_path, prompt)],
            max_length=self.MAX_INPUT_LENGTH,
            min_dimension=256, max_dimension=256,
            patch_size=self.patch_size, merge_size=self.merge_size,
        )
        return batch

    # ---------------------------------------------------- host-side input states
    def build_img_injection(self, batch):
        """Compute projected image-patch features [M,1024] and their token row
        positions [M]. chat.cpp scatters these directly into block_0's input
        mem at the img-token rows (partial s2d by offset), so the full
        [512,1024] embedding never round-trips through the host."""
        tokens = batch["tokens"][0]                      # [L]
        L = tokens.shape[0]
        pixel_values = batch["pixel_values"]             # [N,1,H,W,3]
        pixel_mask = batch["pixel_mask"]                 # [N,H,W]
        if pixel_values is None:
            return (np.zeros((0, self.hidden), np.float32),
                    np.zeros((0,), np.int32), L)
        c = self.config
        pixel_patches = E.rearrange(
            pixel_values,
            "n (t pt) (h ph) (w pw) ch -> n (t h w) (pt ph pw ch)",
            pt=c.temporal_patch_size, ph=c.spatial_patch_size,
            pw=c.spatial_patch_size,
        )                                                # [N, n_patches, 768]
        pixel_patch_mask = E.reduce(
            pixel_mask, "n (t pt) (h ph) (w pw) -> (n t h w)", reduction="any",
            pt=c.temporal_patch_size, ph=c.spatial_patch_size,
            pw=c.spatial_patch_size,
        )                                                # [n_patches]
        flat = pixel_patches.reshape(-1, pixel_patches.shape[-1])
        # numpy matmul (not torch): torch's CPU matmul gives different reduction
        # order across builds (cu124 vs cpu), a ~6e-6 drift that compounds over
        # 28 prefill layers + squared-ReLU into a wrong stop/continue decision.
        pm = pixel_patch_mask.numpy() if hasattr(pixel_patch_mask, "numpy") \
            else np.asarray(pixel_patch_mask)
        valid_patches = flat.numpy().astype(np.float32)[pm]   # [M,768]
        # float64 matmul then cast to f32: cross-arch BLAS (x86 vs aarch64)
        # reduces in different order, a ~5e-6 f32 drift that compounds over
        # 28 prefill layers into a wrong stop/continue. f64 reduction differs
        # by ~1e-15 (<< f32 ULP ~5e-7), so the f32 cast absorbs it -> bit-identical.
        valid_feats = (valid_patches.astype(np.float64)
                        @ self.img_projector.T.astype(np.float64)).astype(np.float32)

        # padded tokens (right-padded with pad_id=0 to MAX_INPUT_LENGTH)
        pad_len = self.MAX_INPUT_LENGTH - L
        padded_tokens = np.concatenate(
            [tokens.numpy().astype(np.int64),
             np.full(pad_len, self.ID_PAD, dtype=np.int64)])
        img_pos = np.where(padded_tokens == self.ID_IMG)[0].astype(np.int32)
        assert img_pos.shape[0] == valid_feats.shape[0], (
            f"img tokens {img_pos.shape[0]} != patches {valid_feats.shape[0]}")
        return valid_feats.astype(np.float32), img_pos, L

    # ---------------------------------------------------------- golden 2D RoPE
    def build_golden_cos_sin(self, batch, L):
        """golden_cos/sin [512,16,32]. img tokens: cos/sin(pos_hw @ freqs).
        text/pad tokens: cos=1, sin=0."""
        pos_hw = batch["pos_hw"][0].float()             # [L,2], NaN for non-img
        gcos = np.ones((self.MAX_INPUT_LENGTH, self.n_heads, self.rope_quart),
                       dtype=np.float32)
        gsin = np.zeros((self.MAX_INPUT_LENGTH, self.n_heads, self.rope_quart),
                        dtype=np.float32)
        valid = ~torch.isnan(pos_hw).any(dim=-1)         # [L]
        if valid.any():
            idx = torch.where(valid)[0]
            pos = pos_hw[idx]                            # [M,2]
            # theta[m,h,f] = sum_p pos[m,p] * freqs[h,f,p]
            theta = torch.einsum(
                "mp,hfp->mhf", pos, self.freqs_cis_golden)   # [M,16,32]
            gcos[idx.numpy()] = np.cos(theta.numpy()).astype(np.float32)
            gsin[idx.numpy()] = np.sin(theta.numpy()).astype(np.float32)
        return gcos, gsin

    # --------------------------------------------------------- attention mask
    def build_prefill_mask(self, batch, L):
        """Additive mask [1,1,512,512]. Composes (per HF attention.py):
        image_prefix OR (causal AND document AND non_left_pad)."""
        tokens = batch["tokens"][0].numpy().astype(np.int64)
        pad_len = self.MAX_INPUT_LENGTH - L
        toks = np.concatenate(
            [tokens, np.full(pad_len, self.ID_PAD, dtype=np.int64)])
        S = self.MAX_INPUT_LENGTH
        allow = np.zeros((S, S), dtype=bool)

        # causal
        causal = np.tril(np.ones((S, S), dtype=bool))

        # document (eos boundaries); last position treated as a boundary
        eos = (toks == self.ID_EOS)
        eos[-1] = True
        seq_idx = np.cumsum(eos.astype(np.int32))
        seq_idx = np.concatenate([[0], seq_idx[:-1]])   # shift like HF
        document = seq_idx[:, None] == seq_idx[None, :]

        # non_left_pad: kv must have cumsum(!=pad) > 0. Right-pad => all >0.
        nonpad = np.cumsum(toks != self.ID_PAD) > 0      # [S]
        non_left_pad = nonpad[None, :]                   # [1,S] broadcast

        block = causal & document & non_left_pad

        # image_prefix: tokens between soi..eoi (exclusive of eoi) of same image
        soi = (toks == self.ID_SOI)
        eoi = (toks == self.ID_EOI)
        acc_soi = np.cumsum(soi)
        acc_eoi = np.cumsum(eoi)
        img_mask = (acc_soi - acc_eoi) > 0
        img_idx = acc_soi * img_mask
        img_prefix = (img_mask[:, None] & img_mask[None, :] &
                      (img_idx[:, None] == img_idx[None, :]))

        allow = img_prefix | block
        mask = np.where(allow, 0.0, MASK_VAL).astype(np.float32)
        return mask[None, None, :, :]                   # [1,1,S,S]

    def build_decode_mask(self, history_length):
        """Additive decode mask [1,1,1,4097].

        kall = concat(history_k[seq_len], new_k[1]) — the just-generated k sits
        at the LAST position (mk-1). The query (current token) must attend to it
        (causal diagonal, like HF's q_idx>=kv_idx and the prefill np.tril). So
        keep positions [0, history_length-2] (prefill kv) and [mk-1] (new k);
        mask only the empty slots [history_length-1, mk-2]."""
        mk = self.SEQLEN + 1
        mask = np.zeros(mk, dtype=np.float32)
        for j in range(history_length - 1, mk - 1):
            mask[j] = MASK_VAL
        return mask[None, None, None, :]                # [1,1,1,4097]

    # -------------------------------------------------------- anyup + heads
    def build_anyup_inputs(self, batch, h_BSD):
        """images [1,3,256,256], lr_tokens [1,1024,16,16] (gather img-token
        hidden PRE conv_segm into the full 16x16 grid, invalid patches = 0).
        Mirrors HF gather_img_tokens. (window_mask is baked into the bmodel.)"""
        c = self.config
        ph = pw = c.spatial_patch_size                         # 16
        tokens = batch["tokens"][0].numpy().astype(np.int64)
        img_pos = np.where(tokens == self.ID_IMG)[0]           # [M] valid img tokens
        valid = h_BSD[img_pos].astype(np.float32)              # [M,1024] in (h,w) order
        # itok grid mask [16,16] from pixel_mask [N,1,H,W] (or [N,H,W])
        pm = batch["pixel_mask"]
        if pm.ndim == 4:
            pm = pm[:, 0]                                      # [N,H,W]
        itok = E.reduce(pm.numpy().astype(bool),
                        "n (h ph) (w pw) -> h w", reduction="any",
                        ph=ph, pw=pw)                          # [16,16]
        grid = np.zeros((itok.shape[0], itok.shape[1], self.hidden),
                        dtype=np.float32)                      # [16,16,1024]
        grid[itok] = valid                                     # scatter valid, rest 0
        lr_tokens = grid.transpose(2, 0, 1)[None].astype(np.float32)  # [1,1024,16,16]
        pv = batch["pixel_values"]                             # [N,1,H,W,3]
        images = pv[0, 0].numpy().transpose(2, 0, 1)[None].astype(np.float32)  # [1,3,256,256]
        return images, lr_tokens

    def run_anyup(self, batch, h_BSD):
        images, lr_tokens = self.build_anyup_inputs(batch, h_BSD)
        hr = np.array(self.model.forward_anyup(images, lr_tokens),
                      dtype=np.float32)                       # [1,256,256,256]
        return hr

    def decode_coord(self, h_last, existing_coords):
        """h_last [1024] -> (x, y) in [0,1] with dedup, returns (xy, fourier_emb)."""
        logits = np.array(self.model.forward_coord(
            h_last.astype(np.float32)), dtype=np.float32).reshape(2, self.num_bins)
        xy = np.zeros(2, dtype=np.float32)
        for d in range(2):
            lg = logits[d].copy()
            for _ in range(self.max_coord_attempts):
                b = int(np.argmax(lg))
                v = b / (self.num_bins - 1)
                if not any(abs(ec[d] - v) < self.coord_repeat_threshold
                           for ec in existing_coords):
                    xy[d] = v
                    break
                lg[b] = -1e9
            else:
                xy[d] = int(np.argmax(logits[d])) / (self.num_bins - 1)
        fourier = np.array(self.model.forward_fourier_coord(
            xy.reshape(1, 2).astype(np.float32)), dtype=np.float32).reshape(1, self.hidden)
        return xy, fourier

    def decode_size(self, h_last):
        """h_last [1024] -> (h, w) in log2 scale, returns (hw, fourier_emb)."""
        logits = np.array(self.model.forward_size(
            h_last.astype(np.float32)), dtype=np.float32).reshape(2, self.num_bins)
        pred = np.argmax(logits, axis=-1).astype(np.float32) / (self.num_bins - 1)
        pred = pred * (0.0 - self.size_min) + self.size_min    # [-10, 0]
        hw = np.power(2.0, pred).astype(np.float32)
        fourier = np.array(self.model.forward_fourier_size(
            hw.reshape(1, 2).astype(np.float32)), dtype=np.float32).reshape(1, self.hidden)
        return hw, fourier

    def decode_seg(self, h_last, hr_features):
        """h_last [1024], hr_features [1,256,256,256] -> mask logits [256,256]."""
        seg_vec = np.array(self.model.forward_seg(
            h_last.astype(np.float32)), dtype=np.float32).reshape(1, self.segm_out_dim)
        k = self.max_segm_tokens
        seg_pad = np.zeros((k, self.segm_out_dim), dtype=np.float32)
        seg_pad[0] = seg_vec[0]
        hr = hr_features.astype(np.float32)                    # [1,256,256,256]
        mask = np.array(self.model.forward_mask(hr, seg_pad),
                        dtype=np.float32).reshape(k, 256, 256)
        return mask[0]                                         # [256,256]

    def _mask_to_binary(self, mask_logits, pixel_mask_hw, orig_hw, threshold=0.5):
        """[256,256] logits (processed space) -> binary [orig_h, orig_w] mask.
        Mirrors HF _postprocess_aux: crop to pixel_mask active region, bilinear
        resize to original size, sigmoid > threshold."""
        import torch.nn.functional as F
        nz = np.argwhere(np.asarray(pixel_mask_hw) > 0)
        if len(nz):
            min_h, min_w = nz.min(0); max_h, max_w = nz.max(0)
            mask_logits = mask_logits[min_h:max_h + 1, min_w:max_w + 1]
        orig_h, orig_w = orig_hw
        m = torch.from_numpy(np.ascontiguousarray(mask_logits.astype(np.float32)))[None, None]
        m = F.interpolate(m, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        return (torch.sigmoid(m[0, 0]) > threshold).numpy().astype(bool)

    def _mask_nms(self, masks, iou_thr=0.6, nms_max_side=256):
        """Vectorised mask NMS (HF _mask_nms): IoU on bilinear-downscaled masks."""
        import torch.nn.functional as F
        N = len(masks)
        if N <= 1:
            return list(range(N))
        h, w = masks[0].shape
        scale = min(1.0, nms_max_side / max(h, w))
        th, tw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
        flat = []
        for m in masks:
            mt = torch.from_numpy(m.astype(np.float32))[None, None]
            if mt.shape[-2:] != (th, tw):
                mt = F.interpolate(mt, size=(th, tw), mode="bilinear", align_corners=False)
            flat.append(mt.flatten())
        bin_ = torch.stack(flat)
        areas = bin_.sum(1)
        inter = bin_ @ bin_.T
        union = areas[:, None] + areas[None, :] - inter
        iou = inter / union.clamp(min=1)
        order = areas.argsort(descending=True).tolist()
        sup = [False] * N
        keep = []
        for idx in order:
            if sup[idx]:
                continue
            keep.append(idx)
            row = (iou[idx] > iou_thr).tolist()
            for j in range(N):
                if row[j]:
                    sup[j] = True
        return keep

    def postprocess(self, aux, pixel_mask_hw, orig_hw, threshold=0.5):
        """Group aux into (xy, hw, mask) triplets, build full-res binary masks,
        mask-NMS, return structured detections (mirrors HF _postprocess_aux)."""
        n_coord = sum(1 for a in aux if isinstance(a, dict) and "x" in a)
        n_size = sum(1 for a in aux if isinstance(a, dict) and "h" in a)
        n_seg = sum(1 for a in aux if isinstance(a, np.ndarray))
        cands = []
        for i in range(0, len(aux) - 2, 3):
            xy, hw, mask = aux[i], aux[i + 1], aux[i + 2]
            if not (isinstance(mask, np.ndarray) and isinstance(xy, dict)
                    and "x" in xy and isinstance(hw, dict) and "h" in hw):
                continue
            bmask = self._mask_to_binary(mask, pixel_mask_hw, orig_hw, threshold)
            cands.append({"x": float(xy["x"]), "y": float(xy["y"]),
                          "h": float(hw["h"]), "w": float(hw["w"]), "mask": bmask})
        keep = self._mask_nms([c["mask"] for c in cands])
        dets = [cands[i] for i in keep]
        return dets, (n_coord, n_size, n_seg, len(cands), len(dets))

    def visualize(self, dets, image_path, out_path):
        """Overlay detection masks + bboxes on the original image, save JPG."""
        from PIL import Image, ImageDraw
        im = Image.open(image_path).convert("RGB")
        W, H = im.size                       # W=orig_w, H=orig_h
        colors = [(255, 0, 0), (0, 200, 0), (0, 0, 255),
                  (255, 255, 0), (255, 0, 255), (0, 200, 200)]
        arr = np.array(im)                   # [H, W, 3]
        for i, d in enumerate(dets):
            col = np.array(colors[i % len(colors)], dtype=np.int32)
            a = d["mask"]                    # [H, W] bool
            arr[a] = np.clip(arr[a].astype(np.int32) * 2 // 5 + col * 3 // 5,
                             0, 255).astype(np.uint8)
        ov = Image.fromarray(arr)
        draw = ImageDraw.Draw(ov)
        for i, d in enumerate(dets):
            col = colors[i % len(colors)]
            cx, cy = d["x"] * W, d["y"] * H
            hw_w, hw_h = d["w"] * W / 2, d["h"] * H / 2
            box = [cx - hw_w, cy - hw_h, cx + hw_w, cy + hw_h]
            draw.rectangle(box, outline=col, width=3)
            draw.text((box[0], max(0, box[1] - 16)), f"#{i}", fill=col)
        ov.save(out_path)
        return out_path


    # ------------------------------------------------------------------ run
    def run_once(self, query, media_path):
        if not os.path.exists(media_path):
            print(f"Can't find image: {media_path}")
            return None
        batch = self.process(media_path, query)
        tokens = batch["tokens"][0]
        L = tokens.shape[0]
        if L > self.MAX_INPUT_LENGTH:
            print(f"Error: token length {L} > MAX_INPUT_LENGTH "
                  f"{self.MAX_INPUT_LENGTH}")
            return None

        print(f"\n[info] tokens={L}  img_tokens={(tokens.numpy()==self.ID_IMG).sum()}")
        t0 = time.time()

        # 1) img-patch injection (projected features + token row positions).
        #    The token embedding stays on device; forward_first scatters these
        #    into block_0's input mem at the img-token rows.
        img_feats, img_pos, L = self.build_img_injection(batch)
        pos_t = batch["pos_t"][0].numpy().astype(np.int32)     # [L]
        pos_pad = np.zeros(self.MAX_INPUT_LENGTH, dtype=np.int32)
        pos_pad[:L] = pos_t
        gcos, gsin = self.build_golden_cos_sin(batch, L)
        attn_mask = self.build_prefill_mask(batch, L)

        # 2) prefill (embed + img scatter + blocks, all on device)
        token = self.model.forward_first(
            tokens.numpy().astype(np.int32), pos_pad, gcos, gsin,
            attn_mask, L, img_feats, img_pos)
        self.max_posid = int(pos_t[-1])
        t1 = time.time()
        # Don't decode the first token here — the decode loop below dispatches
        # every token (incl. the first) and prints it once. Printing it here too
        # double-prints special tokens like <|presence|> (the model emits only
        # one, matching HF; the duplicate was a display artifact).
        print("Answer:\n", end="", flush=True)

        # 3b) full prefill hidden -> anyup (hr_img_features, once) + first h_last
        h_BSD_full = np.array(self.model.forward_first_hidden(),
                              dtype=np.float32).reshape(self.MAX_INPUT_LENGTH,
                                                        self.hidden)
        h_last = h_BSD_full[L - 1].copy()                     # hidden that emitted `token`
        hr_features = self.run_anyup(batch, h_BSD_full[:L])   # [1,256,256,256]

        # 4) decode loop with coord/size/seg head dispatch + Fourier 回灌
        tok_num = 0
        full_word_tokens = []
        text = ""
        aux = []                                              # emitted xy/hw/mask
        existing_coords = []                                  # for coord dedup
        pending_fourier = np.zeros((0,), dtype=np.float32)    # 回灌 embedding for next step
        gcos1 = np.ones((1, self.n_heads, self.rope_quart), dtype=np.float32)
        gsin1 = np.zeros((1, self.n_heads, self.rope_quart), dtype=np.float32)
        while token not in self.stop_ids and \
              self.model.history_length < self.SEQLEN:
            # dispatch heads on `token` using h_last (hidden that emitted it)
            if token == self.ID_COORD:
                xy, fourier = self.decode_coord(h_last, existing_coords)
                existing_coords.append(xy)
                aux.append({"x": float(xy[0]), "y": float(xy[1])})
                pending_fourier = fourier
                print(f"\n[coord] x={xy[0]:.4f} y={xy[1]:.4f}", end="", flush=True)
            elif token == self.ID_SIZE:
                hw, fourier = self.decode_size(h_last)
                aux.append({"h": float(hw[0]), "w": float(hw[1])})
                pending_fourier = fourier
                print(f"\n[size] h={hw[0]:.4f} w={hw[1]:.4f}", end="", flush=True)
            elif token == self.ID_SEG:
                mask = self.decode_seg(h_last, hr_features)
                aux.append(mask)
                print(f"\n[seg] mask_pos={int((1.0/(1.0+np.exp(-np.clip(np.nan_to_num(mask), -50, 50)))>0.5).sum())}",
                      end="", flush=True)
            else:
                full_word_tokens.append(token)
                word = self.tokenizer.decode(full_word_tokens,
                                             skip_special_tokens=False)
                if "�" not in word:
                    text += word
                    print(word, end="", flush=True)
                    full_word_tokens = []
            # advance: forward_next consumes pending_fourier (回灌 at this token)
            self.max_posid += 1
            pos_id = np.array([self.max_posid], dtype=np.int32)
            dec_mask = self.build_decode_mask(self.model.history_length)
            token = self.model.forward_next(token, pos_id, gcos1, gsin1, dec_mask,
                                            pending_fourier)
            h_last = np.array(self.model.forward_hidden(),
                              dtype=np.float32).reshape(self.hidden).copy()
            pending_fourier = np.zeros((0,), dtype=np.float32)
            tok_num += 1

        t2 = time.time()
        # original image size + pixel_mask active region for mask postprocess
        from PIL import Image as _PILImage
        _W, _H = _PILImage.open(media_path).size
        orig_hw = (_H, _W)
        _pm = batch["pixel_mask"]
        _pm = _pm.numpy() if hasattr(_pm, "numpy") else np.asarray(_pm)
        pixel_mask_hw = _pm[0, 0] if _pm.ndim == 4 else _pm[0]
        dets, (n_coord, n_size, n_seg, n_cand, n_keep) = \
            self.postprocess(aux, pixel_mask_hw, orig_hw)
        print(f"\n\n[emission] coord={n_coord} size={n_size} seg={n_seg}  "
              f"[detections] {len(dets)} (NMS kept {n_keep}/{n_cand})")
        for i, d in enumerate(dets):
            mpix = int(d["mask"].sum())
            print(f"  #{i} xy=({d['x']:.3f},{d['y']:.3f}) hw=({d['h']:.3f},{d['w']:.3f}) "
                  f"mask_px={mpix}/{_H *_W}")
        import re as _re
        _tag = _re.sub(r"[^a-z0-9]+", "_", query.lower()).strip("_")[:24] or "query"
        out_jpg = os.path.join(os.path.dirname(media_path) or ".",
                               f"{os.path.splitext(os.path.basename(media_path))[0]}_{_tag}_vis.jpg")
        self.visualize(dets, media_path, out_jpg)
        print(f"  visualization: {out_jpg}")
        print(f"\nFTL: {t1 - t0:.3f} s   decode: {t2 - t1:.3f} s   "
              f"tokens: {tok_num}   TPS: {tok_num / (t2 - t1) if t2 > t1 else 0:.3f}")
        return text

    def chat(self):
        print("""\n=================================================================
1. quit: q / quit / exit
2. new session: clear / new
=================================================================""")
        while True:
            query = input("\nQuery: ")
            if query in ["exit", "q", "quit"]:
                break
            if query in ["clear", "new", "c"]:
                self.model.clear_history()
                continue
            media_path = input("Image Path: ")
            self.run_once(query, media_path)


def main(args):
    model = FalconPerception(args)
    if args.query is not None:
        model.run_once(args.query, args.media_path)
    else:
        model.chat()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', type=str, required=True,
                        help='path to the bmodel file')
    parser.add_argument('-c', '--config_path', type=str, default="../config",
                        help='path to the model config directory (config.json + '
                             'custom processor python + falcon_extra_weights.npz)')
    parser.add_argument('-d', '--devid', type=int, default=0)
    parser.add_argument('-q', '--query', type=str, default=None,
                        help='If set, run a single inference and exit.')
    parser.add_argument('--media_path', type=str, default="",
                        help='Path to an image for programmatic mode.')
    args = parser.parse_args()
    main(args)
