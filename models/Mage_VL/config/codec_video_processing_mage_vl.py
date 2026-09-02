"""Codec-based video preprocessing for MageVL (trust_remote_code).

This module is the codec analogue of ``video_processing_magevl.py``.
It is invoked when a user calls::

    processor(messages=..., video_backend="codec", max_pixels=...)

and is responsible for:

  - Decoding the video and assembling canvas images via ``cv-preinfer``
    (PyPI: ``codec-video-prep``, requires ``ffmpeg`` on PATH).
  - Running the bundled ``Qwen2VLImageProcessor`` on those canvases with a
    pixel budget that is *aligned* to the canvas dimensions (so the
    smart_resize step never desynchronises ``image_grid_thw`` from the
    codec-emitted ``src_patch_position`` array).
  - Producing the per-patch ``patch_positions`` table that
    ``modeling_magevl.py`` reads for the 2D-MRoPE block layout.

The result is a ``BatchFeature``-shaped dict containing the same keys that
the frame-sampling video path produces (``pixel_values`` /
``image_grid_thw`` / ``patch_positions``), so downstream
``modeling_magevl.py`` consumes it without changes.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import warnings
from dataclasses import dataclass, field, fields as _dc_fields
from pathlib import Path
from typing import Optional

try:
    import fcntl
except ImportError:
    fcntl = None  # type: ignore

import numpy as np
import torch
from PIL import Image


VISION_START = "<|vision_start|>"
VISION_END = "<|vision_end|>"
IMAGE_PAD = "<|image_pad|>"


# ----------------------------------------------------------------- config

@dataclass
class DcvcConfig:
    """DCVC-RT neural-codec knobs (nested under ``CodecConfig.dcvc``).

    Only used when ``CodecConfig.engine == "dcvc-rt"``. Checkpoints default to
    the ``DCVC_INTRA_TAR`` / ``DCVC_INTER_TAR`` env vars (environment-specific,
    so better left out of the model config).
    """
    qp: int = 42                          # quantization (0-63); scoring only
    reset_interval: int = 64              # feature-adaptor reset period (frames)
    intra_period: int = -1                # -1 => single I-frame at t=0, rest P
    max_side: int = 0                     # >0 downscales frames before DCVC scoring;
                                          # 0 => full-resolution scoring (matches the
                                          # hevc baseline, which reads h264 bits at
                                          # native stream resolution). Faithful but
                                          # slower to encode.
    seq_len_frames: int = 0               # 0 => CodecConfig.num_sampled_frames()
    patch: int = 16                       # MUST match image processor patch_size
    canvas_token_side: Optional[int] = None  # None => derive from max_pixels
    # ---- readiness grouping knobs (used by the bundled process_video_bitcost_
    # readiness pipeline). Defaults reproduce EXACTLY the offline/cluster DCVC
    # sweep + hevc baseline generation params, so the release DCVC path yields
    # the same assets we evaluated. (These intentionally differ from the release
    # hevc _run_cv_preinfer knobs, e.g. max_group_frames=128 vs CodecConfig's 64.)
    readiness_sum_threshold_mode: str = "auto"
    readiness_coverage_bins: int = 3
    readiness_delta_ratio: float = 0.05
    bitcost_grid: str = "sub"             # DCVC bitmap is at 16px granularity
    bitcost_pct: int = 99
    decode_backsearch_max: int = 16
    max_group_frames: int = 128
    intra_ckpt: str = field(default_factory=lambda: os.getenv("DCVC_INTRA_TAR", ""))
    inter_ckpt: str = field(default_factory=lambda: os.getenv("DCVC_INTER_TAR", ""))
    device: str = field(default_factory=lambda: os.getenv("DCVC_DEVICE", "cuda:0"))
    pkg_dir: str = ""                     # dir with dcvc_rt_engine.py + dcvc_readiness_gen.py + codec_tools/


@dataclass
class CodecConfig:
    """All knobs for the codec preprocessing pipeline.

    ``max_pixels`` is shared with the image_processor / video_processor pixel
    budget. The processor sets it from the user's ``max_pixels=`` kwarg, so
    canvas size and HF smart_resize budget stay consistent.
    """

    target_canvas: int = 32
    group_size: int = 32
    images_per_group: int = 4
    patch: int = 14
    max_pixels: int = 150000
    min_group_frames: int = 8
    max_group_frames: int = 64
    spatial_mask_mode: str = "off"
    cache_root: Path = field(default_factory=lambda: Path(
        os.getenv(
            "ONLINE_CODEC_CACHE_DIR",
            os.path.join(
                os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
                "online_codec",
            ),
        )
    ))
    timeout_seconds: int = int(os.getenv("ONLINE_CODEC_TIMEOUT", "7200"))

    # ---- codec engine selection ------------------------------------------
    # "hevc"    : HEVC/h264 bit-cost readiness (external cv-preinfer binary; the
    #             default). "cv-preinfer" is accepted as a legacy alias.
    # "dcvc-rt" : DCVC-RT neural-codec bit-cost readiness (self-contained, uses
    #             the bundled ``neural_codec/`` package + DCVC-RT checkpoints).
    # Selectable per call via ``codec_config={"engine": "dcvc-rt", ...}``.
    engine: str = field(default_factory=lambda: os.getenv("CODEC_ENGINE", "hevc"))

    # DCVC-RT knobs, nested. In the model config / codec_config this is a plain
    # dict (``"dcvc": {"qp": 21, ...}``); __post_init__ coerces it to DcvcConfig.
    dcvc: "DcvcConfig" = field(default_factory=DcvcConfig)

    def __post_init__(self):
        if isinstance(self.dcvc, dict):
            # Filter to known DcvcConfig fields — preprocessor_config.json's
            # codec.dcvc also carries selection knobs (per_frame_cap_ratio,
            # bottom_atten, threshold_scale, ...) read directly by the readiness
            # pipeline via codec_dcvc_config, which are not DcvcConfig fields.
            _known = {f.name for f in _dc_fields(DcvcConfig)}
            self.dcvc = DcvcConfig(**{k: v for k, v in self.dcvc.items() if k in _known})

    def validate(self) -> None:
        if self.target_canvas <= 0:
            raise ValueError("CodecConfig.target_canvas must be > 0")
        if self.target_canvas % self.images_per_group != 0:
            raise ValueError(
                "CodecConfig.target_canvas must be divisible by images_per_group"
            )
        if self.group_size % self.images_per_group != 0:
            raise ValueError(
                "CodecConfig.group_size must be divisible by images_per_group"
            )

    def num_sampled_frames(self) -> int:
        return (self.target_canvas // self.images_per_group) * self.group_size


# ---------------------------------------------------------- text/position

def _format_timestamp(seconds: float, decimals: int) -> str:
    return f"<{seconds:.{decimals}f} seconds>"


def convert_positions_to_block_layout(
    positions: torch.Tensor, t: int, h: int, w: int, spatial_merge_size: int = 2,
) -> torch.Tensor:
    """Reorder a (T*H*W, 3) patch position table into 2D-MRoPE block layout."""
    sms = int(spatial_merge_size)
    if sms == 1:
        return positions
    total = int(t) * int(h) * int(w)
    indices = torch.arange(total, device=positions.device).view(t, h, w)
    h_m, w_m = int(h) // sms, int(w) // sms
    indices = (
        indices.view(t, h_m, sms, w_m, sms)
        .permute(0, 1, 3, 2, 4).contiguous().view(total)
    )
    return positions[indices]


def codec_positions_for_processor(
    src_positions: np.ndarray, image_grid_thw: torch.Tensor, device: torch.device,
) -> torch.Tensor:
    positions = torch.from_numpy(src_positions).long().to(device)
    expected_total = int(image_grid_thw.prod(dim=1).sum().item())
    if expected_total != positions.shape[0]:
        raise ValueError(
            "codec patch position length mismatch: "
            f"thw_total={expected_total}, positions={positions.shape[0]}"
        )
    chunks, offset = [], 0
    for row in image_grid_thw:
        t, h, w = int(row[0]), int(row[1]), int(row[2])
        n = t * h * w
        chunks.append(convert_positions_to_block_layout(positions[offset: offset + n], t, h, w))
        offset += n
    return torch.cat(chunks, dim=0)


def _timestamp_runs(
    patch_positions: torch.Tensor, fps: float, decimals: int, spatial_merge_size: int = 2,
) -> list[tuple[str, int]]:
    t_values = patch_positions[:, 0]
    unique_t, counts = torch.unique_consecutive(t_values, return_counts=True)
    merge_factor = int(spatial_merge_size) ** 2
    runs = []
    for t_val, count in zip(unique_t.tolist(), counts.tolist()):
        if int(t_val) < 0:
            continue
        token_count = int(count) // merge_factor
        if token_count <= 0:
            continue
        runs.append((_format_timestamp(float(t_val) / float(fps), decimals), token_count))
    return runs


def rewrite_text_with_codec_positions(
    text: str, patch_positions: torch.Tensor, fps: float, decimals: int,
) -> str:
    """Replace the vision span in a chat-template string with codec-aware tokens."""
    parts = []
    for timestamp, token_count in _timestamp_runs(patch_positions, fps, decimals):
        parts.extend([timestamp, VISION_START, IMAGE_PAD * token_count, VISION_END, "\n"])
    vision_text = "".join(parts)
    first_vs, last_ve = text.find(VISION_START), text.rfind(VISION_END)
    if first_vs == -1 or last_ve == -1:
        return text
    tail_start = last_ve + len(VISION_END)
    if tail_start < len(text) and text[tail_start] == "\n":
        tail_start += 1
    return text[:first_vs] + vision_text + text[tail_start:]


def drop_padding_canvases(
    images: list[Image.Image], src_positions: np.ndarray,
) -> tuple[list[Image.Image], np.ndarray, int]:
    """Drop fully-padding canvases (all-negative timestamps) and their patches."""
    n_canvas = len(images)
    if n_canvas == 0:
        return images, src_positions, 0
    total_patches = src_positions.shape[0]
    if total_patches % n_canvas != 0:
        raise ValueError(
            f"src_positions length {total_patches} not divisible by canvas count {n_canvas}"
        )
    ppc = total_patches // n_canvas
    positions = src_positions.reshape(n_canvas, ppc, 3)
    canvas_t = positions[..., 0]
    keep_mask = (canvas_t >= 0).any(axis=1)
    if bool((keep_mask & ~((canvas_t >= 0).all(axis=1))).any()):
        raise ValueError("encountered half-padding canvas; padding is expected to be canvas-granular")
    dropped = int(n_canvas - int(keep_mask.sum()))
    if dropped == 0:
        return images, src_positions, 0
    kept_images = [img for img, keep in zip(images, keep_mask.tolist()) if keep]
    kept_positions = positions[keep_mask].reshape(-1, 3)
    return kept_images, kept_positions, dropped


# ------------------------------------------------------- cv-preinfer driver

def _get_video_total_frames(video_url: str) -> int:
    import cv2
    cap = cv2.VideoCapture(video_url)
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()
    return max(1, total)


def _cache_dir_for(video_url: str, cfg: CodecConfig) -> Path:
    raw = (
        f"{video_url}|eng={cfg.engine}|tc={cfg.target_canvas}|gs={cfg.group_size}"
        f"|ipg={cfg.images_per_group}|patch={cfg.patch}"
        f"|mp={cfg.max_pixels}|mask={cfg.spatial_mask_mode}"
    )
    if cfg.engine == "dcvc-rt":
        d = cfg.dcvc
        raw += (
            f"|dqp={d.qp}|drst={d.reset_interval}|dip={d.intra_period}"
            f"|dms={d.max_side}|dpatch={d.patch}|dseq={d.seq_len_frames}"
            f"|dcts={d.canvas_token_side}"
        )
        # The DCVC subprocess reads the selection-tuning knobs (threshold_scale,
        # bottom_atten / bottom_band, per_frame_cap_ratio, readiness_*, random_*, ...)
        # straight from preprocessor_config.json's codec.dcvc — not via cfg.dcvc — so
        # fold that whole block into the key. Otherwise editing a tuning knob leaves
        # the key unchanged and process_codec_video returns stale cached canvases.
        try:
            _cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "preprocessor_config.json")
            with open(_cfg_path, "r", encoding="utf-8") as _f:
                _dcvc_blk = (json.load(_f).get("codec", {}) or {}).get("dcvc", {}) or {}
            raw += "|dcvcblk=" + json.dumps(_dcvc_blk, sort_keys=True, separators=(",", ":"))
        except (OSError, ValueError):
            pass
    key = hashlib.md5(raw.encode()).hexdigest()
    return cfg.cache_root / f"{Path(video_url).stem}_{key}"


def _load_codec_result(out_dir: Path) -> dict:
    with open(out_dir / "meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    canvas_files = meta.get("canvas_files")
    if not canvas_files:
        for ext in ("npy", "jpg", "png"):
            hits = sorted(p.name for p in out_dir.glob(f"canvas_*.{ext}"))
            if hits:
                canvas_files = hits
                break
        canvas_files = canvas_files or []
    images = []
    for name in canvas_files:
        fp = out_dir / name
        if name.endswith(".npy"):
            images.append(Image.fromarray(np.load(fp)))
        else:
            images.append(Image.open(fp).convert("RGB"))
    src_positions = np.load(out_dir / "src_patch_position.npy")
    fps = float(meta.get("fps") or 30.0)
    return {"images": images, "src_positions": src_positions, "fps": fps,
            "out_dir": str(out_dir), "meta": meta}


def _run_cv_preinfer(video_url: str, out_dir: Path, cfg: CodecConfig) -> dict:
    bin_name = os.environ.get("CV_PREINFER_BIN", "cv-preinfer")
    if shutil.which(bin_name) is None and not os.path.isfile(bin_name):
        raise RuntimeError(
            f"engine='hevc' (traditional codec) needs the external '{bin_name}' binary on "
            "PATH, which is not bundled with this repo. Install it and add it to PATH (or set "
            "CV_PREINFER_BIN to its path), or use the neural codec (engine='dcvc-rt')."
        )
    tmp_dir = Path(tempfile.mkdtemp(dir=str(cfg.cache_root), prefix=f".tmp_{out_dir.name[:48]}_"))
    num_sampled = min(cfg.num_sampled_frames(), _get_video_total_frames(video_url))
    cmd = [
        bin_name, "--video", video_url, "--out_dir", str(tmp_dir),
        "--num_sampled_frames", str(num_sampled),
        "--grouping_mode", "readiness",
        "--group_size", str(cfg.group_size),
        "--images_per_group", str(cfg.images_per_group),
        "--patch", str(cfg.patch),
        "--max_pixels", str(cfg.max_pixels),
        "--readiness_sum_threshold", "0",
        "--min_group_frames", str(cfg.min_group_frames),
        "--max_group_frames", str(cfg.max_group_frames),
        "--avoid_keyframes",
        "--canvas_format", "jpg",
    ]
    try:
        result = subprocess.run(cmd, text=True, capture_output=True, timeout=cfg.timeout_seconds)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout)[-2000:]
            raise RuntimeError(f"online codec failed rc={result.returncode}: {detail}")
        if out_dir.exists():
            shutil.rmtree(out_dir)
        tmp_dir.rename(out_dir)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    return _load_codec_result(out_dir)


# ------------------------------------------------------- dcvc-rt driver
def _run_dcvc_rt(video_url: str, out_dir: Path, cfg: CodecConfig) -> dict:
    """DCVC-RT analogue of ``_run_cv_preinfer``. Generates the codec asset dir
    (canvas_*.jpg + src_patch_position.npy + meta.json) by running the SAME
    readiness pipeline the hevc/offline path uses (``process_video_bitcost_
    readiness``), swapping ONLY the per-frame score source from h264 block bits
    to the DCVC-RT neural-codec bit-cost bitmap.

    Faithful by construction: it shells out to the bundled
    ``neural_codec/dcvc_readiness_gen.py`` — the exact script used for the offline /
    cluster DCVC generation — with the same GEN_PARAMS, so the release
    reproduces the evaluated assets. ffmpeg/ffprobe must be on PATH (same
    requirement as the hevc path); the DCVC engine + checkpoints are loaded by
    the subprocess.
    """
    d = cfg.dcvc
    module_dir = os.path.dirname(os.path.abspath(__file__))
    flat_pkg = os.path.join(module_dir, "neural_codec")
    nested_pkg = os.path.join(os.path.dirname(module_dir), "neural_codec")
    pkg = d.pkg_dir or (flat_pkg if os.path.isdir(flat_pkg) else nested_pkg)
    gen = os.path.join(pkg, "dcvc_readiness_gen.py")
    if not os.path.exists(gen):
        raise RuntimeError(f"dcvc-rt: bundled generator not found at {gen}")
    # Checkpoints default to the copies bundled in neural_codec/ (self-contained);
    # DCVC_INTRA_TAR / DCVC_INTER_TAR still override.
    if not d.intra_ckpt:
        d.intra_ckpt = os.path.join(pkg, "dcvc_rt_intra.tar")
    if not d.inter_ckpt:
        d.inter_ckpt = os.path.join(pkg, "dcvc_rt_inter.tar")
    if not os.path.exists(d.intra_ckpt) or not os.path.exists(d.inter_ckpt):
        raise ValueError(
            f"engine='dcvc-rt' checkpoints not found (intra={d.intra_ckpt!r}, "
            f"inter={d.inter_ckpt!r}); they ship in neural_codec/ — or set "
            "DCVC_INTRA_TAR / DCVC_INTER_TAR."
        )
    # The DCVC-RT source is bundled at neural_codec/DCVC and loaded by dcvc_rt_engine
    # (no env var); sanity-check it so the subprocess fails fast with a clear message.
    if not os.path.isdir(os.path.join(pkg, "DCVC", "src")):
        raise ValueError(
            f"engine='dcvc-rt' bundled DCVC source missing at "
            f"{os.path.join(pkg, 'DCVC')!r} (expected a 'src/' dir); the "
            "neural_codec/DCVC/ folder looks incomplete."
        )

    tmp_dir = Path(tempfile.mkdtemp(dir=str(cfg.cache_root), prefix=f".tmp_{out_dir.name[:48]}_"))
    num_sampled = min(cfg.num_sampled_frames(), _get_video_total_frames(video_url))
    # Readiness CLI args — mirror the offline/cluster GEN_PARAMS exactly so the
    # generated assets match what we evaluated. Shared dims from cfg; DCVC +
    # readiness knobs from cfg.dcvc (patch=16, max_group_frames=128, auto
    # threshold, etc. — intentionally differing from the hevc _run_cv_preinfer).
    cmd = [
        sys.executable, gen,
        "--video", video_url, "--out_dir", str(tmp_dir),
        "--num_sampled_frames", str(num_sampled),
        "--grouping_mode", "readiness",
        "--readiness_sum_threshold_mode", str(d.readiness_sum_threshold_mode),
        "--group_size", str(cfg.group_size),
        "--images_per_group", str(cfg.images_per_group),
        "--patch", str(d.patch),
        "--max_pixels", str(cfg.max_pixels),
        "--min_group_frames", str(cfg.min_group_frames),
        "--max_group_frames", str(d.max_group_frames),
        "--readiness_coverage_bins", str(d.readiness_coverage_bins),
        "--readiness_delta_ratio", str(d.readiness_delta_ratio),
        "--bitcost_grid", str(d.bitcost_grid),
        "--bitcost_pct", str(d.bitcost_pct),
        "--decode_backsearch_max", str(d.decode_backsearch_max),
        "--canvas_format", "jpg",
    ]
    # Only DCVC engine *infra* travels via env (checkpoints / device / repo roots).
    # Selection + scoring params (qp, reset_interval, intra_period, max_side, and
    # the readiness knobs) are read by dcvc_readiness_gen from preprocessor_config.
    # json's codec.dcvc via codec_dcvc_config — NOT from env.
    env = dict(os.environ)
    env.update({
        "DCVC_INTRA_TAR": d.intra_ckpt, "DCVC_INTER_TAR": d.inter_ckpt,
        "DCVC_DEVICE": d.device,
        "DCVC_ENGINE_DIR": pkg, "DCVC_REPO_DIR": pkg,
    })
    try:
        result = subprocess.run(cmd, text=True, capture_output=True,
                                timeout=cfg.timeout_seconds, env=env)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout)[-2000:]
            raise RuntimeError(f"dcvc-rt gen failed rc={result.returncode}: {detail}")
        if not (tmp_dir / "meta.json").exists():
            detail = (result.stderr or result.stdout)[-2000:]
            raise RuntimeError(f"dcvc-rt gen produced no meta.json: {detail}")
        # Tag the score source into meta so downstream can distinguish it.
        try:
            with open(tmp_dir / "meta.json", "r", encoding="utf-8") as f:
                meta = json.load(f)
            meta.update({"engine": "dcvc-rt", "score_source": "dcvc-rt",
                         "qp": d.qp, "reset_interval": d.reset_interval,
                         "dcvc_max_side": d.max_side})
            with open(tmp_dir / "meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        if out_dir.exists():
            shutil.rmtree(out_dir)
        tmp_dir.rename(out_dir)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    return _load_codec_result(out_dir)


def process_codec_video(video_url: str, cfg: CodecConfig) -> dict:
    """Public entrypoint: video URL + config -> dict(images, src_positions, fps, ...).

    Result is cached on disk under ``cfg.cache_root``; concurrent workers
    coordinate via a flock-protected sentinel.

    Soft-warning behaviour (B-mode):
      - If the video has fewer frames than needed to fill ``target_canvas``,
        we emit a one-time UserWarning describing the shortfall but proceed
        normally (cv-preinfer will produce fewer canvases than requested).
      - If the video is so short that cv-preinfer cannot form a single
        group (``< min_group_frames``), we emit a clearer warning and let
        cv-preinfer's own error propagate.
    """
    cfg.validate()
    out_dir = _cache_dir_for(video_url, cfg)
    if (out_dir / "meta.json").exists() and (out_dir / "src_patch_position.npy").exists():
        return _load_codec_result(out_dir)

    _maybe_warn_short_video(video_url, cfg)

    cfg.cache_root.mkdir(parents=True, exist_ok=True)
    lock_path = cfg.cache_root / f".{out_dir.name}.lock"
    lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        if fcntl is not None:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
        if (out_dir / "meta.json").exists() and (out_dir / "src_patch_position.npy").exists():
            return _load_codec_result(out_dir)
        if cfg.engine == "dcvc-rt":
            return _run_dcvc_rt(video_url, out_dir, cfg)
        if cfg.engine in ("hevc", "cv-preinfer"):
            return _run_cv_preinfer(video_url, out_dir, cfg)
        raise ValueError(
            f"unknown codec engine: {cfg.engine!r} (use 'hevc' or 'dcvc-rt')"
        )
    finally:
        try:
            if fcntl is not None:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)


def _maybe_warn_short_video(video_url: str, cfg: CodecConfig) -> None:
    """Soft-warn (B-mode) when a video is too short to fill target_canvas.

    Logic:
      * needed_frames  = num_sampled_frames() = (target_canvas/ipg)*group_size
      * usable_frames  = min(needed_frames, total_frames)
      * expected_canv  = (usable_frames // group_size) * images_per_group
    If ``expected_canv < target_canvas`` we warn. If
    ``total_frames < min_group_frames`` we warn more loudly (cv-preinfer
    will fail downstream and that error is allowed to propagate).
    """
    try:
        total_frames = _get_video_total_frames(video_url)
    except Exception:
        return  # don't fail on probe errors; cv-preinfer will report its own

    needed = cfg.num_sampled_frames()
    usable = min(needed, total_frames)
    expected_canv = (usable // cfg.group_size) * cfg.images_per_group

    if total_frames < cfg.min_group_frames:
        warnings.warn(
            f"[codec] video {video_url!r} has only {total_frames} frames "
            f"(< min_group_frames={cfg.min_group_frames}); cv-preinfer cannot "
            f"form even a single group and will error out. Consider lowering "
            f"min_group_frames or using video_backend='frames' for this clip.",
            UserWarning,
            stacklevel=2,
        )
        return

    if expected_canv < cfg.target_canvas:
        warnings.warn(
            f"[codec] video {video_url!r} has {total_frames} frames; with "
            f"group_size={cfg.group_size}, images_per_group={cfg.images_per_group} "
            f"this yields ~{expected_canv} canvas(es) instead of the requested "
            f"target_canvas={cfg.target_canvas}. Inference will proceed with the "
            f"smaller canvas count.",
            UserWarning,
            stacklevel=2,
        )


# ----------------------------------------------------- processor wiring

def codec_image_processor_outputs(
    image_processor, images: list[Image.Image], max_pixels: int,
) -> dict:
    """Run ``Qwen2VLImageProcessor`` on codec canvases without smart_resize-ing.

    The codec emits canvases already aligned to the patch grid. To keep
    ``image_grid_thw`` consistent with ``src_patch_position``:
      - ``max_pixels`` is clamped up to the largest canvas (never shrinks)
      - ``min_pixels`` is clamped down to the smallest canvas (never upscales)

    Without the ``min_pixels`` clamp, ``Qwen2VLImageProcessor``'s default
    ``min_pixels=200704`` would grow any canvas below that threshold,
    producing extra patches and a chunk/index mismatch downstream.
    """
    canvas_pixels = [im.width * im.height for im in images]
    proc_max = max(int(max_pixels), max(canvas_pixels, default=int(max_pixels)))
    proc_min = min(canvas_pixels) if canvas_pixels else 1
    return image_processor(
        images=images, min_pixels=proc_min, max_pixels=proc_max, return_tensors="pt",
    )


__all__ = [
    "CodecConfig",
    "process_codec_video",
    "drop_padding_canvases",
    "codec_positions_for_processor",
    "rewrite_text_with_codec_positions",
    "codec_image_processor_outputs",
    "VISION_START", "VISION_END", "IMAGE_PAD",
]
