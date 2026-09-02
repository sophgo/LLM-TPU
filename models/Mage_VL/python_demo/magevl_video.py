"""Shared video frame decoder for the Mage-VL demo and its HF reference.

This module is intentionally dependency-light (PyAV + PIL only, no torch, no
``chat``) so it imports identically on the SoC (TPU runtime) and on the CUDA
host (HF reference). Using the *same* decoder + the *same* sampling on both
sides guarantees identical frames, and therefore identical ``pixel_values``,
which is what makes the bmodel-vs-HF precision comparison meaningful.

PyAV is available on the SoC because sophon-mw ships ffmpeg libs and the SoC
python env has ``av`` installed; the CUDA host installs it via pip. Either way
the decoded RGB frames are deterministic for a given file, so the two sides
match as long as the frame *indices* match — which they do, since both call
this one function.
"""
from typing import Callable, List

from PIL import Image


def load_video_frames(
    path: str,
    num_frames: int,
    resize_fn: Callable[[Image.Image], Image.Image],
) -> List[Image.Image]:
    """Decode ``num_frames`` evenly-spaced frames from a video file.

    Frames are selected by linear index spacing over the decoded frame count
    (always including the first and last frame), then passed through
    ``resize_fn`` (the caller's ``resize_to_fixed_pixels``), so every frame is
    resized to exactly ``MAX_PIXELS`` pixels and yields ``MAX_PATCHES`` patches
    — the static vit net encodes each frame in a single ``forward_vit`` call.

    The Mage-VL frames-backend video path treats each frame as an independent
    single-frame image (no cross-frame attention inside the vit), so feeding
    ``len(frames)`` PIL images to ``processor(videos=[frames])`` is exactly the
    model's expected input.

    Args:
        path: video file path.
        num_frames: desired number of frames (>=2 to span the clip).
        resize_fn: maps a PIL frame -> a resized PIL frame (the demo's
            ``resize_to_fixed_pixels``).

    Returns:
        A list of ``num_frames`` (or fewer if the clip is shorter) RGB PIL
        images, each already resized to the fixed pixel budget.
    """
    import av

    container = av.open(path)
    try:
        all_frames = []
        for frame in container.decode(video=0):
            all_frames.append(frame.to_image())
    finally:
        container.close()

    total = len(all_frames)
    if total == 0:
        raise RuntimeError(f"No frames decoded from video: {path!r}")

    if num_frames >= total:
        idxs = list(range(total))
    else:
        # evenly-spaced indices, endpoints included
        idxs = [
            int(round(i * (total - 1) / (num_frames - 1)))
            for i in range(num_frames)
        ]
        # de-duplicate while preserving order (happens for very short clips)
        seen = set()
        idxs = [i for i in idxs if not (i in seen or seen.add(i))]

    return [resize_fn(all_frames[i].convert("RGB")) for i in idxs]


def load_all_video_frames(
    path: str,
    resize_fn: Callable[[Image.Image], Image.Image],
) -> List[Image.Image]:
    """Decode ALL frames from a video file in order, each resized by *resize_fn*.

    Unlike :func:`load_video_frames`, no sub-sampling is performed — every
    decoded frame is returned.  The streaming pipeline uses this to divide the
    video into consecutive segments of ``GATE_FRAMES`` frames for real-time
    gate decisions.

    Args:
        path: video file path.
        resize_fn: maps a PIL frame -> a resized PIL frame (the demo's
            ``resize_to_fixed_pixels``).

    Returns:
        A list of all decoded RGB PIL images, each already resized.
    """
    import av

    container = av.open(path)
    try:
        frames = [frame.to_image().convert("RGB")
                  for frame in container.decode(video=0)]
    finally:
        container.close()

    if not frames:
        raise RuntimeError(f"No frames decoded from video: {path!r}")

    return [resize_fn(f) for f in frames]
