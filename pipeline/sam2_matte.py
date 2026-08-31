"""sam2_matte.py — SAM2 (Segment Anything 2, hiera-tiny) video segmentation
behind the same MatteResult-shaped interface subject_matte.py exposes, so
matte_effects.py's functions can take a precomputed mask instead of only
calling rembg (subject_matte.py's model, ~40-50s/frame -- infeasible past a
couple hundred frames; SAM2 measured ~1s/frame in this fleet's own Pilot 4
bake-off, 2026-08-24, CV_LAB.md).

WHY THIS FILE EXISTS (provenance note): two videos earlier this session
(v-bbox-rickroll-sam2.mp4, v-lasso-rickroll-sam2.mp4) were built from SAM2
masks computed with an ad hoc inline script, never saved as a real module --
a real gap, caught during planning for the 30-second expansion. This is that
adapter, written properly and committed.

CHUNKED, NOT ONE PASS: SAM2's video predictor preallocates the whole clip
into memory (`torch.zeros(num_frames, 3, 1024, 1024, float32)` in its own
frame loader) -- at 900 frames that's ~11.3GB, a real OOM risk on a 16GB
machine, and a single frame-0 box prompt is unlikely to survive hard cuts in
30+ seconds of a produced music video (memory-attention drift, silent mask
corruption with no error). compute_subject_mattes_chunked() processes the
clip in ~300-frame chunks, seeding each chunk's first frame from the
previous chunk's last mask via add_new_mask() (a real conditioning frame,
not a guess) -- bounds peak memory to one chunk's allocation and gives drift
nowhere to silently accumulate past a chunk boundary.
"""
from __future__ import annotations

import glob
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

_REQUIRED_VENV = "/Users/brighamhall/cowork/cv-experiments/venv/bin/python3"
_SAM2_DIR = "/Users/brighamhall/cowork/cv-experiments/sam2"
_CHECKPOINT = f"{_SAM2_DIR}/checkpoints/sam2.1_hiera_tiny.pt"
_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"

try:
    import numpy as np
except ModuleNotFoundError as e:
    raise RuntimeError(
        f"{__name__} requires numpy, installed only in the cv-experiments "
        f"venv. Run with {_REQUIRED_VENV}, not system python3 "
        f"(original error: {e})"
    ) from e

try:
    import cv2
except ModuleNotFoundError as e:
    raise RuntimeError(
        f"{__name__} requires opencv-contrib-python-headless, installed "
        f"only in the cv-experiments venv. Run with {_REQUIRED_VENV}, not "
        f"system python3 (original error: {e})"
    ) from e

try:
    import torch
    sys.path.insert(0, _SAM2_DIR)
    from sam2.build_sam import build_sam2_video_predictor
except ModuleNotFoundError as e:
    raise RuntimeError(
        f"{__name__} requires torch + the sam2 package installed at "
        f"{_SAM2_DIR}, in the cv-experiments venv. Run with "
        f"{_REQUIRED_VENV}, not system python3 (original error: {e})"
    ) from e


@dataclass
class MatteResult:
    """Same shape as subject_matte.MatteResult -- alpha_maps is a list of
    uint8 (H, W) arrays, one per frame, 255=subject."""
    alpha_maps: List["np.ndarray"]
    frame_shape: Tuple[int, int]
    smoothed: bool
    smoothing_window: int


def mask_area_series(result: MatteResult, threshold: int = 30) -> List[int]:
    return [int((m > threshold).sum()) for m in result.alpha_maps]


def bbox_series(result: MatteResult, threshold: int = 30) -> List[Optional[Tuple[int, int, int, int]]]:
    boxes = []
    for m in result.alpha_maps:
        ys, xs = np.where(m > threshold)
        boxes.append((int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())) if len(xs) else None)
    return boxes


def _extract_frames(video_path: str, out_dir: str) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", video_path,
         "-q:v", "2", "-start_number", "0", os.path.join(out_dir, "%05d.jpg")],
        check=True,
    )
    return sorted(glob.glob(os.path.join(out_dir, "*.jpg")))


def compute_subject_mattes_chunked(
    video_path: str,
    seed_box,
    chunk_frames: int = 300,
    quality_gate=None,
) -> MatteResult:
    """seed_box: (x0, y0, x1, y1) prompt for the FIRST chunk's frame 0 only.
    Every later chunk is seeded from the previous chunk's own last computed
    mask via add_new_mask() -- a real conditioning frame, never re-prompted
    from the original box (which would ignore how the subject has moved).

    quality_gate: optional callable(chunk_idx, alpha_maps_for_chunk) -> None,
    raise to abort. Default gate rejects a chunk whose first mask is empty
    (mirrors subject_matte.py's own shadow-band regression guard) -- this
    function is written to be called as the gate this plan requires BEFORE
    any render is launched, not just left as an inline assertion."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        frame_paths = _extract_frames(video_path, tmp)
        n_total = len(frame_paths)
        if n_total == 0:
            raise RuntimeError(f"no frames extracted from {video_path}")

        first_frame = cv2.imread(frame_paths[0])
        h, w = first_frame.shape[:2]

        predictor = build_sam2_video_predictor(_CONFIG, _CHECKPOINT, device="cpu")

        all_alpha: List[np.ndarray] = []
        prev_chunk_last_mask = None

        chunk_starts = list(range(0, n_total, chunk_frames))
        for ci, start in enumerate(chunk_starts):
            end = min(start + chunk_frames, n_total)
            chunk_dir = os.path.join(tmp, f"chunk_{ci}")
            os.makedirs(chunk_dir, exist_ok=True)
            chunk_paths = frame_paths[start:end]
            for i, p in enumerate(chunk_paths):
                os.symlink(p, os.path.join(chunk_dir, f"{i:05d}.jpg"))

            state = predictor.init_state(video_path=chunk_dir)
            if prev_chunk_last_mask is None:
                x0, y0, x1, y1 = seed_box
                predictor.add_new_points_or_box(
                    inference_state=state, frame_idx=0, obj_id=1,
                    box=np.array([x0, y0, x1, y1], dtype=np.float32),
                )
            else:
                predictor.add_new_mask(
                    inference_state=state, frame_idx=0, obj_id=1,
                    mask=prev_chunk_last_mask,
                )

            chunk_alpha = [None] * len(chunk_paths)
            for out_idx, out_obj_ids, out_logits in predictor.propagate_in_video(state):
                m = (out_logits[0] > 0.0).cpu().numpy()[0]
                chunk_alpha[out_idx] = (m.astype(np.uint8)) * 255

            if quality_gate is not None:
                quality_gate(ci, chunk_alpha)
            else:
                first_mask = chunk_alpha[0]
                if first_mask is None or (first_mask > 30).sum() == 0:
                    raise RuntimeError(
                        f"chunk {ci} (frames {start}-{end}) produced an "
                        f"empty first mask -- aborting before any render "
                        f"launches, per this module's quality-gate contract"
                    )

            all_alpha.extend(chunk_alpha)
            prev_chunk_last_mask = (chunk_alpha[-1] > 30)

        return MatteResult(all_alpha, (h, w), smoothed=False, smoothing_window=1)


def compute_subject_mattes_from_point(
    video_path: str,
    seed_point,
    chunk_frames: int = 300,
    quality_gate=None,
) -> MatteResult:
    """THE LITERAL "MAGIC LASSO" MECHANISM (CV_LAB.md Pilot 6, 2026-08-24).

    The operator's original complaint that started this whole lab was "that's
    what i meant by magic lasso (aka photoshop stuff)" -- a single click that
    selects the whole subject. Every other SAM2 path in this lab
    (compute_subject_mattes_chunked, above) seeds from a BOX
    (x0, y0, x1, y1) -- four coordinates, not a click, and in practice
    produced by a saliency detector or a manual scrub, not a literal
    single-click UX. This function seeds from ONE (x, y) point instead,
    which is the actual single-click Photoshop-lasso analog: SAM2's own
    `add_new_points_or_box(points=[[x, y]], labels=[1])` API (points =
    foreground/background click coordinates, labels 1=foreground/0=
    background -- confirmed directly against the predictor source at
    ~/cowork/cv-experiments/sam2/sam2/sam2_video_predictor.py:161-201, not
    assumed) accepts a bare point exactly like Photoshop's magic-wand/lasso
    click. Everything else -- chunking, re-seeding each later chunk from the
    previous chunk's own last mask via add_new_mask() -- is identical to
    compute_subject_mattes_chunked(); only the FIRST chunk's FIRST-frame seed
    prompt differs (one point instead of a box).

    REAL COMPARISON RUN (2026-08-24, rickroll_short.mp4, 50 frames, 640x360,
    single human subject, CPU-only hiera-tiny), against the existing
    BOX-seeded masks already on disk at ~/blueduck-grants/datamoshing/work/
    sam2_rickroll_masks/ (0/50 empty, mean area 51,435px):

    - seed point (355, 230), picked by eye on frame 0
      (~/blueduck-grants/datamoshing/work/rickroll_frames/00000.jpg) landing
      on the subject's striped shirt. Result: 0/50 empty masks (0.0% empty
      rate) -- BUT mean mask area only 3,511px (6.8% of the box run's area),
      and visual inspection of the saved masks
      (~/blueduck-grants/datamoshing/work/sam2_rickroll_masks_point_seed/)
      shows SAM2 tracked one or two shirt stripes across all 50 frames, not
      the whole person. Mean per-frame IoU against the box-seeded mask:
      0.066. The empty-mask-rate criterion alone is BLIND to this failure --
      0.0% empty in both runs -- because a wrong-but-nonempty mask and a
      correct mask both count as "not empty." IoU against a known-good
      baseline is the metric that actually catches it.
    - This is real, reproducible SAM2 single-point ambiguity, not a bad
      click: the code path SAM2 uses for a single init-frame point
      (`_use_multimask`, sam2_base.py:881) requests 3 candidate masks and
      auto-picks the model's own highest-estimated-IoU one -- for a
      textured/multi-part object like a striped shirt, the model's own
      best guess can legitimately be a texture-coherent sub-region, not the
      whole subject. (Confirmed: this code path exists and runs an
      argmax-over-IoU selection. Not independently confirmed: that this
      specific mechanism, rather than something else, is *why* these two
      particular masks came out small -- that would need instrumenting the
      3 candidate masks/IoU scores directly, not done here.)
    - A SECOND real run, seed point (270, 270) on the solid black jacket
      (away from any texture edge, the theoretically "safer" click), was run
      to check whether the stripe click was just a bad pick. It was worse,
      not better: **46/50 empty masks (92.0% empty rate)** -- SAM2 produced
      a nonempty mask for only frames 0-3, then an empty mask for the
      remaining 46 frames of this same clean, uncut, single-subject 50-frame
      clip (consistent with SAM2's own no-object/occlusion score flipping
      negative partway through propagation; not independently instrumented
      here, so stated as consistent-with rather than confirmed). Mean area
      511px (1.0% of the box run's area), mean IoU against the box-seeded
      mask: 0.006. Even frame 0's own mask was a noisy speckle pattern
      bleeding into the background wall, not a clean jacket silhouette --
      masks on disk at ~/blueduck-grants/datamoshing/work/
      sam2_rickroll_masks_point_seed_jacket/.
    - Practical consequence, stated plainly: single-point seeding on this
      clip is CLICK-LOCATION-DEPENDENT and, at both tested locations, worse
      than the box seed on every metric that isn't raw empty-rate (which the
      jacket click also fails outright, 92% vs the box's 0%). The box
      prompt's structural advantage -- it bounds the WHOLE subject's extent
      by construction, point prompts do not -- is not a minor implementation
      detail; it is decisive on this clip. The default quality_gate this
      module ships (both compute_subject_mattes_chunked() and this
      function use the same one, inline below) inspects ONLY
      `chunk_alpha[0]` -- frame 0 of the chunk -- for emptiness, and both
      real runs above were called with `quality_gate=None` (the default),
      so it ran unmodified on both. It passed BOTH: the shirt run because
      frame 0 was genuinely nonempty (just wrong), and the jacket run
      because frame 0 was ALSO nonempty -- the 46-frame collapse happened
      on frames 1-49, entirely outside what a frame-0-only gate can see. A
      real point-seed gate needs to check more than frame 0, and needs an
      area/IoU sanity check against something, not just non-emptiness (not
      built here -- flagged for whoever picks this path up next).
    - Bottom line on the literal ask: single-click "magic lasso" IS a real,
      confirmed SAM2 mechanism (`points`/`labels` on
      `add_new_points_or_box`, verified against the predictor source, not
      assumed) -- the operator's request is technically answerable. But
      measured against the SAME clip and the SAME empty-mask-rate
      methodology this lab already uses, it is NOT currently a safe
      drop-in replacement for the box path: box = 0/50 empty, mean area
      51,435px; point (shirt) = 0/50 empty but IoU 0.066 (wrong region);
      point (jacket) = 46/50 empty (92%), IoU 0.006. Two honest attempts,
      two different failure modes, zero clean wins for the point path on
      this clip.

    seed_point: (x, y) pixel coordinate on the FIRST chunk's frame 0, the
    single click. labels is hardcoded to [1] (foreground) -- a pure
    single-click seed, not a multi-point disambiguation session (that would
    no longer be "one click").
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        frame_paths = _extract_frames(video_path, tmp)
        n_total = len(frame_paths)
        if n_total == 0:
            raise RuntimeError(f"no frames extracted from {video_path}")

        first_frame = cv2.imread(frame_paths[0])
        h, w = first_frame.shape[:2]

        predictor = build_sam2_video_predictor(_CONFIG, _CHECKPOINT, device="cpu")

        all_alpha: List[np.ndarray] = []
        prev_chunk_last_mask = None

        chunk_starts = list(range(0, n_total, chunk_frames))
        for ci, start in enumerate(chunk_starts):
            end = min(start + chunk_frames, n_total)
            chunk_dir = os.path.join(tmp, f"chunk_{ci}")
            os.makedirs(chunk_dir, exist_ok=True)
            chunk_paths = frame_paths[start:end]
            for i, p in enumerate(chunk_paths):
                os.symlink(p, os.path.join(chunk_dir, f"{i:05d}.jpg"))

            state = predictor.init_state(video_path=chunk_dir)
            if prev_chunk_last_mask is None:
                x, y = seed_point
                predictor.add_new_points_or_box(
                    inference_state=state, frame_idx=0, obj_id=1,
                    points=np.array([[x, y]], dtype=np.float32),
                    labels=np.array([1], dtype=np.int32),
                )
            else:
                predictor.add_new_mask(
                    inference_state=state, frame_idx=0, obj_id=1,
                    mask=prev_chunk_last_mask,
                )

            chunk_alpha = [None] * len(chunk_paths)
            for out_idx, out_obj_ids, out_logits in predictor.propagate_in_video(state):
                m = (out_logits[0] > 0.0).cpu().numpy()[0]
                chunk_alpha[out_idx] = (m.astype(np.uint8)) * 255

            if quality_gate is not None:
                quality_gate(ci, chunk_alpha)
            else:
                first_mask = chunk_alpha[0]
                if first_mask is None or (first_mask > 30).sum() == 0:
                    raise RuntimeError(
                        f"chunk {ci} (frames {start}-{end}) produced an "
                        f"empty first mask -- aborting before any render "
                        f"launches, per this module's quality-gate contract"
                    )

            all_alpha.extend(chunk_alpha)
            prev_chunk_last_mask = (chunk_alpha[-1] > 30)

        return MatteResult(all_alpha, (h, w), smoothed=False, smoothing_window=1)
