"""whatsapp_dive_pipeline.py -- reconstructed montage driver, 2026-08-24.

Applies the same pipeline that built montage_30s_ascii_thermal.mp4 (CV_LAB.md,
Pilot 4 SAM2 bake-off + Pilot 7 background library) to a real WhatsApp video:
a diver at a Basque coastal swimming pool, one continuous locked-off shot,
90.0s-123.785s excerpt of "WhatsApp Video 2026-08-12 at 07.24.37.mp4"
(1013 frames, ~30fps, 576x1024 portrait).

RECONSTRUCTED, NOT COPIED: the ad hoc driver that built the four
montage_30s_*.mp4 outputs from the Rick Roll clip was never committed to
this repo (sam2_matte.py's own docstring calls this out as "a real gap").
This script is that missing driver, written properly this time:

  1. Extract the excerpt's frames to disk as %05d.jpg (same convention
     sam2_matte.py's internal extractor uses, so index order matches).
  2. Run compute_subject_mattes_chunked() with a hand-picked frame-0 seed
     box. SAM2 needs a box prompt and there is no reliable classical
     detector to supply one automatically -- CV_LAB.md's own root-cause
     finding is that the saliency detector on this project's other subject
     locks onto a cast SHADOW, not the subject, and dense optical flow
     measured as an anti-signal (0.78x) on generative footage. For a real,
     unseen phone video the correct substitute is a human (or a
     multimodal model) looking at frame 0 and drawing the box by eye --
     that's how this run's seed_box was picked.
  3. Save per-frame alpha masks to disk as %05d.png.
  4. Render three background-effect passes off the same masks:
       - ascii_background.render_ascii_background(palette="thermal_drift")
         -- the direct analog of montage_30s_ascii_thermal.mp4
       - layer_effects.render_layer_video("thermal_falsecolor")
       - layer_effects.render_layer_video("chromatic_shift")
     plus a hand-rolled contour-lasso visualization straight off the SAM2
     masks (matte_effects.draw_subject_lasso() calls subject_matte.py's
     rembg backend, ~40-50s/frame -- at 1013 frames that is 11-14 hours,
     infeasible; the lasso videos in the original 4-output set were
     likewise built off SAM2 masks directly, per sam2_matte.py's own
     provenance note about "an ad hoc inline script").
  5. Print this lab's two standing honesty numbers for every render:
     empty-mask rate (the gate that excluded 4 of 6 candidate shots in the
     original rickroll run) and subject/background mean-abs-diff (the
     proof the subject stayed byte-identical while the background changed).

Run with the cv-experiments venv (numpy/cv2/torch/sam2 live there only):
    /Users/brighamhall/cowork/cv-experiments/venv/bin/python3 \
        whatsapp_dive_pipeline.py
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, "/Users/brighamhall/cowork/cv-skills")
sys.path.insert(0, os.path.dirname(__file__))

import cv2
import numpy as np

import sam2_matte as s2
import layer_effects as le
import ascii_background as ab

# ascii_layer.py hardcodes ROWS=45, COLS=128 -- chosen because that grid
# evenly divides the fleet's landscape test clips (720x1280 / 640x360),
# not because it's resolution-agnostic (ascii_conv's _cell_sum requires
# exact H/rows and W/cols division, no remainder). Our source is a portrait
# 576x1024 phone video -- neither dimension divides by the hardcoded grid
# (1024 % 45 = 34, 576 % 128 = 64), which crashed the first attempt at this
# render with a numpy reshape error. Fix: override the grid to values that
# divide 576x1024 exactly (64 divides both 1024 and 576 cleanly) via a
# runtime monkeypatch of ascii_layer's module globals -- frames_to_ascii()
# looks these up at call time, not at def time, so this works without
# touching the shared library file (other pipelines' landscape clips are
# unaffected; this only changes behavior for the rest of THIS process).
ab.al.ROWS, ab.al.COLS = 64, 64

WORK = "/Users/brighamhall/blueduck-grants/datamoshing/work/wa_clip"
# 2026-08-24 UPDATE: the full 33.8s/1067-frame clip was run first. SAM2's own
# quality gate caught real degradation past the "sitting on the railing"
# portion -- chunk 2 (frames 600-900, the jump/splash) came back 58.7% empty
# masks, chunk 3 (900-1067, swimming away) came back 100% empty and the gate
# correctly aborted before any render launched. Water occlusion + rapid
# motion breaks a single frame-0 box seed. Per this lab's own rickroll
# precedent (ship the shots that hold up, exclude the ones that don't rather
# than silently rendering at degraded quality), this run uses a trimmed
# clip covering only the two chunks that measured 0% empty (frames 0-600 of
# the original extraction, ~19.0s) -- see PLAYBOOK.md.
SOURCE_CLIP = os.path.join(WORK, "wa_dive_clip_clean.mp4")
FRAMES_DIR = os.path.join(WORK, "frames_clean")
MASKS_DIR = os.path.join(WORK, "masks_clean")
FPS = 30.0
SEED_BOX = (325, 515, 485, 745)  # hand-picked from frame0.png, see docstring

DROPBOX_OUT = (
    "/Users/brighamhall/Library/CloudStorage/Dropbox-BlueDuckLLC/"
    "Brigham Hall/datamoshing-cv-lab/cv-lab-demos"
)


def extract_frames():
    os.makedirs(FRAMES_DIR, exist_ok=True)
    if glob.glob(f"{FRAMES_DIR}/*.jpg"):
        print(f"[skip] frames already extracted in {FRAMES_DIR}")
        return
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", SOURCE_CLIP,
         "-q:v", "2", "-start_number", "0", f"{FRAMES_DIR}/%05d.jpg"],
        check=True,
    )
    n = len(glob.glob(f"{FRAMES_DIR}/*.jpg"))
    print(f"[extract] {n} frames -> {FRAMES_DIR}")


def compute_masks():
    os.makedirs(MASKS_DIR, exist_ok=True)
    if glob.glob(f"{MASKS_DIR}/*.png"):
        print(f"[skip] masks already computed in {MASKS_DIR}")
        return
    t0 = time.time()
    empty_chunks = []

    def quality_gate(chunk_idx, alpha_maps_for_chunk):
        empty = sum(1 for m in alpha_maps_for_chunk if m is None or (m > 30).sum() == 0)
        rate = empty / len(alpha_maps_for_chunk)
        empty_chunks.append((chunk_idx, len(alpha_maps_for_chunk), empty, rate))
        print(f"[sam2] chunk {chunk_idx}: {len(alpha_maps_for_chunk)} frames, "
              f"{empty} empty ({rate*100:.1f}%)")
        first_mask = alpha_maps_for_chunk[0]
        if first_mask is None or (first_mask > 30).sum() == 0:
            raise RuntimeError(f"chunk {chunk_idx} produced an empty first mask -- aborting")

    result = s2.compute_subject_mattes_chunked(
        SOURCE_CLIP, SEED_BOX, chunk_frames=300, quality_gate=quality_gate,
    )
    dt = time.time() - t0
    total = len(result.alpha_maps)
    total_empty = sum(1 for m in result.alpha_maps if m is None or (m > 30).sum() == 0)
    print(f"[sam2] DONE {total} frames in {dt:.1f}s ({dt/total:.2f}s/frame) "
          f"-- overall empty-mask rate {total_empty/total*100:.2f}%")

    for i, m in enumerate(result.alpha_maps):
        cv2.imwrite(os.path.join(MASKS_DIR, f"{i:05d}.png"), m)
    print(f"[sam2] wrote {total} masks -> {MASKS_DIR}")

    with open(os.path.join(WORK, "sam2_quality_gate.json"), "w") as f:
        json.dump({
            "total_frames": total,
            "wall_clock_s": round(dt, 1),
            "s_per_frame": round(dt / total, 3),
            "overall_empty_mask_rate_pct": round(total_empty / total * 100, 2),
            "chunks": [
                {"chunk": c, "frames": n, "empty": e, "rate_pct": round(r * 100, 2)}
                for c, n, e, r in empty_chunks
            ],
        }, f, indent=2)


def render_lasso():
    """Hand-rolled contour-lasso off the SAM2 masks already on disk --
    same recipe as matte_effects.draw_subject_lasso() (threshold at 127,
    3x3 open+close, largest external contour, approxPolyDP epsilon=0.5%
    arc length) but reading precomputed masks instead of calling rembg."""
    out_path = os.path.join(WORK, "whatsapp_pier_lasso_19s.mp4")
    frame_files = sorted(glob.glob(f"{FRAMES_DIR}/*.jpg"))
    mask_files = sorted(glob.glob(f"{MASKS_DIR}/*.png"))
    assert len(frame_files) == len(mask_files)

    first = cv2.imread(frame_files[0])
    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    tmp_path = out_path.replace(".mp4", "_raw.mp4")
    writer = cv2.VideoWriter(tmp_path, fourcc, FPS, (w, h))
    kernel = np.ones((3, 3), np.uint8)

    contours_missing = 0
    for ff, mf in zip(frame_files, mask_files):
        frame = cv2.imread(ff)
        mask = cv2.imread(mf, cv2.IMREAD_GRAYSCALE)
        _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        out = frame.copy()
        if contours:
            largest = max(contours, key=cv2.contourArea)
            eps = 0.005 * cv2.arcLength(largest, True)
            simplified = cv2.approxPolyDP(largest, eps, True)
            cv2.drawContours(out, [simplified], -1, (0, 255, 0), 3)
        else:
            contours_missing += 1
        le.draw_engine_label(out, le.ENGINE_LABEL, "viz: contour lasso (SAM2 mask)")
        writer.write(out)

    writer.release()
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", tmp_path,
         "-c:v", "libx264", "-pix_fmt", "yuv420p", out_path],
        check=True,
    )
    os.remove(tmp_path)
    print(f"[lasso] {len(frame_files)} frames, {contours_missing} with no contour -> {out_path}")
    return out_path


def main():
    extract_frames()
    compute_masks()

    results = {}

    print("[render] ascii thermal_drift ...")
    ascii_out = os.path.join(WORK, "whatsapp_pier_ascii_thermal_19s.mp4")
    results["ascii_thermal_drift"] = ab.render_ascii_background(
        FRAMES_DIR, MASKS_DIR, ascii_out, fps=FPS, palette="thermal_drift",
    )
    print(f"  -> {results['ascii_thermal_drift']}")

    print("[render] thermal_falsecolor ...")
    thermal_out = os.path.join(WORK, "whatsapp_pier_thermal_19s.mp4")
    results["thermal_falsecolor"] = le.render_layer_video(
        FRAMES_DIR, MASKS_DIR, "thermal_falsecolor", thermal_out, fps=FPS,
    )
    print(f"  -> {results['thermal_falsecolor']}")

    print("[render] chromatic_shift ...")
    chroma_out = os.path.join(WORK, "whatsapp_pier_chroma_19s.mp4")
    results["chromatic_shift"] = le.render_layer_video(
        FRAMES_DIR, MASKS_DIR, "chromatic_shift", chroma_out, fps=FPS,
    )
    print(f"  -> {results['chromatic_shift']}")

    print("[render] contour lasso ...")
    lasso_out = render_lasso()

    with open(os.path.join(WORK, "render_stats.json"), "w") as f:
        json.dump(results, f, indent=2)

    os.makedirs(DROPBOX_OUT, exist_ok=True)
    for src in [ascii_out, thermal_out, chroma_out, lasso_out]:
        dst = os.path.join(DROPBOX_OUT, os.path.basename(src))
        subprocess.run(["cp", src, dst], check=True)
        print(f"[dropbox] {dst}")

    print("\n=== RENDER STATS (honesty numbers) ===")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
