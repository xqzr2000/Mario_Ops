"""
MP4 encoding and annotated highlight frames.

Deliberately a near-copy of the equivalent functions in
Mario_AWS/play.py rather than an import: the two projects sit in
different folders with different requirements files, and a shared module
would mean Mario_AWS could not be built or moved without considering
this experiment. The duplication is about 60 lines and buys full
independence.

The one thing that MUST stay identical is the output contract:

    runs/run_0001/run.mp4        H.264 / yuv420p, browser-previewable
    runs/run_0001/summary.json
    runs/run_0001/frames/*.png   sparse, annotated

`run.mp4` is a fixed name because Mario_AWS/cloud/storage.py's
upload_run() looks for exactly that. This project does not upload to S3
today, but keeping the name means it can, later, with no changes here.
"""

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np


def encode_video(frames_rgb: list, output_path: Path, fps: int) -> None:
    """Encode raw RGB frames to H.264/yuv420p via ffmpeg.

    ffmpeg is installed in Dockerfile.develop specifically for this.
    The OpenCV mp4v fallback exists only so a run on a machine without
    ffmpeg still produces *something*; that file will not play in a
    browser or in the S3/Drive preview, which looks like a broken model
    rather than a missing codec -- hence the loud warning.
    """
    if not frames_rgb:
        print("No frames captured; video was not created.")
        return

    height, width, _ = frames_rgb[0].shape

    if shutil.which("ffmpeg"):
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}", "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(output_path),
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for frame in frames_rgb:
            proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError("ffmpeg encoding failed")
    else:
        print("WARNING: ffmpeg not found -- falling back to mp4v; "
              "the clip will not preview in browsers. "
              "Add ffmpeg to the container image.")
        writer = cv2.VideoWriter(
            str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
            fps, (width, height),
        )
        for frame in frames_rgb:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()

    print(f"Video saved to: {output_path}")


def write_highlights(frames_rgb: list, meta: list, frames_dir: Path,
                     every_n: int) -> None:
    """Sparse annotated PNGs, for debugging a run after the fact.

    The overlay carries the DECISION INDEX, not just the frame index --
    that is the number you look up in trace.jsonl to see the exact
    prompt and reply that produced this moment.
    """
    frames_dir.mkdir(parents=True, exist_ok=True)

    for i in range(0, len(frames_rgb), every_n):
        m = meta[i]
        frame_bgr = cv2.cvtColor(frames_rgb[i], cv2.COLOR_RGB2BGR)
        line1 = f"d{m['decision']:03d} x{m['x_pos']:04d} t{m['time']}"
        line2 = f"{m['action_name']}"
        cv2.putText(frame_bgr, line1, (6, 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame_bgr, line2, (6, 32), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(
            str(frames_dir / f"frame_{i:04d}_x{m['x_pos']:04d}.png"),
            frame_bgr,
        )

    print(f"Highlight frames saved to: {frames_dir}")


def next_run_dir(runs_root: Path) -> Path:
    """Allocate runs/run_NNNN, probing upward past gaps.

    Same logic as Mario_AWS/play.py and for the same reason: counting
    existing folders collides after a deletion (remove run_0003 and the
    next run is also 0003, silently merged). mkdir without exist_ok so a
    genuine race fails loudly instead of interleaving two runs.
    """
    runs_root.mkdir(parents=True, exist_ok=True)
    n = len(list(runs_root.glob("run_*"))) + 1
    while (runs_root / f"run_{n:04d}").exists():
        n += 1
    run_dir = runs_root / f"run_{n:04d}"
    run_dir.mkdir(parents=True)
    return run_dir
