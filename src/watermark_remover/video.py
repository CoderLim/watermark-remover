from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Callable

import cv2
import numpy as np


def sample_video_frames(path: str | Path, sample_count: int = 18) -> list[np.ndarray]:
    path = str(path)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")

    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            raise RuntimeError("video reports no frames")

        count = max(4, min(sample_count, frame_count))
        indices = np.linspace(0, frame_count - 1, num=count, dtype=np.int64)

        frames: list[np.ndarray] = []
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if ok and frame is not None:
                frames.append(frame)

        if len(frames) < 4:
            raise RuntimeError("could not decode at least 4 sample frames")
        return frames
    finally:
        cap.release()


def video_metadata(path: str | Path) -> tuple[int, int, float, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if width <= 0 or height <= 0 or fps <= 0:
            raise RuntimeError("invalid video metadata")
        return width, height, fps, frames
    finally:
        cap.release()


def encode_processed_video(
    input_path: str | Path,
    output_path: str | Path,
    process_frame: Callable[[np.ndarray], np.ndarray],
    *,
    crf: int = 18,
    preset: str = "medium",
) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required on PATH")

    input_path = str(input_path)
    output_path = str(output_path)
    width, height, fps, _ = video_metadata(input_path)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {input_path}")

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s:v",
        f"{width}x{height}",
        "-r",
        f"{fps:.8f}",
        "-i",
        "pipe:0",
        "-i",
        input_path,
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-c:v",
        "libx264",
        "-crf",
        str(crf),
        "-preset",
        preset,
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        "-shortest",
        output_path,
    ]

    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    assert process.stdin is not None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            processed = process_frame(frame)
            if processed.shape != frame.shape:
                raise RuntimeError("frame processor changed frame dimensions")
            process.stdin.write(np.ascontiguousarray(processed).tobytes())
    except BrokenPipeError as exc:
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        raise RuntimeError(f"ffmpeg terminated early: {stderr}") from exc
    finally:
        cap.release()
        try:
            process.stdin.close()
        except BrokenPipeError:
            pass

    stderr_bytes = process.stderr.read() if process.stderr else b""
    return_code = process.wait()
    if return_code != 0:
        stderr = stderr_bytes.decode("utf-8", errors="replace")
        raise RuntimeError(f"ffmpeg failed with exit code {return_code}: {stderr}")
