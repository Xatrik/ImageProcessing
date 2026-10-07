"""Video frame reading/writing helpers."""

from collections.abc import Iterator

import cv2
import numpy as np


def read_frames(path: str, resize: tuple[int, int] | None = None, max_frames: int | None = None) -> Iterator[np.ndarray]:
    """Yield BGR frames from a video file, optionally resized and capped in count."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {path}")
    count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if resize is not None:
                frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)
            yield frame
            count += 1
            if max_frames is not None and count >= max_frames:
                break
    finally:
        cap.release()


def video_properties(path: str) -> dict:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {path}")
    props = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS) or 30.0,
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return props


class VideoWriter:
    """Thin wrapper around cv2.VideoWriter for writing a BGR frame sequence."""

    def __init__(self, path: str, fps: float, size: tuple[int, int]):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(path, fourcc, fps, size)
        if not self._writer.isOpened():
            raise IOError(f"could not open video writer for: {path}")

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def release(self) -> None:
        self._writer.release()

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.release()
