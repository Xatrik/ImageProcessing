"""Watch a folder for dropped-in videos and auto-run the smear compositor on each."""

import argparse
import json
import os
import time
import traceback

import cv2

from .smear import LightPaintCompositor, SmearConfig
from .video_io import VideoWriter, read_frames, video_properties

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--watch-dir", "-w", default="watch", help="folder to drop videos into")
    p.add_argument("--output-dir", "-o", default="output", help="folder composites are written to")
    p.add_argument("--poll-interval", type=float, default=2.0, help="seconds between folder scans")
    p.add_argument("--threshold", type=float, default=1.0)
    p.add_argument("--min-steps", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument("--flow-method", choices=["farneback", "dis"], default="farneback")
    p.add_argument("--resize", type=str, default=None, help="WxH, e.g. 960x540")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--preview-video", action="store_true", help="also write a <name>_preview.mp4 of the composite building up")
    return p.parse_args(argv)


def _state_path(watch_dir: str) -> str:
    return os.path.join(watch_dir, ".smear_state.json")


def _load_state(watch_dir: str) -> dict:
    path = _state_path(watch_dir)
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_state(watch_dir: str, state: dict) -> None:
    with open(_state_path(watch_dir), "w") as f:
        json.dump(state, f, indent=2)


def _is_stable(path: str) -> bool:
    """True if the file's size isn't changing (i.e. a copy/download has finished)."""
    try:
        size1 = os.path.getsize(path)
        time.sleep(0.5)
        size2 = os.path.getsize(path)
        return size1 == size2 and size1 > 0
    except OSError:
        return False


def _process_video(path: str, output_dir: str, args: argparse.Namespace) -> None:
    stem = os.path.splitext(os.path.basename(path))[0]
    out_image = os.path.join(output_dir, f"{stem}.png")

    resize = None
    if args.resize:
        w, h = args.resize.lower().split("x")
        resize = (int(w), int(h))

    props = video_properties(path)
    width, height = resize if resize else (props["width"], props["height"])

    config = SmearConfig(
        threshold=args.threshold,
        min_steps=args.min_steps,
        max_steps=args.max_steps,
        flow_method=args.flow_method,
    )
    compositor = LightPaintCompositor(width, height, config)

    preview_writer = None
    if args.preview_video:
        preview_writer = VideoWriter(os.path.join(output_dir, f"{stem}_preview.mp4"), props["fps"], (width, height))

    try:
        for frame in read_frames(path, resize=resize, max_frames=args.max_frames):
            compositor.process_frame(frame)
            if preview_writer is not None:
                preview_writer.write(compositor.composite())
    finally:
        if preview_writer is not None:
            preview_writer.release()

    cv2.imwrite(out_image, compositor.composite())
    print(f"  -> wrote {out_image}")


def main(argv=None) -> None:
    args = parse_args(argv)
    os.makedirs(args.watch_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    state = _load_state(args.watch_dir)
    print(f"watching {os.path.abspath(args.watch_dir)} (drop a video in to process it)")
    print(f"composites written to {os.path.abspath(args.output_dir)}")

    try:
        while True:
            for name in sorted(os.listdir(args.watch_dir)):
                ext = os.path.splitext(name)[1].lower()
                if ext not in VIDEO_EXTS:
                    continue
                path = os.path.join(args.watch_dir, name)
                if not os.path.isfile(path):
                    continue

                try:
                    stat = os.stat(path)
                except OSError:
                    continue
                fingerprint = f"{stat.st_mtime}:{stat.st_size}"

                if state.get(name) == fingerprint:
                    continue  # already processed this exact file
                if not _is_stable(path):
                    continue  # still being copied/downloaded, check again next poll

                print(f"[{time.strftime('%H:%M:%S')}] processing {name}")
                try:
                    _process_video(path, args.output_dir, args)
                except Exception:
                    print(f"  !! failed to process {name}:")
                    traceback.print_exc()

                state[name] = fingerprint
                _save_state(args.watch_dir, state)

            time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
