"""CLI entry point: video -> long-exposure motion-smear composite."""

import argparse

import cv2
from tqdm import tqdm

from .smear import LightPaintCompositor, SmearConfig
from .video_io import VideoWriter, read_frames, video_properties


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", "-i", required=True, help="input video path")
    p.add_argument("--output", "-o", required=True, help="output composite image path (e.g. out.png)")
    p.add_argument("--threshold", type=float, default=1.0, help="min per-pixel flow magnitude (px) to smear")
    p.add_argument("--min-steps", type=int, default=2, help="trail samples for barely-moving pixels")
    p.add_argument("--max-steps", type=int, default=20, help="cap on trail samples for fast motion")
    p.add_argument("--flow-method", choices=["farneback", "dis"], default="farneback")
    p.add_argument("--resize", type=str, default=None, help="WxH, e.g. 960x540")
    p.add_argument("--max-frames", type=int, default=None, help="stop after N frames")
    p.add_argument("--preview-video", type=str, default=None, help="optional path to write the composite building up over time")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    resize = None
    if args.resize:
        w, h = args.resize.lower().split("x")
        resize = (int(w), int(h))

    props = video_properties(args.input)
    width, height = resize if resize else (props["width"], props["height"])

    config = SmearConfig(
        threshold=args.threshold,
        min_steps=args.min_steps,
        max_steps=args.max_steps,
        flow_method=args.flow_method,
    )
    compositor = LightPaintCompositor(width, height, config)

    total = args.max_frames or props["frame_count"] or None
    preview_writer = None
    if args.preview_video:
        preview_writer = VideoWriter(args.preview_video, props["fps"], (width, height))

    try:
        for frame in tqdm(read_frames(args.input, resize=resize, max_frames=args.max_frames), total=total, desc="smearing"):
            compositor.process_frame(frame)
            if preview_writer is not None:
                preview_writer.write(compositor.composite())
    finally:
        if preview_writer is not None:
            preview_writer.release()

    result = compositor.composite()
    cv2.imwrite(args.output, result)
    print(f"wrote {args.output}")
    if args.preview_video:
        print(f"wrote {args.preview_video}")


if __name__ == "__main__":
    main()
