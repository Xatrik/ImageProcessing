"""CLI entry point: recolor photo(s) using a palette.

Every pixel of each input image is remapped to its nearest color in a
palette, which comes from (in order): --colors "#ff0000,#00ff00,..."
(explicit list), --palette path/to/image (every unique color that image
contains), or otherwise the first image found in ./palette/ (created
automatically if missing — just drop a reference image in there). Matching
is dithered by default (an ordered Bayer pattern, --no-dither disables it)
to break up hard banding between palette colors. Contrast correction and a
light film grain are applied afterward by default, the same finishing
touches pixelsort.py uses.

With no --input/--output, processes every image in ./photos into
./recolored (both created automatically if missing).
"""

import argparse
import os

import cv2

from .pixelsort import add_film_grain, adjust_contrast
from .recolor import palette_from_hex, palette_from_image, recolor_with_palette

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", "-i", default="photos", help="input image file, or a folder of images (default: ./photos)")
    p.add_argument("--output", "-o", default="recolored", help="output image path, or output folder if --input is a folder (default: ./recolored)")
    p.add_argument("--palette", type=str, default=None, help="reference image to extract the palette from (every unique color it contains)")
    p.add_argument("--palette-dir", type=str, default="palette", help="fallback folder to look for a reference image in, if neither --palette nor --colors is given (default: ./palette)")
    p.add_argument("--colors", type=str, default=None, help="comma-separated explicit hex colors instead of --palette, e.g. '#ff0000,#00ff00,#0000ff'")
    p.add_argument("--no-lab", dest="use_lab", action="store_false", default=True,
                    help="match colors in raw BGR instead of perceptual Lab space")
    p.add_argument("--no-dither", dest="dither", action="store_false", default=True,
                    help="don't dither — hard color bands between palette colors")
    p.add_argument("--dither-amplitude", type=float, default=12.0, help="strength of the dither pattern, only used unless --no-dither")
    p.add_argument("--no-contrast-adjust", dest="contrast_adjust", action="store_false", default=True,
                    help="don't restore normal-photo contrast at the end (CLAHE on lightness)")
    p.add_argument("--contrast-clip-limit", type=float, default=2.0, help="CLAHE clip limit, only used unless --no-contrast-adjust (higher = stronger contrast boost)")
    p.add_argument("--contrast-tile-size", type=int, default=1, help="CLAHE tile grid size (pixels), only used unless --no-contrast-adjust")
    p.add_argument("--no-film-grain", dest="film_grain", action="store_false", default=True,
                    help="don't sprinkle a light monochromatic noise over the final image")
    p.add_argument("--film-grain-amount", type=float, default=15.0, help="strength of the grain noise, only used unless --no-film-grain")
    return p.parse_args(argv)


def _load_palette(args: argparse.Namespace):
    if args.colors:
        return palette_from_hex(args.colors.split(","))

    if args.palette:
        ref = cv2.imread(args.palette)
        if ref is None:
            raise FileNotFoundError(f"could not read palette image: {args.palette}")
        palette = palette_from_image(ref)
        print(f"using {len(palette)} unique colors from {args.palette}")
        return palette

    if not os.path.exists(args.palette_dir):
        os.makedirs(args.palette_dir, exist_ok=True)
        print(f"{args.palette_dir} didn't exist, created it — drop a reference image in there and run this again")
        return None

    names = sorted(n for n in os.listdir(args.palette_dir) if os.path.splitext(n)[1].lower() in IMAGE_EXTS)
    if not names:
        print(f"no images found in {args.palette_dir} — drop a reference image in there (or pass --palette/--colors)")
        return None

    palette_path = os.path.join(args.palette_dir, names[0])
    ref = cv2.imread(palette_path)
    if ref is None:
        raise FileNotFoundError(f"could not read palette image: {palette_path}")
    palette = palette_from_image(ref)
    print(f"using {len(palette)} unique colors from {palette_path}")
    return palette


def _process_one(input_path: str, output_path: str, palette, args: argparse.Namespace) -> None:
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError(f"could not read image: {input_path}")

    result = recolor_with_palette(image, palette, args.use_lab, args.dither, args.dither_amplitude)
    if args.contrast_adjust:
        result = adjust_contrast(result, args.contrast_clip_limit, args.contrast_tile_size)
    if args.film_grain:
        result = add_film_grain(result, args.film_grain_amount)

    cv2.imwrite(output_path, result)
    print(f"  -> wrote {output_path}")


def main(argv=None) -> None:
    args = parse_args(argv)
    palette = _load_palette(args)
    if palette is None:
        return

    if not os.path.exists(args.input):
        if os.path.splitext(args.input)[1].lower() in IMAGE_EXTS:
            raise FileNotFoundError(f"input file not found: {args.input}")
        os.makedirs(args.input, exist_ok=True)
        print(f"{args.input} didn't exist, created it — drop images in there and run this again")
        return

    if os.path.isdir(args.input):
        os.makedirs(args.output, exist_ok=True)

        names = sorted(n for n in os.listdir(args.input) if os.path.splitext(n)[1].lower() in IMAGE_EXTS)
        if not names:
            print(f"no images found in {args.input}")
            return

        for name in names:
            in_path = os.path.join(args.input, name)
            out_path = os.path.join(args.output, name)
            print(f"processing {name}")
            try:
                _process_one(in_path, out_path, palette, args)
            except ValueError as exc:
                print(f"  !! skipped: {exc}")
    else:
        _process_one(args.input, args.output, palette, args)


if __name__ == "__main__":
    main()
