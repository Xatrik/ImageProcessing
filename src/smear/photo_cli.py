"""CLI entry point: photo(s) -> pixel sort.

By default the image is first split into light and dark pixels and each
class is blurred using only its own pixels (tune with
--region-blur-kernel, disable with --no-region-blur), then posterized to
its 10 most common colors (k-means clustering; tune with --quantize-k,
disable with --no-quantize-colors). Then every pixel is sorted along a
spiral winding out from the image center (tune how tight with
--spiral-turns) in one continuous run, except that a section restarts
wherever there's a big enough jump in hue between neighboring pixels (tune
with --hue-gap-threshold, disable with --no-hue-gap) OR a local-contrast
edge (tune with --contrast-kernel, --edge-threshold, disable with
--no-edges) — whichever comes first. Use --axis rows/columns/both/radial
for the other sort geometries instead.

With no arguments, processes every image in ./photos into ./sorted (both
created automatically if missing). --input can instead point at a single
image file, or a different folder — in which case every image in that
folder gets the effect applied and written into --output (created as a
folder too).
"""

import argparse
import os

import cv2

from .pixelsort import SortConfig, dominant_palette, edge_mask_preview, palette_image, pixel_sort

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def parse_args(argv=None) -> argparse.Namespace:
    # Every SortConfig-mapped flag below defaults to None, meaning "not set on the
    # command line." main() only forwards the ones that aren't None, so the actual
    # defaults always come from SortConfig itself (in pixelsort.py) — the single
    # source of truth. Edit values there and they take effect with no flags needed.
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", "-i", default="photos", help="input image file, or a folder of images (default: ./photos)")
    p.add_argument("--output", "-o", default="sorted", help="output image path, or output folder if --input is a folder (default: ./sorted)")
    p.add_argument("--region-blur", dest="region_blur", action="store_true", default=None, help="blur light/dark pixels separately using only their own class before sorting")
    p.add_argument("--no-region-blur", dest="region_blur", action="store_false", help="don't blur light/dark pixels separately using only their own class before sorting")
    p.add_argument("--region-blur-kernel", type=int, default=None, help="blur kernel size (pixels), only used unless --no-region-blur")
    p.add_argument("--quantize-colors", dest="quantize_colors", action="store_true", default=None, help="posterize to the K most common colors before sorting")
    p.add_argument("--no-quantize-colors", dest="quantize_colors", action="store_false", help="don't posterize to the K most common colors before sorting")
    p.add_argument("--quantize-k", type=int, default=None, help="number of colors to posterize down to, only used unless --no-quantize-colors")
    p.add_argument("--dither", dest="dither", action="store_true", default=None, help="ordered-dither the posterization so hard color bands break up into a fine pattern")
    p.add_argument("--no-dither", dest="dither", action="store_false", help="don't dither — hard color bands between posterized clusters")
    p.add_argument("--dither-amplitude", type=float, default=None, help="strength of the dither pattern, only used unless --no-dither")
    p.add_argument("--hue-gap", dest="use_hue_gap", action="store_true", default=None, help="start a new section on a big hue jump between neighboring pixels")
    p.add_argument("--no-hue-gap", dest="use_hue_gap", action="store_false", help="don't start a new section on a big hue jump between neighboring pixels")
    p.add_argument("--hue-gap-threshold", type=float, default=None, help="hue difference (0-179 scale) between neighboring pixels that starts a new section")
    p.add_argument("--edges", dest="use_edges", action="store_true", default=None, help="split on high-local-contrast regions")
    p.add_argument("--no-edges", dest="use_edges", action="store_false", help="don't also split on high-local-contrast regions")
    p.add_argument("--contrast-kernel", type=int, default=None, help="neighborhood size (pixels) local contrast is measured over, only used unless --no-edges")
    p.add_argument("--edge-threshold", type=float, default=None, help="local max-min contrast (0-255 scale) that counts as an edge, only used unless --no-edges (higher = fewer/stronger edges = longer, more aggressive sort streaks)")
    p.add_argument("--denoise", dest="denoise", action="store_true", default=None, help="smooth noise out of the grayscale before contrast-edge detection")
    p.add_argument("--no-denoise", dest="denoise", action="store_false", help="don't denoise before contrast-edge detection")
    p.add_argument("--denoise-d", type=int, default=None, help="bilateral filter neighborhood diameter, only used unless --no-denoise")
    p.add_argument("--denoise-sigma-color", type=float, default=None, help="bilateral filter color sigma, only used unless --no-denoise")
    p.add_argument("--denoise-sigma-space", type=float, default=None, help="bilateral filter space sigma, only used unless --no-denoise")
    p.add_argument("--mask-blur", type=float, default=None, help="blur radius applied to the detected edge mask before it's used to sort, thickening/softening the boundary band (0 disables)")
    p.add_argument("--mask-blur-threshold", type=float, default=None, help="re-threshold level (0-1, as a fraction of a fresh edge's blurred peak) applied to the blurred edge mask")
    p.add_argument("--key", choices=["brightness", "hue", "saturation", "red", "green", "blue"], default=None)
    p.add_argument("--axis", choices=["rows", "columns", "both", "radial", "spiral"], default=None)
    p.add_argument("--spiral-turns", type=float, default=None, help="full rotations the spiral makes from center to edge (only used with --axis spiral)")
    p.add_argument("--reverse", action="store_true", default=None, help="sort descending instead of ascending")
    p.add_argument("--min-length", type=int, default=None, help="skip sections shorter than this many pixels (1 = sort every section)")
    p.add_argument("--edge-blur", type=float, default=None, help="softens the hard seams where sorted sections meet (pixel radius; 0 disables)")
    p.add_argument("--contrast-adjust", dest="contrast_adjust", action="store_true", default=None, help="restore normal-photo contrast at the end (CLAHE on lightness)")
    p.add_argument("--no-contrast-adjust", dest="contrast_adjust", action="store_false", help="don't restore contrast at the end")
    p.add_argument("--contrast-clip-limit", type=float, default=None, help="CLAHE clip limit, only used unless --no-contrast-adjust (higher = stronger contrast boost)")
    p.add_argument("--contrast-tile-size", type=int, default=None, help="CLAHE tile grid size (pixels), only used unless --no-contrast-adjust")
    p.add_argument("--film-grain", dest="film_grain", action="store_true", default=None, help="sprinkle a light monochromatic noise over the final image")
    p.add_argument("--no-film-grain", dest="film_grain", action="store_false", help="don't add film grain")
    p.add_argument("--film-grain-amount", type=float, default=None, help="strength of the grain noise, only used unless --no-film-grain")
    p.add_argument("--save-edges", type=str, default=None,
                    help="optional path (or folder, in folder mode) to also write the combined section-boundary mask, for tuning contrast-kernel/threshold/hue-gap")
    p.add_argument("--no-palette", dest="save_palette", action="store_false", default=True,
                    help="don't also write each image's dominant-color palette as <name>_palette.png")
    p.add_argument("--palette-k", type=int, default=150, help="number of colors shown in the palette swatch")
    return p.parse_args(argv)


def _palette_path(output_path: str) -> str:
    base, _ext = os.path.splitext(output_path)
    return f"{base}_palette.png"


def _process_one(input_path: str, output_path: str, config: SortConfig, save_edges_path: str | None,
                  save_palette: bool, palette_k: int) -> None:
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError(f"could not read image: {input_path}")

    if save_edges_path:
        cv2.imwrite(save_edges_path, edge_mask_preview(image, config))
        print(f"  -> wrote {save_edges_path}")

    if save_palette:
        palette_path = _palette_path(output_path)
        cv2.imwrite(palette_path, palette_image(dominant_palette(image, palette_k)))
        print(f"  -> wrote {palette_path}")

    result = pixel_sort(image, config)
    cv2.imwrite(output_path, result)
    print(f"  -> wrote {output_path}")


def main(argv=None) -> None:
    args = parse_args(argv)

    non_config_keys = {"input", "output", "save_edges", "save_palette", "palette_k"}
    overrides = {k: v for k, v in vars(args).items() if k not in non_config_keys and v is not None}
    config = SortConfig(**overrides)

    if not os.path.exists(args.input):
        if os.path.splitext(args.input)[1].lower() in IMAGE_EXTS:
            raise FileNotFoundError(f"input file not found: {args.input}")
        os.makedirs(args.input, exist_ok=True)
        print(f"{args.input} didn't exist, created it — drop images in there and run this again")
        return

    if os.path.isdir(args.input):
        os.makedirs(args.output, exist_ok=True)
        if args.save_edges:
            os.makedirs(args.save_edges, exist_ok=True)

        names = sorted(n for n in os.listdir(args.input) if os.path.splitext(n)[1].lower() in IMAGE_EXTS)
        if not names:
            print(f"no images found in {args.input}")
            return

        for name in names:
            in_path = os.path.join(args.input, name)
            out_path = os.path.join(args.output, name)
            edges_path = os.path.join(args.save_edges, name) if args.save_edges else None
            print(f"processing {name}")
            try:
                _process_one(in_path, out_path, config, edges_path, args.save_palette, args.palette_k)
            except ValueError as exc:
                print(f"  !! skipped: {exc}")
    else:
        _process_one(args.input, args.output, config, args.save_edges, args.save_palette, args.palette_k)


if __name__ == "__main__":
    main()
