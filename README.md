# Motion Smear

Turns a video into a single long-exposure "light painting" image: dense
optical flow gives every pixel a per-frame motion vector, and pixels moving
faster than a threshold get smeared into a persistent trail along that
vector. Static/background pixels are averaged into the backdrop instead.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python main.py --input path/to/video.mp4 --output composite.png
```

### Or just drop videos into a folder

```bash
python watch.py
```

This watches `./watch/` and writes a composite to `./output/` for every
video dropped in (checks every 2s, waits for the file size to stop changing
so a still-copying file isn't grabbed mid-write). Leave it running in a
terminal and just drag videos into `watch/`. Already-processed files are
tracked in `watch/.smear_state.json` so restarting the watcher won't redo
them; dropping in a new file with the same name reprocesses it. Use
`--watch-dir` / `--output-dir` to point at different folders, and
`--preview-video` to also get a `<name>_preview.mp4` of the composite
building up. All the `main.py` tuning flags (`--threshold`, `--resize`,
etc.) work here too.

Useful flags:

- `--threshold` (default 1.0): minimum per-pixel flow magnitude in pixels to
  count as "moving" and get smeared. Lower catches subtler motion but picks
  up more noise; higher isolates only fast subjects.
- `--min-steps` / `--max-steps`: how many samples are drawn along a moving
  pixel's trail per frame (auto-scaled by how far it moved, clamped to this
  range).
- `--flow-method`: `farneback` (default, no extra deps) or `dis` (OpenCV's
  DIS optical flow, sharper but needs `opencv-contrib-python`).
- `--resize WxH`: downscale before processing (faster, smaller flow noise).
- `--max-frames N`: only process the first N frames.
- `--preview-video out.mp4`: also write a video of the composite building up
  frame by frame, useful for seeing the trails accumulate over time.

## Photo mode: pixel sorting

Turns a still photo into a pixel-sorting glitch image. By default every
pixel is sorted along a spiral winding out from the image center, in one
continuous run — except that a section restarts fresh wherever there's a
big jump in hue between neighboring pixels, OR a local-contrast edge,
whichever comes first — so the sorted streaks stay contained around real
object boundaries instead of bleeding across the whole image.

```bash
python pixelsort.py
```

With no arguments it just processes everything in `./photos/` (created
automatically the first run if it doesn't exist yet — drop images in and
run again) and writes results, same filenames, into `./sorted/`. Point
`--input`/`--output` at a specific file or a different folder to override:

```bash
python pixelsort.py --input photo.jpg --output result.png
python pixelsort.py --input other_photos/ --output other_sorted/
```

Every run also writes each photo's dominant-color palette next to its
output, as `<name>_palette.png` — a proportional swatch bar of its most
common colors, widest-first. `--no-palette` disables this, `--palette-k`
changes how many colors it shows (default 10).

Most flags below are `SortConfig` fields in `src/smear/pixelsort.py` —
their defaults live there, not in the CLI, so editing a default there takes
effect immediately with no flag needed. Run `--help` for the current
defaults. Useful flags:

- `--axis`: `spiral` (default, winds out from center), `radial` (straight
  rays from center), `rows`, `columns`, or `both`.
- `--spiral-turns` (default 3.0): full rotations the spiral makes from
  center to edge — only used with `--axis spiral`.
- `--hue-gap-threshold` (default 25.0): hue difference (0-179 scale)
  between neighboring pixels that starts a new section. `--no-hue-gap`
  disables this boundary source.
- `--contrast-kernel` (default 5) / `--edge-threshold` (default 20.0):
  local-contrast edge detection — the neighborhood size contrast is
  measured over, and how strong that contrast must be to count as an edge.
  Higher threshold = fewer/stronger edges = longer, more aggressive sort
  streaks. `--no-edges` disables this boundary source.
- `--mask-blur` (default 3.0) / `--mask-blur-threshold` (default 0.3):
  blurs the detected edge mask before it's used to sort, thickening/
  softening the boundary band instead of a razor-thin line. `0` disables.
- `--edge-blur` (default 2.0): after sorting, feathers a light diagonal
  blur into the hard seams where sorted sections meet. `0` disables.
- `--key`: `brightness` (default), `hue`, `saturation`, `red`, `green`, or
  `blue` — what each section gets sorted by.
- `--reverse`: sort descending instead of ascending.
- `--min-length`: skip sections shorter than this many pixels (default 1,
  i.e. every section gets sorted).
- `--save-edges out.png`: also dump the combined section-boundary mask,
  handy for tuning the edge/hue-gap/blur flags before committing to a full
  sort.

## Photo mode: recolor with a palette

Remaps every pixel of a photo to its nearest color in a palette — either
every unique color found in another image, or an explicit list you give it.

```bash
python recolor.py
```

With no arguments it processes `./photos/` into `./recolored/` (same as
`pixelsort.py`), using a palette taken from the first image dropped into
`./palette/` (created automatically the first run if it doesn't exist —
drop a reference image in and run again). That reference can be anything,
including one of the `_palette.png` swatches `pixelsort.py` writes, so you
can lift a look from one photo and apply it to another. Override with:

```bash
python recolor.py --palette reference.jpg
python recolor.py --colors "#ff0000,#00ff00,#0000ff,#ffffff,#111111"
```

`--palette path/to/image` extracts every unique color that image contains
instead of using `./palette/` — feed it one of `pixelsort.py`'s flat-color
`_palette.png` swatches to get exactly the colors shown there. `--colors
"#rrggbb,..."` uses an exact palette instead of extracting one. Matching is
done in perceptual Lab color space by default;
`--no-lab` switches to raw BGR distance.

## How it works

1. Dense optical flow (`cv2.calcOpticalFlowFarneback` or DIS) is computed
   between each consecutive frame pair, giving a backward displacement
   vector per pixel — where that pixel's content came from a moment ago.
2. Pixels whose flow magnitude exceeds `--threshold` are treated as moving.
   For those pixels, the current frame is resampled (`cv2.remap`) at several
   points along the vector and max-blended into a trail canvas, painting a
   streak the length and direction of the true motion.
3. Every frame also feeds a running mean, used as the static background.
4. The final image overlays the trail canvas onto the background wherever
   any motion was ever recorded at that pixel.
