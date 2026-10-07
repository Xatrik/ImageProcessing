"""Pixel sorting for still photos.

By default every pixel in each row (or column) gets sorted by a chosen key
(brightness, hue, saturation, or a single channel) in one continuous run —
except that a large jump in hue between neighboring pixels starts a fresh
section, so a sort doesn't bleed across a sharp color change. Optionally
(config.use_edges), a local-contrast mask also splits each row/column into
sections.
"""

from dataclasses import dataclass

import cv2
import numpy as np

HUE_RANGE = 180  # OpenCV's 8-bit HSV hue is 0-179


@dataclass
class SortConfig:
    contrast_kernel: int = 5
    edge_threshold: float = 20.0
    key: str = "brightness"
    axis: str = "columns"        # rows | columns | both | radial | spiral
    spiral_turns: float = 10.0   # how many full rotations the spiral makes from center to edge
    reverse: bool = False
    min_length: int = 1
    use_edges: bool = True      # if True, also split on high-local-contrast regions
    denoise: bool = True        # if True, smooth noise out of the grayscale before contrast-edge detection
    denoise_d: int = 5              # bilateral filter neighborhood diameter
    denoise_sigma_color: float = 50.0
    denoise_sigma_space: float = 50.0
    use_hue_gap: bool = False    # if True, split wherever neighboring pixels' hue jumps too much
    hue_gap_threshold: float = 25.0
    mask_blur: float = 0.0             # blur radius applied to the detected edge mask before it's used to sort; 0 disables
    mask_blur_threshold: float = 0.3   # re-threshold level, as a fraction of a fresh edge's blurred peak
    edge_blur: float = 0.0      # softens the hard seams where sorted sections meet; 0 disables
    quantize_colors: bool = True  # if True, posterize to the K most common colors before doing anything else
    quantize_k: int = 2
    dither: bool = True         # if True, ordered-dither the quantization so bands break up into a fine pattern
    dither_amplitude: float = 40.0
    region_blur: bool = True    # if True, blur light pixels using only other light pixels, and dark using only dark
    region_blur_kernel: int = 30
    contrast_adjust: bool = True  # if True, restore normal-photo contrast at the end (CLAHE on lightness)
    contrast_clip_limit: float = 2.0
    contrast_tile_size: int = 1
    film_grain: bool = True     # if True, sprinkle a light monochromatic noise over the final image
    film_grain_amount: float = 15.0


def _kmeans_clusters(image_bgr: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """k-means over the image's pixels. Returns (labels, centers) — centers are the
    k cluster-average BGR colors, labels assigns each pixel to one of them."""
    data = image_bgr.reshape((-1, 3)).astype(np.float32)
    k = max(1, min(k, len(data)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, attempts=3, flags=cv2.KMEANS_PP_CENTERS)
    return labels.flatten(), np.clip(centers, 0, 255).astype(np.uint8)


_BAYER_8X8 = np.array([
    [0, 32, 8, 40, 2, 34, 10, 42],
    [48, 16, 56, 24, 50, 18, 58, 26],
    [12, 44, 4, 36, 14, 46, 6, 38],
    [60, 28, 52, 20, 62, 30, 54, 22],
    [3, 35, 11, 43, 1, 33, 9, 41],
    [51, 19, 59, 27, 49, 17, 57, 25],
    [15, 47, 7, 39, 13, 45, 5, 37],
    [63, 31, 55, 23, 61, 29, 53, 21],
], dtype=np.float32)
_BAYER_8X8 = (_BAYER_8X8 + 0.5) / 64.0 - 0.5  # centered in [-0.5, 0.5)


def _dither_pattern(height: int, width: int) -> np.ndarray:
    """An (height, width) tiling of the 8x8 Bayer ordered-dither matrix, values in
    [-0.5, 0.5)."""
    reps_y, reps_x = height // 8 + 1, width // 8 + 1
    return np.tile(_BAYER_8X8, (reps_y, reps_x))[:height, :width]


def quantize_colors(image_bgr: np.ndarray, k: int, dither: bool = False, dither_amplitude: float = 12.0) -> np.ndarray:
    """Cluster the image's colors into k groups (k-means) and replace every pixel
    with its cluster's average color — indexes the image down to its k most
    common colors. With dither=True, an ordered (Bayer) dither pattern perturbs
    each pixel before assigning it to a color, breaking up hard banding between
    clusters into a fine dithered pattern instead."""
    labels, centers = _kmeans_clusters(image_bgr, k)

    if not dither:
        return centers[labels].reshape(image_bgr.shape)

    height, width = image_bgr.shape[:2]
    noise = (_dither_pattern(height, width) * dither_amplitude)[..., None]
    perturbed = np.clip(image_bgr.astype(np.float32) + noise, 0, 255)

    centers_f = centers.astype(np.float32)
    best_dist = np.full((height, width), np.inf, dtype=np.float32)
    best_idx = np.zeros((height, width), dtype=np.int32)
    for i, color in enumerate(centers_f):
        dist = np.sum((perturbed - color) ** 2, axis=-1)
        better = dist < best_dist
        best_dist = np.where(better, dist, best_dist)
        best_idx = np.where(better, i, best_idx)

    return centers[best_idx]


def dominant_palette(image_bgr: np.ndarray, k: int) -> list[tuple[np.ndarray, float]]:
    """The image's k most common colors (k-means cluster averages), as a list of
    (bgr_color, fraction_of_pixels) sorted most-common first."""
    labels, centers = _kmeans_clusters(image_bgr, k)
    counts = np.bincount(labels, minlength=len(centers))
    order = np.argsort(-counts)
    total = counts.sum()
    return [(centers[i], counts[i] / total) for i in order]


def palette_image(palette: list[tuple[np.ndarray, float]], width: int = 500, height: int = 80) -> np.ndarray:
    """Render a dominant_palette() result as a horizontal swatch bar, BGR uint8,
    each color's width proportional to how much of the image it covers."""
    out = np.zeros((height, width, 3), dtype=np.uint8)
    x = 0
    for i, (color, fraction) in enumerate(palette):
        is_last = i == len(palette) - 1
        w = (width - x) if is_last else int(round(fraction * width))
        out[:, x:x + w] = color
        x += w
    return out


def _masked_box_blur(image_f32: np.ndarray, mask_f32: np.ndarray, kernel_size: int) -> np.ndarray:
    """Box-blur image_f32, weighting each contributing pixel by mask_f32 (normalized
    convolution) — pixels outside the mask don't leak into the blurred result."""
    ksize = (kernel_size, kernel_size)
    weighted_sum = cv2.boxFilter(image_f32 * mask_f32[..., None], ddepth=-1, ksize=ksize, normalize=True)
    weight = cv2.boxFilter(mask_f32, ddepth=-1, ksize=ksize, normalize=True)
    return weighted_sum / np.maximum(weight[..., None], 1e-6)


def region_blur(image_bgr: np.ndarray, kernel_size: int) -> np.ndarray:
    """Split the image into light and dark pixels (Otsu threshold on brightness),
    then blur each class using only its own pixels — light areas smooth among
    themselves, dark areas smooth among themselves, without bleeding across the
    light/dark boundary."""
    kernel_size = max(1, kernel_size | 1)  # odd, >= 1
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, light_mask_u8 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    light_mask = light_mask_u8 > 0

    img_f = image_bgr.astype(np.float32)
    light_blurred = _masked_box_blur(img_f, light_mask.astype(np.float32), kernel_size)
    dark_blurred = _masked_box_blur(img_f, (~light_mask).astype(np.float32), kernel_size)

    out = np.where(light_mask[..., None], light_blurred, dark_blurred)
    return np.clip(out, 0, 255).astype(np.uint8)


def contrast_mask(gray: np.ndarray, kernel_size: int, threshold: float) -> np.ndarray:
    """Boolean edge mask from local contrast: the gap between the brightest and
    darkest pixel in each pixel's neighborhood (a morphological gradient)."""
    kernel_size = max(1, kernel_size | 1)  # odd, >= 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    local_max = cv2.dilate(gray, kernel).astype(np.float32)
    local_min = cv2.erode(gray, kernel).astype(np.float32)
    return (local_max - local_min) > threshold


def compute_key_map(image_bgr: np.ndarray, key: str) -> np.ndarray:
    """Per-pixel scalar sort key, same shape as image_bgr[..., 0]."""
    b = image_bgr[..., 0].astype(np.float32)
    g = image_bgr[..., 1].astype(np.float32)
    r = image_bgr[..., 2].astype(np.float32)

    if key == "brightness":
        return 0.114 * b + 0.587 * g + 0.299 * r
    if key == "red":
        return r
    if key == "green":
        return g
    if key == "blue":
        return b
    if key in ("hue", "saturation"):
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        return hsv[..., 0] if key == "hue" else hsv[..., 1]
    raise ValueError(f"unknown sort key: {key}")


def hue_gap_mask(image_bgr: np.ndarray, threshold: float, axis: int) -> np.ndarray:
    """Boolean mask, True where a pixel's hue differs from its predecessor along `axis`
    (1 = along a row / x-direction, 0 = along a column / y-direction) by more than
    `threshold`, accounting for hue wraparound (e.g. 179 next to 0 is a tiny gap)."""
    hue = compute_key_map(image_bgr, "hue")
    diff = np.abs(np.diff(hue, axis=axis))
    circular_diff = np.minimum(diff, HUE_RANGE - diff)
    boundary = circular_diff > threshold

    pad_shape = list(boundary.shape)
    pad_shape[axis] = 1
    pad = np.zeros(pad_shape, dtype=bool)
    return np.concatenate([pad, boundary], axis=axis)


def _boundary_mask(image_bgr: np.ndarray, config: SortConfig, direction_axis: int) -> np.ndarray:
    """Combined section-boundary mask for one sort pass (direction_axis: 1 for a
    row-wise/horizontal pass, 0 for a column-wise/vertical pass): find the edges,
    blur that mask to soften/thicken it, then re-threshold — so what actually
    drives the sort is a blurred edge band rather than a razor-thin line."""
    height, width = image_bgr.shape[:2]
    mask = np.zeros((height, width), dtype=bool)

    if config.use_edges:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        if config.denoise:
            gray = cv2.bilateralFilter(gray, config.denoise_d, config.denoise_sigma_color, config.denoise_sigma_space)
        mask |= contrast_mask(gray, config.contrast_kernel, config.edge_threshold)

    if config.use_hue_gap:
        mask |= hue_gap_mask(image_bgr, config.hue_gap_threshold, axis=direction_axis)

    if config.mask_blur > 0:
        blurred = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), config.mask_blur)
        peak = 1.0 / (config.mask_blur * np.sqrt(2 * np.pi))  # peak value a 1px-wide edge blurs down to
        mask = blurred > peak * config.mask_blur_threshold

    return mask


def _sort_lines(image: np.ndarray, boundary_mask: np.ndarray, key_map: np.ndarray,
                 reverse: bool, min_length: int) -> np.ndarray:
    """Sort pixels within each boundary-bounded run, along axis 1 (columns within a row)."""
    out = image.copy()
    height, width = boundary_mask.shape
    for y in range(height):
        boundaries = np.flatnonzero(boundary_mask[y])
        bounds = np.unique(np.concatenate(([0], boundaries, [width])))
        for i in range(len(bounds) - 1):
            start, end = int(bounds[i]), int(bounds[i + 1])
            if end - start < min_length:
                continue
            order = np.argsort(key_map[y, start:end])
            if reverse:
                order = order[::-1]
            out[y, start:end] = out[y, start:end][order]
    return out


def _polar_geometry(width: int, height: int) -> tuple[tuple[float, float], float, tuple[int, int]]:
    """Center, max radius, and polar-image (width, height) for warping this image size
    to/from polar coordinates, where polar rows = angle and polar columns = radius."""
    center = (width / 2.0, height / 2.0)
    max_radius = float(np.hypot(width / 2.0, height / 2.0))
    radius_res = max(1, int(round(max_radius)))
    angle_res = int(np.clip(round(2 * np.pi * max_radius), 360, 2000))
    return center, max_radius, (radius_res, angle_res)


def _to_polar(image_bgr: np.ndarray, center: tuple[float, float], max_radius: float, dsize: tuple[int, int]) -> np.ndarray:
    return cv2.warpPolar(image_bgr, dsize, center, max_radius, cv2.INTER_LINEAR + cv2.WARP_POLAR_LINEAR)


def _from_polar(polar_bgr: np.ndarray, center: tuple[float, float], max_radius: float, size: tuple[int, int]) -> np.ndarray:
    return cv2.warpPolar(polar_bgr, size, center, max_radius,
                          cv2.INTER_LINEAR + cv2.WARP_POLAR_LINEAR + cv2.WARP_INVERSE_MAP)


def _spiral_forward_maps(center: tuple[float, float], max_radius: float, dsize: tuple[int, int],
                          turns: float) -> tuple[np.ndarray, np.ndarray]:
    """map_x/map_y (shape angle_res x radius_res) sampling Cartesian source pixels for
    each (angle, radius) position along a spiral: angle twists by `turns` full
    rotations as radius goes from 0 to max_radius, so each "row" traces a spiral arm
    instead of a straight ray."""
    radius_res, angle_res = dsize
    rows = np.arange(angle_res, dtype=np.float32).reshape(-1, 1)
    cols = np.arange(radius_res, dtype=np.float32).reshape(1, -1)
    theta = 2 * np.pi * rows / angle_res + turns * 2 * np.pi * (cols / radius_res)
    r = max_radius * cols / radius_res
    map_x = (center[0] + r * np.cos(theta)).astype(np.float32)
    map_y = (center[1] + r * np.sin(theta)).astype(np.float32)
    return map_x, map_y


def _spiral_inverse_maps(width: int, height: int, center: tuple[float, float], max_radius: float,
                          dsize: tuple[int, int], turns: float) -> tuple[np.ndarray, np.ndarray]:
    """map_x/map_y (shape height x width) sampling spiral-space (col=radius, row=angle)
    source pixels for each Cartesian output pixel — the inverse of _spiral_forward_maps."""
    radius_res, angle_res = dsize
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    dx = xs - center[0]
    dy = ys - center[1]
    r = np.hypot(dx, dy)
    theta = np.mod(np.arctan2(dy, dx), 2 * np.pi)

    col = np.clip(r / max_radius * radius_res, 0, radius_res - 1)
    row = np.mod((theta - turns * 2 * np.pi * (col / radius_res)) / (2 * np.pi) * angle_res, angle_res)
    return col.astype(np.float32), row.astype(np.float32)


def _to_spiral(image_bgr: np.ndarray, center: tuple[float, float], max_radius: float,
                dsize: tuple[int, int], turns: float) -> np.ndarray:
    map_x, map_y = _spiral_forward_maps(center, max_radius, dsize, turns)
    return cv2.remap(image_bgr, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _from_spiral(spiral_bgr: np.ndarray, width: int, height: int, center: tuple[float, float],
                  max_radius: float, dsize: tuple[int, int], turns: float) -> np.ndarray:
    map_x, map_y = _spiral_inverse_maps(width, height, center, max_radius, dsize, turns)
    return cv2.remap(spiral_bgr, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _diagonal_kernel(length: int) -> np.ndarray:
    """A normalized streak kernel along the main diagonal (top-left to bottom-right)."""
    length = max(3, length | 1)  # odd, >= 3
    kernel = np.eye(length, dtype=np.float32)
    return kernel / kernel.sum()


def soften_edges(image_bgr: np.ndarray, config: SortConfig, radius: float) -> np.ndarray:
    """Feather a light diagonal blur into the image right around each detected
    section boundary, so the hard seams between sorted runs soften a bit while
    the interior of each streak stays sharp."""
    if radius <= 0:
        return image_bgr

    boundary = _boundary_mask(image_bgr, config, direction_axis=1) | _boundary_mask(image_bgr, config, direction_axis=0)
    alpha = cv2.GaussianBlur(boundary.astype(np.float32), (0, 0), radius)
    alpha = np.clip(alpha * 3.0, 0.0, 1.0)[..., None]

    kernel = _diagonal_kernel(int(round(radius)) * 2 + 1)
    blurred = cv2.filter2D(image_bgr, -1, kernel).astype(np.float32)
    out = blurred * alpha + image_bgr.astype(np.float32) * (1.0 - alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def adjust_contrast(image_bgr: np.ndarray, clip_limit: float, tile_size: int) -> np.ndarray:
    """Restore normal-photo contrast via CLAHE on the Lab lightness channel only —
    boosts local contrast/dynamic range without skewing hue or saturation, useful
    after steps (region blur, quantization) that tend to flatten the image out."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def add_film_grain(image_bgr: np.ndarray, amount: float) -> np.ndarray:
    """Sprinkle a light monochromatic noise over the image — the same random offset
    on all three channels per pixel, like film grain, rather than independent
    per-channel noise which reads as color static."""
    if amount <= 0:
        return image_bgr
    noise = np.random.normal(0.0, amount, image_bgr.shape[:2]).astype(np.float32)
    grainy = image_bgr.astype(np.float32) + noise[..., None]
    return np.clip(grainy, 0, 255).astype(np.uint8)


def _finalize(image_bgr: np.ndarray, config: SortConfig) -> np.ndarray:
    """Shared post-processing applied after sorting, regardless of axis mode."""
    out = soften_edges(image_bgr, config, config.edge_blur)
    if config.contrast_adjust:
        out = adjust_contrast(out, config.contrast_clip_limit, config.contrast_tile_size)
    if config.film_grain:
        out = add_film_grain(out, config.film_grain_amount)
    return out


def pixel_sort(image_bgr: np.ndarray, config: SortConfig | None = None) -> np.ndarray:
    """Return a pixel-sorted copy of image_bgr."""
    config = config or SortConfig()

    if config.region_blur:
        image_bgr = region_blur(image_bgr, config.region_blur_kernel)

    if config.quantize_colors:
        image_bgr = quantize_colors(image_bgr, config.quantize_k, config.dither, config.dither_amplitude)

    if config.axis == "radial":
        height, width = image_bgr.shape[:2]
        center, max_radius, dsize = _polar_geometry(width, height)
        polar = _to_polar(image_bgr, center, max_radius, dsize)

        boundary_mask = _boundary_mask(polar, config, direction_axis=1)
        key_map = compute_key_map(polar, config.key)
        polar_sorted = _sort_lines(polar, boundary_mask, key_map, config.reverse, config.min_length)

        out = _from_polar(polar_sorted, center, max_radius, (width, height))
        return _finalize(out, config)

    if config.axis == "spiral":
        height, width = image_bgr.shape[:2]
        center, max_radius, dsize = _polar_geometry(width, height)
        spiral = _to_spiral(image_bgr, center, max_radius, dsize, config.spiral_turns)

        boundary_mask = _boundary_mask(spiral, config, direction_axis=1)
        key_map = compute_key_map(spiral, config.key)
        spiral_sorted = _sort_lines(spiral, boundary_mask, key_map, config.reverse, config.min_length)

        out = _from_spiral(spiral_sorted, width, height, center, max_radius, dsize, config.spiral_turns)
        return _finalize(out, config)

    out = image_bgr
    if config.axis in ("rows", "both"):
        boundary_mask = _boundary_mask(out, config, direction_axis=1)
        key_map = compute_key_map(out, config.key)
        out = _sort_lines(out, boundary_mask, key_map, config.reverse, config.min_length)

    if config.axis in ("columns", "both"):
        boundary_mask_v = _boundary_mask(out, config, direction_axis=0)
        key_map_t = compute_key_map(out, config.key).T
        out_t = _sort_lines(out.transpose(1, 0, 2), boundary_mask_v.T, key_map_t, config.reverse, config.min_length)
        out = out_t.transpose(1, 0, 2)

    return _finalize(out, config)


def edge_mask_preview(image_bgr: np.ndarray, config: SortConfig | None = None) -> np.ndarray:
    """Visualize the combined section-boundary mask as a white-on-black BGR image, for
    tuning contrast-kernel/threshold/hue-gap."""
    config = config or SortConfig()

    if config.region_blur:
        image_bgr = region_blur(image_bgr, config.region_blur_kernel)

    if config.quantize_colors:
        image_bgr = quantize_colors(image_bgr, config.quantize_k, config.dither, config.dither_amplitude)

    if config.axis == "radial":
        height, width = image_bgr.shape[:2]
        center, max_radius, dsize = _polar_geometry(width, height)
        polar = _to_polar(image_bgr, center, max_radius, dsize)
        mask = _boundary_mask(polar, config, direction_axis=1)
        mask_img = (mask.astype(np.uint8) * 255)
        mask_img = _from_polar(cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR), center, max_radius, (width, height))
        return mask_img

    if config.axis == "spiral":
        height, width = image_bgr.shape[:2]
        center, max_radius, dsize = _polar_geometry(width, height)
        spiral = _to_spiral(image_bgr, center, max_radius, dsize, config.spiral_turns)
        mask = _boundary_mask(spiral, config, direction_axis=1)
        mask_img = (mask.astype(np.uint8) * 255)
        mask_img = _from_spiral(cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR), width, height, center, max_radius, dsize, config.spiral_turns)
        return mask_img

    mask = _boundary_mask(image_bgr, config, direction_axis=1)
    return cv2.cvtColor((mask.astype(np.uint8) * 255), cv2.COLOR_GRAY2BGR)
