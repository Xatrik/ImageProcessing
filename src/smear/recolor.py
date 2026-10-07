"""Recolor a photo by remapping every pixel to its nearest color in a palette.

The palette can come from another image (every unique color it contains) or
from an explicit list of hex colors.
"""

import cv2
import numpy as np

from .pixelsort import _dither_pattern


def palette_from_image(image_bgr: np.ndarray) -> np.ndarray:
    """Every unique color in image_bgr, as an (n, 3) uint8 BGR array sorted by how
    much of the image it covers (most common first)."""
    pixels = image_bgr.reshape(-1, 3)
    colors, counts = np.unique(pixels, axis=0, return_counts=True)
    order = np.argsort(-counts)
    return colors[order].astype(np.uint8)


def palette_from_hex(hex_colors: list[str]) -> np.ndarray:
    """Parse '#RRGGBB' (or 'RRGGBB') strings into a (k, 3) uint8 BGR array."""
    colors = []
    for h in hex_colors:
        h = h.strip().lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        colors.append((b, g, r))
    return np.array(colors, dtype=np.uint8)


_BRUTE_FORCE_MAX_PALETTE = 8  # below this, a plain loop beats building a search tree


def _nearest_indices(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Index into `reference` of the nearest row to each row of `query` (both
    (n, 3) float32). Uses a KD-tree so this stays fast even with thousands of
    reference colors; falls back to brute force for a handful of colors, where
    building a tree isn't worth it."""
    if len(reference) <= _BRUTE_FORCE_MAX_PALETTE:
        best_dist = np.full(len(query), np.inf, dtype=np.float32)
        best_idx = np.zeros(len(query), dtype=np.int32)
        for i, color in enumerate(reference):
            dist = np.sum((query - color) ** 2, axis=-1)
            better = dist < best_dist
            best_dist = np.where(better, dist, best_dist)
            best_idx = np.where(better, i, best_idx)
        return best_idx

    index = cv2.flann.Index(reference, dict(algorithm=1, trees=4))  # FLANN_INDEX_KDTREE
    matches, _distances = index.knnSearch(query, 1, params={})
    return matches[:, 0]


def recolor_with_palette(image_bgr: np.ndarray, palette: np.ndarray, use_lab: bool = True,
                          dither: bool = False, dither_amplitude: float = 12.0) -> np.ndarray:
    """Replace every pixel with the closest color in `palette` (an (k, 3) uint8 BGR
    array — k can be anything up to every color in an image). Distance is
    measured in Lab space by default (perceptually closer matches than raw
    BGR), or BGR directly if use_lab is False. With dither=True, an ordered
    (Bayer) pattern perturbs each pixel before matching, breaking up hard
    banding between palette colors into a fine dithered pattern instead.

    Without dithering, only distinct pixel colors are searched once each (then
    broadcast back), which is fast even with thousands of unique colors —
    dithering makes every pixel's perturbed value unique, so that shortcut is
    skipped and every pixel is searched individually."""
    if use_lab:
        base = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        pal = cv2.cvtColor(palette.reshape(1, -1, 3), cv2.COLOR_BGR2LAB).astype(np.float32).reshape(-1, 3)
    else:
        base = image_bgr.astype(np.float32)
        pal = palette.astype(np.float32)

    height, width = image_bgr.shape[:2]

    if dither:
        noise = (_dither_pattern(height, width) * dither_amplitude)[..., None]
        pixels = (base + noise).reshape(-1, 3)
        nearest_per_pixel = _nearest_indices(pixels, pal)
    else:
        pixels = base.reshape(-1, 3)
        unique_pixels, inverse = np.unique(pixels, axis=0, return_inverse=True)
        nearest_per_unique = _nearest_indices(unique_pixels, pal)
        nearest_per_pixel = nearest_per_unique[inverse]

    return palette[nearest_per_pixel].reshape(height, width, 3)
