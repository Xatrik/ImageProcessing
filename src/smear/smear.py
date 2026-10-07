"""Long-exposure light-painting compositor.

For each frame, dense optical flow gives a per-pixel motion vector. Pixels
whose motion magnitude exceeds a threshold are "smeared": the pixel's local
neighborhood along its motion vector is sampled and max-blended into a
persistent trail canvas, so fast-moving subjects leave a streak whose length
and direction track their real motion. Slow/static pixels instead feed a
running mean that becomes the background plate. The final composite overlays
the trail wherever anything was ever smeared, on top of the background.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from .flow import backward_flow


@dataclass
class SmearConfig:
    threshold: float = 1.0       # min flow magnitude (px) to count as "moving"
    min_steps: int = 2           # trail samples for barely-moving pixels
    max_steps: int = 20          # cap on trail samples for very fast motion
    flow_method: str = "farneback"


class LightPaintCompositor:
    def __init__(self, width: int, height: int, config: SmearConfig | None = None):
        self.width = width
        self.height = height
        self.config = config or SmearConfig()

        self._bg_sum = np.zeros((height, width, 3), dtype=np.float64)
        self._bg_count = 0
        self._trail = np.zeros((height, width, 3), dtype=np.float32)
        self._painted = np.zeros((height, width), dtype=bool)

        x_grid, y_grid = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
        self._x_grid = x_grid
        self._y_grid = y_grid

        self._prev_gray: np.ndarray | None = None

    def process_frame(self, frame_bgr: np.ndarray) -> None:
        """Feed the next frame (BGR, uint8) into the compositor."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        frame_f = frame_bgr.astype(np.float32)

        self._bg_sum += frame_bgr.astype(np.float64)
        self._bg_count += 1

        if self._prev_gray is not None:
            flow = backward_flow(self._prev_gray, gray, method=self.config.flow_method)
            magnitude = np.linalg.norm(flow, axis=-1)
            motion_mask = magnitude > self.config.threshold

            if np.any(motion_mask):
                max_mag = float(magnitude[motion_mask].max())
                steps = int(np.clip(np.ceil(max_mag), self.config.min_steps, self.config.max_steps))

                for t in np.linspace(0.0, 1.0, steps):
                    map_x = (self._x_grid + t * flow[..., 0]).astype(np.float32)
                    map_y = (self._y_grid + t * flow[..., 1]).astype(np.float32)
                    warped = cv2.remap(
                        frame_f, map_x, map_y,
                        interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
                    )
                    self._trail[motion_mask] = np.maximum(self._trail[motion_mask], warped[motion_mask])

                self._painted |= motion_mask

        self._prev_gray = gray

    def composite(self) -> np.ndarray:
        """Return the current composite as a BGR uint8 image."""
        if self._bg_count == 0:
            background = np.zeros((self.height, self.width, 3), dtype=np.float32)
        else:
            background = (self._bg_sum / self._bg_count).astype(np.float32)

        out = background.copy()
        out[self._painted] = self._trail[self._painted]
        return np.clip(out, 0, 255).astype(np.uint8)
