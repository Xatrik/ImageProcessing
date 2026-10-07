"""Dense optical flow computation between consecutive frames."""

import cv2
import numpy as np


def backward_flow(prev_gray: np.ndarray, curr_gray: np.ndarray, method: str = "farneback") -> np.ndarray:
    """Compute, for each pixel of curr_gray, the displacement vector pointing back
    to where that pixel's content came from in prev_gray.

    This is the direction a moving pixel trails behind itself, which is what we
    want to paint the smear streak along.

    Returns an (H, W, 2) float32 array of (dx, dy) vectors.
    """
    if method == "dis":
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        flow = dis.calc(curr_gray, prev_gray, None)
    elif method == "farneback":
        flow = cv2.calcOpticalFlowFarneback(
            curr_gray, prev_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )
    else:
        raise ValueError(f"unknown flow method: {method}")
    return flow
