from __future__ import annotations
import numpy as np


def blit(dst: np.ndarray, src: np.ndarray, x: int, y: int) -> None:
    """Copy src onto dst at (x, y), clipping to dst bounds. In-place."""
    dw, dh = dst.shape[:2]
    sw, sh = src.shape[:2]

    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + sw, dw), min(y + sh, dh)

    if x0 >= x1 or y0 >= y1:
        return

    dst[x0:x1, y0:y1] = src[x0 - x : x1 - x, y0 - y : y1 - y]


def scale(img: np.ndarray, factor: int | tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour scale by an integer factor."""
    fx, fy = factor if isinstance(factor, tuple) else (factor, factor)
    w, h = img.shape[:2]
    return resize(img, (w * fx, h * fy))


def resize(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resize to (w, h)."""
    w, h = size
    sw, sh = img.shape[:2]
    xs = (np.arange(w) * sw / w).astype(np.int32)
    ys = (np.arange(h) * sh / h).astype(np.int32)
    return img[np.ix_(xs, ys)]
