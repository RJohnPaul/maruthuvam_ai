from __future__ import annotations
from PIL import Image
import numpy as np
import cv2  # type: ignore


def _load_gray(path: str) -> np.ndarray:
    img = Image.open(path).convert("L")
    return np.array(img)


def is_grayscale_image(path: str, tol: float = 2.5) -> bool:
    """Return True if image is effectively grayscale (low channel variance).
    Loads color if present and compares channels. For robustness, we just check L-mode conversion stability.
    """
    try:
        im = Image.open(path)
        if im.mode in ("L", "I;16", "1"):
            return True
        im_rgb = im.convert("RGB")
        r, g, b = im_rgb.split()
        r, g, b = np.array(r, float), np.array(g, float), np.array(b, float)
        diff_rg = np.abs(r - g).mean()
        diff_gb = np.abs(g - b).mean()
        diff_rb = np.abs(r - b).mean()
        return (diff_rg + diff_gb + diff_rb) / 3.0 < tol
    except Exception:
        return True


def edge_density(path: str) -> float:
    """Canny edge density in [0,1]."""
    try:
        g = _load_gray(path)
        g = cv2.resize(g, (512, 512), interpolation=cv2.INTER_AREA)
        edges = cv2.Canny(g, 50, 150)
        return float((edges > 0).mean())
    except Exception:
        return 0.2


def is_likely_chest_xray(path: str) -> tuple[bool, str]:
    """Heuristic to reject non-chest X-rays (e.g., hand/forearm films).
    Relax color requirement (some X-rays are RGB). Use:
    - Aspect ratio roughly portrait/square [0.6, 1.6]
    - Moderate edge density (hands have many fine edges): threshold ~ 0.22
    - Dark field proportion within range (lungs occupy sizable dark area): ~ [0.12, 0.65]
    """
    try:
        im = Image.open(path)
        w, h = im.size
        ar = w / max(h, 1)
        g = _load_gray(path)
        ed = edge_density(path)
        # proportion of dark pixels
        dark_ratio = float((g < 70).mean())
        if not (0.6 <= ar <= 1.6):
            return False, f"Aspect ratio {ar:.2f} out of chest range"
        if ed > 0.22:
            return False, f"Edge density {ed:.2f} too high for chest"
        if not (0.12 <= dark_ratio <= 0.65):
            return False, f"Dark field ratio {dark_ratio:.2f} atypical for chest"
        return True, "looks like chest X-ray"
    except Exception as e:
        return False, f"error: {e}"


def is_likely_ultrasound(path: str) -> tuple[bool, str]:
    """Heuristic for ultrasound: grayscale and speckle-like texture (moderate local variance).
    We compute local std over patches and require mean std in range.
    """
    try:
        g = _load_gray(path)
        g = cv2.resize(g, (384, 384), interpolation=cv2.INTER_AREA)
        # local std via blur approximations
        g_f = g.astype(np.float32)
        mean = cv2.GaussianBlur(g_f, (11, 11), 0)
        sq = cv2.GaussianBlur(g_f * g_f, (11, 11), 0)
        var = np.maximum(sq - mean * mean, 0)
        std = np.sqrt(var + 1e-6)
        mstd = float(std.mean())
        # ultrasound tends to be grayscale with moderate std
        if not is_grayscale_image(path):
            return False, "Not grayscale"
        if not (8.0 <= mstd <= 38.0):
            return False, f"Local std {mstd:.1f} not in ultrasound range"
        return True, "looks like ultrasound"
    except Exception as e:
        return False, f"error: {e}"
