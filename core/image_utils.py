from pathlib import Path
import io

import numpy as np
from PIL import Image


def load_image_as_array(path: str | Path) -> np.ndarray:
    """Load image file → return RGB numpy array (H, W, 3, uint8)."""
    img = Image.open(path).convert("RGB")
    return np.array(img)


def get_file_size_mb(path: str | Path) -> float:
    """Return actual file size in MB."""
    p = Path(path)
    return p.stat().st_size / (1024 * 1024)


def save_array_to_jpeg_bytes(arr: np.ndarray, quality: int = 90) -> bytes:
    """
    Convert numpy array to JPEG format (stored in memory, not written to disk),
    return byte string.
    """
    arr = np.clip(arr, 0, 255).astype("uint8")
    img = Image.fromarray(arr)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def estimate_jpeg_size_mb(arr: np.ndarray, quality: int = 90) -> float:
    """
    Estimate the file size in MB if the image is saved as JPEG with the specified quality.
    Converts to actual JPEG bytes in memory, so the estimate is accurate.
    """
    data = save_array_to_jpeg_bytes(arr, quality=quality)
    return len(data) / (1024 * 1024)