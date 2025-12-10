from pathlib import Path
import io

import numpy as np
from PIL import Image


def load_image_as_array(path: str | Path) -> np.ndarray:
    """讀檔案 → 回傳 RGB numpy array（H, W, 3, uint8）"""
    img = Image.open(path).convert("RGB")
    return np.array(img)


def get_file_size_mb(path: str | Path) -> float:
    """回傳檔案實際大小（MB）。"""
    p = Path(path)
    return p.stat().st_size / (1024 * 1024)


def save_array_to_jpeg_bytes(arr: np.ndarray, quality: int = 90) -> bytes:
    """把 numpy array 存成 JPEG bytes（存到記憶體 buffer）。"""
    arr = np.clip(arr, 0, 255).astype("uint8")
    img = Image.fromarray(arr)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()

def estimate_compressed_size_mb(
    orig_size_mb: float,
    k: int,
    max_rank: int,
) -> float:
    """
    粗估「SVD 壓縮後」檔案大小（MB）。

    這裡採用一個簡單線性模型：
        estimated_size ≈ orig_size_mb * (0.1 + 0.9 * k/max_rank)

    直覺：
      - k 越大 → 保留越多資訊 → 檔案越大
      - 當 k 非常小時，檔案不會趨近 0，而是大約原本的 10%
    """
    if orig_size_mb <= 0 or max_rank <= 0:
        return -1.0

    k = max(1, min(int(k), int(max_rank)))
    ratio = k / float(max_rank)

    # 這個 0.1 / 0.9 是可以之後微調的超參數
    scale = 0.1 + 0.9 * ratio
    return orig_size_mb * scale

def estimate_jpeg_size_mb(arr: np.ndarray, quality: int = 90) -> float:
    """估計圖片若存成 JPEG 時的大小（MB）。"""
    data = save_array_to_jpeg_bytes(arr, quality=quality)
    return len(data) / (1024 * 1024)