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
    """
    把 numpy array 存成 JPEG（存在記憶體，不寫入硬碟），
    回傳 byte 串。
    """
    arr = np.clip(arr, 0, 255).astype("uint8")
    img = Image.fromarray(arr)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def estimate_jpeg_size_mb(arr: np.ndarray, quality: int = 90) -> float:
    """
    估計這張圖若用指定 JPEG 品質（quality）儲存時的檔案大小（MB）。
    這裡是「真的轉成 JPEG 放在記憶體」，所以會很準。
    """
    data = save_array_to_jpeg_bytes(arr, quality=quality)
    return len(data) / (1024 * 1024)