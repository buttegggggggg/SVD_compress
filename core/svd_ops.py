# core/svd_ops.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class SVDRGBDecomp:
    """儲存一張 RGB 圖片三個通道的 SVD 結果"""
    U_R: np.ndarray
    S_R: np.ndarray
    Vt_R: np.ndarray
    U_G: np.ndarray
    S_G: np.ndarray
    Vt_G: np.ndarray
    U_B: np.ndarray
    S_B: np.ndarray
    Vt_B: np.ndarray
    max_rank: int  # 可以用的最大 k 值 (min(height, width))


# ===== 1. 做 full SVD：只跑一次，之後可以重複用 =====
def svd_decompose_rgb(img_rgb: np.ndarray) -> SVDRGBDecomp:
    """
    img_rgb: shape = (H, W, 3), dtype 可以是 uint8 或 float
    回傳：每個通道的 U, S, Vt，以及 max_rank
    """
    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise ValueError("svd_decompose_rgb 只接受 RGB 圖片 (H, W, 3)。")

    # 轉 float 比較好算
    img = img_rgb.astype(float)
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]

    U_R, S_R, Vt_R = np.linalg.svd(R, full_matrices=False)
    U_G, S_G, Vt_G = np.linalg.svd(G, full_matrices=False)
    U_B, S_B, Vt_B = np.linalg.svd(B, full_matrices=False)

    max_rank = int(min(len(S_R), len(S_G), len(S_B)))

    return SVDRGBDecomp(
        U_R=U_R, S_R=S_R, Vt_R=Vt_R,
        U_G=U_G, S_G=S_G, Vt_G=Vt_G,
        U_B=U_B, S_B=S_B, Vt_B=Vt_B,
        max_rank=max_rank,
    )


# ===== 2. 給一個 k，用上面的分解結果重建 =====
def _reconstruct_channel(U: np.ndarray, S: np.ndarray, Vt: np.ndarray, k: int) -> np.ndarray:
    """跟你作業裡的 reconstruct_channel 一樣邏輯。"""
    k = int(k)
    k = max(1, min(k, len(S)))  # clamp 到 [1, len(S)]

    U_k = U[:, :k]
    S_k = S[:k]
    Vt_k = Vt[:k, :]

    return U_k @ (np.diag(S_k) @ Vt_k)


def reconstruct_rgb(decomp: SVDRGBDecomp, k: int) -> np.ndarray:
    """
    用前 k 個奇異值重建一張 RGB 圖片。
    回傳: uint8 的 (H, W, 3) array，可以直接丟去顯示。
    """
    k = int(k)
    k = max(1, min(k, decomp.max_rank))

    R_approx = _reconstruct_channel(decomp.U_R, decomp.S_R, decomp.Vt_R, k)
    G_approx = _reconstruct_channel(decomp.U_G, decomp.S_G, decomp.Vt_G, k)
    B_approx = _reconstruct_channel(decomp.U_B, decomp.S_B, decomp.Vt_B, k)

    img_approx = np.stack([R_approx, G_approx, B_approx], axis=2)

    # clip 到 [0, 255] 然後轉 uint8
    img_approx = np.clip(img_approx, 0, 255).astype(np.uint8)
    return img_approx


# ===== 3. (先備好，之後你要顯示 MSE/PSNR 再用) =====
def mse_rgb(original: np.ndarray, compressed: np.ndarray) -> float:
    diff = original.astype(float) - compressed.astype(float)
    return float(np.mean(diff ** 2))


def psnr_rgb(original: np.ndarray, compressed: np.ndarray, max_val: float = 255.0) -> float:
    m = mse_rgb(original, compressed)
    if m == 0:
        return float("inf")
    return float(10 * np.log10((max_val ** 2) / m))

def energy_for_k(decomp: SVDRGBDecomp, k: int) -> float:
    """
    回傳 0~1 之間的 energy 比例（越接近 1 表示保留的資訊越多）
    用的是三個通道奇異值平方和的比例。
    """
    k = int(k)
    k = max(1, min(k, decomp.max_rank))

    # 三個通道總能量
    total = (
        np.sum(decomp.S_R ** 2) +
        np.sum(decomp.S_G ** 2) +
        np.sum(decomp.S_B ** 2)
    )

    # 前 k 個能量
    keep = (
        np.sum(decomp.S_R[:k] ** 2) +
        np.sum(decomp.S_G[:k] ** 2) +
        np.sum(decomp.S_B[:k] ** 2)
    )

    if total == 0:
        return 0.0
    return float(keep / total)

def energy_for_k(decomp, k: int) -> float:
    """
    回傳 0~1 之間的 energy 比例（越接近 1 表示保留的資訊越多）。
    使用 RGB 三個通道奇異值平方和的比例計算：
        sum_{i=1}^k sigma_i^2  /  sum_{i=1}^r sigma_i^2
    """
    # 保險：確保 k 在合法範圍
    k = int(k)
    k = max(1, min(k, int(decomp.max_rank)))

    # 三通道總能量
    total = (
        np.sum(decomp.S_R ** 2) +
        np.sum(decomp.S_G ** 2) +
        np.sum(decomp.S_B ** 2)
    )

    # 三通道前 k 個能量
    keep = (
        np.sum(decomp.S_R[:k] ** 2) +
        np.sum(decomp.S_G[:k] ** 2) +
        np.sum(decomp.S_B[:k] ** 2)
    )

    if total == 0:
        return 0.0

    return float(keep / total)

def find_k_for_energy(decomp, target_energy: float) -> int:

    """
    找到最小的 k，使得 energy_for_k(decomp, k) >= target_energy。
    target_energy 例如 0.90 / 0.95 / 0.99。
    """
    target_energy = float(target_energy)
    max_k = int(decomp.max_rank)

    for k in range(1, max_k + 1):
        e = energy_for_k(decomp, k)
        if e >= target_energy:
            return k

    return max_k


