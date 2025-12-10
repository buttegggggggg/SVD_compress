# core/svd_ops.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class SVDRGBDecomp:
    """Stores SVD decomposition results for three channels of an RGB image"""
    U_R: np.ndarray
    S_R: np.ndarray
    Vt_R: np.ndarray
    U_G: np.ndarray
    S_G: np.ndarray
    Vt_G: np.ndarray
    U_B: np.ndarray
    S_B: np.ndarray
    Vt_B: np.ndarray
    max_rank: int  # Maximum usable k value (min(height, width))


# ===== 1. Perform full SVD: run once, reuse afterwards =====
def svd_decompose_rgb(img_rgb: np.ndarray) -> SVDRGBDecomp:
    """
    img_rgb: shape = (H, W, 3), dtype can be uint8 or float
    Returns: U, S, Vt for each channel, and max_rank
    """
    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise ValueError("svd_decompose_rgb only accepts RGB images (H, W, 3).")

    # Convert to float for better computation
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


# ===== 2. Given k, reconstruct using the decomposition results above =====
def _reconstruct_channel(U: np.ndarray, S: np.ndarray, Vt: np.ndarray, k: int) -> np.ndarray:
    """Same logic as the reconstruct_channel in your assignment."""
    k = int(k)
    k = max(1, min(k, len(S)))  # clamp to [1, len(S)]

    U_k = U[:, :k]
    S_k = S[:k]
    Vt_k = Vt[:k, :]

    return U_k @ (np.diag(S_k) @ Vt_k)


def reconstruct_rgb(decomp: SVDRGBDecomp, k: int) -> np.ndarray:
    """
    Reconstruct an RGB image using the first k singular values.
    Returns: uint8 (H, W, 3) array, ready for display.
    """
    k = int(k)
    k = max(1, min(k, decomp.max_rank))

    R_approx = _reconstruct_channel(decomp.U_R, decomp.S_R, decomp.Vt_R, k)
    G_approx = _reconstruct_channel(decomp.U_G, decomp.S_G, decomp.Vt_G, k)
    B_approx = _reconstruct_channel(decomp.U_B, decomp.S_B, decomp.Vt_B, k)

    img_approx = np.stack([R_approx, G_approx, B_approx], axis=2)

    # Clip to [0, 255] and convert to uint8
    img_approx = np.clip(img_approx, 0, 255).astype(np.uint8)
    return img_approx


# ===== 3. (Prepare in advance, use when displaying MSE/PSNR later) =====
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
    Returns energy ratio between 0 and 1 (closer to 1 means more information retained).
    Calculated using the sum of squared singular values of three channels.
    """
    k = int(k)
    k = max(1, min(k, decomp.max_rank))

    # Total energy of three channels
    total = (
        np.sum(decomp.S_R ** 2) +
        np.sum(decomp.S_G ** 2) +
        np.sum(decomp.S_B ** 2)
    )

    # Energy of first k singular values across three channels
    keep = (
        np.sum(decomp.S_R[:k] ** 2) +
        np.sum(decomp.S_G[:k] ** 2) +
        np.sum(decomp.S_B[:k] ** 2)
    )

    if total == 0:
        return 0.0
    return float(keep / total)


def find_k_for_energy(decomp: SVDRGBDecomp, target_energy: float) -> int:
    """
    Find the minimum k such that energy_for_k(decomp, k) >= target_energy.
    target_energy can be, for example, 0.90 / 0.95 / 0.99.
    """
    target_energy = float(target_energy)
    max_k = int(decomp.max_rank)

    for k in range(1, max_k + 1):
        e = energy_for_k(decomp, k)
        if e >= target_energy:
            return k

    return max_k
