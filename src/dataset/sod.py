from __future__ import annotations

import numpy as np


def sod_encode(zn: np.ndarray, K: int, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """zn: (T, 9) normalised. Returns spikes (T, K, 18) float32 and reconstruction r (T, 9)
    at the end of each sample (for diagnostics)."""
    T, C = zn.shape
    spikes = np.zeros((T, K, 2 * C), dtype=np.float32)
    recon = np.zeros((T, C), dtype=np.float64)
    r = zn[0].astype(np.float64).copy()
    for t in range(T):
        x = zn[t]
        for s in range(K):
            d = x - r
            on = d > threshold
            off = d < -threshold
            r = r + threshold * on - threshold * off
            spikes[t, s, :C] = on
            spikes[t, s, C:] = off
        recon[t] = r
    return spikes, recon


def pad_spikes(spikes: np.ndarray, T_pad: int, dtype=np.float32) -> np.ndarray:
    out = np.zeros((T_pad,) + spikes.shape[1:], dtype=dtype)
    out[: spikes.shape[0]] = spikes
    return out


def encoder_stats(zn: np.ndarray, spikes: np.ndarray, recon: np.ndarray, rate_hz: float) -> dict:
    T = zn.shape[0]
    return {
        "enc_recon_rmse_norm": float(np.sqrt(np.mean((recon - zn) ** 2))),
        "enc_spikes_per_sample": float(spikes.sum() / T),
        "enc_rate_hz_per_channel": float(spikes.sum() / (T / rate_hz) / spikes.shape[-1]),
    }
