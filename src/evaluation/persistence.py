from __future__ import annotations

import numpy as np

from dataset.records import SPLITS, Record
from evaluation.metrics import RANGES, finalize, metric_sums, range_masks


def eval_host(rec: Record, pred_n: np.ndarray, tau: np.ndarray) -> dict:
    """pred_n: (M, 9) normalised absolute forecasts for target times tau. -> {range: metrics}."""
    b1, b2 = SPLITS[rec.rate]
    P = pred_n * rec.std + rec.mean
    X = rec.z_mm[tau]
    masks = range_masks(np, tau, rec.T, b1, b2)
    return {r: finalize(np, metric_sums(np, P, X, masks[r])) for r in RANGES}


def persistence(rec: Record, h: int) -> dict:
    """Forecast z(tau) by the last observed sample z(tau - h). Causal by construction."""
    tau = np.arange(h, rec.T)
    return eval_host(rec, rec.zn[tau - h], tau)
