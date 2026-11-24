from __future__ import annotations

METRICS = ("mae", "rmse", "nrmse", "max", "jitter", "frac_gt2")
SUM_KEYS = ("n", "sum_d", "sum_d2", "max_d", "n_gt2", "jit_sum", "jit_n", "den", "nonfinite")
RANGES = ("cv", "test")
PROTOCOLS = ("pohl", "causal")
GT_MM = 2.0


def marker_dist(xp, A):
    """A: (n, 9) channel order x1 x2 x3 y1 y2 y3 z1 z2 z3 -> (n, 3) per-marker norms."""
    A3 = A.reshape(A.shape[0], 3, 3)  # [time, coord, marker]
    return xp.sqrt(xp.sum(A3 * A3, axis=1))


def metric_sums(xp, P, X, mask):
    """P, X: (n, 9) forecast and true positions (mm); mask: (n,) bool evaluation range."""
    dt = P.dtype
    md = mask[:, None]
    d = marker_dist(xp, P - X)
    nonfinite = xp.sum(xp.where(md, ~xp.isfinite(d), False))
    d0 = xp.where(md, d, 0.0)
    cnt = xp.sum(mask).astype(dt)
    s = {
        "n": 3.0 * cnt,
        "sum_d": xp.sum(d0),
        "sum_d2": xp.sum(d0 * d0),
        "max_d": xp.max(xp.where(md, d, -xp.inf)),
        "n_gt2": xp.sum(xp.where(md, d > GT_MM, False)).astype(dt),
        "nonfinite": nonfinite.astype(dt),
    }
    pair = mask[1:] & mask[:-1]
    jd = marker_dist(xp, P[1:] - P[:-1])
    s["jit_sum"] = xp.sum(xp.where(pair[:, None], jd, 0.0))
    s["jit_n"] = 3.0 * xp.sum(pair).astype(dt)
    mu = xp.sum(xp.where(md, X, 0.0), axis=0) / xp.maximum(cnt, 1.0)
    dev = xp.where(md, X - mu, 0.0)
    s["den"] = xp.sum(dev * dev)
    return s


def finalize(xp, s):
    n = xp.maximum(s["n"], 1.0)
    out = {
        "mae": s["sum_d"] / n,
        "rmse": xp.sqrt(s["sum_d2"] / n),
        "nrmse": xp.sqrt(s["sum_d2"]) / xp.sqrt(s["den"]),
        "max": s["max_d"],
        "jitter": s["jit_sum"] / xp.maximum(s["jit_n"], 1.0),
        "frac_gt2": s["n_gt2"] / n,
    }
    out["finite"] = (s["nonfinite"] == 0) & xp.isfinite(out["rmse"]) & xp.isfinite(out["max"])
    return out


def window_targets(xp, zn_pad, k: int, h, L: int):
    """True (normalised) target for each window index i in [0, L): z(i+k-1+h), clipped."""
    T_pad = zn_pad.shape[0]
    tau = xp.arange(L) + (k - 1) + h
    return tau, zn_pad[xp.clip(tau, 0, T_pad - 1)]


def range_masks(xp, tau, T, b1: int, b2: int):
    valid = tau <= T - 1
    return {
        "cv": valid & (tau >= b1) & (tau < b2),
        "test": valid & (tau >= b2),
    }


def evaluate_windows(xp, pred_n, zn_pad, mean, std, k: int, h, T, b1: int, b2: int):
    """pred_n: (L, 9) normalised absolute forecasts indexed by window. Returns {range: metrics}."""
    L = pred_n.shape[0]
    tau, X_n = window_targets(xp, zn_pad, k, h, L)
    P = pred_n * std + mean
    X = X_n * std + mean
    masks = range_masks(xp, tau, T, b1, b2)
    return {r: finalize(xp, metric_sums(xp, P, X, masks[r])) for r in RANGES}
