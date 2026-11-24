from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation.metrics import METRICS, PROTOCOLS
from training.config import lr_allowed

ID_COLS = ["experiment", "method", "rate", "record", "static_key", "h", "lr", "phase", "n_seeds"]


def _allowed_mask(df: pd.DataFrame, select_lr: dict, proto: str) -> pd.Series:
    if not select_lr or not select_lr.get(proto):
        return pd.Series(True, index=df.index)
    return df["lr"].apply(lambda v: (not np.isfinite(v)) or lr_allowed(select_lr, proto, float(v)))


def stable_lr_mask(df: pd.DataFrame, proto: str, base_cv: dict, kappa: float) -> pd.Series:
    """Stable-lr rule: per record, drop every row of an lr whose mean CV RMSE exceeds kappa x persistence's CV
    RMSE (base_cv[(record, h)]) at any static config and horizon of that record. A record where every lr
    would be dropped keeps all of them. Uses CV data only."""
    base = np.array([base_cv.get((int(r), int(h)), np.nan) for r, h in zip(df["record"], df["h"])])
    bad = (df[f"{proto}_cv_rmse_mean"].to_numpy() > kappa * base)
    unstable = pd.Series(bad, index=df.index).groupby([df["record"], df["lr"]]).transform("any")
    all_bad = unstable.groupby(df["record"]).transform("all")
    return ~unstable | all_bad


def selectable_rows(cv_rows: pd.DataFrame, select_lr: dict | None, proto: str, stable: dict | None) -> pd.DataFrame:
    """CV rows the selection may pick from under ``proto``: the protocol's lr grid, then the stable-lr rule
    (applied among those lrs, so the all-excluded fallback sees the right set)."""
    sub = cv_rows[_allowed_mask(cv_rows, select_lr or {}, proto)]
    if stable:
        sub = sub[stable_lr_mask(sub, proto, stable["base"], stable["kappa"])]
    return sub


def select(cv_rows: pd.DataFrame, select_lr: dict | None = None, protocols=PROTOCOLS,
           stable: dict | None = None) -> pd.DataFrame:
    """cv_rows: phase == 'cv' rows. Returns one row per (record, h, protocol) with the chosen config.
    stable: {"kappa": float, "base": {(record, h): persistence CV RMSE}} applies stable_lr_mask."""
    select_lr = select_lr or {}
    out = []
    for proto in protocols:
        sub = selectable_rows(cv_rows, select_lr, proto, stable)
        for _, g in sub.groupby(["record", "h"], sort=True):
            ndiv = g[f"{proto}_n_div"]
            cand = g[ndiv == ndiv.min()]
            score = cand[f"{proto}_cv_rmse_mean"]
            if not np.isfinite(score).any():
                row = cand.iloc[0].copy()
                row["selection_note"] = "all diverged"
            else:
                row = cand.loc[score.idxmin()].copy()
                row["selection_note"] = ""
            row["protocol"] = proto
            out.append(row)
    return pd.DataFrame(out).reset_index(drop=True)


def best_table(chosen: pd.DataFrame, test_rows: pd.DataFrame | None) -> pd.DataFrame:
    """Combine the selection (CV phase) with phase-2 test rows (if any) into the _best table."""
    rows = []
    for _, r in chosen.iterrows():
        proto = r["protocol"]
        rec = {
            "experiment": r["experiment"],
            "method": r["method"],
            "rate": r["rate"],
            "record": int(r["record"]),
            "regular": bool(r["regular"]),
            "h": int(r["h"]),
            "h_s": float(r["h_s"]),
            "protocol": proto,
            "static_key": r["static_key"],
            "lr": r.get("lr", np.nan),
            "selection_note": r.get("selection_note", ""),
            "cv_rmse": r[f"{proto}_cv_rmse_mean"],
            "cv_n_div": r[f"{proto}_n_div"],
            "cv_n_seeds": r["n_seeds"],
        }
        for c in r.index:
            if c.startswith("cfg."):
                rec[c] = r[c]
        src = r
        source = "cv-phase"
        if test_rows is not None and len(test_rows):
            m = test_rows[
                (test_rows["record"] == r["record"])
                & (test_rows["h"] == r["h"])
                & (test_rows["static_key"] == r["static_key"])
                & np.isclose(test_rows["lr"].astype(float), float(r.get("lr", np.nan)), equal_nan=True)
            ]
            if len(m):
                src = m.iloc[-1]
                source = "test-phase"
        for met in METRICS:
            rec[f"test_{met}"] = src[f"{proto}_test_{met}_mean"]
            rec[f"test_{met}_std"] = src[f"{proto}_test_{met}_std"]
        rec["test_n_div"] = src[f"{proto}_n_div"]
        rec["test_n_seeds"] = src["n_seeds"]
        rec["test_source"] = source
        for c in src.index:
            if c.startswith(("cost_", "enc_")) or c.endswith("_per_sample_mean") or c in (
                "mean_rate_hz_mean",
                "silent_frac_mean",
            ):
                rec[c] = src[c]
        rows.append(rec)
    return pd.DataFrame(rows)


def summarize(best: pd.DataFrame) -> dict:
    """Averages over records x horizons (Pohl's Table 3 format), per protocol, plus subsets."""
    out: dict = {}
    for proto, g in best.groupby("protocol"):
        for met in METRICS:
            out[f"{proto}/test_{met}"] = float(np.nanmean(g[f"test_{met}"]))
        out[f"{proto}/cv_rmse"] = float(np.nanmean(g["cv_rmse"]))
        out[f"{proto}/n_div_total"] = float(np.nansum(g["test_n_div"]))
        for name, sub in (("regular", g[g["regular"]]), ("irregular", g[~g["regular"]])):
            if len(sub):
                out[f"{proto}/test_rmse_{name}"] = float(np.nanmean(sub["test_rmse"]))
                out[f"{proto}/test_nrmse_{name}"] = float(np.nanmean(sub["test_nrmse"]))
    return out


def per_horizon(best: pd.DataFrame) -> pd.DataFrame:
    cols = [f"test_{m}" for m in METRICS]
    return best.groupby(["protocol", "h", "h_s"])[cols].mean().reset_index()

