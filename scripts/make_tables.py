from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from training.selection import select, selectable_rows  # noqa: E402

LABELS = {
    "persistence": "Persistence",
    "lms": "LMS filter",
    "snap1": "SnAp-1 RNN",
    "lsnn_analog": "LSNN, analog",
    "lsnn_sod": "LSNN, SOD",
    "hcrsnn_analog": "HC-RSNN, analog",
    "hcrsnn_sod": "HC-RSNN, SOD",
}
RSNNS = ["lsnn_analog", "lsnn_sod", "hcrsnn_analog", "hcrsnn_sod"]
LEARNERS = ["lms", "snap1"] + RSNNS
SHLS = (1.2, 2.4, 3.6, 4.8, 6.0)
THRESHOLDS = ("0.01", "0.02", "0.05")
# Published immediate-feedback RMSE of Pohl et al. (2024), 10 Hz, mean over 9 records and h 0.1-2.1 s
PUBLISHED = {"persistence": 4.3753, "lms": 1.4192, "snap1": 0.6994,
             "UORO RNN": 0.7549, "DNI RNN": 0.7522, "RTRL RNN": 0.9260}
# Latencies of motion-compensated delivery systems (h in samples at 10 Hz covering them); see the paper.
LATENCY_H = (1, 2, 3, 4, 5, 6)
# ReckOn (Frenkel & Indiveri, ISSCC 2022): energy per synaptic operation, operating points, size limits
PJ_PER_SOP = 5.3
CLOCKS = {"13 MHz": 13e6, "115 MHz": 115e6}
RECKON_MAX_IN, RECKON_MAX_N = 256, 256

RES = ROOT / "results"


# ----------------------------------------------------------------------------- loading
def best(name: str, proto: str) -> pd.DataFrame:
    b = pd.read_csv(RES / f"{name}_best.csv")
    sub = b[b.protocol == proto]
    if sub.empty:  # persistence: no learning, identical under both protocols
        sub = b[b.protocol == "causal"]
    return sub.set_index(["record", "h"]).sort_index()


def cv_rows(name: str) -> pd.DataFrame:
    g = pd.read_csv(RES / f"{name}_grid.csv")
    return g[g.phase == "cv"]


def stable_rule() -> dict:
    """Stable-lr rule of the IFeed selection: kappa 2 x persistence validation RMSE per (record, h)."""
    g = pd.read_csv(RES / "persistence_grid.csv")
    return {"kappa": 2.0, "base": {(int(r), int(h)): v for r, h, v in zip(g.record, g.h, g.causal_cv_rmse_mean)}}


def immediate(model: str) -> pd.DataFrame:
    return best("persistence" if model == "persistence" else f"{model}_if", "pohl")


def delayed_variants(model: str) -> dict[str, pd.DataFrame]:
    return {t: best(f"{model}_df_{t}", "causal") for t in ("abs", "disp")}


def delayed(model: str) -> tuple[str, pd.DataFrame]:
    """DFeed result with the output chosen on the mean validation RMSE -> (output, best rows)."""
    if model == "persistence":
        return "--", best("persistence", "causal")
    v = delayed_variants(model)
    out = min(v, key=lambda t: v[t].cv_rmse.mean())
    return out, v[out]


def fmt(v, nd=3) -> str:
    return "--" if v is None or (isinstance(v, float) and np.isnan(v)) else (v if isinstance(v, str) else f"{v:.{nd}f}")


def md_table(header: list[str], rows: list[list]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + \
           ["| " + " | ".join(map(str, r)) + " |" for r in rows]


# ----------------------------------------------------------------------------- tables
def accuracy(out: Path) -> dict:
    pers = best("persistence", "causal").test_rmse
    rows, sel = [], {}
    for m in ["persistence"] + LEARNERS:
        ifs = immediate(m).test_rmse
        tgt, d = delayed(m)
        sel[m] = (ifs, d.test_rmse)
        worse = "--" if m == "persistence" else f"{int((d.test_rmse > pers).sum())}/{len(d)}"
        rows.append([LABELS[m], fmt(PUBLISHED.get(m)), fmt(ifs.mean()), fmt(d.test_rmse.mean()), tgt,
                     fmt(d.test_rmse.median()), worse])
    for name in ("UORO RNN", "DNI RNN", "RTRL RNN"):
        rows.append([f"{name} (cited)", fmt(PUBLISHED[name]), "--", "--", "--", "--", "--"])
    md = ["# Test RMSE (mm), mean over 9 records x 21 horizons, 50 test seeds per selected configuration", "",
          "DFeed output chosen per model on the mean validation RMSE; '> persistence': DFeed cells worse than "
          "persistence. Published: Pohl et al. (2024), immediate feedback.", ""]
    md += md_table(["Model", "Published", "IFeed", "DFeed", "DFeed output", "DFeed median", "> persistence"], rows)
    rob = [[LABELS[m], fmt(v["abs"].cv_rmse.mean()), fmt(v["disp"].cv_rmse.mean()), fmt(v["abs"].test_rmse.mean()),
            fmt(v["disp"].test_rmse.mean())] for m in LEARNERS for v in [delayed_variants(m)]]
    md += ["", "## DFeed, both outputs (validation and test RMSE)", ""]
    md += md_table(["Model", "abs CV", "disp CV", "abs test", "disp test"], rob)
    (out / "accuracy.md").write_text("\n".join(md) + "\n")
    print("\n".join(md), "\n")
    return sel


def latency(out: Path, sel: dict) -> None:
    reg = best("persistence", "causal").regular.astype(bool)
    md = ["# DFeed test RMSE (mm) at the horizons covering delivery-system latencies", "",
          "Best learner per horizon in bold.", ""]
    models = ["persistence"] + LEARNERS
    for title, mask in (("All 9 recordings", reg | ~reg), ("Regular breathing (records 2, 5, 6, 8, 9)", reg)):
        rows = []
        for h in LATENCY_H + (10, 21):
            v = {m: sel[m][1][mask & (sel[m][1].index.get_level_values("h") == h)].mean() for m in models}
            top = min(LEARNERS, key=v.get)
            rows.append([f"{h / 10:.1f}"] + [f"**{v[m]:.2f}**" if m == top else f"{v[m]:.2f}" for m in models])
        md += [f"## {title}", ""] + md_table(["h (s)"] + [LABELS[m] for m in models], rows) + [""]
    (out / "latency.md").write_text("\n".join(md))


def tests(out: Path, sel: dict) -> None:
    pairs = [(a, b) for a in RSNNS for b in ("snap1", "lms", "persistence")] + [
        ("lsnn_sod", "lsnn_analog"), ("hcrsnn_sod", "hcrsnn_analog"),      # encoding at a fixed network
        ("hcrsnn_analog", "lsnn_analog"), ("hcrsnn_sod", "lsnn_sod"),      # constraints at a fixed encoding
        ("hcrsnn_sod", "lsnn_analog"), ("snap1", "lms")]
    md = ["# Paired Wilcoxon signed-rank tests over the 189 (record, horizon) cells of test RMSE", "",
          "Cells share records and neighbouring horizons: read p as indicative.", ""]
    for i, title in ((0, "Immediate feedback (IFeed)"), (1, "Delayed feedback (DFeed)")):
        rows = []
        for a, b in pairs:
            x, y = sel[a][i].align(sel[b][i], join="inner")
            rec = x.groupby(level="record").mean() < y.groupby(level="record").mean()
            rows.append([LABELS[a], LABELS[b], fmt(x.mean()), fmt(y.mean()), f"{(1 - x.mean() / y.mean()) * 100:+.1f} %",
                         f"{int((x < y).sum())}/{len(x)}", f"{int(rec.sum())}/{len(rec)}", f"{wilcoxon(x, y).pvalue:.1e}"])
        md += [f"## {title}", ""] + md_table(["A", "B", "RMSE A", "RMSE B", "A lower by", "cells A < B",
                                              "records A < B", "p"], rows) + [""]
    x = sel["hcrsnn_sod"][1]
    rows = []
    for m in ("snap1", "lsnn_analog", "lms"):
        y = delayed_variants(m)["abs"].test_rmse
        x2, y2 = x.align(y, join="inner")
        rows.append([LABELS[m], fmt(y2.mean()), f"{(1 - x2.mean() / y2.mean()) * 100:+.1f} %", f"{wilcoxon(x2, y2).pvalue:.1e}"])
    md += ["## DFeed: SOD HC-RSNN against absolute-output variants", ""]
    md += md_table(["B (absolute outputs)", "RMSE B", "HC-RSNN, SOD lower by", "p"], rows)
    (out / "tests.md").write_text("\n".join(md) + "\n")


def cost_row(df: pd.DataFrame, rmse_col: str) -> dict:
    k, K, N = df["cfg.k"], df["cfg.K"], df["cfg.N"]
    n_in = 18 * k + 1  # SOD: ON and OFF line per coordinate and tap, plus the bias line
    active = (df["in_spikes_per_sample_mean"] + df["rec_spikes_per_sample_mean"]) / K
    m = np.maximum(n_in, N)  # ReckOn cycle model (per timestep), read from the RTL
    cycles = ((2 * m + active * (N + 1) + np.ceil(N / 16) * (32 + 2 * m)) * K).mean()
    row = {"RMSE (mm)": df[rmse_col].mean(), "mean SHL (s)": df["cfg.shl_s"].mean(), "input lines": n_in.mean(),
           "cells fitting ReckOn": f"{int(((n_in <= RECKON_MAX_IN) & (N <= RECKON_MAX_N)).sum())}/{len(df)}",
           "synaptic ops (k)": df["cost_syn_ops"].mean() / 1e3,
           "energy (uJ)": df["cost_syn_ops"].mean() * PJ_PER_SOP / 1e6}
    for name, f in CLOCKS.items():
        row[f"time @ {name} (ms)"] = cycles / f * 1e3
    return row


def cost(out: Path) -> None:
    cols = {}
    cols["IFeed"] = cost_row(immediate("hcrsnn_sod").reset_index(), "test_rmse")
    tgt, d = delayed("hcrsnn_sod")
    cols["DFeed"] = cost_row(d.reset_index(), "test_rmse")
    cv = selectable_rows(cv_rows("hcrsnn_sod_if"), None, "pohl", stable_rule())  # stable rule over all SHLs
    cols["IFeed, chip-sized*"] = cost_row(select(cv[np.isclose(cv["cfg.shl_s"], 1.2)], protocols=["pohl"]),
                                          "pohl_test_rmse_mean")
    cv = cv_rows(f"hcrsnn_sod_df_{tgt}")
    cols["DFeed, chip-sized*"] = cost_row(select(cv[np.isclose(cv["cfg.shl_s"], 1.2)], protocols=["causal"]),
                                          "causal_test_rmse_mean")
    t = pd.DataFrame(cols)
    g = lambda v: v if isinstance(v, str) else (f"{v:,.3f}" if abs(v) < 10 else f"{v:,.0f}")
    md = ["# Estimated cost per forecast of the SOD HC-RSNN on a ReckOn-class processor", "",
          f"Means over the 189 selected cells. Energy = synaptic ops x {PJ_PER_SOP} pJ; times from a cycle model of "
          "the ReckOn design, K = 10 timesteps per forecast; sampling period 100 ms. *chip-sized: selected within "
          "SHL 1.2 s (217 input lines), RMSE from the 20 validation-phase seeds.", ""]
    md += md_table([""] + list(t.columns), [[i] + [g(v) for v in r] for i, r in t.iterrows()])
    (out / "cost.md").write_text("\n".join(md) + "\n")


def threshold(out: Path) -> None:
    stable = stable_rule()
    pers = best("persistence", "causal").test_rmse
    md = ["# SOD threshold ablation, HC-RSNN with SOD input", "",
          "Per-SHL columns: lr selected per (record, h) on validation for that SHL, 20-seed validation-phase test; "
          "'SHL selected': 50-seed test. Reference row: the analog LSNN.", ""]
    for proto, title, names in (
            ("pohl", "Immediate feedback (IFeed), absolute outputs, stable-lr rule",
             [("0.005", "hcrsnn_sod_if")] + [(t, f"hcrsnn_sod_if_thr{t}") for t in THRESHOLDS]
             + [("analog LSNN", "lsnn_analog_if")]),
            ("causal", "Delayed feedback (DFeed), displacement outputs",
             [("0.005", "hcrsnn_sod_df_disp")] + [(t, f"hcrsnn_sod_df_disp_thr{t}") for t in THRESHOLDS]
             + [("analog LSNN", "lsnn_analog_df_disp")])):
        rows = []
        for label, name in names:
            cv = cv_rows(name)
            if proto == "pohl":
                cv = selectable_rows(cv, None, "pohl", stable)
            b = best(name, proto)
            per_shl = [select(cv[np.isclose(cv["cfg.shl_s"], s)], protocols=[proto])[f"{proto}_test_rmse_mean"].mean()
                       for s in SHLS]
            events = cv["enc_spikes_per_sample"].mean() if "enc_spikes_per_sample" in cv else np.nan
            sop = b["cost_syn_ops"].mean() / 1e3 if label != "analog LSNN" else np.nan
            rows.append([label] + [fmt(v) for v in per_shl] + [
                fmt(b.test_rmse.mean()), fmt(b.groupby(level="record").test_rmse.mean().median()),
                str(int((b.test_rmse > 2 * pers.loc[b.index]).sum())), fmt(events, 1), fmt(sop, 0)])
        md += [f"## {title}", ""] + md_table(
            ["threshold"] + [f"SHL {s:g}" for s in SHLS] + ["SHL selected", "median over records",
                                                            "cells > 2x persistence", "events / sample", "SOP (k)"],
            rows) + [""]
    (out / "threshold.md").write_text("\n".join(md))


def recompute(out: Path) -> None:
    rows = []
    for m in ("snap1", "lsnn_analog", "hcrsnn_sod"):
        st = best(f"{m}_df_disp", "causal").test_rmse.mean()
        rc = best(f"{m}_df_disp_recompute", "causal").test_rmse.mean()
        rows.append([LABELS[m], fmt(st), fmt(rc)])
    md = ["# Supplementary: delayed feedback with stored sensitivities vs. two-network recomputation", "",
          "DFeed test RMSE (mm), displacement outputs. 'recompute': each delayed update is recomputed with a second "
          "copy of the network at twice the forward cost (coupled loop).", ""]
    md += md_table(["Model", "stored (paper)", "recompute"], rows)
    (out / "recompute.md").write_text("\n".join(md) + "\n")


def main() -> int:
    global RES
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results", default=str(RES), help="directory with the sweep CSV files")
    args = p.parse_args()
    RES = Path(args.results)
    out = RES / "tables"
    out.mkdir(parents=True, exist_ok=True)
    sel = accuracy(out)
    latency(out, sel)
    tests(out, sel)
    cost(out)
    threshold(out)
    recompute(out)
    files = ", ".join(sorted(p.name for p in out.glob("*.md")))
    print(f"wrote {out}/: {files}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
