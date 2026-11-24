"""Sweep driver (scripts/sweep.py): one sweep config (configs/sweeps/NAME.json) -> CSV files.

Outputs (``results/NAME`` by default):
    NAME_grid.csv     every (record, static config, horizon, lr, phase) row; resumable
    NAME_best.csv     selected configuration per (record, horizon, protocol) + its test metrics
    NAME_config.json  copy of the sweep config used

Two phases for seeded methods:
    phase "cv":   every grid point with --seeds-cv seeds  -> selection on the validation (CV) RMSE
    phase "test": the selected points only, with --seeds-test fresh seeds -> reported test metrics
Each online run covers the full record, so CV and test numbers come from the same runs; the selection only
ever looks at CV numbers. Protocol keys: "pohl" = immediate feedback, "causal" = delayed feedback.
"""

from __future__ import annotations

import argparse
import math
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from dataset.records import RATE_HZ, parse_horizons, parse_seqs
from evaluation.metrics import METRICS, PROTOCOLS, RANGES
from paths import RESULTS_DIR, ROOT
from training.config import Grid, load_grid, static_key
from training.progress import Progress, hms

RATE = "10"  # the paper uses the 10 Hz records only
METHOD_NAMES = {"persistence": "Persistence", "lms": "LMS filter", "snap1": "tanh RNN + SnAp-1", "rsnn": "RSNN + e-prop"}
NETWORK_NAMES = {"lsnn": "LSNN", "hcrsnn": "HC-RSNN"}
LOOP_NAMES = {"coupled": "immediate feedback (coupled loop, also gives the two-network delayed stream)",
              "stored": "delayed feedback (stored sensitivities)"}
SEEDED = ("snap1", "rsnn")
METHODS = ("persistence", "lms", "snap1", "rsnn")


# ----------------------------------------------------------------------------- CSV helpers
def append_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    new = pd.DataFrame(rows)
    if path.exists():
        old_cols = list(pd.read_csv(path, nrows=0).columns)
        if set(new.columns) <= set(old_cols):
            new.reindex(columns=old_cols).to_csv(path, mode="a", header=False, index=False)
            return
        df = pd.concat([pd.read_csv(path), new], ignore_index=True)
        df.to_csv(path, index=False)
    else:
        new.to_csv(path, index=False)


def read_rows(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _key(record, skey, h, lr, phase, n_seeds) -> tuple:
    f = lambda v: "nan" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{float(v):.6g}"
    return (int(record), str(skey), int(h), f(lr), str(phase), int(n_seeds))


def done_keys(df: pd.DataFrame) -> set:
    if df.empty:
        return set()
    cols = ["record", "static_key", "h", "lr", "phase", "n_seeds"]
    return {_key(*r) for r in df[cols].itertuples(index=False)}


# ----------------------------------------------------------------------------- rows
def base_row(ctx, rec, skey, resolved, h, lr, phase, n_seeds) -> dict:
    row = {
        "experiment": ctx.grid.name,
        "method": ctx.grid.method,
        "rate": RATE,
        "record": rec.seq,
        "regular": rec.regular,
        "static_key": skey,
        "k": resolved.get("k", np.nan),
        "h": int(h),
        "h_s": float(h) / RATE_HZ[RATE],
        "lr": float(lr) if lr is not None else np.nan,
        "phase": phase,
        "n_seeds": int(n_seeds),
    }
    for k, v in resolved.items():
        row[f"cfg.{k}"] = v
    return row


def cost_cols(method, model, resolved, res: dict) -> dict:
    from evaluation.costs import lms_costs, rnn_costs, snn_costs

    if method == "snap1":
        c = rnn_costs(resolved["q"], resolved["k"])
    elif method == "lms":
        c = lms_costs(resolved["k"])
    elif method == "rsnn":
        c = snn_costs(model.cfg, resolved["k"], res.get("rec_spikes_per_sample_mean", np.nan),
                      res.get("in_spikes_per_sample_mean", np.nan))
    else:
        c = {}
    return {f"cost_{k}": v for k, v in c.items()}


# ----------------------------------------------------------------------------- runners
def run_persistence(ctx) -> None:
    from evaluation.persistence import persistence

    for rec in ctx.records:
        resolved = {"k": 1}
        skey = static_key("persistence", resolved)
        group = []
        for h in ctx.horizons:
            if _key(rec.seq, skey, h, None, "cv", 1) in ctx.done:
                continue
            res = persistence(rec, h)
            row = base_row(ctx, rec, skey, resolved, h, None, "cv", 1)
            for proto in PROTOCOLS:  # no learning: identical under both protocols
                row[f"{proto}_n_div"] = 0 if all(bool(res[r]["finite"]) for r in RANGES) else 1
                for r in RANGES:
                    for m in METRICS:
                        row[f"{proto}_{r}_{m}_mean"] = float(res[r][m])
                        row[f"{proto}_{r}_{m}_std"] = 0.0
            group.append(row)
        append_rows(ctx.grid_csv, group)
    print(f"  persistence computed for {len(ctx.records)} records x {len(ctx.horizons)} horizons", flush=True)


def build_plan(ctx) -> dict:
    """{static_key: {"model", "resolved", "cells": {record: [(h, lr), ...]}}}"""
    from models import build_model

    plan = {}
    for pt in ctx.grid.static_points:
        model, resolved = build_model(ctx.grid.method, ctx.grid.model_layers, pt, RATE)
        skey = static_key(ctx.grid.method, resolved)
        lrs = ctx.grid.lrs or [0.0]
        cells = {rec.seq: [(h, lr) for h in ctx.horizons for lr in lrs] for rec in ctx.records}
        plan[skey] = {"model": model, "resolved": resolved, "cells": cells}
    return plan


def _todo(ctx, plan: dict, phase: str, n_seeds: int) -> dict:
    """{static_key: {record: [(h, lr), ...]}} of the cells not yet in the grid CSV."""
    return {skey: {rec.seq: [c for c in item["cells"].get(rec.seq, [])
                             if _key(rec.seq, skey, c[0], c[1], phase, n_seeds) not in ctx.done]
                   for rec in ctx.records}
            for skey, item in plan.items()}


def _point_label(resolved: dict) -> str:
    out = [f"SHL {resolved['shl_s']:g} s"]
    if "q" in resolved:
        out.append(f"q {resolved['q']}")
    return ", ".join(out)


def run_online_phase(ctx, phase: str, plan: dict) -> None:
    import jax.numpy as jnp

    from training.engine import TEST_SEED_OFFSET, BatchRunner, device_data, jobs_for_memory

    n_seeds = ctx.seeds_cv if phase == "cv" else ctx.seeds_test
    seeds = list(range(n_seeds)) if phase == "cv" else [TEST_SEED_OFFSET + s for s in range(n_seeds)]
    todo = _todo(ctx, plan, phase, n_seeds)
    total = sum(len(c) for per in todo.values() for c in per.values()) * n_seeds
    name = "validation" if phase == "cv" else "test"
    if total == 0:
        print(f"  {name} phase: nothing to do (all rows already in {ctx.grid_csv.name})", flush=True)
        return
    prog = Progress(total, f"{name} ({n_seeds} seeds)")
    for skey, item in plan.items():
        model, resolved = item["model"], item["resolved"]
        runner = BatchRunner(model, RATE, stored_nbuf=ctx.stored_nbuf)
        for rec in ctx.records:
            cells = todo[skey][rec.seq]
            if not cells:
                continue
            prog.status(f"record {rec.seq}, {_point_label(resolved)}")
            data, enc_info = device_data(rec, model, jnp.float32)
            max_jobs = min(ctx.max_batch, jobs_for_memory(model, rec.T_pad, ctx.mem_gb, nbuf=ctx.stored_nbuf))
            results, _ = runner.run(data, rec.T, cells, seeds, max_jobs, on_batch=prog.update)
            group = []
            for (h, lr), res in zip(cells, results):
                row = base_row(ctx, rec, skey, resolved, h, lr, phase, n_seeds)
                row.update(res)
                row.update(cost_cols(ctx.grid.method, model, resolved, res))
                row.update(enc_info)
                group.append(row)
            append_rows(ctx.grid_csv, group)
    wall = prog.close()
    print(f"  {name} phase done in {hms(wall)} ({total / max(wall, 1e-9):,.0f} runs/s)", flush=True)


# ----------------------------------------------------------------------------- selection
def stable_rule(ctx) -> dict | None:
    """Stable-lr rule (config "stable_select"): kappa and persistence CV RMSE per (record, h)."""
    if ctx.grid.stable_select is None:
        return None
    from evaluation.persistence import persistence

    base = {(r.seq, h): float(persistence(r, h)["cv"]["rmse"]) for r in ctx.records for h in ctx.horizons}
    return {"kappa": ctx.grid.stable_select, "base": base}


def finalize_outputs(ctx, best_csv: Path) -> None:
    from training.selection import best_table, select, summarize

    allrows = read_rows(ctx.grid_csv)
    if allrows.empty:
        print("no rows; nothing to summarise")
        return
    rows = allrows[allrows["record"].isin(ctx.seqs) & allrows["h"].isin(ctx.horizons)]
    cvr = rows[(rows["phase"] == "cv") & (rows["n_seeds"] == ctx.seeds_cv)]
    chosen = select(cvr, ctx.grid.select_lr, ctx.protocols, stable_rule(ctx))
    testr = rows[(rows["phase"] == "test") & (rows["n_seeds"] == ctx.seeds_test)] if ctx.seeds_test else None
    best = best_table(chosen, testr)
    best.to_csv(best_csv, index=False)
    summ = summarize(best)
    print(f"  result, selected configuration per (record, horizon), mean over {best.record.nunique()} records x "
          f"{best.h.nunique()} horizons, test range:", flush=True)
    for proto in ctx.protocols:
        if f"{proto}/test_rmse" in summ:
            name = {"pohl": "immediate feedback", "causal": "delayed feedback"}[proto]
            if ctx.grid.loop == "coupled" and proto == "causal":
                name = "two-network delayed"
            print(f"    {name:20s} RMSE {summ[f'{proto}/test_rmse']:.3f} mm   "
                  f"diverged runs {summ[f'{proto}/n_div_total']:.0f}", flush=True)


# ----------------------------------------------------------------------------- main
class Ctx:
    pass


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Run one sweep config (configs/sweeps/*.json); see README.md.")
    p.add_argument("config", help="sweep config, e.g. configs/sweeps/hcrsnn_sod_df_disp.json")
    p.add_argument("--out", default=None, help="output prefix (default results/<config name>)")
    p.add_argument("--seqs", default="all", help="records, e.g. 'all', '2,5,8', '1-9'")
    p.add_argument("--horizons", default="all", help="'all', samples '1,5,10' or seconds '0.3s,1.0s'")
    p.add_argument("--seeds-cv", type=int, default=20)
    p.add_argument("--seeds-test", type=int, default=50)
    p.add_argument("--max-batch", type=int, default=8192, help="max vmapped jobs per device call")
    p.add_argument("--mem-gb", type=float, default=20.0, help="device memory budget for batch sizing")
    p.add_argument("--allow-cpu", action="store_true", help="do not fail when JAX has no GPU")
    p.add_argument("--tag", default="", help="prefix for the header line, e.g. '[3/28]' (scripts/run_paper.sh)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    grid: Grid = load_grid(args.config)
    method = grid.method
    if method not in METHODS:
        raise SystemExit(f"{grid.path.name}: unknown method {method!r}; expected one of {METHODS}")
    out = Path(args.out) if args.out else RESULTS_DIR / grid.name
    out = out if out.is_absolute() else ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)

    ctx = Ctx()
    ctx.grid = grid
    ctx.seqs = parse_seqs(args.seqs)
    ctx.horizons = parse_horizons(args.horizons, RATE)
    ctx.max_batch, ctx.mem_gb = args.max_batch, args.mem_gb
    ctx.protocols = PROTOCOLS
    ctx.stored_nbuf = 0
    if grid.loop == "stored":
        if method == "persistence":
            raise SystemExit("loop 'stored' applies to online learners only")
        ctx.protocols = ("causal",)
        ctx.stored_nbuf = max(ctx.horizons)
    deterministic = method not in SEEDED
    ctx.seeds_cv = 1 if deterministic else args.seeds_cv
    ctx.seeds_test = 0 if deterministic else args.seeds_test
    ctx.grid_csv = out.with_name(out.name + "_grid.csv")
    best_csv = out.with_name(out.name + "_best.csv")
    shutil.copyfile(grid.path, out.with_name(out.name + "_config.json"))

    t_start = time.perf_counter()
    device = "host (NumPy)"
    if method != "persistence":
        import jax

        backend = jax.default_backend()
        if backend != "gpu" and not args.allow_cpu:
            raise SystemExit("JAX sees no GPU (README, Setup). Fix the install or pass --allow-cpu.")
        device = f"{jax.devices()[0].device_kind} (JAX {jax.__version__}, {backend})"

    from dataset.records import load_records

    ctx.records = load_records(RATE, ctx.seqs)
    ctx.done = done_keys(read_rows(ctx.grid_csv))
    plan = build_plan(ctx) if method != "persistence" else {}
    print_header(ctx, args, plan, device, out)

    if method == "persistence":
        run_persistence(ctx)
    else:
        run_online_phase(ctx, "cv", plan)
        if ctx.seeds_test > 0:
            from training.selection import select

            allrows = read_rows(ctx.grid_csv)
            cvr = allrows[(allrows["phase"] == "cv") & (allrows["n_seeds"] == ctx.seeds_cv)]
            cvr = cvr[cvr["static_key"].isin(plan) & cvr["record"].isin(ctx.seqs) & cvr["h"].isin(ctx.horizons)]
            chosen = select(cvr, grid.select_lr, ctx.protocols, stable_rule(ctx))
            for item in plan.values():
                item["cells"] = {}
            for _, r in chosen.iterrows():
                cells = plan[r["static_key"]]["cells"].setdefault(int(r["record"]), [])
                cell = (int(r["h"]), float(r["lr"]))
                if cell not in cells:
                    cells.append(cell)
            run_online_phase(ctx, "test", plan)

    finalize_outputs(ctx, best_csv)
    print(f"  finished {grid.name} in {hms(time.perf_counter() - t_start)} -> {best_csv.name}, "
          f"{ctx.grid_csv.name}", flush=True)
    return 0


def print_header(ctx, args, plan: dict, device: str, out: Path) -> None:
    grid = ctx.grid
    method = grid.method
    what = METHOD_NAMES[method]
    if method == "rsnn":
        cfg = next(iter(plan.values()))["resolved"]
        enc = {"analog": "analog", "sod": "SOD event"}[cfg["encoding"]]
        what = f"{NETWORK_NAMES[cfg['network']]} with {enc} input, e-prop"
        if cfg["encoding"] == "sod":
            what += f", SOD threshold {cfg['sod_threshold']:g}"
    output = ""
    if plan:
        output = next(iter(plan.values()))["resolved"].get("target", "")
    rule = f", stable-lr rule (kappa {grid.stable_select:g})" if grid.stable_select is not None else ""
    width = 100
    print("\n" + "=" * width, flush=True)
    print(f"{args.tag + ' ' if args.tag else ''}{grid.name}", flush=True)
    if grid.raw.get("description"):
        import textwrap

        print(textwrap.fill(grid.raw["description"], width, initial_indent="  ", subsequent_indent="  "), flush=True)
    print("-" * width, flush=True)
    rows = [("model", what + (f", {output} outputs" if output else "")),
            ("training", LOOP_NAMES[grid.loop] if method != "persistence" else "none (no learning)")]
    if plan:
        shls = sorted({it["resolved"]["shl_s"] for it in plan.values()})
        static = f"{len(plan)} static configs (SHL {', '.join(f'{v:g}' for v in shls)} s"
        qs = sorted({it["resolved"]["q"] for it in plan.values() if "q" in it["resolved"]})
        static += f"; q {', '.join(map(str, qs))})" if qs else ")"
        lrs = ", ".join(f"{v:g}" for v in (grid.lrs or [0.0]))
        n_cv = len(plan) * len(ctx.records) * len(ctx.horizons) * max(len(grid.lrs), 1) * ctx.seeds_cv
        rows += [("grid", f"{static} x learning rates {lrs}"),
                 ("selection", f"per (record, horizon): fewest diverged seeds, then lowest validation RMSE{rule}"),
                 ("seeds", f"{ctx.seeds_cv} validation, {ctx.seeds_test} test (selected configurations only)"),
                 ("validation", f"{n_cv:,} runs")]
    hs = ctx.horizons
    horizons = (f"{', '.join(f'{h / 10:g}' for h in hs)} s" if len(hs) <= 6
                else f"{len(hs)}, from {hs[0] / 10:g} to {hs[-1] / 10:g} s")
    target = out.with_name(out.name + "_{grid,best}.csv")
    try:
        target = target.relative_to(ROOT)
    except ValueError:
        pass
    rows += [("records", f"{', '.join(map(str, ctx.seqs))}"),
             ("horizons", horizons),
             ("device", device),
             ("output", str(target))]
    if ctx.done:
        rows.append(("resuming", f"{len(ctx.done):,} rows already in {ctx.grid_csv.name} are skipped"))
    for k, v in rows:
        print(f"  {k:<11}{v}", flush=True)
    print("-" * width, flush=True)
