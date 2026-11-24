from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

from dataset.records import SPLITS, Record
from dataset.sod import encoder_stats, pad_spikes, sod_encode
from evaluation.metrics import METRICS, PROTOCOLS, RANGES, evaluate_windows
from training.loops import run_coupled, run_stored

BASE_SEED = 20261028
TEST_SEED_OFFSET = 100_000  # phase-2 (test) seeds are disjoint from CV seeds


def _next_pow2(n: int) -> int:
    return 1 << max(0, int(n - 1).bit_length())


_ENC_CACHE: dict = {}


def device_data(rec: Record, model, dtype=jnp.float32) -> tuple[dict, dict]:
    """Device arrays for one record (+ encoder diagnostics for spike-input models)."""
    np_dtype = np.dtype(dtype)
    data = {
        "z": jnp.asarray(rec.zn_pad(np_dtype)),
        "mean": jnp.asarray(rec.mean.astype(np_dtype)),
        "std": jnp.asarray(rec.std.astype(np_dtype)),
    }
    info: dict = {}
    if getattr(model, "sod", False):
        K, thr = model.cfg.K, model.cfg.sod_threshold
        key = (rec.rate, rec.seq, K, thr)
        if key not in _ENC_CACHE:
            spk, recon = sod_encode(rec.zn, K, thr)
            _ENC_CACHE[key] = (spk, encoder_stats(rec.zn, spk, recon, model.rate_hz))
        spk, info = _ENC_CACHE[key]
        data["spk"] = jnp.asarray(pad_spikes(spk, rec.T_pad, np_dtype))
    return data, dict(info)


def jobs_for_memory(model, T_pad: int, mem_gb: float, dtype_bytes: int = 4, nbuf: int = 0) -> int:
    L = T_pad - model.k + 1
    floats = model.job_floats(L) + L * (2 * 9 * 3 + 8)
    if nbuf:  # stored-sensitivity ring buffer
        floats += nbuf * model.sens_floats()
    per_job = floats * dtype_bytes * 1.5
    return max(1, int(mem_gb * 1e9 // per_job))


def aggregate(flat: dict, S: int) -> dict:
    res = {}
    for proto in PROTOCOLS:
        fin = flat[f"{proto}/finite"]
        cnt = fin.sum(axis=1)
        denom = jnp.maximum(cnt, 1)
        res[f"{proto}_n_div"] = S - cnt
        for r in RANGES:
            for m in METRICS:
                v = flat[f"{proto}/{r}/{m}"]
                mean = jnp.where(fin, v, 0.0).sum(axis=1) / denom
                var = jnp.where(fin, (v - mean[:, None]) ** 2, 0.0).sum(axis=1) / denom
                res[f"{proto}_{r}_{m}_mean"] = jnp.where(cnt > 0, mean, jnp.nan)
                res[f"{proto}_{r}_{m}_std"] = jnp.where(cnt > 0, jnp.sqrt(var), jnp.nan)
    fin = flat["causal/finite"]
    denom = jnp.maximum(fin.sum(axis=1), 1)
    for key, v in flat.items():
        if key.startswith("cost/"):
            res[key[5:] + "_mean"] = jnp.where(fin, v, 0.0).sum(axis=1) / denom
    return res


class BatchRunner:
    """Runs many (h, lr, seed) jobs of one static configuration on one record at a time."""

    def __init__(self, model, rate: str, base_seed: int = BASE_SEED, stored_nbuf: int = 0):
        """stored_nbuf > 0: delayed feedback with stored sensitivities (loops.run_stored, ring buffer of
        that length, >= the largest horizon); only the causal stream is meaningful (the 'pohl' columns
        then repeat it and must not be used). stored_nbuf = 0: loops.run_coupled."""
        self.stored_nbuf = int(stored_nbuf)
        self.model = model
        self.rate = rate
        self.b1, self.b2 = SPLITS[rate]
        self.base_seed = base_seed
        self._fns: dict = {}
        self.compile_s = 0.0

    def _job(self, data, T, h, lr, seed):
        model = self.model
        key = jax.random.fold_in(jax.random.PRNGKey(self.base_seed), seed)
        dtype = data["z"].dtype
        init = model.init(key, dtype)
        dyn = {"lr": lr.astype(dtype)}
        if self.stored_nbuf:
            y_causal, aux, final = run_stored(model, data, T, h, dyn, init, self.b2, self.stored_nbuf)
            y_pohl = y_causal
        else:
            y_pohl, y_causal, aux, final = run_coupled(model, data, T, h, dyn, init, self.b2)
        out = {}
        for proto, Y in (("pohl", y_pohl), ("causal", y_causal)):
            ev = evaluate_windows(jnp, Y, data["z"], data["mean"], data["std"], model.k, h, T, self.b1, self.b2)
            out[f"{proto}/finite"] = ev["cv"]["finite"] & ev["test"]["finite"]
            for r in RANGES:
                for m in METRICS:
                    out[f"{proto}/{r}/{m}"] = ev[r][m]
        if hasattr(model, "cost_stats"):
            for k, v in model.cost_stats(aux, final[3]).items():
                out[f"cost/{k}"] = v
        return out

    def _fn(self, n_c: int, S: int):
        key = (n_c, S)
        if key not in self._fns:

            def batch(data, T, h, lr, seed):
                flat = jax.vmap(self._job, in_axes=(None, None, 0, 0, 0))(
                    data, T, h.reshape(-1), lr.reshape(-1), seed.reshape(-1)
                )
                flat = {k: v.reshape(n_c, S) for k, v in flat.items()}
                return aggregate(flat, S)

            self._fns[key] = jax.jit(batch)
        return self._fns[key]

    def run(self, data: dict, T: int, configs: list[tuple[int, float]], seeds: list[int], max_jobs: int,
            on_batch=None):
        """configs: list of (h, lr). Returns (list of result dicts aligned with configs, stats).
        on_batch(n_runs): optional callback after each device call (n_runs = configs x seeds just finished)."""
        S = len(seeds)
        n_cfg = len(configs)
        if n_cfg == 0:
            return [], {"jobs": 0, "wall_s": 0.0, "batch_jobs": 0}
        n_c = max(1, min(_next_pow2(n_cfg), max_jobs // max(S, 1)))
        fn = self._fn(n_c, S)
        T_arr = jnp.asarray(T, jnp.int32)
        lr_dtype = np.dtype(data["z"].dtype)
        results: list[dict] = []
        t0 = time.perf_counter()
        first = True
        for start in range(0, n_cfg, n_c):
            chunk = list(configs[start : start + n_c])
            chunk_p = chunk + [chunk[0]] * (n_c - len(chunk))
            h = np.array([[c[0]] * S for c in chunk_p], dtype=np.int32)
            lr = np.array([[c[1]] * S for c in chunk_p], dtype=lr_dtype)
            sd = np.tile(np.asarray(seeds, dtype=np.int32), (n_c, 1))
            tc = time.perf_counter()
            out = jax.device_get(fn(data, T_arr, h, lr, sd))
            if first:
                self.compile_s += time.perf_counter() - tc  # includes compilation on first call
                first = False
            for i in range(len(chunk)):
                results.append({k: float(np.asarray(v)[i]) for k, v in out.items()})
            if on_batch is not None:
                on_batch(len(chunk) * S)
        wall = time.perf_counter() - t0
        L = int(data["z"].shape[0]) - self.model.k + 1
        jobs = n_cfg * S
        stats = {
            "jobs": jobs,
            "batch_jobs": n_c * S,
            "wall_s": wall,
            "jobs_per_s": jobs / max(wall, 1e-9),
            "window_steps_per_s": jobs * L / max(wall, 1e-9),
        }
        return results, stats
