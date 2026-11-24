"""Recurrent spiking network of LIF / ALIF neurons trained online with e-prop (Bellec et al. 2020).

One implementation, two networks (NETWORK_PRESETS) and two input encodings (``encoding``):

    lsnn     LIF + ALIF neurons, symmetric feedback (B = W_out^T), full eligibility traces, float32 weights
    hcrsnn   hardware-constrained (ReckOn-like): LIF only, fixed random feedback, local pre x post traces,
             8-bit stochastic-rounded weights
    analog   the input window is fed as an input current (9k + 1 lines, zero-order hold)
    sod      send-on-delta ON/OFF events per coordinate, presented through a spike tapped delay line
             (18k + 1 lines)

Per substep (dt = 1 / (rate K)), input held constant within a marker sample:

    v <- alpha v + W_in x + W_rec z - z v_th          (z = previous spikes; reset by subtraction)
    a <- rho a + z
    A = v_th + beta a
    z = H(v - A),  psi = (gamma / v_th) max(0, 1 - |v - A| / v_th)
    y <- kappa y + (1 - kappa)(W_out z + b)

e-prop traces (bar = alpha-filtered presynaptic trace; p = x for inputs, p = previous z for recurrent):

    bar      <- alpha bar + p
    eps_a    <- psi_old (x) bar_old + (rho - psi_old beta) (.) eps_a          (ALIF rows only)
    e         = psi (x) bar - (psi beta) (.) eps_a
    ebar     <- kappa ebar + (1 - kappa) e
    zbar     <- kappa zbar + (1 - kappa) z

Update once per marker sample (the loss exists only at sample instants):

    err = y - target;  L_j = sum_k B_jk err_k + c_reg (fbar_j - f_target)
    dW_in = -eta L (x) ebar_in;  dW_rec = -eta (L (x) ebar_rec) (.) mask
    dW_out = -eta_out err (x) zbar;  db = -eta_out err          (global-norm clipped)

hcrsnn local traces: ebar_ji ~= psibar_j bar_i (psibar = kappa-filtered psi), memory O(N + n_in).
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp

from dataset.records import RATE_HZ, rate_key
from models.common import analog_window, clip_by_global_norm

NETWORK_PRESETS: dict[str, dict] = {
    "lsnn": {"feedback": "symmetric", "traces": "full", "quant_bits": 0},
    "hcrsnn": {"feedback": "random", "traces": "local", "quant_bits": 8, "frac_adaptive": 0.0},
}


@dataclass(frozen=True)
class SNNConfig:
    network: str = "lsnn"  # lsnn | hcrsnn (structural switches in NETWORK_PRESETS)
    N: int = 200
    K: int = 10  # substeps per marker sample
    frac_adaptive: float = 0.4
    tau_a_min: float = 0.5  # s, log-uniform
    tau_a_max: float = 6.0
    f_target: float = 10.0  # Hz
    tau_m: float = 0.020  # s
    tau_out: float = 0.030  # s
    v_th: float = 1.0
    gamma: float = 0.3
    p_rec: float = 0.2
    in_gain: float = 0.4  # W_in ~ N(0, in_gain^2 / (n_in - 1))
    bias_std: float = 0.3  # per-neuron bias column
    rec_gain: float = 0.5  # W_rec ~ N(0, rec_gain^2 / (p_rec N))
    out_var: float = 0.01  # W_out ~ N(0, out_var / N)
    c_reg: float = 0.0  # firing-rate regulariser weight (learning-signal units per Hz)
    f_ema_tau: float = 5.0  # s, firing-rate estimate
    out_lr_ratio: float = 1.0  # eta_out = out_lr_ratio * eta
    clip: float = 100.0
    encoding: str = "analog"  # analog | sod
    sod_threshold: float = 0.005  # send-on-delta threshold (normalised units)
    feedback: str = "symmetric"  # symmetric | random
    feedback_std: float = 0.0  # random feedback std; 0 -> 1/sqrt(N)
    traces: str = "full"  # full | local
    quant_bits: int = 0  # 0 = float32 weights
    quant_mult_in: float = 4.0  # quantisation range = mult x init std, per matrix
    quant_mult_bias: float = 4.0
    quant_mult_rec: float = 4.0
    quant_mult_out: float = 4.0
    target: str = "displacement"  # displacement | absolute
    extra: dict = field(default_factory=dict, compare=False, hash=False)

    @classmethod
    def from_dict(cls, d: dict) -> SNNConfig:
        names = {f.name for f in dataclasses.fields(cls)}
        known = {k: v for k, v in d.items() if k in names and k != "extra"}
        unknown = {k: v for k, v in d.items() if k not in names}
        if unknown:
            raise ValueError(f"unknown SNN config keys: {sorted(unknown)}")
        return cls(**known)

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d.pop("extra", None)
        return d

    @property
    def n_adaptive(self) -> int:
        return int(round(self.frac_adaptive * self.N))


def resolve_snn_config(*layers: dict) -> SNNConfig:
    """defaults < layers[0] (configs/rsnn_base.json) < NETWORK_PRESETS[network] < later layers (sweep config).

    ``network`` is taken from the last layer that sets it (default lsnn)."""
    network = "lsnn"
    for lay in layers:
        network = lay.get("network", network)
    if network not in NETWORK_PRESETS:
        raise ValueError(f"unknown network {network!r}; expected one of {sorted(NETWORK_PRESETS)}")
    merged: dict = {}
    if layers:
        merged.update(layers[0])
    merged.update(NETWORK_PRESETS[network])
    for lay in layers[1:]:
        merged.update(lay)
    merged["network"] = network
    return SNNConfig.from_dict(merged)


def stochastic_round(w, step, qmax, key):
    u = jax.random.uniform(key, w.shape, w.dtype)
    q = jnp.clip(jnp.floor(w / step + u), -qmax, qmax)
    return q * step


def nearest_round(w, step, qmax):
    return jnp.clip(jnp.round(w / step), -qmax, qmax) * step


class ALIFNetwork:
    def __init__(self, cfg: SNNConfig, k: int, rate: str, n_ch: int = 9):
        self.cfg = cfg
        self.n_ch = n_ch  # coordinates in and out
        self.k = int(k)
        self.rate = rate_key(rate)
        self.rate_hz = RATE_HZ[self.rate]
        self.dt = 1.0 / (self.rate_hz * cfg.K)
        self.displacement = cfg.target == "displacement"
        self.sod = cfg.encoding == "sod"
        self.n_in = (2 * n_ch if self.sod else n_ch) * self.k + 1
        self.n_ad = cfg.n_adaptive
        self.full = cfg.traces == "full"
        for name, val, allowed in (
            ("encoding", cfg.encoding, ("analog", "sod")),
            ("feedback", cfg.feedback, ("symmetric", "random")),
            ("traces", cfg.traces, ("full", "local")),
            ("target", cfg.target, ("displacement", "absolute")),
        ):
            if val not in allowed:
                raise ValueError(f"{name}={val!r}, expected one of {allowed}")
        self.alpha = math.exp(-self.dt / cfg.tau_m)
        self.kappa = math.exp(-self.dt / cfg.tau_out)
        self.lam_f = math.exp(-self.dt / cfg.f_ema_tau)
        N = cfg.N
        self.std_in = cfg.in_gain / math.sqrt(max(self.n_in - 1, 1))
        self.std_rec = cfg.rec_gain / math.sqrt(max(cfg.p_rec * N, 1e-12))
        self.std_out = math.sqrt(cfg.out_var / N)
        self.qmax = float(2 ** (cfg.quant_bits - 1) - 1) if cfg.quant_bits else 0.0

    # ------------------------------------------------------------------ inputs
    def window(self, data, i):
        if self.sod:
            spk = data["spk"]  # (T_pad, K, 18)
            return jax.lax.dynamic_slice(spk, (i, 0, 0), (self.k, spk.shape[1], spk.shape[2]))
        return analog_window(data["z"], i, self.k)

    def inputs(self, params, xwin):
        """-> X (K, n_in) presynaptic input per substep, I (K, N) input current per substep."""
        K = self.cfg.K
        if self.sod:
            Xs = jnp.transpose(xwin, (1, 0, 2)).reshape(K, -1)  # (K, k*18): samples i..i+k-1 at substep s
            X = jnp.concatenate([jnp.ones((K, 1), Xs.dtype), Xs], axis=1)
            return X, X @ params["W_in"].T
        X = jnp.broadcast_to(xwin, (K, xwin.shape[0]))
        I = params["W_in"] @ xwin  # zero-order hold: once per sample
        return X, jnp.broadcast_to(I, (K, I.shape[0]))

    # ------------------------------------------------------------------ init
    def init(self, key, dtype):
        cfg, N, n_in = self.cfg, self.cfg.N, self.n_in
        ks = jax.random.split(key, 8)
        W = self.std_in * jax.random.normal(ks[0], (N, n_in - 1), dtype)
        bias = cfg.bias_std * jax.random.normal(ks[1], (N, 1), dtype)
        W_in = jnp.concatenate([bias, W], axis=1)  # column 0 multiplies the constant-1 input
        mask = (jax.random.uniform(ks[2], (N, N)) < cfg.p_rec) & ~jnp.eye(N, dtype=bool)
        mask = mask.astype(dtype)
        W_rec = self.std_rec * jax.random.normal(ks[3], (N, N), dtype) * mask
        W_out = self.std_out * jax.random.normal(ks[4], (self.n_ch, N), dtype)
        b = jnp.zeros((self.n_ch,), dtype)
        log_tau = jax.random.uniform(
            ks[5], (N,), dtype, math.log(cfg.tau_a_min), math.log(cfg.tau_a_max)
        )
        tau_a = jnp.exp(log_tau)
        adaptive = jnp.arange(N) < self.n_ad  # adaptive neurons are the first n_ad
        beta = jnp.where(adaptive, cfg.v_th / (cfg.f_target * tau_a), 0.0).astype(dtype)
        rho = jnp.exp(-self.dt / tau_a).astype(dtype)
        fb_std = cfg.feedback_std if cfg.feedback_std > 0 else 1.0 / math.sqrt(N)
        consts = {
            "mask": mask,
            "beta": beta,
            "rho": rho,
            "B": fb_std * jax.random.normal(ks[6], (N, self.n_ch), dtype),
            "qkey": ks[7],
        }
        params = {"W_in": W_in, "W_rec": W_rec, "W_out": W_out, "b": b}
        if cfg.quant_bits:
            params = self.quantize(params, consts, None)
        zN = jnp.zeros((N,), dtype)
        pstate = {"v": zN, "a": zN, "z": zN, "y": jnp.zeros((self.n_ch,), dtype), "counts": zN}
        lstate = {
            "v": zN,
            "a": zN,
            "z": zN,
            "psi": zN,
            "y": jnp.zeros((self.n_ch,), dtype),
            "bar_in": jnp.zeros((n_in,), dtype),
            "bar_rec": zN,
            "zbar": zN,
            "fbar": jnp.full((N,), cfg.f_target, dtype),
        }
        if self.full:
            lstate["ebar_in"] = jnp.zeros((N, n_in), dtype)
            lstate["ebar_rec"] = jnp.zeros((N, N), dtype)
            if self.n_ad > 0:
                lstate["eps_in"] = jnp.zeros((self.n_ad, n_in), dtype)
                lstate["eps_rec"] = jnp.zeros((self.n_ad, N), dtype)
        else:
            lstate["psibar"] = zN
        return params, consts, lstate, pstate

    # ------------------------------------------------------------------ quantisation
    def q_steps(self):
        c = self.cfg
        ranges = {
            "in": c.quant_mult_in * self.std_in,
            "bias": c.quant_mult_bias * c.bias_std,
            "rec": c.quant_mult_rec * self.std_rec,
            "out": c.quant_mult_out * self.std_out,
        }
        return {k: r / self.qmax for k, r in ranges.items()}

    def quantize(self, params, consts, j):
        """Nearest rounding at init (j is None), stochastic rounding after every update."""
        st = self.q_steps()
        qmax = self.qmax
        W_in = params["W_in"]
        parts = {"bias": W_in[:, :1], "in": W_in[:, 1:], "rec": params["W_rec"], "out": params["W_out"]}
        out = {}
        for i, (name, w) in enumerate(parts.items()):
            if j is None:
                out[name] = nearest_round(w, st[name], qmax)
            else:
                key = jax.random.fold_in(jax.random.fold_in(consts["qkey"], j), i)
                out[name] = stochastic_round(w, st[name], qmax, key)
        return {
            "W_in": jnp.concatenate([out["bias"], out["in"]], axis=1),
            "W_rec": out["rec"] * consts["mask"],
            "W_out": out["out"],
            "b": params["b"],  # readout bias (9 values) kept in float; reported in the cost table
        }

    # ------------------------------------------------------------------ dynamics
    def neuron_step(self, params, consts, v, a, z_prev, I):
        c = self.cfg
        v = self.alpha * v + I + params["W_rec"] @ z_prev - z_prev * c.v_th
        a = consts["rho"] * a + z_prev
        x = v - (c.v_th + consts["beta"] * a)
        z = (x > 0).astype(v.dtype)
        psi = (c.gamma / c.v_th) * jnp.maximum(0.0, 1.0 - jnp.abs(x) / c.v_th)
        return v, a, z, psi

    def readout(self, params, y, z):
        return self.kappa * y + (1.0 - self.kappa) * (params["W_out"] @ z + params["b"])

    def predict(self, params, consts, pstate, xwin, dyn, count):
        X, I = self.inputs(params, xwin)

        def sub(carry, I_s):
            v, a, z, y = carry
            v, a, z, _ = self.neuron_step(params, consts, v, a, z, I_s)
            y = self.readout(params, y, z)
            return (v, a, z, y), z

        (v, a, z, y), zs = jax.lax.scan(sub, (pstate["v"], pstate["a"], pstate["z"], pstate["y"]), I)
        zsum = zs.sum(axis=0)
        counts = pstate["counts"] + jnp.where(count, zsum, 0.0)
        aux = {
            "rec_spikes": zsum.sum(),
            "in_spikes": X[:, 1:].sum() if self.sod else jnp.zeros((), zsum.dtype),
        }
        return {"v": v, "a": a, "z": z, "y": y, "counts": counts}, y, aux

    def learner_forward(self, params, consts, lstate, X, I):
        """Run the K substeps of one window with fixed weights, updating all traces."""
        al, ka, lam = self.alpha, self.kappa, self.lam_f
        beta, rho, n_ad = consts["beta"], consts["rho"], self.n_ad
        full = self.full

        def sub(s, inp):
            x_s, I_s = inp
            z_prev, psi_old = s["z"], s["psi"]
            bar_in_old, bar_rec_old = s["bar_in"], s["bar_rec"]
            bar_in = al * bar_in_old + x_s
            bar_rec = al * bar_rec_old + z_prev
            v, a, z, psi = self.neuron_step(params, consts, s["v"], s["a"], z_prev, I_s)
            new = dict(s)
            if full:
                e_in = psi[:, None] * bar_in[None, :]
                e_rec = psi[:, None] * bar_rec[None, :]
                if n_ad > 0:
                    po, ba, ro = psi_old[:n_ad], beta[:n_ad], rho[:n_ad]
                    eps_in = po[:, None] * bar_in_old[None, :] + (ro - po * ba)[:, None] * s["eps_in"]
                    eps_rec = po[:, None] * bar_rec_old[None, :] + (ro - po * ba)[:, None] * s["eps_rec"]
                    pb = (psi[:n_ad] * ba)[:, None]
                    e_in = e_in.at[:n_ad].add(-pb * eps_in)
                    e_rec = e_rec.at[:n_ad].add(-pb * eps_rec)
                    new["eps_in"], new["eps_rec"] = eps_in, eps_rec
                new["ebar_in"] = ka * s["ebar_in"] + (1.0 - ka) * e_in
                new["ebar_rec"] = ka * s["ebar_rec"] + (1.0 - ka) * e_rec
            else:
                new["psibar"] = ka * s["psibar"] + (1.0 - ka) * psi
            new.update(
                v=v,
                a=a,
                z=z,
                psi=psi,
                bar_in=bar_in,
                bar_rec=bar_rec,
                zbar=ka * s["zbar"] + (1.0 - ka) * z,
                y=self.readout(params, s["y"], z),
                fbar=lam * s["fbar"] + (1.0 - lam) * z / self.dt,
            )
            return new, z

        lstate, zs = jax.lax.scan(sub, lstate, (X, I))
        return lstate, lstate["y"], zs

    def eprop_grads(self, params, consts, lstate, y, target):
        """Raw e-prop gradient estimates at a sample instant (before clipping)."""
        c = self.cfg
        err = y - target
        B = params["W_out"].T if c.feedback == "symmetric" else consts["B"]
        Lsig = B @ err + c.c_reg * (lstate["fbar"] - c.f_target)
        if self.full:
            g_in = Lsig[:, None] * lstate["ebar_in"]
            g_rec = Lsig[:, None] * lstate["ebar_rec"]
        else:
            lp = Lsig * lstate["psibar"]
            g_in = lp[:, None] * lstate["bar_in"][None, :]
            g_rec = lp[:, None] * lstate["bar_rec"][None, :]
        return {
            "W_in": g_in,
            "W_rec": g_rec * consts["mask"],
            "W_out": jnp.outer(err, lstate["zbar"]),
            "b": err,
        }

    def learn(self, params, consts, lstate, xwin, target, dyn, j):
        X, I = self.inputs(params, xwin)
        lstate, y, _ = self.learner_forward(params, consts, lstate, X, I)
        return self._apply_grads(params, consts, lstate, y, target, dyn, j), lstate, y

    # delayed feedback with stored sensitivities (training/loops.py, run_stored): the eligibility traces, readout trace and
    # rate estimate of each forecast are stored when it is issued and used for the e-prop update once its target
    # is observed (full traces: N (n_in + N) values per stored forecast; local traces: O(N + n_in)).
    SENS_FULL = ("ebar_in", "ebar_rec", "zbar", "fbar")
    SENS_LOCAL = ("psibar", "bar_in", "bar_rec", "zbar", "fbar")

    def sense(self, params, consts, lstate, xwin, dyn, count):
        X, I = self.inputs(params, xwin)
        lstate, y, zs = self.learner_forward(params, consts, lstate, X, I)
        zsum = zs.sum(axis=0)
        aux = {
            "rec_spikes": zsum.sum(),
            "in_spikes": X[:, 1:].sum() if self.sod else jnp.zeros((), zsum.dtype),
            "zsum": jnp.where(count, zsum, 0.0),
        }
        sens = {key: lstate[key] for key in (self.SENS_FULL if self.full else self.SENS_LOCAL)}
        sens["ylin"] = params["W_out"] @ lstate["zbar"] + params["b"]  # readout of zbar at issue time
        return lstate, y, sens, aux

    def apply(self, params, consts, sens, y, target, dyn, j):
        y = y + (params["W_out"] @ sens["zbar"] + params["b"]) - sens["ylin"]  # re-read with the current readout
        return self._apply_grads(params, consts, sens, y, target, dyn, j)

    def sens_floats(self) -> int:
        N, n_in = self.cfg.N, self.n_in
        return (N * n_in + N * N + 2 * N if self.full else 4 * N + n_in) + 2 * self.n_ch

    def _apply_grads(self, params, consts, traces, y, target, dyn, j):
        grads = self.eprop_grads(params, consts, traces, y, target)
        grads, _ = clip_by_global_norm(grads, self.cfg.clip)
        lr = dyn["lr"]
        lr_out = lr * self.cfg.out_lr_ratio
        params = {
            "W_in": params["W_in"] - lr * grads["W_in"],
            "W_rec": params["W_rec"] - lr * grads["W_rec"],
            "W_out": params["W_out"] - lr_out * grads["W_out"],
            "b": params["b"] - lr_out * grads["b"],
        }
        if self.cfg.quant_bits:
            params = self.quantize(params, consts, j)
        return params

    # ------------------------------------------------------------------ bookkeeping
    def cost_stats(self, aux, pstate):
        """Spike statistics of the predictor network over the test range (per marker sample)."""
        n = jnp.maximum(aux["counted"].sum(), 1.0)
        c = aux["counted"]
        rates = pstate["counts"] / (n / self.rate_hz)
        return {
            "rec_spikes_per_sample": (aux["rec_spikes"] * c).sum() / n,
            "in_spikes_per_sample": (aux["in_spikes"] * c).sum() / n,
            "mean_rate_hz": rates.mean(),
            "silent_frac": (rates < 0.1).astype(rates.dtype).mean(),
        }

    def job_floats(self, L: int) -> int:
        N, n_in, n_ad, K = self.cfg.N, self.n_in, self.n_ad, self.cfg.K
        params = N * n_in + 2 * N * N + 2 * self.n_ch * N
        if self.full:
            traces = 2 * (N * n_in + N * N) + n_ad * (n_in + N)
        else:
            traces = 3 * N + n_in
        temps = 2 * (N * n_in + N * N) + K * (n_in + N)
        return 2 * params + 2 * traces + temps

