from __future__ import annotations

from typing import Any, Protocol

import jax
import jax.numpy as jnp


class OnlineModel(Protocol):
    k: int
    displacement: bool

    def window(self, data: dict, i) -> Any: ...

    def init(self, key, dtype) -> tuple[Any, Any, Any, Any]:
        """-> (params, consts, learner_state, predictor_state)."""

    def learn(self, params, consts, lstate, xwin, target, dyn: dict, j) -> tuple[Any, Any, Any]:
        """Forward window, compute error against ``target``, update. -> (params, lstate, y)."""

    def predict(self, params, consts, pstate, xwin, dyn: dict, count) -> tuple[Any, Any, dict]:
        """Forward window with fixed params. -> (pstate, y, aux scalars)."""


def tree_where(cond, new, old):
    return jax.tree.map(lambda n, o: jnp.where(cond, n, o), new, old)


def run_coupled(model: OnlineModel, data: dict, T, h, dyn: dict, init_state, b2: int):
    """Scan j over 0 .. T_pad - k.

    data: {"z": (T_pad, 9) normalised record (zero padded), ...model-specific arrays}
    T: real record length (traced int); h: horizon in samples (traced int).
    b2: start of the test range (target time, samples) used for predictor spike counters.

    Returns (y_pohl, y_causal, aux, final_state); y_* are (L, 9) normalised *absolute* forecasts
    indexed by window i (target time i+k-1+h).
    """
    z = data["z"]
    k = model.k
    T_pad = z.shape[0]
    L = T_pad - k + 1
    M = T - k - h + 1
    disp = model.displacement

    def step(carry, j):
        params, consts, lstate, pstate = carry
        # learner: window i = j - h, target z[j+k-1] (just observed)
        i = j - h
        l_active = (i >= 0) & (i < M)
        ic = jnp.clip(i, 0, L - 1)
        x_l = model.window(data, ic)
        last_l = z[ic + k - 1]
        tgt = z[j + k - 1]
        if disp:
            tgt = tgt - last_l
        new_params, new_l, y_l = model.learn(params, consts, lstate, x_l, tgt, dyn, j)
        params = tree_where(l_active, new_params, params)
        lstate = tree_where(l_active, new_l, lstate)
        # predictor: window j with the weights just updated
        p_active = j < M
        count = p_active & (j + k - 1 + h >= b2)
        x_p = model.window(data, j)
        new_p, y_p, aux = model.predict(params, consts, pstate, x_p, dyn, count)
        pstate = tree_where(p_active, new_p, pstate)
        last_p = z[j + k - 1]
        if disp:
            y_l = y_l + last_l
            y_p = y_p + last_p
        aux = dict(aux)
        aux["counted"] = count.astype(z.dtype)
        return (params, consts, lstate, pstate), (y_l, y_p, aux)

    final, (y_l, y_p, aux) = jax.lax.scan(step, init_state, jnp.arange(L))
    idx = jnp.clip(jnp.arange(L) + h, 0, L - 1)
    y_pohl = jnp.take(y_l, idx, axis=0)
    return y_pohl, y_p, aux, final


def run_stored(model: OnlineModel, data: dict, T, h, dyn: dict, init_state, b2: int, nbuf: int):
    """Delayed feedback with stored sensitivities: one network, no recomputation.

    At step j the forecast for window j is issued and stored together with the sensitivity snapshot
    the model needs to learn from it (model.sense: LMS input, SnAp-1 influence matrix, e-prop traces).
    The target of window i = j - h, z[j+k-1], has just been observed: before forecasting, the weights
    are updated from the stored forecast and snapshot of window i (model.apply). The gradient is thus
    h steps stale, and it belongs to the forecast that was actually issued and evaluated.
    The error is recomputed at update time from the stored activity read out with the current readout weights
    (LMS: W u; SnAp-1: Wc x; RSNN: stored forecast corrected by the readout change), so readout updates made
    since the forecast was issued are not over-corrected; the sensitivities stay the stored ones.
    nbuf (static) >= h is the ring-buffer length. At h = 1 this is exactly Pohl et al.'s loop, and it
    equals run_coupled's causal stream.

    Returns (y_causal, aux, final_state); y_causal is (L, 9) normalised absolute forecasts by window,
    final_state[3] = {"counts": per-neuron spike counts over the test range} for spiking models.
    """
    z = data["z"]
    k = model.k
    L = z.shape[0] - k + 1
    M = T - k - h + 1
    disp = model.displacement
    params, consts, lstate, _ = init_state
    x0 = model.window(data, 0)
    shapes = jax.eval_shape(lambda: model.sense(params, consts, lstate, x0, dyn, False))
    _, y_shape, sens_shape, aux_shape = shapes
    sbuf = jax.tree.map(lambda a: jnp.zeros((nbuf,) + a.shape, a.dtype), sens_shape)
    ybuf = jnp.zeros((nbuf,) + y_shape.shape, y_shape.dtype)
    counts = jnp.zeros(aux_shape["zsum"].shape, z.dtype) if "zsum" in aux_shape else jnp.zeros((0,), z.dtype)

    def step(carry, j):
        params, lstate, sbuf, ybuf, counts = carry
        # 1) target of window i = j - h observed now: update from the stored forecast and snapshot
        i = j - h
        u_active = (i >= 0) & (i < M)
        slot_i = jnp.mod(i, nbuf)
        ic = jnp.clip(i, 0, L - 1)
        tgt = z[j + k - 1]
        if disp:
            tgt = tgt - z[ic + k - 1]
        sens_i = jax.tree.map(lambda b: b[slot_i], sbuf)
        new_params = model.apply(params, consts, sens_i, ybuf[slot_i], tgt, dyn, j)
        params = tree_where(u_active, new_params, params)
        # 2) issue and store the forecast of window j
        p_active = j < M
        count = p_active & (j + k - 1 + h >= b2)
        new_l, y, sens, aux = model.sense(params, consts, lstate, model.window(data, j), dyn, count)
        lstate = tree_where(p_active, new_l, lstate)
        slot_j = jnp.mod(j, nbuf)
        sbuf = jax.tree.map(lambda b, v: b.at[slot_j].set(jnp.where(p_active, v, b[slot_j])), sbuf, sens)
        ybuf = ybuf.at[slot_j].set(jnp.where(p_active, y, ybuf[slot_j]))
        aux = dict(aux)
        if "zsum" in aux:
            counts = counts + aux.pop("zsum")
        aux["counted"] = count.astype(z.dtype)
        y_out = y + z[j + k - 1] if disp else y
        return (params, lstate, sbuf, ybuf, counts), (y_out, aux)

    (params, lstate, _, _, counts), (y_c, aux) = jax.lax.scan(
        step, (params, lstate, sbuf, ybuf, counts), jnp.arange(L))
    return y_c, aux, (params, consts, lstate, {"counts": counts})

