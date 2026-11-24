from __future__ import annotations

import jax.numpy as jnp

from models.common import analog_window, clip_by_global_norm


class LMS:
    """y = W u; e = target - y; g = -e u^T, global-norm clipped; W <- W - lr g. W init 0."""

    displacement = False

    def __init__(self, k: int, clip: float = 100.0, target: str = "absolute"):
        self.k = k
        self.m = 9 * k + 1
        self.clip = float(clip)
        self.displacement = target == "displacement"

    def window(self, data, i):
        return analog_window(data["z"], i, self.k)

    def init(self, key, dtype):
        return jnp.zeros((9, self.m), dtype), {}, {}, {}

    def learn(self, W, consts, lstate, u, target, dyn, j):
        y = W @ u
        e = target - y
        g, _ = clip_by_global_norm(-jnp.outer(e, u), self.clip)
        return W - dyn["lr"] * g, lstate, y

    def predict(self, W, consts, pstate, u, dyn, count):
        return pstate, W @ u, {}

    # stored-sensitivity delayed feedback (training/loops.py, run_stored): the forecast and its input are stored when the
    # forecast is issued; the update uses them once the target is observed.
    def sense(self, W, consts, lstate, u, dyn, count):
        return lstate, W @ u, {"u": u}, {}

    def apply(self, W, consts, sens, y, target, dyn, j):
        y = W @ sens["u"]  # error of the stored input under the current weights (= recomputing, for LMS)
        e = target - y
        g, _ = clip_by_global_norm(-jnp.outer(e, sens["u"]), self.clip)
        return W - dyn["lr"] * g

    def sens_floats(self) -> int:
        return self.m + 9

    def job_floats(self, L: int) -> int:
        return 4 * 9 * self.m + 2 * self.m
