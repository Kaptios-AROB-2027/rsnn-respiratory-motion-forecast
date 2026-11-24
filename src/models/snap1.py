from __future__ import annotations

import jax
import jax.numpy as jnp

from models.common import analog_window, clip_by_global_norm


class SnAp1RNN:
    displacement = False

    def __init__(self, k: int, q: int, clip: float = 100.0, init_std: float = 0.02, target: str = "absolute"):
        self.k = int(k)
        self.q = int(q)
        self.m = 9 * self.k + 1
        self.clip = float(clip)
        self.init_std = float(init_std)
        self.displacement = target == "displacement"

    def window(self, data, i):
        return analog_window(data["z"], i, self.k)

    def init(self, key, dtype):
        q, m = self.q, self.m
        ka, kb, kc = jax.random.split(key, 3)
        params = {
            "Wa": self.init_std * jax.random.normal(ka, (q, q), dtype),
            "Wb": self.init_std * jax.random.normal(kb, (q, m), dtype),
            "Wc": self.init_std * jax.random.normal(kc, (9, q), dtype),
        }
        lstate = {"x": jnp.zeros((q,), dtype), "J": jnp.zeros((q, q + m), dtype)}
        pstate = {"x": jnp.zeros((q,), dtype)}
        return params, {}, lstate, pstate

    @staticmethod
    def forward(params, x, u):
        x_new = jnp.tanh(params["Wa"] @ x + params["Wb"] @ u)
        return x_new, params["Wc"] @ x_new

    def _forward_snap(self, params, lstate, u):
        """Forward one window and advance the SnAp-1 influence matrix J (q, q + m)."""
        x_old = lstate["x"]
        x_new, y = self.forward(params, x_old, u)
        dphi = 1.0 - x_new * x_new
        D = dphi * jnp.diagonal(params["Wa"])
        I = dphi[:, None] * jnp.concatenate([x_old, u])[None, :]
        J = I + D[:, None] * lstate["J"]
        return x_new, y, J

    def _update(self, params, J, x_new, y, target, dyn):
        q = self.q
        e = target - y
        dx = -(params["Wc"].T @ e)
        gW = dx[:, None] * J
        grads = {"Wa": gW[:, :q], "Wb": gW[:, q:], "Wc": -jnp.outer(e, x_new)}
        grads, _ = clip_by_global_norm(grads, self.clip)
        lr = dyn["lr"]
        return jax.tree.map(lambda p, g: p - lr * g, params, grads)

    def learn(self, params, consts, lstate, u, target, dyn, j):
        x_new, y, J = self._forward_snap(params, lstate, u)
        params = self._update(params, J, x_new, y, target, dyn)
        return params, {"x": x_new, "J": J}, y

    # stored-sensitivity delayed feedback (training/loops.py, run_stored): one network; J and the hidden state of each
    # forecast are stored when it is issued and used for the update once its target is observed. The error
    # uses the stored hidden state read out with the current Wc and is back-projected through the current Wc.
    def sense(self, params, consts, lstate, u, dyn, count):
        x_new, y, J = self._forward_snap(params, lstate, u)
        return {"x": x_new, "J": J}, y, {"J": J, "x": x_new}, {}

    def apply(self, params, consts, sens, y, target, dyn, j):
        y = params["Wc"] @ sens["x"]  # stored hidden state read out with the current Wc
        return self._update(params, sens["J"], sens["x"], y, target, dyn)

    def sens_floats(self) -> int:
        return self.q * (self.q + self.m) + self.q + 9

    def predict(self, params, consts, pstate, u, dyn, count):
        x_new, y = self.forward(params, pstate["x"], u)
        return {"x": x_new}, y, {}

    def job_floats(self, L: int) -> int:
        q, m = self.q, self.m
        # params, J, gradient temporaries
        return 4 * (q * q + q * m + 9 * q) + 3 * q * (q + m)
