"""Online models and the factory used by the sweep engine."""

from __future__ import annotations

from dataset.records import shl_samples
from models.lms import LMS
from models.rsnn import NETWORK_PRESETS, ALIFNetwork, SNNConfig, resolve_snn_config
from models.snap1 import SnAp1RNN

ONLINE_METHODS = ("lms", "snap1", "rsnn")

__all__ = [
    "LMS",
    "SnAp1RNN",
    "ALIFNetwork",
    "SNNConfig",
    "NETWORK_PRESETS",
    "resolve_snn_config",
    "build_model",
    "ONLINE_METHODS",
]


def build_model(method: str, model_layers: list[dict], static: dict, rate: str):
    """Build an online model.

    model_layers: model config dicts from the sweep config (base file first), static: this static grid point
    (may contain 'shl_s' in seconds, converted to k samples, plus model-field overrides).
    Returns (model, resolved) where resolved is the full flat config used for logging / hashing.
    """
    static = dict(static)
    shl_s = static.pop("shl_s", None)
    merged: dict = {}
    for lay in model_layers:
        merged.update(lay)
    if shl_s is None:
        shl_s = merged.pop("shl_s", 0.0)
    else:
        merged.pop("shl_s", None)
    k = shl_samples(shl_s, rate)
    if method == "lms":
        merged.update(static)
        model = LMS(k, clip=merged.get("clip", 100.0), target=merged.get("target", "absolute"))
        resolved = {"k": k, "clip": model.clip, "target": "displacement" if model.displacement else "absolute"}
    elif method == "snap1":
        merged.update(static)
        model = SnAp1RNN(
            k,
            q=int(merged["q"]),
            clip=merged.get("clip", 100.0),
            init_std=merged.get("init_std", 0.02),
            target=merged.get("target", "absolute"),
        )
        resolved = {
            "k": k,
            "q": model.q,
            "clip": model.clip,
            "init_std": model.init_std,
            "target": "displacement" if model.displacement else "absolute",
        }
    elif method == "rsnn":
        layers = list(model_layers) + [static]
        cfg = resolve_snn_config(*[{k_: v for k_, v in lay.items() if k_ != "shl_s"} for lay in layers])
        model = ALIFNetwork(cfg, k, rate)
        resolved = {"k": k, **cfg.to_dict()}
    else:
        raise ValueError(f"unknown online method {method}; expected one of {ONLINE_METHODS}")
    resolved["shl_s"] = float(shl_s)
    return model, resolved
