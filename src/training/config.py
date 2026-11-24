from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path

from paths import CONFIG_DIR, resolve_config

META_KEYS = {"base", "description", "method", "notes"}
LOOPS = ("coupled", "stored")


def _strip(d: dict) -> dict:
    return {k: v for k, v in d.items() if k not in META_KEYS and not k.startswith("_")}


def load_model_chain(name: str | Path, _seen=None) -> list[dict]:
    """Follow "base" links; returns layers from the most basic to the most specific."""
    _seen = set() if _seen is None else _seen
    p = resolve_config(name) if not Path(name).is_absolute() else Path(name)
    if p in _seen:
        raise ValueError(f"cyclic base chain at {p}")
    _seen.add(p)
    d = json.loads(p.read_text())
    layers = []
    if "base" in d:
        base = d["base"]
        base_path = (p.parent / base) if (p.parent / base).exists() else (CONFIG_DIR / base)
        layers = load_model_chain(base_path, _seen)
    layers.append(_strip(d))
    return layers


@dataclass
class Grid:
    path: Path
    raw: dict
    method: str
    model_layers: list[dict]
    static_points: list[dict]
    loop: str = "coupled"
    stable_select: float | None = None
    lrs: list[float] = field(default_factory=list)
    select_lr: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.path.stem


def load_grid(path: str | Path) -> Grid:
    p = resolve_config(path)
    raw = json.loads(p.read_text())
    method = raw.get("method")
    if method is None:
        raise ValueError(f"{p}: 'method' missing")
    loop = raw.get("loop", "coupled")
    if loop not in LOOPS:
        raise ValueError(f"{p}: loop {loop!r}, expected one of {LOOPS}")
    layers: list[dict] = []
    if raw.get("model_file"):
        layers.extend(load_model_chain(raw["model_file"]))
    if raw.get("model"):
        layers.append(dict(raw["model"]))
    static = raw.get("static", {}) or {}
    names = list(static)
    values = [v if isinstance(v, list) else [v] for v in static.values()]
    points = []
    for combo in itertools.product(*values) if names else [()]:
        points.append(dict(zip(names, combo)))
    stable = raw.get("stable_select")
    return Grid(
        path=p,
        raw=raw,
        method=method,
        model_layers=layers,
        static_points=points,
        loop=loop,
        stable_select=float(stable) if stable is not None else None,
        lrs=[float(x) for x in raw.get("lr", [])],
        select_lr={k: [float(x) for x in v] for k, v in raw.get("select_lr", {}).items()},
    )


def static_key(method: str, resolved: dict) -> str:
    blob = json.dumps({"method": method, **resolved}, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


def lr_allowed(select_lr: dict, protocol: str, lr: float) -> bool:
    allowed = select_lr.get(protocol)
    if not allowed:
        return True
    return any(abs(lr - a) <= 1e-12 + 1e-9 * abs(a) for a in allowed)
