from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    env = os.environ.get("RSNN_REPO_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists() and (parent / "src" / "training").exists():
            return parent
    return Path.cwd().resolve()


ROOT = repo_root()
EXTERNAL_DIR = ROOT / "external"
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
CONFIG_DIR = ROOT / "configs"


def data_file() -> Path:
    env = os.environ.get("RSNN_DATA")
    return Path(env).resolve() if env else DATA_DIR / "extmarker.npz"


def resolve_config(path: str | os.PathLike) -> Path:
    """Accept 'configs/x.json', 'x.json' or an absolute path."""
    p = Path(path)
    if p.is_absolute() and p.exists():
        return p
    for cand in (Path.cwd() / p, ROOT / p, CONFIG_DIR / p):
        if cand.exists():
            return cand.resolve()
    raise FileNotFoundError(f"config not found: {path} (looked in CWD, {ROOT}, {CONFIG_DIR})")
