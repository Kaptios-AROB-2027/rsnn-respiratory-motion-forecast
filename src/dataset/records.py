from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np

from paths import data_file

RATE_KEYS = ("3.33", "10", "30")
RATE_HZ = {"3.33": 10.0 / 3.0, "10": 10.0, "30": 30.0}
# (start of CV, start of test) in samples of target time = (30 s, 60 s)
SPLITS = {"3.33": (100, 200), "10": (300, 600), "30": (900, 1800)}
# horizons in samples, 0.1 (0.3) s .. 2.1 s
HORIZONS = {
    "3.33": tuple(range(1, 8)),
    "10": tuple(range(1, 22)),
    "30": tuple(range(3, 64, 3)),
}
SEQS = tuple(range(1, 10))
LENGTHS_10HZ = {1: 2220, 2: 1383, 3: 1297, 4: 1422, 5: 1307, 6: 1171, 7: 726, 8: 3199, 9: 3061}
# speaking / laughing records (original file codes T, NO, LT, O)
IRREGULAR = (1, 3, 4, 7)
REGULAR = (2, 5, 6, 8, 9)
PAD_MULTIPLE = 256


def rate_key(rate: str | float | int) -> str:
    """Normalise '10', 10, '10.0', '10hz', 3.33, '3.33' -> canonical key."""
    s = str(rate).lower().replace("hz", "").strip()
    try:
        v = float(s)
    except ValueError as exc:
        raise ValueError(f"unknown rate {rate!r}") from exc
    for key, hz in RATE_HZ.items():
        if abs(v - hz) < 0.01 or abs(v - float(key)) < 1e-9:
            return key
    raise ValueError(f"unknown rate {rate!r}; expected one of {RATE_KEYS}")


def npz_key(rate: str, seq: int) -> str:
    return f"f{rate_key(rate)}_seq{seq}"


def shl_samples(shl_s: float, rate: str) -> int:
    """Signal history length in samples; shl_s = 0 means a single sample."""
    return max(1, int(round(float(shl_s) * RATE_HZ[rate_key(rate)])))


def horizon_seconds(h: int, rate: str) -> float:
    return float(h) / RATE_HZ[rate_key(rate)]


def parse_horizons(spec: str | None, rate: str) -> tuple[int, ...]:
    """'all' / None -> all horizons; '1,5,10' -> samples; '0.3s,1.0s' -> seconds rounded to samples."""
    rate = rate_key(rate)
    if spec is None or str(spec).strip().lower() in ("", "all"):
        return HORIZONS[rate]
    out = []
    for tok in str(spec).split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok.endswith("s"):
            h = int(round(float(tok[:-1]) * RATE_HZ[rate]))
        else:
            h = int(tok)
        if h not in HORIZONS[rate]:
            raise ValueError(f"horizon {tok} -> {h} samples is not in the {rate} Hz horizon set {HORIZONS[rate]}")
        out.append(h)
    return tuple(dict.fromkeys(out))


def parse_seqs(spec: str | None) -> tuple[int, ...]:
    if spec is None or str(spec).strip().lower() in ("", "all"):
        return SEQS
    out = []
    for tok in str(spec).split(","):
        tok = tok.strip()
        if "-" in tok:
            a, b = tok.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif tok:
            out.append(int(tok))
    bad = [s for s in out if s not in SEQS]
    if bad:
        raise ValueError(f"unknown records {bad}")
    return tuple(dict.fromkeys(out))


def pad_length(T: int, multiple: int = PAD_MULTIPLE) -> int:
    return int(-(-T // multiple) * multiple)


@dataclass(frozen=True)
class Record:
    rate: str
    seq: int
    z_mm: np.ndarray = field(repr=False)  # (T, 9) float64, millimetres

    @property
    def T(self) -> int:
        return int(self.z_mm.shape[0])

    @property
    def regular(self) -> bool:
        return self.seq in REGULAR

    @cached_property
    def stats(self) -> tuple[np.ndarray, np.ndarray]:
        n30 = SPLITS[self.rate][0]
        head = self.z_mm[:n30]
        return head.mean(axis=0), head.std(axis=0)  # population std (ddof=0)

    @property
    def mean(self) -> np.ndarray:
        return self.stats[0]

    @property
    def std(self) -> np.ndarray:
        return self.stats[1]

    @cached_property
    def zn(self) -> np.ndarray:
        return (self.z_mm - self.mean) / self.std

    @property
    def T_pad(self) -> int:
        return pad_length(self.T)

    def zn_pad(self, dtype=np.float32) -> np.ndarray:
        out = np.zeros((self.T_pad, 9), dtype=dtype)
        out[: self.T] = self.zn
        return out

    def n_windows(self, k: int, h: int) -> int:
        return self.T - k - h + 1


def load_npz(path: Path | None = None) -> dict[str, np.ndarray]:
    path = Path(path) if path else data_file()
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `uv run python scripts/fetch_data.py` first (see README step 3)."
        )
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def load_records(rate: str, seqs=SEQS, path: Path | None = None) -> list[Record]:
    rate = rate_key(rate)
    arrs = load_npz(path)
    out = []
    for s in seqs:
        key = npz_key(rate, s)
        if key not in arrs:
            raise KeyError(f"{key} missing from data file; re-run scripts/fetch_data.py")
        z = np.asarray(arrs[key], dtype=np.float64)
        if z.ndim != 2 or z.shape[1] != 9:
            raise ValueError(f"{key}: expected (T, 9), got {z.shape}")
        out.append(Record(rate=rate, seq=int(s), z_mm=z))
    return out
