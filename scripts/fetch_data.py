from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dataset.records import LENGTHS_10HZ, SEQS, npz_key  # noqa: E402
from paths import DATA_DIR, EXTERNAL_DIR  # noqa: E402

REPO = "https://github.com/pohl-michel/2D-MR-image-prediction"
SUBDIR = "Time_series_forecasting"
INPUT_DIR = "a. Input time series sequences"
# folder name patterns (note the two spaces before 3.33 Hz)
FOLDERS = {
    "10": "Ext markers seq {k}",
    "3.33": "Ext markers seq {k}  3.33 Hz",
    "30": "Ext markers seq {k} 30 Hz",
}


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def checkout(dest: Path) -> None:
    if (dest / ".git").exists():
        print(f"{dest} exists; refreshing sparse checkout")
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", REPO, str(dest)])
    run(["git", "sparse-checkout", "set", SUBDIR], cwd=dest)


def load_mat(path: Path) -> np.ndarray:
    from scipy.io import loadmat

    try:
        d = loadmat(path)
    except NotImplementedError:  # MATLAB v7.3 (HDF5)
        try:
            import h5py
        except ImportError:
            sys.exit(f"{path} is a MATLAB v7.3 file; run `uv add h5py` and retry.")
        with h5py.File(path, "r") as f:
            return np.array(f["org_data"]).T
    if "org_data" not in d:
        sys.exit(f"{path}: variable 'org_data' not found (keys: {[k for k in d if not k.startswith('__')]})")
    return np.asarray(d["org_data"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-git", action="store_true", help="do not clone / update; convert only")
    ap.add_argument("--dest", default=str(EXTERNAL_DIR / "pohl_new"))
    ap.add_argument("--out", default=str(DATA_DIR / "extmarker.npz"))
    args = ap.parse_args()
    dest = Path(args.dest)
    if not args.skip_git:
        checkout(dest)
    base = dest / SUBDIR / INPUT_DIR
    if not base.exists():
        sys.exit(f"input folder not found: {base}")

    arrays: dict[str, np.ndarray] = {}
    problems: list[str] = []
    for rate, pattern in FOLDERS.items():
        for k in SEQS:
            p = base / pattern.format(k=k) / "data.mat"
            if not p.exists():
                problems.append(f"missing {p}")
                continue
            org = load_mat(p).astype(np.float64)
            if org.ndim != 2 or org.shape[0] != 9:
                problems.append(f"{p}: org_data shape {org.shape}, expected (9, T)")
                continue
            z = org.T.astype(np.float32)  # (T, 9): x1 x2 x3 y1 y2 y3 z1 z2 z3
            if not np.isfinite(z).all():
                problems.append(f"{p}: non-finite values")
            if np.abs(z[0]).max() > 1e-3:
                print(f"  note: {npz_key(rate, k)} first frame is not ~0 (max |z(0)| = {np.abs(z[0]).max():.3g} mm)")
            arrays[npz_key(rate, k)] = z

    for k, T in LENGTHS_10HZ.items():
        key = npz_key("10", k)
        if key in arrays and arrays[key].shape[0] != T:
            problems.append(f"{key}: length {arrays[key].shape[0]}, expected {T}")
    for k, T in LENGTHS_10HZ.items():
        for rate, factor in (("3.33", 1 / 3), ("30", 3.0)):
            key = npz_key(rate, k)
            if key in arrays and abs(arrays[key].shape[0] - T * factor) > 3:
                print(f"  warning: {key} length {arrays[key].shape[0]} vs ~{T * factor:.0f} expected from 10 Hz")
    if problems:
        print("\nFAILED:", *problems, sep="\n  ")
        return 1

    try:
        commit = subprocess.run(
            ["git", "-C", str(dest), "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
    except Exception:
        commit = "unknown"
    arrays["_source_commit"] = np.array(commit)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)
    print(f"\nwrote {out} ({len(arrays) - 1} arrays, source commit {commit[:10]})")
    for rate in FOLDERS:
        lens = [arrays[npz_key(rate, k)].shape[0] for k in SEQS]
        print(f"  {rate:>4} Hz lengths: {lens}")
    print("10 Hz lengths match the expected values. OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
