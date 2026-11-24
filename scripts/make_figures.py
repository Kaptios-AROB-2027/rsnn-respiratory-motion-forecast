from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import make_tables as mt  # noqa: E402

BLUE, ORANGE, TEAL = "#2a78d6", "#eb6834", "#1a9e8f"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
STYLE = [  # (model, legend label, colour, line style, width)
    ("persistence", "Persistence", MUTED, (0, (1, 1.5)), 1.2),
    ("lms", "LMS filter", INK2, (0, (5, 1.5, 1, 1.5)), 1.2),
    ("snap1", "RNN (SnAp-1)", ORANGE, "-", 1.5),
    ("lsnn_analog", "LSNN, analog", TEAL, (0, (3, 1.5)), 1.5),
    ("lsnn_sod", "LSNN, SOD", TEAL, "-", 1.5),
    ("hcrsnn_analog", "HC-RSNN, analog", BLUE, (0, (3, 1.5)), 1.5),
    ("hcrsnn_sod", "HC-RSNN, SOD", BLUE, "-", 2.0),
]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results", default=str(mt.RES), help="directory with the sweep CSV files")
    args = p.parse_args()
    mt.RES = Path(args.results)
    out = mt.RES / "figures"

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
        "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6, "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.85), sharey=True, constrained_layout=True)
    panels = ((axes[0], "(a) immediate feedback (IFeed)", lambda m: mt.immediate(m)),
              (axes[1], "(b) delayed feedback (DFeed)", lambda m: mt.delayed(m)[1]))
    for ax, title, get in panels:
        for model, label, color, ls, lw in STYLE:
            curve = get(model).groupby("h_s").test_rmse.mean()
            ax.plot(curve.index, curve.values, color=color, ls=ls, lw=lw, label=label, solid_capstyle="round")
        ax.set_xlim(0.05, 2.15)
        ax.set_ylim(0, 7.4)
        ax.set_xlabel("forecast horizon h (s)")
        ax.set_title(title, fontsize=8, color=INK, loc="left")
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("test RMSE (mm)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False, fontsize=7,
               handlelength=2.4, columnspacing=1.1)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "rmse.pdf"
    fig.savefig(path, dpi=600)
    fig.savefig(path.with_suffix(".png"), dpi=220)
    print(f"wrote {path} (+ .png)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
