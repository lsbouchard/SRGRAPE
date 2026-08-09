from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def wilson_interval(success: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials <= 0:
        return (0.0, 0.0)
    phat = success / trials
    denom = 1.0 + z * z / trials
    center = (phat + z * z / (2.0 * trials)) / denom
    half = z * np.sqrt(phat * (1.0 - phat) / trials + z * z / (4.0 * trials * trials)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def read_rows(summary: Path) -> list[dict[str, str]]:
    with summary.open(newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--summary",
        type=Path,
        default=Path("reproduction/data/scaling_summary_n2_n7.csv"),
        help="Scaling-summary CSV produced by assemble_scaling_summary.py",
    )
    ap.add_argument(
        "--outdir",
        type=Path,
        default=Path("reproduction/figures"),
        help="Directory for figure1_scaling.pdf and .png",
    )
    args = ap.parse_args()
    rows = sorted(read_rows(args.summary), key=lambda r: int(r["N"]))

    x = np.array([int(r["N"]) for r in rows], dtype=float)
    sr_mean = np.array([float(r["sr_mean"]) for r in rows])
    sr_std = np.array([float(r["sr_std"]) for r in rows])
    grape_mean = np.array([float(r["grape_mean"]) for r in rows])
    grape_std = np.array([float(r["grape_std"]) for r in rows])
    trials = np.array([int(r["trials"]) for r in rows])
    sr_success = np.array([int(r["sr_success"]) for r in rows])
    grape_success = np.array([int(r["grape_success"]) for r in rows])

    sr_p = sr_success / trials
    grape_p = grape_success / trials

    sr_ci = np.array([wilson_interval(int(s), int(t)) for s, t in zip(sr_success, trials)])
    grape_ci = np.array([wilson_interval(int(s), int(t)) for s, t in zip(grape_success, trials)])

    blue = "#2563eb"
    orange = "#d97706"
    gray = "#374151"

    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 20,
            "axes.titlesize": 21,
            "legend.fontsize": 14,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "figure.dpi": 240,
            "savefig.dpi": 320,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), constrained_layout=True)
    ax = axes[0]
    ax.errorbar(
        x - 0.07,
        sr_mean,
        yerr=sr_std,
        fmt="o-",
        color=blue,
        ecolor=blue,
        elinewidth=2.4,
        capsize=5,
        markersize=8,
        lw=2.6,
        label="SR-GRAPE",
        zorder=4,
    )
    ax.errorbar(
        x + 0.07,
        grape_mean,
        yerr=grape_std,
        fmt="s--",
        color=orange,
        ecolor=orange,
        elinewidth=2.4,
        capsize=5,
        markersize=8,
        lw=2.6,
        label="direct GRAPE",
        zorder=3,
    )
    ax.axhline(0.95, color=gray, ls=":", lw=1.7, alpha=0.9, label="0.95")
    ax.axhline(0.99, color="#6b7280", ls=":", lw=1.5, alpha=0.75, label="0.99")
    ax.set_title("(a) Mean Final Fidelity")
    ax.set_xlabel(r"Number of Qubits $N$")
    ax.set_ylabel(r"Final Physical Fidelity $\mathcal{F}$")
    ax.set_xlim(1.75, 7.35)
    ax.set_ylim(0.45, 1.075)
    ax.set_xticks([2, 3, 4, 5, 6, 7])
    ax.grid(True, alpha=0.25, lw=1.0)
    ax.legend(loc="lower left", frameon=False, fontsize=13)

    ax = axes[1]
    ax.errorbar(
        x - 0.07,
        sr_p,
        yerr=[sr_p - sr_ci[:, 0], sr_ci[:, 1] - sr_p],
        fmt="o-",
        color=blue,
        ecolor=blue,
        elinewidth=2.4,
        capsize=5,
        markersize=8,
        lw=2.6,
        label="SR-GRAPE",
        zorder=4,
    )
    ax.errorbar(
        x + 0.07,
        grape_p,
        yerr=[grape_p - grape_ci[:, 0], grape_ci[:, 1] - grape_p],
        fmt="s--",
        color=orange,
        ecolor=orange,
        elinewidth=2.4,
        capsize=5,
        markersize=8,
        lw=2.6,
        label="direct GRAPE",
        zorder=3,
    )
    for xi, yi, s, t in zip(x - 0.07, sr_p, sr_success, trials):
        if int(round(xi + 0.07)) >= 4:
            ax.text(xi, 1.035, f"{s}/{t}", color=blue, ha="center", va="bottom", fontsize=10.5)
    for xi, yi, s, t in zip(x + 0.07, grape_p, grape_success, trials):
        if int(round(xi - 0.07)) >= 4:
            if yi > 0.8:
                ax.text(xi + 0.22, 0.875, f"{s}/{t}", color=orange, ha="left", va="center", fontsize=10.5)
            else:
                ax.text(xi, 0.205, f"{s}/{t}", color=orange, ha="center", va="bottom", fontsize=10.5)
    ax.set_title("(b) Empirical Success Probability")
    ax.set_xlabel(r"Number of Qubits $N$")
    ax.set_ylabel(r"Success Fraction at $\mathcal{F}\geq 0.95$")
    ax.set_xlim(1.75, 7.35)
    ax.set_ylim(-0.04, 1.12)
    ax.set_xticks([2, 3, 4, 5, 6, 7])
    ax.grid(True, alpha=0.25, lw=1.0)
    ax.legend(loc="lower left", frameon=False, fontsize=13)

    out_png = args.outdir / "figure1_scaling.png"
    out_pdf = args.outdir / "figure1_scaling.pdf"
    args.outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"wrote {out_png}")
    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    main()
