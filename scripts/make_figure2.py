                      
"""Generate the cumulative-degree convergence figure from validated artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "reproduction" / "data"


def relpath(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def as_float(value: str | None) -> float:
    if value is None or value == "":
        return float("nan")
    return float(value)


def normal_half_width(std: float, n: int, z: float = 1.96) -> float:
    if n <= 0 or not math.isfinite(std):
        return float("nan")
    return z * std / math.sqrt(n)


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    phat = successes / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2.0 * n)) / denom
    half = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * n)) / n) / denom
    return center - half, center + half


def load_cumulative(summary_path: Path) -> list[dict[str, float]]:
    summary = json.loads(summary_path.read_text())
    rows = []
    for key in sorted(summary["by_degree_cap"], key=lambda x: int(x)):
        item = summary["by_degree_cap"][key]
        n = int(item["seed_count"])
        succ = int(item["success_count"])
        lo, hi = wilson_interval(succ, n)
        rows.append(
            {
                "degree_cap": int(key),
                "seed_count": n,
                "success_count": succ,
                "success_rate": float(item["success_rate"]),
                "success_lo": lo,
                "success_hi": hi,
                "mean_final": float(item["mean_final"]),
                "sample_std_final": float(item["sample_std_final"]),
                "sem_final": float(item["sem_final"]),
                "ci95_half_width": normal_half_width(float(item["sample_std_final"]), n),
            }
        )
    return rows


def load_increment_rows(cumulative_rows_path: Path) -> tuple[list[int], np.ndarray, np.ndarray]:
    rows = read_csv(cumulative_rows_path)
    by_seed: dict[int, dict[int, float]] = {}
    for row in rows:
        seed = int(row["seed"])
        cap = int(row["degree_cap"])
        by_seed.setdefault(seed, {})[cap] = as_float(row["best_final_physical_fidelity"])
    seeds = sorted(by_seed)
    d32 = []
    d43 = []
    for seed in seeds:
        vals = by_seed[seed]
        d32.append(vals.get(3, vals.get(2, np.nan)) - vals.get(2, np.nan))
        d43.append(vals.get(4, vals.get(3, np.nan)) - vals.get(3, np.nan))
    return seeds, np.asarray(d32, dtype=float), np.asarray(d43, dtype=float)


def load_trace(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = read_csv(path)
    iters = []
    vals = []
    for row in rows:
        iters.append(as_float(row.get("iter")))
        vals.append(as_float(row.get("best_fidelity") or row.get("fidelity")))
    return np.asarray(iters, dtype=float), np.asarray(vals, dtype=float)


def log_infidelity(fid: np.ndarray) -> np.ndarray:
    return np.log10(np.maximum(1.0 - fid, 1e-12))


def make_fig3(
    cumulative_summary: Path,
    cumulative_rows: Path,
    trace_p2: Path,
    trace_p3: Path,
    out_pdf: Path,
    out_png: Path,
    manifest_path: Path,
) -> None:
    cum = load_cumulative(cumulative_summary)
    seeds, d32, d43 = load_increment_rows(cumulative_rows)
    p2_x, p2_f = load_trace(trace_p2)
    p3_x, p3_f = load_trace(trace_p3)

    plt.rcParams.update(
        {
            "font.size": 9.8,
            "axes.labelsize": 10.1,
            "axes.titlesize": 10.4,
            "legend.fontsize": 9.0,
            "xtick.labelsize": 9.4,
            "ytick.labelsize": 9.4,
            "axes.linewidth": 0.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.05), constrained_layout=True)
    colors = {"blue": "#2563eb", "orange": "#d97706", "green": "#15803d", "gray": "#374151"}

    ax = axes[0]
    x = np.array([row["degree_cap"] for row in cum], dtype=float)
    y = np.array([row["mean_final"] for row in cum], dtype=float)
    yerr = np.array([row["ci95_half_width"] for row in cum], dtype=float)
    ax.errorbar(
        x,
        y,
        yerr=yerr,
        color=colors["blue"],
        marker="o",
        linewidth=2.0,
        markersize=5.0,
        capsize=4,
        elinewidth=1.4,
        capthick=1.4,
    )
    ax.axhline(0.95, color=colors["gray"], linewidth=1.0, linestyle=":", label="success threshold")
    ax.set_xticks([2, 3, 4])
    ax.set_ylim(0.90, 1.005)
    ax.set_xlabel("Degree Cap $P$")
    ax.set_ylabel(r"Mean $\mathcal{F}_{\leq P}$")
    ax.set_title("(a) Physical Fidelity")
    ax.grid(True, alpha=0.23)
    ax.legend(loc="lower right", frameon=False)

    ax = axes[1]
    rate = np.array([row["success_rate"] for row in cum], dtype=float)
    err_low = rate - np.array([row["success_lo"] for row in cum], dtype=float)
    err_high = np.array([row["success_hi"] for row in cum], dtype=float) - rate
    ax.errorbar(
        x,
        rate,
        yerr=np.vstack([err_low, err_high]),
        color=colors["green"],
        marker="s",
        linewidth=2.0,
        markersize=4.8,
        capsize=4,
        elinewidth=1.4,
        capthick=1.4,
    )
    ax.set_xticks([2, 3, 4])
    ax.set_ylim(0.80, 1.055)
    ax.set_xlabel("Degree Cap $P$")
    ax.set_ylabel("Success Fraction")
    ax.set_title("(b) Success at $\\mathcal{F}\\geq0.95$")
    ax.grid(True, alpha=0.23)
    for row in cum:
        ax.text(
            row["degree_cap"],
            1.018,
            f"{int(row['success_count'])}/{int(row['seed_count'])}",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

    ax = axes[2]
    offset = p2_x.max() + 20.0
    ax.plot(p2_x, log_infidelity(p2_f), color=colors["orange"], linewidth=2.0, label="$p=2$ stage")
    ax.plot(p3_x + offset, log_infidelity(p3_f), color=colors["blue"], linewidth=2.0, linestyle="--", label="$p=3$ continuation")
    ax.axvline(offset, color=colors["gray"], linewidth=0.9, linestyle=":")
    ax.text(offset, -0.35, "restart", rotation=90, va="top", ha="right", fontsize=7.8, color=colors["gray"])
    ax.set_xlabel("EA Iteration")
    ax.set_ylabel(r"$\log_{10}(1-\mathcal{F}_{\mathrm{EA,best}})$")
    ax.set_title("(c) Nested Continuation")
    ax.set_ylim(-3.45, -0.15)
    ax.grid(True, alpha=0.23)
    ax.legend(frameon=False, loc="lower left")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=500, bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "figure": "figure2_cumulative_degree",
        "outputs": {"pdf": relpath(out_pdf), "png": relpath(out_png)},
        "sources": {
            "cumulative_summary": relpath(cumulative_summary),
            "cumulative_rows": relpath(cumulative_rows),
            "trace_p2": relpath(trace_p2),
            "trace_p3": relpath(trace_p3),
        },
        "panel_a": {
            "quantity": "sample mean of F_{<=P}(omega)=max_{2<=p<=P} F_p(omega)",
            "ci": "normal 95% half-width, 1.96*s/sqrt(n)",
        },
        "panel_b": {
            "quantity": "sample success fraction at F >= 0.95",
            "ci": "Wilson 95% interval",
        },
        "panel_c": {
            "quantity": "representative nested EA trace, log10 best infidelity",
            "note": "Trace is illustrative; panels A-B are the statistical 25-seed result.",
        },
        "increments": {
            "seeds": seeds,
            "F_le_3_minus_F_le_2": d32.tolist(),
            "F_le_4_minus_F_le_3": d43.tolist(),
            "mean_F_le_3_minus_F_le_2": float(np.nanmean(d32)),
            "mean_F_le_4_minus_F_le_3": float(np.nanmean(d43)),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[FIG] wrote {out_pdf}")
    print(f"[FIG] wrote {out_png}")
    print(f"[FIG] wrote {manifest_path}")


def sample_std(values: np.ndarray) -> float:
    clean = values[np.isfinite(values)]
    if clean.size <= 1:
        return 0.0
    return float(np.std(clean, ddof=1))


def summarize_compare_csv(N: int, path: Path, threshold: float = 0.95) -> dict[str, float | int | str]:
    rows = read_csv(path)
    method = np.asarray([as_float(row["method_finalF_phys"]) for row in rows], dtype=float)
    baseline = np.asarray([as_float(row["baseline_finalF_phys"]) for row in rows], dtype=float)
    out: dict[str, float | int | str] = {
        "N": N,
        "trials": len(rows),
        "source_csv": str(path),
        "success_threshold": threshold,
        "method_mean": float(np.nanmean(method)),
        "method_std": sample_std(method),
        "method_sem": sample_std(method) / math.sqrt(len(rows)) if rows else float("nan"),
        "method_success": int(np.sum(method >= threshold)),
        "method_success_rate": float(np.mean(method >= threshold)),
        "method_success_0999": int(np.sum(method >= 0.999)),
        "method_success_rate_0999": float(np.mean(method >= 0.999)),
        "baseline_mean": float(np.nanmean(baseline)),
        "baseline_std": sample_std(baseline),
        "baseline_sem": sample_std(baseline) / math.sqrt(len(rows)) if rows else float("nan"),
        "baseline_success": int(np.sum(baseline >= threshold)),
        "baseline_success_rate": float(np.mean(baseline >= threshold)),
        "baseline_success_0999": int(np.sum(baseline >= 0.999)),
        "baseline_success_rate_0999": float(np.mean(baseline >= 0.999)),
    }
    return out


def write_summary_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_fig2(
    n2_csv: Path,
    n3_csv: Path,
    n4_csv: Path,
    n5_csv: Path,
    out_pdf: Path,
    out_png: Path,
    summary_csv: Path,
    manifest_path: Path,
) -> None:
    rows = [
        summarize_compare_csv(2, n2_csv),
        summarize_compare_csv(3, n3_csv),
        summarize_compare_csv(4, n4_csv),
        summarize_compare_csv(5, n5_csv),
    ]
    write_summary_csv(summary_csv, rows)

    plt.rcParams.update(
        {
            "font.size": 9.4,
            "axes.labelsize": 9.8,
            "axes.titlesize": 10.2,
            "legend.fontsize": 8.8,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "axes.linewidth": 0.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75), constrained_layout=True)
    colors = {"sr": "#2563eb", "grape": "#d97706", "gray": "#374151"}
    x = np.asarray([float(row["N"]) for row in rows], dtype=float)
    offset = 0.075

    ax = axes[0]
    method_mean = np.asarray([float(row["method_mean"]) for row in rows])
    method_std = np.asarray([float(row["method_std"]) for row in rows])
    baseline_mean = np.asarray([float(row["baseline_mean"]) for row in rows])
    baseline_std = np.asarray([float(row["baseline_std"]) for row in rows])
    ax.errorbar(
        x - offset,
        method_mean,
        yerr=method_std,
        color=colors["sr"],
        marker="o",
        linewidth=2.0,
        markersize=4.8,
        capsize=4,
        elinewidth=1.35,
        capthick=1.35,
        label="SR-GRAPE",
    )
    ax.errorbar(
        x + offset,
        baseline_mean,
        yerr=baseline_std,
        color=colors["grape"],
        marker="s",
        linewidth=2.0,
        markersize=4.8,
        capsize=4,
        elinewidth=1.35,
        capthick=1.35,
        linestyle="--",
        label="direct GRAPE",
    )
    ax.axhline(0.95, color=colors["gray"], linewidth=1.0, linestyle=":", label="success threshold")
    ax.set_xticks([2, 3, 4, 5])
    ax.set_ylim(0.45, 1.025)
    ax.set_xlabel("number of qubits $N$")
    ax.set_ylabel(r"final physical fidelity $\mathcal{F}$")
    ax.set_title("(a) Mean final fidelity")
    ax.grid(True, alpha=0.23)
    ax.legend(frameon=False, loc="lower left")

    ax = axes[1]
    method_rate = np.asarray([float(row["method_success_rate"]) for row in rows])
    baseline_rate = np.asarray([float(row["baseline_success_rate"]) for row in rows])
    method_err_low = []
    method_err_high = []
    base_err_low = []
    base_err_high = []
    for row in rows:
        n = int(row["trials"])
        m_lo, m_hi = wilson_interval(int(row["method_success"]), n)
        b_lo, b_hi = wilson_interval(int(row["baseline_success"]), n)
        method_err_low.append(float(row["method_success_rate"]) - m_lo)
        method_err_high.append(m_hi - float(row["method_success_rate"]))
        base_err_low.append(float(row["baseline_success_rate"]) - b_lo)
        base_err_high.append(b_hi - float(row["baseline_success_rate"]))
    ax.errorbar(
        x - offset,
        method_rate,
        yerr=np.vstack([method_err_low, method_err_high]),
        color=colors["sr"],
        marker="o",
        linewidth=2.0,
        markersize=4.8,
        capsize=4,
        elinewidth=1.35,
        capthick=1.35,
        label="SR-GRAPE",
    )
    ax.errorbar(
        x + offset,
        baseline_rate,
        yerr=np.vstack([base_err_low, base_err_high]),
        color=colors["grape"],
        marker="s",
        linewidth=2.0,
        markersize=4.8,
        capsize=4,
        elinewidth=1.35,
        capthick=1.35,
        linestyle="--",
        label="direct GRAPE",
    )
    for row in rows:
        n = int(row["trials"])
        N = float(row["N"])
        ax.text(N - offset, min(float(row["method_success_rate"]) + 0.035, 1.02), f"{int(row['method_success'])}/{n}", ha="center", fontsize=7.4)
        ax.text(N + offset, max(float(row["baseline_success_rate"]) - 0.055, 0.02), f"{int(row['baseline_success'])}/{n}", ha="center", fontsize=7.4)
    ax.set_xticks([2, 3, 4, 5])
    ax.set_ylim(-0.04, 1.08)
    ax.set_xlabel("number of qubits $N$")
    ax.set_ylabel(r"success fraction at $\mathcal{F}\geq0.95$")
    ax.set_title("(b) Empirical success probability")
    ax.grid(True, alpha=0.23)
    ax.legend(frameon=False, loc="lower left")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=500, bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "figure": "figure_2",
        "outputs": {"pdf": relpath(out_pdf), "png": relpath(out_png), "summary_csv": relpath(summary_csv)},
        "sources": {
            "N2": relpath(n2_csv),
            "N3": relpath(n3_csv),
            "N4": relpath(n4_csv),
            "N5": relpath(n5_csv),
        },
        "summary_rows": rows,
        "success_threshold": 0.95,
        "fidelity_error_bars": "sample standard deviation over 25 trials/seeds",
        "success_error_bars": "Wilson 95% intervals",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[FIG] wrote {out_pdf}")
    print(f"[FIG] wrote {out_png}")
    print(f"[FIG] wrote {summary_csv}")
    print(f"[FIG] wrote {manifest_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", type=Path, default=ROOT / "reproduction" / "figures")
    ap.add_argument("--figure", choices=["fig2", "fig3", "all"], default="fig3")
    ap.add_argument(
        "--fig2-n2-csv",
        type=Path,
        default=REPRO / "n2_budget_compare25.csv",
    )
    ap.add_argument(
        "--fig2-n3-csv",
        type=Path,
        default=REPRO / "n3_budget_compare25.csv",
    )
    ap.add_argument(
        "--fig2-n4-csv",
        type=Path,
        default=REPRO / "n4_budget_compare25.csv",
    )
    ap.add_argument(
        "--fig2-n5-csv",
        type=Path,
        default=REPRO / "n5_budget_compare_seed25_thr99_rows.csv",
    )
    ap.add_argument(
        "--cumulative-summary",
        type=Path,
        default=REPRO / "n5_cumulative_degree_seed25_summary.json",
    )
    ap.add_argument(
        "--cumulative-rows",
        type=Path,
        default=REPRO / "n5_cumulative_degree_seed25_rows.csv",
    )
    ap.add_argument(
        "--trace-p2",
        type=Path,
        default=ROOT / "reproduction" / "degree_cap" / "traces" / "ea_p2_nested_trace.csv",
    )
    ap.add_argument(
        "--trace-p3",
        type=Path,
        default=ROOT / "reproduction" / "degree_cap" / "traces" / "ea_p3_continuation_trace.csv",
    )
    ns = ap.parse_args()
    if ns.figure in ("fig2", "all"):
        make_fig2(
            ns.fig2_n2_csv,
            ns.fig2_n3_csv,
            ns.fig2_n4_csv,
            ns.fig2_n5_csv,
            ns.outdir / "figure1_legacy_scaling.pdf",
            ns.outdir / "figure1_legacy_scaling.png",
            ns.outdir / "figure1_legacy_scaling_summary.csv",
            ns.outdir / "figure1_legacy_scaling_manifest.json",
        )
    if ns.figure in ("fig3", "all"):
        make_fig3(
            ns.cumulative_summary,
            ns.cumulative_rows,
            ns.trace_p2,
            ns.trace_p3,
            ns.outdir / "figure2_cumulative_degree.pdf",
            ns.outdir / "figure2_cumulative_degree.png",
            ns.outdir / "figure2_cumulative_degree_manifest.json",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
