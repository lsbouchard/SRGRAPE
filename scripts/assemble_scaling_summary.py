                      
"""Assemble the Figure 1/Table I summary CSV from raw result artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def sample_std(values: list[float]) -> float:
    return float(np.std(np.asarray(values, dtype=float), ddof=1)) if len(values) > 1 else 0.0


def avg(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=float)))


def compare_row(path: Path, N: int, threshold: float, note: str) -> dict:
    rows = read_csv(path)
    method = [float(row["method_finalF_phys"]) for row in rows]
    baseline = [float(row["baseline_finalF_phys"]) for row in rows]
    return {
        "N": N,
        "trials": len(rows),
        "kind": "matched",
        "success_threshold": threshold,
        "sr_mean": avg(method),
        "sr_std": sample_std(method),
        "sr_success": sum(x >= threshold for x in method),
        "grape_mean": avg(baseline),
        "grape_std": sample_std(baseline),
        "grape_success": sum(x >= threshold for x in baseline),
        "note": note,
    }


def state_grow_values(path: Path) -> list[float]:
    rows = read_csv(path)
    return [float(row["final_child_fidelity"]) for row in rows]


def baseline_summary(path: Path) -> tuple[int, float, float, int]:
    data = json.loads(path.read_text())
    summary = data["summary"]
    return int(summary["n"]), float(summary["mean"]), float(summary["std_sample"]), int(summary["successes"])


def n7_sr_summary(path: Path) -> tuple[int, float, float, int]:
    data = json.loads(path.read_text())
    if "reported_mean" not in data:
        return (
            int(data["n_completed_with_fidelity"]),
            float(data["final_mean"]),
            float(data["final_std_sample"]),
            int(data["success_count"]),
        )
    return (
        int(data["n_seeds"]),
        float(data["reported_mean"]),
        float(data["reported_std_sample"]),
        int(data["reported_success_count"]),
    )


def n7_grape_summary(path: Path) -> tuple[int, float, float, int]:
    data = json.loads(path.read_text())
    return (
        int(data["n_with_fidelity"]),
        float(data["mean_finalF"]),
        float(data["std_finalF_sample"]),
        int(data["success_count"]),
    )


def state_grow_row(
    N: int,
    sr_rows_path: Path,
    grape_summary_path: Path,
    threshold: float,
    note: str,
) -> dict:
    sr_vals = state_grow_values(sr_rows_path)
    n_g, grape_mean, grape_std, grape_success = baseline_summary(grape_summary_path)
    return {
        "N": N,
        "trials": len(sr_vals),
        "kind": "state-growing-extension",
        "success_threshold": threshold,
        "sr_mean": avg(sr_vals),
        "sr_std": sample_std(sr_vals),
        "sr_success": sum(x >= threshold for x in sr_vals),
        "grape_mean": grape_mean,
        "grape_std": grape_std,
        "grape_success": grape_success,
        "note": note,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("data/scaling_summary_n2_n7.csv"))
    args = ap.parse_args()
    d = args.data_dir

    rows = [
        compare_row(d / "n2_budget_compare25.csv", 2, 0.95, "budget-matched SR-GRAPE/direct-GRAPE comparison"),
        compare_row(d / "n3_budget_compare25.csv", 3, 0.95, "budget-matched SR-GRAPE/direct-GRAPE comparison"),
        compare_row(d / "n4_budget_compare25.csv", 4, 0.95, "budget-matched SR-GRAPE/direct-GRAPE comparison"),
        state_grow_row(
            5,
            d / "n5_t8_state_grow_seed25_rows.csv",
            d / "n5_t8_direct_grape_baseline_summary.json",
            0.99,
            "SR-GRAPE state-growing extension and full-budget direct-GRAPE random-start baseline",
        ),
        state_grow_row(
            6,
            d / "n6_t8_state_grow_seed1_25_rows.csv",
            d / "n6_direct_grape_baseline_summary.json",
            0.99,
            "SR-GRAPE state-growing extension and direct-GRAPE random-start baseline",
        ),
    ]

    n_sr, sr_mean, sr_std, sr_success = n7_sr_summary(d / "n7_amp3_state_growing_summary.json")
    n_g, grape_mean, grape_std, grape_success = n7_grape_summary(d / "n7_amp3_direct_grape_summary.json")
    if n_sr != n_g:
        raise ValueError(f"N=7 trial mismatch: SR={n_sr}, GRAPE={n_g}")
    rows.append(
        {
            "N": 7,
            "trials": n_sr,
            "kind": "state-growing-extension-u_max_3",
            "success_threshold": 0.99,
            "sr_mean": sr_mean,
            "sr_std": sr_std,
            "sr_success": sr_success,
            "grape_mean": grape_mean,
            "grape_std": grape_std,
            "grape_success": grape_success,
            "note": "SR-GRAPE state-growing extension and direct-GRAPE random-start baseline at u_max=3",
        }
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
