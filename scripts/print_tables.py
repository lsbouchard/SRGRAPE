from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scaling", type=Path, default=Path("reproduction/data/scaling_summary_n2_n7.csv"))
    ap.add_argument("--degree-summary", type=Path, default=Path("reproduction/data/n5_cumulative_degree_seed25_summary.json"))
    args = ap.parse_args()
    scaling_path = args.scaling
    degree_path = args.degree_summary

    print("Scaling summary")
    print("N,trials,threshold,SR_mean,SR_std,SR_success,GRAPE_mean,GRAPE_std,GRAPE_success")
    with scaling_path.open(newline="") as f:
        for row in csv.DictReader(f):
            print(
                "{N},{trials},{success_threshold},{sr_mean:.12f},{sr_std:.12g},{sr_success}/{trials},"
                "{grape_mean:.12f},{grape_std:.12g},{grape_success}/{trials}".format(
                    N=row["N"],
                    trials=row["trials"],
                    success_threshold=float(row["success_threshold"]),
                    sr_mean=float(row["sr_mean"]),
                    sr_std=float(row["sr_std"]),
                    sr_success=row["sr_success"],
                    grape_mean=float(row["grape_mean"]),
                    grape_std=float(row["grape_std"]),
                    grape_success=row["grape_success"],
                )
            )

    print()
    print("N=5 cumulative-degree summary")
    print("P,seeds,mean_final,sample_std,successes")
    data = json.loads(degree_path.read_text())
    for key in sorted(data["by_degree_cap"], key=lambda x: int(x)):
        row = data["by_degree_cap"][key]
        print(
            "{P},{n},{mean:.12f},{std:.12g},{succ}/{n}".format(
                P=key,
                n=int(row["seed_count"]),
                mean=float(row["mean_final"]),
                std=float(row["sample_std_final"]),
                succ=int(row["success_count"]),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
