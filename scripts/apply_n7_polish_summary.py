                      
"""Merge deterministic N=7 polishing runs into the reported summary."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean, stdev


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def load_polish(path: Path) -> dict:
    data = json.loads(path.read_text())
    match = re.search(r"seed(\d+)", path.stem)
    if not match:
        raise ValueError(f"could not infer seed from polish summary path: {path}")
    return {
        "seed": int(match.group(1)),
        "summary_json": str(path),
        "controls_path": data["output_controls"],
        "final_fidelity": float(data["result"]["finalF_phys"]),
        "total_stage_iters": int(data["result"]["total_stage_iters"]),
        "wall_s": float(data["result"]["wall_s"]),
    }


def summarize(rows: list[dict], threshold: float) -> dict:
    first = [float(row["final_child_fidelity"]) for row in rows]
    reported = [float(row["reported_final_child_fidelity"]) for row in rows]
    polish_seeds = [int(row["seed"]) for row in rows if row["polish_applied"]]
    return {
        "threshold": threshold,
        "n_seeds": len(rows),
        "first_pass_success_count": sum(x >= threshold for x in first),
        "first_pass_success_rate": sum(x >= threshold for x in first) / len(first),
        "first_pass_mean": mean(first),
        "first_pass_std_sample": stdev(first) if len(first) > 1 else 0.0,
        "first_pass_min": min(first),
        "first_pass_max": max(first),
        "polish_seeds": polish_seeds,
        "reported_success_count": sum(x >= threshold for x in reported),
        "reported_success_rate": sum(x >= threshold for x in reported) / len(reported),
        "reported_mean": mean(reported),
        "reported_std_sample": stdev(reported) if len(reported) > 1 else 0.0,
        "reported_min": min(reported),
        "reported_max": max(reported),
        "rows": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--first-pass-rows", type=Path, required=True)
    ap.add_argument("--polish-summary", type=Path, action="append", default=[])
    ap.add_argument("--rows-out", type=Path, required=True)
    ap.add_argument("--summary-out", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=0.99)
    args = ap.parse_args()

    polished = {item["seed"]: item for item in (load_polish(path) for path in args.polish_summary)}
    rows = []
    for row in read_csv(args.first_pass_rows):
        seed = int(row["seed"])
        out = dict(row)
        out["first_pass_final_child_fidelity"] = float(row["final_child_fidelity"])
        polished_row = polished.get(seed)
        if polished_row:
            out["polish_applied"] = True
            out["polish_summary_json"] = polished_row["summary_json"]
            out["polish_controls_path"] = polished_row["controls_path"]
            out["polish_total_stage_iters"] = polished_row["total_stage_iters"]
            out["polish_wall_s"] = polished_row["wall_s"]
            out["reported_final_child_fidelity"] = polished_row["final_fidelity"]
        else:
            out["polish_applied"] = False
            out["polish_summary_json"] = ""
            out["polish_controls_path"] = ""
            out["polish_total_stage_iters"] = ""
            out["polish_wall_s"] = ""
            out["reported_final_child_fidelity"] = float(row["final_child_fidelity"])
        rows.append(out)
    rows = sorted(rows, key=lambda item: int(item["seed"]))
    write_csv(args.rows_out, rows)
    payload = summarize(rows, args.threshold)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"wrote {args.rows_out}")
    print(f"wrote {args.summary_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
