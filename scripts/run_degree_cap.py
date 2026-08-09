"""Rerun the five-qubit cumulative degree-cap diagnostic.

The script launches one method-only calculation for each requested seed and
degree cap, then writes the cumulative statistic
F_{<=P}(seed) = max_{2 <= p <= P} F_p(seed)
used by the paper's degree-cap figure. All generated files belong in the
caller-selected output directory; no result data are committed to the repo.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[1]


def parse_int_list(raw: str) -> list[int]:
    values: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, stop = (int(item) for item in part.split("-", 1))
            values.update(range(start, stop + 1))
        else:
            values.add(int(part))
    if not values:
        raise ValueError(f"empty integer list: {raw!r}")
    return sorted(values)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize(values: list[float], degree_cap: int, threshold: float) -> dict[str, object]:
    n = len(values)
    std = stdev(values) if n > 1 else 0.0
    return {
        "N": 5,
        "degree_cap": degree_cap,
        "seed_count": n,
        "mean_final": mean(values) if values else None,
        "sample_std_final": std,
        "sem_final": std / (n**0.5) if n else None,
        "min_final": min(values) if values else None,
        "max_final": max(values) if values else None,
        "success_count": sum(value >= threshold for value in values),
        "success_rate": (sum(value >= threshold for value in values) / n) if n else 0.0,
    }


def build_command(args: argparse.Namespace, degree: int, seed: int, metadata: Path, trace_dir: Path) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "srgrape",
        "--d",
        "2",
        "--N",
        "5",
        "--Nt",
        str(args.Nt),
        "--T",
        str(args.T),
        "--p",
        str(degree),
        "--target",
        "ghz",
        "--drives",
        "xy",
        "--drift-strength-ea",
        "1.0",
        "--drift-strength-hw",
        "1.0",
        "--homotopy",
        "1.0",
        "--homotopy-mode",
        "mult",
        "--ea-iters",
        str(args.ea_iters),
        "--S1",
        str(args.S1),
        "--S2",
        str(args.S2),
        "--iters1",
        str(args.iters1),
        "--iters2",
        str(args.iters2),
        "--lr1",
        str(args.lr1),
        "--lr2",
        str(args.lr2),
        "--amp",
        str(args.amp),
        "--hard-amp",
        str(args.amp),
        "--compiler-mode",
        "ea_target_continuation_project",
        "--compiler-eps",
        "0.02",
        "--compiler-budget-frac",
        "1.0",
        "--compiler-max-terms",
        "24",
        "--accept-mode",
        "soft",
        "--accept-drop",
        "0.002",
        "--backtracks",
        "5",
        "--clip",
        "5.0",
        "--stall-enable",
        "1",
        "--stall-gnorm",
        "1e-6",
        "--stall-max-kicks",
        "12",
        "--stall-kick-sigma",
        "0.05",
        "--baseline-mode",
        "budget",
        "--compare",
        "0",
        "--threshold",
        str(args.threshold),
        "--seed",
        str(seed),
        "--verbose",
        "0",
        "--progress",
        "off",
        "--save-traces",
        "1",
        "--save-compiler-diagnostics",
        "0",
        "--save-metadata",
        "1",
        "--trace-dir",
        str(trace_dir),
        "--metadata-path",
        str(metadata),
        "--outdir",
        str(metadata.parent),
        "--tag",
        f"p{degree}_seed{seed:02d}",
    ]
    if args.nested:
        command.extend(
            [
                "--ea-nested-p2",
                "1",
                "--ea-nested-p2-iters",
                str(args.ea_iters),
                "--ea-nested-p3-iters",
                str(args.ea_iters),
                "--ea-nested-guard",
                "1",
            ]
        )
    return command


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", type=Path, default=Path("reproduction/degree_cap"))
    ap.add_argument("--seeds", default="1-25")
    ap.add_argument("--degrees", default="2,3,4")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-completed", action="store_true")
    ap.add_argument("--Nt", type=int, default=20)
    ap.add_argument("--T", type=float, default=5.0)
    ap.add_argument("--ea-iters", type=int, default=600)
    ap.add_argument("--S1", type=int, default=28)
    ap.add_argument("--S2", type=int, default=48)
    ap.add_argument("--iters1", type=int, default=800)
    ap.add_argument("--iters2", type=int, default=1200)
    ap.add_argument("--lr1", type=float, default=0.08)
    ap.add_argument("--lr2", type=float, default=0.06)
    ap.add_argument("--amp", type=float, default=2.0)
    ap.add_argument("--threshold", type=float, default=0.95)
    ap.add_argument("--nested", type=int, default=1, help="Use guarded p=2 continuation for p>=3")
    args = ap.parse_args()

    seeds = parse_int_list(args.seeds)
    degrees = parse_int_list(args.degrees)
    if any(degree < 2 or degree > 4 for degree in degrees):
        raise SystemExit("--degrees must be selected from 2,3,4")

    args.outdir.mkdir(parents=True, exist_ok=True)
    raw_rows: list[dict[str, object]] = []
    environment = os.environ.copy()
    source_path = str(ROOT / "src")
    environment["PYTHONPATH"] = source_path + os.pathsep + environment.get("PYTHONPATH", "")

    for degree in degrees:
        for seed in seeds:
            run_dir = args.outdir / "runs" / f"p{degree}" / f"seed{seed:02d}"
            metadata = run_dir / "metadata.json"
            trace_dir = args.outdir / "traces" / f"p{degree}" / f"seed{seed:02d}"
            command = build_command(args, degree, seed, metadata, trace_dir)
            print("COMMAND:", " ".join(command), flush=True)
            if args.dry_run:
                continue
            if args.skip_completed and metadata.exists():
                payload = json.loads(metadata.read_text())
            else:
                run_dir.mkdir(parents=True, exist_ok=True)
                completed = subprocess.run(command, env=environment)
                if completed.returncode != 0:
                    raise SystemExit(completed.returncode)
                payload = json.loads(metadata.read_text())
            result = payload.get("method_only", {}).get("result", {})
            if "finalF_phys" not in result:
                raise RuntimeError(f"metadata has no method-only result: {metadata}")
            raw_rows.append(
                {
                    "seed": seed,
                    "degree": degree,
                    "final_physical_fidelity": float(result["finalF_phys"]),
                    "metadata": str(metadata),
                }
            )

            if degree == 2:
                source = trace_dir / "ea_trace.csv"
                target = args.outdir / "traces" / "ea_p2_nested_trace.csv"
            elif degree == 3:
                source = trace_dir / "ea_p3_nested_trace.csv"
                if not source.exists():
                    source = trace_dir / "ea_trace.csv"
                target = args.outdir / "traces" / "ea_p3_continuation_trace.csv"
            else:
                continue
            if source.exists() and not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)

    if args.dry_run:
        return 0

    by_seed: dict[int, dict[int, float]] = {}
    for row in raw_rows:
        by_seed.setdefault(int(row["seed"]), {})[int(row["degree"])] = float(row["final_physical_fidelity"])

    cumulative_rows: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "N": 5,
        "definition": "F_le_P(seed)=max_{2<=p<=P} F_p(seed)",
        "settings": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
        "by_degree_cap": {},
    }
    for cap in degrees:
        values: list[float] = []
        for seed in seeds:
            values_for_seed = [by_seed[seed][degree] for degree in degrees if degree <= cap and degree in by_seed[seed]]
            if not values_for_seed:
                continue
            best = max(values_for_seed)
            source_degree = max(
                degree for degree in degrees if degree <= cap and degree in by_seed[seed] and by_seed[seed][degree] == best
            )
            values.append(best)
            cumulative_rows.append(
                {
                    "N": 5,
                    "seed": seed,
                    "degree_cap": cap,
                    "best_final_physical_fidelity": best,
                    "best_source_p": source_degree,
                    "success": best >= args.threshold,
                    "success_threshold": args.threshold,
                }
            )
        summary["by_degree_cap"][str(cap)] = summarize(values, cap, args.threshold)

    write_csv(args.outdir / "n5_cumulative_degree_seed25_rows.csv", cumulative_rows)
    (args.outdir / "n5_cumulative_degree_seed25_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    write_csv(args.outdir / "n5_degree_cap_raw_rows.csv", raw_rows)
    print(f"wrote {args.outdir / 'n5_cumulative_degree_seed25_rows.csv'}")
    print(f"wrote {args.outdir / 'n5_cumulative_degree_seed25_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
