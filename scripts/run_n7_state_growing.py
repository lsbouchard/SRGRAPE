                      
"""N=7 amp=3 SR-GRAPE state-growing validation runner.

This script launches the fixed srgrape state-growing calculation for each
saved N=6 parent seed and aggregates the resulting N=7 fidelities.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRGRAPE = PACKAGE_ROOT / "src" / "srgrape" / "srgrape.py"
DEFAULT_PARENT_ROWS = Path("reproduction/data/n6_t8_state_grow_seed1_25_rows.csv")


def resolve_parent_controls(path: Path, seed: int) -> Path:
    """Resolve parent controls from the current reproduction output."""
    if path.exists():
        return path
    repo_candidate = PACKAGE_ROOT / path
    if repo_candidate.exists():
        return repo_candidate
    return path


@dataclass(frozen=True)
class ParentSeed:
    seed: int
    parent_final_fidelity: float
    controls_path: Path


def read_parent_rows(path: Path) -> list[ParentSeed]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    parents: list[ParentSeed] = []
    for row in rows:
        seed = int(row["seed"])
        controls_path = resolve_parent_controls(Path(row["final_controls_path"]), seed)
        if not controls_path.exists():
            raise FileNotFoundError(f"missing parent controls for seed {row['seed']}: {controls_path}")
        parents.append(
            ParentSeed(
                seed=seed,
                parent_final_fidelity=float(row["final_child_fidelity"]),
                controls_path=controls_path,
            )
        )
    return sorted(parents, key=lambda r: r.seed)


def latest_summary(seed_dir: Path) -> Path | None:
    summaries = sorted(seed_dir.glob("state_grow_summary_*.json"))
    return summaries[-1] if summaries else None


def build_command(srgrape: Path, parent: ParentSeed, seed_dir: Path) -> list[str]:
    tag = f"n7_amp3_seed{parent.seed:02d}"
    return [
        sys.executable,
        str(srgrape),
        "--state-grow",
        "1",
        "--state-grow-parent-controls",
        str(parent.controls_path),
        "--d",
        "2",
        "--N",
        "7",
        "--Nt",
        "20",
        "--T",
        "8.0",
        "--p",
        "4",
        "--target",
        "ghz",
        "--drives",
        "xy",
        "--drift-strength-ea",
        "1.0",
        "--drift-strength-hw",
        "1.0",
        "--state-grow-newqubit-sigma",
        "0.05",
        "--state-grow-newqubit-seed",
        "123",
        "--state-grow-homotopy",
        "1.0",
        "--S1",
        "96",
        "--S2",
        "128",
        "--iters1",
        "160",
        "--iters2",
        "220",
        "--lr1",
        "0.025",
        "--lr2",
        "0.018",
        "--amp",
        "3.0",
        "--hard-amp",
        "3.0",
        "--l2",
        "0.0",
        "--accept-mode",
        "soft",
        "--accept-drop",
        "0.01",
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
        "0.04",
        "--threshold",
        "0.99",
        "--seed",
        str(parent.seed),
        "--verbose",
        "0",
        "--progress",
        "off",
        "--save-traces",
        "0",
        "--save-compiler-diagnostics",
        "0",
        "--save-metadata",
        "1",
        "--outdir",
        str(seed_dir),
        "--tag",
        tag,
        "--metadata-path",
        str(seed_dir / "metadata.json"),
    ]


def parse_summary(summary_path: Path, parent: ParentSeed, wall_s: float | None = None) -> dict:
    data = json.loads(summary_path.read_text())
    child = data["child"]
    parent_data = data["parent"]
    return {
        "seed": parent.seed,
        "parent_controls_path": str(parent.controls_path),
        "parent_manifest_fidelity": parent.parent_final_fidelity,
        "parent_physical_seed_fidelity": float(parent_data["physical_seed_fidelity"]),
        "embedded_child_fidelity": float(child["embedded_fidelity"]),
        "dithered_child_fidelity": float(child["dithered_fidelity"]),
        "final_child_fidelity": float(child["final_fidelity"]),
        "iters_to_threshold": child["iters_to_threshold"],
        "total_stage_iters": int(child["total_stage_iters"]),
        "child_wall_s": float(child["wall_s"]),
        "wrapper_wall_s": wall_s,
        "summary_json": str(summary_path),
        "final_controls_path": str(child["final_controls_path"]),
    }


def run_one(args_tuple: tuple[Path, ParentSeed, Path, bool]) -> dict:
    srgrape, parent, root, skip_completed = args_tuple
    seed_dir = root / f"seed{parent.seed:02d}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    log_path = seed_dir / "run.log"
    summary_path = latest_summary(seed_dir)
    if skip_completed and summary_path is not None:
        row = parse_summary(summary_path, parent, wall_s=None)
        row["returncode"] = 0
        row["status"] = "skipped_existing"
        row["log"] = str(log_path)
        return row

    cmd = build_command(srgrape, parent, seed_dir)
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(seed_dir / "mplconfig")
    start = time.time()
    with log_path.open("w") as log:
        log.write("COMMAND: " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    wall_s = time.time() - start

    summary_path = latest_summary(seed_dir)
    if summary_path is None:
        return {
            "seed": parent.seed,
            "returncode": proc.returncode,
            "status": "missing_summary",
            "log": str(log_path),
            "parent_controls_path": str(parent.controls_path),
            "error": "no state_grow_summary_*.json produced",
            "wrapper_wall_s": wall_s,
        }

    row = parse_summary(summary_path, parent, wall_s=wall_s)
    row["returncode"] = proc.returncode
    row["status"] = "ok" if proc.returncode == 0 else "nonzero_return"
    row["log"] = str(log_path)
    return row


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict], threshold: float) -> dict:
    finals = [float(r["final_child_fidelity"]) for r in rows if "final_child_fidelity" in r]
    success = [f >= threshold for f in finals]
    below_threshold = [
        int(r["seed"])
        for r in rows
        if "final_child_fidelity" in r and float(r["final_child_fidelity"]) < threshold
    ]
    if finals:
        mean = sum(finals) / len(finals)
        var = sum((f - mean) ** 2 for f in finals) / max(1, len(finals) - 1)
        std = var ** 0.5
    else:
        mean = std = float("nan")
    return {
        "threshold": threshold,
        "n_requested": len(rows),
        "n_completed_with_fidelity": len(finals),
        "success_count": int(sum(success)),
        "success_rate": float(sum(success) / len(finals)) if finals else 0.0,
        "final_mean": mean,
        "final_std_sample": std,
        "final_min": min(finals) if finals else None,
        "final_max": max(finals) if finals else None,
        "below_threshold_seeds": below_threshold,
        "failed_or_incomplete_seeds": [
            int(r["seed"]) for r in rows if r.get("status") not in {"ok", "skipped_existing"}
        ],
        "rows": rows,
    }


def write_status(path: Path, summary: dict) -> None:
    lines = [
        "# N=7 amp=3 First-Pass SR-GRAPE Validation",
        "",
        f"- Threshold: `{summary['threshold']}`",
        f"- Completed with fidelity: `{summary['n_completed_with_fidelity']}/{summary['n_requested']}`",
        f"- Successes: `{summary['success_count']}/{summary['n_completed_with_fidelity']}`",
        f"- Success rate: `{summary['success_rate']:.3f}`",
        f"- Final fidelity mean: `{summary['final_mean']:.12f}`",
        f"- Final fidelity sample std: `{summary['final_std_sample']:.12f}`",
        f"- Final fidelity range: `{summary['final_min']:.12f}` to `{summary['final_max']:.12f}`",
        f"- Below-threshold seeds: `{summary['below_threshold_seeds']}`",
        "",
        "| seed | parent F | embedded N=7 F | final N=7 F | iters | wall min | status |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in sorted(summary["rows"], key=lambda r: int(r["seed"])):
        wall = row.get("child_wall_s", row.get("wrapper_wall_s"))
        wall_min = float(wall) / 60.0 if wall is not None else float("nan")
        lines.append(
            "| {seed} | {parent:.12f} | {embedded:.12f} | {final:.12f} | {iters} | {wall:.2f} | {status} |".format(
                seed=int(row["seed"]),
                parent=float(row.get("parent_physical_seed_fidelity", float("nan"))),
                embedded=float(row.get("embedded_child_fidelity", float("nan"))),
                final=float(row.get("final_child_fidelity", float("nan"))),
                iters=row.get("total_stage_iters", ""),
                wall=wall_min,
                status=row.get("status", ""),
            )
        )
    path.write_text("\n".join(lines) + "\n")


def parse_seeds(raw: str, available: Iterable[int]) -> list[int]:
    available_set = set(available)
    if raw.strip().lower() in {"all", "1-25"}:
        return sorted(available_set)
    seeds: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = [int(x) for x in part.split("-", 1)]
            seeds.update(range(a, b + 1))
        else:
            seeds.add(int(part))
    missing = sorted(seeds - available_set)
    if missing:
        raise ValueError(f"requested seeds not in parent manifest: {missing}")
    return sorted(seeds)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--srgrape", type=Path, default=DEFAULT_SRGRAPE)
    ap.add_argument("--parent-rows", type=Path, default=DEFAULT_PARENT_ROWS)
    ap.add_argument("--outdir", type=Path, default=Path("runs/n7_amp3_state_growing25"))
    ap.add_argument("--seeds", default="1-25")
    ap.add_argument("--max-workers", type=int, default=2)
    ap.add_argument("--threshold", type=float, default=0.99)
    ap.add_argument("--skip-completed", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.srgrape.exists():
        raise FileNotFoundError(args.srgrape)
    parents_all = read_parent_rows(args.parent_rows)
    seed_list = parse_seeds(args.seeds, [p.seed for p in parents_all])
    parents = [p for p in parents_all if p.seed in seed_list]
    args.outdir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "srgrape": str(args.srgrape),
        "parent_rows": str(args.parent_rows),
        "outdir": str(args.outdir),
        "seeds": seed_list,
        "max_workers": int(args.max_workers),
        "threshold": float(args.threshold),
        "skip_completed": bool(args.skip_completed),
        "parents": [
            {
                "seed": p.seed,
                "parent_final_fidelity": p.parent_final_fidelity,
                "controls_path": str(p.controls_path),
            }
            for p in parents
        ],
    }
    (args.outdir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return 0

    rows: list[dict] = []
    tasks = [(args.srgrape, p, args.outdir, bool(args.skip_completed)) for p in parents]
    started = time.time()
                                                                                
                                                                                   
    with ThreadPoolExecutor(max_workers=max(1, int(args.max_workers))) as ex:
        futures = {ex.submit(run_one, task): task[1].seed for task in tasks}
        for fut in as_completed(futures):
            seed = futures[fut]
            try:
                row = fut.result()
            except Exception as exc:                                                     
                row = {"seed": seed, "status": "runner_exception", "error": repr(exc)}
            rows.append(row)
            rows_sorted = sorted(rows, key=lambda r: int(r["seed"]))
            write_csv(args.outdir / "n7_amp3_state_growing_rows_partial.csv", rows_sorted)
            partial_summary = summarize(rows_sorted, args.threshold)
            partial_summary["elapsed_s"] = time.time() - started
            (args.outdir / "n7_amp3_state_growing_summary_partial.json").write_text(
                json.dumps(partial_summary, indent=2)
            )
            print(
                "seed {seed:02d} {status} final={final} completed={done}/{total}".format(
                    seed=int(row["seed"]),
                    status=row.get("status", ""),
                    final=(
                        f"{float(row['final_child_fidelity']):.12f}"
                        if "final_child_fidelity" in row
                        else "NA"
                    ),
                    done=len(rows),
                    total=len(tasks),
                ),
                flush=True,
            )

    rows = sorted(rows, key=lambda r: int(r["seed"]))
    summary = summarize(rows, args.threshold)
    summary["elapsed_s"] = time.time() - started
    rows_path = args.outdir / "n7_amp3_state_growing_rows.csv"
    summary_path = args.outdir / "n7_amp3_state_growing_summary.json"
    status_path = args.outdir / "N7_AMP3_STATE_GROWING_STATUS.md"
    write_csv(rows_path, rows)
    summary_path.write_text(json.dumps(summary, indent=2))
    write_status(status_path, summary)
    print(f"wrote {rows_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {status_path}")
    return 0 if not summary["failed_or_incomplete_seeds"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
