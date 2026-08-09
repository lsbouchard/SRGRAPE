                      
"""Direct-GRAPE N=7 amp=3 baseline for the SR-GRAPE comparison.

The baseline uses the same physical Hamiltonian, horizon, amplitude bound, grid
schedule, optimizer, and iteration budget as the N=7 SR-GRAPE validation.  The
only removed ingredient is the SR/state-growing initialization: controls are
random physical XY waveforms, RMS-matched to the corresponding SR starting
waveform for the same seed.
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
from pathlib import Path

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SR_CONTROLS_ROOT = Path("reproduction/n7_amp3_state_growing25")


def load_sr_module():
    src_root = PACKAGE_ROOT / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from srgrape import srgrape as sr

    return sr


def load_awg_controls(path: Path, expected_labels: list[str]) -> np.ndarray:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"empty CSV: {path}")
        labels = [x for x in reader.fieldnames if x != "t"]
        if labels != expected_labels:
            raise ValueError(f"label mismatch in {path}\nloaded={labels}\nexpected={expected_labels}")
        rows = [[float(row[label]) for label in labels] for row in reader]
    if not rows:
        raise ValueError(f"no rows in {path}")
    return np.asarray(rows, dtype=float)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def run_one_seed(args: argparse.Namespace) -> int:
    sr = load_sr_module()
    seed = int(args.one_seed)
    outdir = Path(args.outdir) / f"seed{seed:02d}"
    outdir.mkdir(parents=True, exist_ok=True)

    d = 2
    N = 7
    Nt = int(args.Nt)
    T = float(args.T)
    amp = float(args.amp)
    threshold = float(args.threshold)
    S1 = int(args.S1)
    S2 = int(args.S2)

    psi0, target = sr.build_targets(d, N, "ghz")
    axes, labels = sr.build_single_site_axes(d, N, "xy")
    H0_base = sr.build_drift_base(d, N)
    H0_dense = sr.qobj_to_dense_matrix(H0_base)
    ctrl_stack = sr.build_dense_operator_stack(axes)
    psi0_vec = sr.qobj_to_dense_vector(psi0)
    target_vec = sr.qobj_to_dense_vector(target)

    ref_path = (
        Path(args.sr_controls_root)
        / f"seed{seed:02d}"
        / f"state_grow_dithered_controls_n7_amp3_seed{seed:02d}.csv"
    )
    if not ref_path.exists():
        raise FileNotFoundError(ref_path)
    u_ref = load_awg_controls(ref_path, labels)
    M0 = Nt * S1
    ref_rms = float(sr.rms(sr.resample_controls(u_ref, M0)))

    rng = np.random.RandomState(seed)
    u0 = sr.random_controls(
        M0,
        len(labels),
        rng,
        amp_bound=amp,
        init_mode=str(args.init),
        sigma=float(args.sigma),
        target_rms=ref_rms,
    )

    stages = [
        sr.Stage(scale=1.0, S=S1, iters=int(args.iters1), lr=float(args.lr1)),
        sr.Stage(scale=1.0, S=S2, iters=int(args.iters2), lr=float(args.lr2)),
    ]
    cfg = sr.OptConfig(
        lr=float(args.lr1),
        l2=0.0,
        amp=amp,
        clip=5.0,
        backtracks=5,
        accept_mode="soft",
        accept_drop=float(args.accept_drop),
        threshold=threshold,
        verbose=0,
        stall_enable=True,
        stall_gnorm=1e-6,
        stall_max_kicks=12,
        stall_kick_sigma=0.04,
    )

    start = time.time()
    rr, u_final = sr.run_method_stages(
        name="DIRECT_GRAPE",
        psi0=psi0,
        target=target,
        H0_base=H0_base,
        drift_strength_hw=1.0,
        ctrl_axes=axes,
        T=T,
        Nt=Nt,
        stages=stages,
        homotopy_mode="mult",
        u0=u0,
        opt_cfg_template=cfg,
        threshold=threshold,
        seed=seed,
        jitter_list=None,
        verbose=0,
        reporter=sr.ConsoleReporter("off"),
        progress_every=int(args.progress_every),
        H0_base_dense=H0_dense,
        ctrl_stack_dense=ctrl_stack,
        psi0_vec=psi0_vec,
        target_vec=target_vec,
    )
    wall_s = time.time() - start

    controls_path = outdir / f"direct_grape_controls_seed{seed:02d}.csv"
    summary_path = outdir / f"direct_grape_summary_seed{seed:02d}.json"
    sr.write_awg_csv(controls_path, u_final, T / u_final.shape[0], labels)
    summary = {
        "seed": seed,
        "mode": "direct_grape_rmsmatched",
        "target": "ghz",
        "N": N,
        "T": T,
        "Nt": Nt,
        "amp": amp,
        "threshold": threshold,
        "reference_sr_dithered_controls": str(ref_path),
        "reference_rms": ref_rms,
        "random_init_rms": float(sr.rms(u0)),
        "stages": [{"S": st.S, "iters": st.iters, "lr": st.lr, "scale": st.scale} for st in stages],
        "result": {
            "initF_phys": rr.initF_phys,
            "finalF_phys": rr.finalF_phys,
            "bestF_last_stage": rr.bestF_last_stage,
            "iters_to_threshold": rr.iters_to_threshold,
            "total_stage_iters": rr.total_stage_iters,
            "wall_s": rr.wall_s,
        },
        "wrapper_wall_s": wall_s,
        "controls_path": str(controls_path),
    }
    write_json(summary_path, summary)
    print(
        f"seed {seed:02d} final={rr.finalF_phys:.12f} init={rr.initF_phys:.12f} "
        f"iters={rr.total_stage_iters} wall_min={wall_s / 60:.2f}",
        flush=True,
    )
    return 0


def parse_seed_list(raw: str) -> list[int]:
    if raw.strip().lower() in {"all", "1-25"}:
        return list(range(1, 26))
    out: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = [int(x) for x in part.split("-", 1)]
            out.update(range(a, b + 1))
        else:
            out.add(int(part))
    return sorted(out)


def latest_summary(seed_dir: Path, seed: int) -> Path | None:
    path = seed_dir / f"direct_grape_summary_seed{seed:02d}.json"
    return path if path.exists() else None


def launch_one(script: Path, args: argparse.Namespace, seed: int) -> dict:
    seed_dir = Path(args.outdir) / f"seed{seed:02d}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    summary_path = latest_summary(seed_dir, seed)
    if int(args.skip_completed) and summary_path is not None:
        data = json.loads(summary_path.read_text())
        return row_from_summary(data, "skipped_existing", 0, str(seed_dir / "run.log"))

    cmd = [
        sys.executable,
        str(script),
        "--one-seed",
        str(seed),
        "--outdir",
        str(args.outdir),
        "--sr-controls-root",
        str(args.sr_controls_root),
        "--init",
        str(args.init),
        "--sigma",
        str(args.sigma),
        "--T",
        str(args.T),
        "--Nt",
        str(args.Nt),
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
        "--accept-drop",
        str(args.accept_drop),
        "--threshold",
        str(args.threshold),
    ]
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(seed_dir / "mplconfig")
    log_path = seed_dir / "run.log"
    start = time.time()
    with log_path.open("w") as log:
        log.write("COMMAND: " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    wall_s = time.time() - start

    summary_path = latest_summary(seed_dir, seed)
    if summary_path is None:
        return {
            "seed": seed,
            "status": "missing_summary",
            "returncode": proc.returncode,
            "wrapper_wall_s": wall_s,
            "log": str(log_path),
        }
    data = json.loads(summary_path.read_text())
    row = row_from_summary(data, "ok" if proc.returncode == 0 else "nonzero_return", proc.returncode, str(log_path))
    row["wrapper_wall_s"] = wall_s
    return row


def row_from_summary(data: dict, status: str, returncode: int, log_path: str) -> dict:
    result = data["result"]
    return {
        "seed": int(data["seed"]),
        "status": status,
        "returncode": int(returncode),
        "initF_phys": float(result["initF_phys"]),
        "finalF_phys": float(result["finalF_phys"]),
        "bestF_last_stage": float(result["bestF_last_stage"]),
        "iters_to_threshold": result["iters_to_threshold"],
        "total_stage_iters": int(result["total_stage_iters"]),
        "child_wall_s": float(result["wall_s"]),
        "reference_rms": float(data["reference_rms"]),
        "random_init_rms": float(data["random_init_rms"]),
        "summary_json": str(Path(log_path).with_name(f"direct_grape_summary_seed{int(data['seed']):02d}.json")),
        "controls_path": data["controls_path"],
        "log": log_path,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
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
    finals = [float(r["finalF_phys"]) for r in rows if "finalF_phys" in r]
    mean = sum(finals) / len(finals) if finals else float("nan")
    std = (
        (sum((x - mean) ** 2 for x in finals) / max(1, len(finals) - 1)) ** 0.5
        if finals
        else float("nan")
    )
    return {
        "threshold": float(threshold),
        "n": len(rows),
        "n_with_fidelity": len(finals),
        "success_count": sum(x >= threshold for x in finals),
        "success_rate": (sum(x >= threshold for x in finals) / len(finals)) if finals else 0.0,
        "mean_finalF": mean,
        "std_finalF_sample": std,
        "min_finalF": min(finals) if finals else None,
        "max_finalF": max(finals) if finals else None,
        "rows": rows,
    }


def write_status(path: Path, summary: dict) -> None:
    lines = [
        "# N=7 amp=3 Direct-GRAPE Baseline",
        "",
        "Direct GRAPE uses random physical XY controls RMS-matched to the corresponding SR-GRAPE starting controls.",
        "",
        f"- Threshold: `{summary['threshold']}`",
        f"- Successes: `{summary['success_count']}/{summary['n_with_fidelity']}`",
        f"- Success rate: `{summary['success_rate']:.3f}`",
        f"- Mean final fidelity: `{summary['mean_finalF']:.12f}`",
        f"- Sample std: `{summary['std_finalF_sample']:.12f}`",
        f"- Final fidelity range: `{summary['min_finalF']:.12f}` to `{summary['max_finalF']:.12f}`",
        "",
        "| seed | init F | final F | iters | wall min |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(summary["rows"], key=lambda r: int(r["seed"])):
        lines.append(
            f"| {int(row['seed'])} | {float(row['initF_phys']):.12f} | "
            f"{float(row['finalF_phys']):.12f} | {int(row['total_stage_iters'])} | "
            f"{float(row['child_wall_s']) / 60.0:.2f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--one-seed", type=int, default=0)
    ap.add_argument("--outdir", type=Path, default=Path("runs/n7_amp3_grape_baseline25"))
    ap.add_argument("--sr-controls-root", type=Path, default=DEFAULT_SR_CONTROLS_ROOT)
    ap.add_argument("--seeds", default="1-25")
    ap.add_argument("--max-workers", type=int, default=2)
    ap.add_argument("--skip-completed", type=int, default=1)
    ap.add_argument("--init", default="rmsmatch", choices=["rmsmatch", "gauss", "uniform"])
    ap.add_argument("--sigma", type=float, default=0.2)
    ap.add_argument("--T", type=float, default=8.0)
    ap.add_argument("--Nt", type=int, default=20)
    ap.add_argument("--S1", type=int, default=96)
    ap.add_argument("--S2", type=int, default=128)
    ap.add_argument("--iters1", type=int, default=160)
    ap.add_argument("--iters2", type=int, default=220)
    ap.add_argument("--lr1", type=float, default=0.025)
    ap.add_argument("--lr2", type=float, default=0.018)
    ap.add_argument("--amp", type=float, default=3.0)
    ap.add_argument("--accept-drop", type=float, default=0.01)
    ap.add_argument("--threshold", type=float, default=0.99)
    ap.add_argument("--progress-every", type=int, default=50)
    if len(sys.argv) == 1:
        ap.print_help()
        return 2
    args = ap.parse_args()

    if int(args.one_seed) > 0:
        return run_one_seed(args)

    args.outdir.mkdir(parents=True, exist_ok=True)
    seeds = parse_seed_list(args.seeds)
    manifest = {
        "mode": "direct_grape_rmsmatched",
        "seeds": seeds,
        "outdir": str(args.outdir),
        "sr_controls_root": str(args.sr_controls_root),
        "max_workers": int(args.max_workers),
        "schedule": {
            "T": float(args.T),
            "Nt": int(args.Nt),
            "S1": int(args.S1),
            "S2": int(args.S2),
            "iters1": int(args.iters1),
            "iters2": int(args.iters2),
            "lr1": float(args.lr1),
            "lr2": float(args.lr2),
            "amp": float(args.amp),
        },
    }
    write_json(args.outdir / "run_manifest.json", manifest)
    script = Path(__file__).resolve()
    rows: list[dict] = []
    start = time.time()
    with ThreadPoolExecutor(max_workers=max(1, int(args.max_workers))) as ex:
        futures = {ex.submit(launch_one, script, args, seed): seed for seed in seeds}
        for fut in as_completed(futures):
            seed = futures[fut]
            try:
                row = fut.result()
            except Exception as exc:
                row = {"seed": seed, "status": "runner_exception", "error": repr(exc)}
            rows.append(row)
            rows_sorted = sorted(rows, key=lambda r: int(r["seed"]))
            write_csv(args.outdir / "direct_grape_rows_partial.csv", rows_sorted)
            summary = summarize(rows_sorted, float(args.threshold))
            summary["elapsed_s"] = time.time() - start
            write_json(args.outdir / "direct_grape_summary_partial.json", summary)
            final_txt = f"{float(row['finalF_phys']):.12f}" if "finalF_phys" in row else "NA"
            print(
                f"seed {int(row['seed']):02d} {row.get('status')} final={final_txt} "
                f"completed={len(rows)}/{len(seeds)}",
                flush=True,
            )

    rows = sorted(rows, key=lambda r: int(r["seed"]))
    write_csv(args.outdir / "direct_grape_rows.csv", rows)
    summary = summarize(rows, float(args.threshold))
    summary["elapsed_s"] = time.time() - start
    write_json(args.outdir / "direct_grape_summary.json", summary)
    write_status(args.outdir / "N7_AMP3_DIRECT_GRAPE_STATUS.md", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
