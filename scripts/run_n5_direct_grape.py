                      
"""Run the matched direct-GRAPE baseline for the N=5, T=8 GHZ point.

This is the orange baseline paired with the N=5 state-growing SR-GRAPE
benchmark run. It uses the same physical Hamiltonian, target, time horizon,
amplitude box, micro-grid schedule, optimizer, and terminal physical GRAPE
budget.  It intentionally omits the EA/state-growing/product-formula front end,
because that front end is the SR-GRAPE control-design advantage being tested.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PACKAGE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from srgrape import srgrape as sr


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(values: list[float], threshold: float) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    n = int(arr.size)
    return {
        "n": n,
        "mean": float(np.mean(arr)) if n else None,
        "std_population": float(np.std(arr)) if n else None,
        "std_sample": float(np.std(arr, ddof=1)) if n > 1 else 0.0,
        "sem": float(np.std(arr, ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
        "min": float(np.min(arr)) if n else None,
        "max": float(np.max(arr)) if n else None,
        "success_threshold": float(threshold),
        "successes": int(np.sum(arr >= float(threshold))) if n else 0,
        "success_rate": float(np.mean(arr >= float(threshold))) if n else None,
    }


def _build_dense_problem(d: int, N: int, target_name: str, drives: str) -> dict[str, Any]:
    psi0, target = sr.build_targets(d, N, target_name)
    H0_base = sr.build_drift_base(d, N)
    ctrl_axes, ctrl_labels = sr.build_single_site_axes(d, N, drives)
    return {
        "psi0": psi0,
        "target": target,
        "H0_base": H0_base,
        "ctrl_axes": ctrl_axes,
        "ctrl_labels": ctrl_labels,
        "H0_base_dense": sr.qobj_to_dense_matrix(H0_base),
        "ctrl_stack_dense": sr.build_dense_operator_stack(ctrl_axes),
        "psi0_vec": sr.qobj_to_dense_vector(psi0),
        "target_vec": sr.qobj_to_dense_vector(target),
    }


def run_trial(config: dict[str, Any], trial: int) -> dict[str, Any]:
    d = int(config["d"])
    N = int(config["N"])
    T = float(config["T"])
    Nt = int(config["Nt"])
    amp = float(config["amp"])
    threshold = float(config["threshold"])
    trial_seed = int(config["seed"]) + int(trial) - 1

    problem = _build_dense_problem(d=d, N=N, target_name=str(config["target"]), drives=str(config["drives"]))
    stages = sr.build_stages(
        [1.0],
        S1=int(config["S1"]),
        S2=int(config["S2"]),
        iters1=int(config["iters1"]),
        iters2=int(config["iters2"]),
        lr1=float(config["lr1"]),
        lr2=float(config["lr2"]),
    )
    opt_cfg = sr.OptConfig(
        lr=float(config["lr2"]),
        l2=0.0,
        amp=amp,
        clip=float(config["clip"]),
        backtracks=int(config["backtracks"]),
        accept_mode=str(config["accept_mode"]),
        accept_drop=float(config["accept_drop"]),
        threshold=threshold,
        verbose=0,
        stall_enable=bool(config["stall_enable"]),
        stall_gnorm=float(config["stall_gnorm"]),
        stall_max_kicks=int(config["stall_max_kicks"]),
        stall_kick_sigma=float(config["stall_kick_sigma"]),
    )

    rng = np.random.RandomState(trial_seed)
    M0 = Nt * stages[0].S
    u0 = sr.random_controls(
        M=M0,
        m=len(problem["ctrl_axes"]),
        rng=rng,
        amp_bound=amp,
        init_mode=str(config["init_mode"]),
        sigma=float(config["sigma"]),
        target_rms=None,
    )

    t0 = time.time()
    rr, u_final = sr.run_method_stages(
        name="DIRECT_GRAPE_N5_T8",
        psi0=problem["psi0"],
        target=problem["target"],
        H0_base=problem["H0_base"],
        drift_strength_hw=float(config["drift_strength_hw"]),
        ctrl_axes=problem["ctrl_axes"],
        T=T,
        Nt=Nt,
        stages=stages,
        homotopy_mode="mult",
        u0=u0,
        opt_cfg_template=opt_cfg,
        threshold=threshold,
        seed=trial_seed + 10_000,
        jitter_list=[0.0],
        verbose=0,
        reporter=None,
        progress_every=int(config["progress_every"]),
        trace_dir=None,
        H0_base_dense=problem["H0_base_dense"],
        ctrl_stack_dense=problem["ctrl_stack_dense"],
        psi0_vec=problem["psi0_vec"],
        target_vec=problem["target_vec"],
    )
    row = {
        "trial": int(trial),
        "seed": int(trial_seed),
        "initF_phys": float(rr.initF_phys),
        "finalF_phys": float(rr.finalF_phys),
        "bestF_last_stage": float(rr.bestF_last_stage),
        "iters_to_threshold": "" if rr.iters_to_threshold is None else int(rr.iters_to_threshold),
        "total_stage_iters": int(rr.total_stage_iters),
        "wall_s": float(time.time() - t0),
    }
    return {"row": row, "u_final": np.asarray(u_final, dtype=float), "ctrl_labels": problem["ctrl_labels"]}


def write_summary(
    path: Path,
    rows_path: Path,
    best_controls_path: Path,
    config: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    summary = {
        "settings": {
            "d": int(config["d"]),
            "N": int(config["N"]),
            "target": str(config["target"]),
            "T": float(config["T"]),
            "Nt": int(config["Nt"]),
            "drives": str(config["drives"]),
            "drift_strength_hw": float(config["drift_strength_hw"]),
            "amp": float(config["amp"]),
            "homotopy": [1.0],
            "S1": int(config["S1"]),
            "S2": int(config["S2"]),
            "iters1": int(config["iters1"]),
            "iters2": int(config["iters2"]),
            "lr1": float(config["lr1"]),
            "lr2": float(config["lr2"]),
            "threshold": float(config["threshold"]),
            "baseline_init": str(config["init_mode"]),
            "base_seed": int(config["seed"]),
            "trials_requested": int(config["trials"]),
            "workers": int(config["workers"]),
            "comparison_note": "same terminal physical GRAPE refinement budget as the N=5 T=8 state-growing SR-GRAPE run",
        },
        "summary": summarize([float(r["finalF_phys"]) for r in rows], float(config["threshold"])),
        "rows_csv": str(rows_path),
        "best_controls_csv": str(best_controls_path),
        "rows": rows,
    }
    with path.open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=25)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--outdir", type=Path, default=Path("runs/n5_t8_direct_grape_baseline"))
    ap.add_argument("--progress-every", type=int, default=50)
    ap.add_argument("--iters1", type=int, default=500)
    ap.add_argument("--iters2", type=int, default=800)
    ap.add_argument("--S1", type=int, default=64)
    ap.add_argument("--S2", type=int, default=96)
    ap.add_argument("--lr1", type=float, default=0.04)
    ap.add_argument("--lr2", type=float, default=0.025)
    ap.add_argument("--threshold", type=float, default=0.99)
    ap.add_argument("--init-mode", default="rmsmatch")
    ap.add_argument("--sigma", type=float, default=0.2)
    args = ap.parse_args()

    config: dict[str, Any] = {
        "d": 2,
        "N": 5,
        "target": "ghz",
        "T": 8.0,
        "Nt": 20,
        "drives": "xy",
        "drift_strength_hw": 1.0,
        "amp": 2.0,
        "S1": int(args.S1),
        "S2": int(args.S2),
        "iters1": int(args.iters1),
        "iters2": int(args.iters2),
        "lr1": float(args.lr1),
        "lr2": float(args.lr2),
        "threshold": float(args.threshold),
        "seed": int(args.seed),
        "trials": int(args.trials),
        "workers": max(1, int(args.workers)),
        "progress_every": int(args.progress_every),
        "init_mode": str(args.init_mode),
        "sigma": float(args.sigma),
        "accept_mode": "soft",
        "accept_drop": 0.02,
        "backtracks": 5,
        "clip": 5.0,
        "stall_enable": True,
        "stall_gnorm": 1e-6,
        "stall_max_kicks": 12,
        "stall_kick_sigma": 0.04,
    }

    args.outdir.mkdir(parents=True, exist_ok=True)
    rows_path = args.outdir / "n5_t8_direct_grape_baseline_rows.csv"
    summary_path = args.outdir / "n5_t8_direct_grape_baseline_summary.json"
    best_controls_path = args.outdir / "n5_t8_direct_grape_best_controls.csv"

    print(
        "[N5 DIRECT] trials={} workers={} T=8.0 Nt=20 S=({}, {}) iters=({}, {}) lr=({}, {})".format(
            int(args.trials),
            config["workers"],
            config["S1"],
            config["S2"],
            config["iters1"],
            config["iters2"],
            config["lr1"],
            config["lr2"],
        ),
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    best_f = -1.0
    best_u: np.ndarray | None = None
    ctrl_labels: list[str] | None = None
    t0 = time.time()

    if config["workers"] == 1:
        for trial in range(1, int(args.trials) + 1):
            result = run_trial(config, trial)
            row = result["row"]
            rows.append(row)
            rows.sort(key=lambda r: int(r["trial"]))
            if float(row["finalF_phys"]) > best_f:
                best_f = float(row["finalF_phys"])
                best_u = result["u_final"]
                ctrl_labels = list(result["ctrl_labels"])
                sr.write_awg_csv(best_controls_path, best_u, float(config["T"]) / best_u.shape[0], ctrl_labels)
            write_csv(rows_path, rows)
            write_summary(summary_path, rows_path, best_controls_path, config, rows)
            print(
                f"trial {int(row['trial']):2d}/{int(args.trials)} seed={int(row['seed']):3d} "
                f"initF={float(row['initF_phys']):.9f} finalF={float(row['finalF_phys']):.9f} "
                f"iters={int(row['total_stage_iters'])} wall={float(row['wall_s']):.1f}s "
                f"elapsed={time.time() - t0:.1f}s",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=config["workers"]) as pool:
            futures = {pool.submit(run_trial, config, trial): trial for trial in range(1, int(args.trials) + 1)}
            for fut in as_completed(futures):
                result = fut.result()
                row = result["row"]
                rows.append(row)
                rows.sort(key=lambda r: int(r["trial"]))
                if float(row["finalF_phys"]) > best_f:
                    best_f = float(row["finalF_phys"])
                    best_u = result["u_final"]
                    ctrl_labels = list(result["ctrl_labels"])
                    sr.write_awg_csv(best_controls_path, best_u, float(config["T"]) / best_u.shape[0], ctrl_labels)
                write_csv(rows_path, rows)
                write_summary(summary_path, rows_path, best_controls_path, config, rows)
                print(
                    f"trial {int(row['trial']):2d}/{int(args.trials)} seed={int(row['seed']):3d} "
                    f"initF={float(row['initF_phys']):.9f} finalF={float(row['finalF_phys']):.9f} "
                    f"iters={int(row['total_stage_iters'])} wall={float(row['wall_s']):.1f}s "
                    f"completed={len(rows)}/{int(args.trials)} elapsed={time.time() - t0:.1f}s",
                    flush=True,
                )

    write_summary(summary_path, rows_path, best_controls_path, config, rows)
    print(json.dumps(summarize([float(r["finalF_phys"]) for r in rows], float(config["threshold"])), indent=2))


if __name__ == "__main__":
    main()
