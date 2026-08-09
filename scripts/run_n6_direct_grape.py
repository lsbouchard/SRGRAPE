                      
"""Run a direct-GRAPE baseline for the N=6 GHZ benchmark.

This script intentionally reuses the srgrape physical model, random-control
initializer, stage scheduler, and GRAPE optimizer.  It does not run EA or
state-growing; it is the orange direct-GRAPE baseline for Figure 1.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PACKAGE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from srgrape import srgrape as sr


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(values: list[float], threshold: float) -> dict:
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=25)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--outdir", type=Path, default=Path("runs/n6_direct_grape_baseline"))
    ap.add_argument("--progress-every", type=int, default=50)
    ap.add_argument("--quick-iters1", type=int, default=0, help="Override stage-1 iters for timing/debug only")
    ap.add_argument("--quick-iters2", type=int, default=0, help="Override stage-2 iters for timing/debug only")
    args = ap.parse_args()

    d = 2
    N = 6
    T = 8.0
    Nt = 20
    amp = 2.0
    threshold = 0.99
    drift_strength_hw = 1.0
    homotopy = [1.0]
    S1, S2 = 64, 96
    iters1 = int(args.quick_iters1) if int(args.quick_iters1) > 0 else 500
    iters2 = int(args.quick_iters2) if int(args.quick_iters2) > 0 else 800

    args.outdir.mkdir(parents=True, exist_ok=True)
    rows_path = args.outdir / "n6_direct_grape_baseline_rows.csv"
    summary_path = args.outdir / "n6_direct_grape_baseline_summary.json"
    best_controls_path = args.outdir / "n6_direct_grape_best_controls.csv"

    psi0, target = sr.build_targets(d, N, "ghz")
    H0_base = sr.build_drift_base(d, N)
    ctrl_axes, ctrl_labels = sr.build_single_site_axes(d, N, "xy")
    H0_base_dense = sr.qobj_to_dense_matrix(H0_base)
    ctrl_stack_dense = sr.build_dense_operator_stack(ctrl_axes)
    psi0_vec = sr.qobj_to_dense_vector(psi0)
    target_vec = sr.qobj_to_dense_vector(target)

    stages = sr.build_stages(
        homotopy,
        S1=S1,
        S2=S2,
        iters1=iters1,
        iters2=iters2,
        lr1=0.08,
        lr2=0.06,
    )
    opt_cfg = sr.OptConfig(
        lr=0.06,
        l2=0.0,
        amp=amp,
        clip=5.0,
        backtracks=5,
        accept_mode="soft",
        accept_drop=2e-3,
        threshold=threshold,
        verbose=0,
        stall_enable=True,
        stall_gnorm=1e-6,
        stall_max_kicks=12,
        stall_kick_sigma=0.05,
    )

    rows: list[dict] = []
    best_f = -1.0
    best_u = None
    t0 = time.time()
    for trial in range(1, int(args.trials) + 1):
        trial_seed = int(args.seed) + trial - 1
        rng = np.random.RandomState(trial_seed)
        M0 = Nt * stages[0].S
        u0 = sr.random_controls(
            M=M0,
            m=len(ctrl_axes),
            rng=rng,
            amp_bound=amp,
            init_mode="rmsmatch",
            sigma=0.2,
            target_rms=None,
        )
        run_t0 = time.time()
        rr, u_final = sr.run_method_stages(
            name="DIRECT_GRAPE_N6",
            psi0=psi0,
            target=target,
            H0_base=H0_base,
            drift_strength_hw=drift_strength_hw,
            ctrl_axes=ctrl_axes,
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
            progress_every=int(args.progress_every),
            trace_dir=None,
            H0_base_dense=H0_base_dense,
            ctrl_stack_dense=ctrl_stack_dense,
            psi0_vec=psi0_vec,
            target_vec=target_vec,
        )
        row = {
            "trial": trial,
            "seed": trial_seed,
            "initF_phys": rr.initF_phys,
            "finalF_phys": rr.finalF_phys,
            "bestF_last_stage": rr.bestF_last_stage,
            "iters_to_threshold": "" if rr.iters_to_threshold is None else rr.iters_to_threshold,
            "total_stage_iters": rr.total_stage_iters,
            "wall_s": rr.wall_s,
        }
        rows.append(row)
        write_csv(rows_path, rows)
        if rr.finalF_phys > best_f:
            best_f = float(rr.finalF_phys)
            best_u = np.asarray(u_final, dtype=float).copy()
            sr.write_awg_csv(best_controls_path, best_u, T / best_u.shape[0], ctrl_labels)
        elapsed = time.time() - t0
        print(
            f"trial {trial:2d}/{int(args.trials)} seed={trial_seed:3d} "
            f"initF={rr.initF_phys:.9f} finalF={rr.finalF_phys:.9f} "
            f"iters={rr.total_stage_iters} wall={time.time() - run_t0:.1f}s "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )

        summary = {
            "settings": {
                "d": d,
                "N": N,
                "target": "ghz",
                "T": T,
                "Nt": Nt,
                "drives": "xy",
                "drift_strength_hw": drift_strength_hw,
                "amp": amp,
                "homotopy": homotopy,
                "S1": S1,
                "S2": S2,
                "iters1": iters1,
                "iters2": iters2,
                "threshold": threshold,
                "baseline_init": "rmsmatch",
                "base_seed": int(args.seed),
            },
            "summary": summarize([float(r["finalF_phys"]) for r in rows], threshold),
            "rows_csv": str(rows_path),
            "best_controls_csv": str(best_controls_path),
            "rows": rows,
        }
        with summary_path.open("w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
