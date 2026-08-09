                      
"""Continue polishing an existing N=7 physical-control CSV."""

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


def load_awg_controls(path: Path, expected_labels: list[str]) -> tuple[np.ndarray, list[str]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"empty CSV: {path}")
        labels = [x for x in reader.fieldnames if x != "t"]
        rows = []
        for row in reader:
            rows.append([float(row[label]) for label in labels])
    if not rows:
        raise ValueError(f"no control rows in {path}")
    if len(labels) != len(expected_labels):
        raise ValueError(f"{path} has {len(labels)} controls, expected {len(expected_labels)}")
    if labels != expected_labels:
        raise ValueError(f"control labels mismatch in {path}\nloaded={labels}\nexpected={expected_labels}")
    return np.asarray(rows, dtype=float), labels


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-controls", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--tag", default="polish")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--threshold", type=float, default=0.99)
    ap.add_argument("--T", type=float, default=8.0)
    ap.add_argument("--Nt", type=int, default=20)
    ap.add_argument("--amp", type=float, default=3.0)
    ap.add_argument("--stages", default="128:260:0.012,160:320:0.009")
    ap.add_argument("--accept-drop", type=float, default=0.005)
    ap.add_argument("--progress", type=int, default=20)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    psi0, target = sr.build_targets(2, 7, "ghz")
    axes, labels = sr.build_single_site_axes(2, 7, "xy")
    H0_base = sr.build_drift_base(2, 7)
    H0_dense = sr.qobj_to_dense_matrix(H0_base)
    ctrl_stack = sr.build_dense_operator_stack(axes)
    psi0_vec = sr.qobj_to_dense_vector(psi0)
    target_vec = sr.qobj_to_dense_vector(target)

    u0, loaded_labels = load_awg_controls(args.input_controls, labels)
    stages = []
    for item in args.stages.split(","):
        s_txt, it_txt, lr_txt = item.split(":")
        stages.append(sr.Stage(scale=1.0, S=int(s_txt), iters=int(it_txt), lr=float(lr_txt)))

    cfg = sr.OptConfig(
        lr=stages[0].lr,
        l2=0.0,
        amp=float(args.amp),
        clip=5.0,
        backtracks=5,
        accept_mode="soft",
        accept_drop=float(args.accept_drop),
        threshold=float(args.threshold),
        verbose=0,
        stall_enable=True,
        stall_gnorm=1e-6,
        stall_max_kicks=12,
        stall_kick_sigma=0.04,
    )

    reporter = sr.ConsoleReporter("on")
    start = time.time()
    rr, u_final = sr.run_method_stages(
        name="N7_POLISH",
        psi0=psi0,
        target=target,
        H0_base=H0_base,
        drift_strength_hw=1.0,
        ctrl_axes=axes,
        T=float(args.T),
        Nt=int(args.Nt),
        stages=stages,
        homotopy_mode="mult",
        u0=u0,
        opt_cfg_template=cfg,
        threshold=float(args.threshold),
        seed=int(args.seed),
        jitter_list=None,
        verbose=0,
        reporter=reporter,
        progress_every=int(args.progress),
        H0_base_dense=H0_dense,
        ctrl_stack_dense=ctrl_stack,
        psi0_vec=psi0_vec,
        target_vec=target_vec,
    )
    wall_s = time.time() - start

    tag = f"_{args.tag}" if args.tag else ""
    controls_path = args.outdir / f"n7_polish_controls{tag}.csv"
    summary_path = args.outdir / f"n7_polish_summary{tag}.json"
    sr.write_awg_csv(controls_path, u_final, float(args.T) / u_final.shape[0], labels)
    summary = {
        "input_controls": str(args.input_controls),
        "output_controls": str(controls_path),
        "stages": [{"S": st.S, "iters": st.iters, "lr": st.lr, "scale": st.scale} for st in stages],
        "threshold": float(args.threshold),
        "result": {
            "initF_phys": rr.initF_phys,
            "finalF_phys": rr.finalF_phys,
            "bestF_last_stage": rr.bestF_last_stage,
            "iters_to_threshold": rr.iters_to_threshold,
            "total_stage_iters": rr.total_stage_iters,
            "wall_s": rr.wall_s,
        },
        "wrapper_wall_s": wall_s,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[POLISH] finalF={rr.finalF_phys:.12f} initF={rr.initF_phys:.12f}")
    print(f"[POLISH] wrote {summary_path}")
    print(f"[POLISH] wrote {controls_path}")
    return 0 if rr.finalF_phys >= float(args.threshold) else 2


if __name__ == "__main__":
    raise SystemExit(main())
