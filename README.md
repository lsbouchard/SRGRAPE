# SR-GRAPE

Reference implementation of sub-Riemannian gradient ascent pulse engineering
(SR-GRAPE) for the quantum-control calculations in:

Ryan Choi, Vwani Roychowdhury, and Louis-S. Bouchard, “Sub-Riemannian lift
advantage in quantum control,” *Physical Review A* (2026),
<https://doi.org/10.1103/cvhq-kvhx>.

The repository contains software and reproduction helpers only. It does not
contain manuscript data, archived numerical results, saved controls, or
generated figures. Long-running commands below regenerate those files under
the ignored `reproduction/` directory.

## Layout

```text
.
├── LICENSE
├── README.md
├── pyproject.toml
├── src/
│   └── srgrape/
│       ├── __init__.py
│       ├── __main__.py
│       └── srgrape.py
└── scripts/
    ├── aggregate_state_growing.py
    ├── assemble_scaling_summary.py
    ├── make_figure1.py
    ├── make_figure2.py
    ├── print_tables.py
    ├── run_degree_cap.py
    ├── run_n5_direct_grape.py
    ├── run_n6_direct_grape.py
    ├── run_n7_direct_grape.py
    ├── run_n7_state_growing.py
    ├── polish_n7_controls.py
    └── apply_n7_polish_summary.py
```

`src/srgrape/srgrape.py` is the implementation. The installed `srgrape`
command and `python -m srgrape` both call its `main()` function. The scripts
are deliberately limited to benchmark orchestration, aggregation, and figure
generation; they do not contain hidden data.

## Installation

From the repository root on macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[reproduce]"
export MPLCONFIGDIR="$PWD/.mplconfig"
mkdir -p reproduction/data reproduction/runs reproduction/figures
```

The core package requires Python 3.10+, NumPy, SciPy, and QuTiP. The
`reproduce` extra adds Matplotlib for figure generation.

## Quick checks

```bash
srgrape --self-test
srgrape --list-presets
srgrape --preset smoke --dry-run
python -m py_compile src/srgrape/*.py scripts/*.py
```

These checks do not launch the paper-scale benchmark suite.

## Reproducibility and interpretation

The paper reports finite-sample stochastic calculations: 25 seeds for the
benchmark rows and 25 seeds for the degree-cap diagnostic. The commands below
use the reported model, grids, iteration budgets, optimizer settings, and seed
conventions. They regenerate the calculations and figures, but they are not
expected to reproduce every floating-point value bit-for-bit across Python,
QuTiP, BLAS/LAPACK, CPU, and process-scheduling environments.

The historical result CSV/JSON files and saved N=6/N=7 controls are not part
of this lean repository. Therefore, the commands below are independent
recomputations, not a claim that the original archived numbers can be recovered
without rerunning the jobs. The degree-cap command makes the diagnostic
protocol explicit; the accepted manuscript did not print every CLI parameter
used for that auxiliary analysis.

All generated artifacts remain outside version control:

```bash
rm -rf reproduction
mkdir -p reproduction/data reproduction/runs reproduction/figures
```

## Table-I scaling benchmarks

The N=2, N=3, and N=4 rows are budget-matched SR-GRAPE/direct-GRAPE
comparisons. The `--threshold` value is the optimizer stopping threshold; the
aggregation step reports the paper’s success threshold of 0.95 for these rows.

### N=2

```bash
srgrape \
  --d 2 --N 2 --Nt 10 --T 5.0 --p 4 \
  --drives xy \
  --drift-strength-ea 1.0 --drift-strength-hw 1.0 \
  --homotopy 2.0,1.5,1.0 --homotopy-mode mult \
  --ea-iters 600 \
  --S1 20 --S2 40 --iters1 300 --iters2 600 \
  --amp 2.0 --hard-amp 2.0 --l2 0.0 \
  --compiler-mode ea_target_continuation_project \
  --compiler-eps 0.02 --compiler-budget-frac 0.8 --compiler-max-terms 24 \
  --accept-mode soft --accept-drop 2e-3 --backtracks 5 --clip 5.0 \
  --baseline-mode budget --baseline-init rmsmatch --baseline-sigma 0.2 \
  --compare 1 --trials 25 --threshold 0.99 --seed 1 \
  --verbose 0 --progress off --save-traces 0 \
  --outdir reproduction/runs/n2_compare25 \
  --metadata-path reproduction/runs/n2_compare25_metadata.json \
  --results-csv reproduction/data/n2_budget_compare25.csv \
  --tag n2_compare25
```

### N=3

```bash
srgrape \
  --d 2 --N 3 --Nt 20 --T 5.0 --p 4 \
  --drives xy \
  --drift-strength-ea 1.0 --drift-strength-hw 1.0 \
  --homotopy 1.0 --homotopy-mode mult \
  --ea-iters 600 \
  --S1 24 --S2 24 --iters1 120 --iters2 0 \
  --amp 2.0 --hard-amp 2.0 --l2 0.0 \
  --compiler-mode ea_target_continuation_project \
  --compiler-eps 0.02 --compiler-budget-frac 1.0 --compiler-max-terms 24 \
  --accept-mode soft --accept-drop 2e-3 --backtracks 5 --clip 5.0 \
  --baseline-mode budget --baseline-init rmsmatch --baseline-sigma 0.2 \
  --compare 1 --trials 25 --threshold 0.999 --seed 1 \
  --verbose 0 --progress off --save-traces 0 \
  --outdir reproduction/runs/n3_compare25 \
  --metadata-path reproduction/runs/n3_compare25_metadata.json \
  --results-csv reproduction/data/n3_budget_compare25.csv \
  --tag n3_compare25
```

### N=4

```bash
srgrape \
  --d 2 --N 4 --Nt 20 --T 5.0 --p 4 \
  --drives xy \
  --drift-strength-ea 1.0 --drift-strength-hw 1.0 \
  --homotopy 1.0 --homotopy-mode mult \
  --ea-iters 600 \
  --S1 32 --S2 32 --iters1 180 --iters2 0 \
  --amp 2.0 --hard-amp 2.0 --l2 0.0 \
  --compiler-mode ea_target_continuation_project \
  --compiler-eps 0.02 --compiler-budget-frac 1.0 --compiler-max-terms 24 \
  --ea-nested-p2 1 --ea-nested-p2-iters 500 --ea-nested-p3-iters 500 --ea-nested-guard 1 \
  --accept-mode soft --accept-drop 2e-3 --backtracks 5 --clip 5.0 \
  --baseline-mode budget --baseline-init rmsmatch --baseline-sigma 0.2 \
  --compare 1 --trials 25 --threshold 0.999 --seed 1 \
  --verbose 0 --progress off --save-traces 0 \
  --outdir reproduction/runs/n4_compare25 \
  --metadata-path reproduction/runs/n4_compare25_metadata.json \
  --results-csv reproduction/data/n4_budget_compare25.csv \
  --tag n4_compare25
```

## N=5 and N=6 state-growing SR-GRAPE

Each seed is an independent state-growing calculation. The parent controls
needed for the next system size are written by the preceding stage and are
aggregated into a CSV.

### N=5

```bash
for seed in $(seq 1 25); do
  seed2=$(printf "%02d" "$seed")
  srgrape \
    --state-grow 1 \
    --d 2 --N 5 --Nt 20 --T 8.0 --p 4 --target ghz --drives xy \
    --drift-strength-ea 1.0 --drift-strength-hw 1.0 \
    --state-grow-parent-iters 700 --state-grow-parent-S 64 \
    --state-grow-newqubit-sigma 0.05 --state-grow-newqubit-seed 123 \
    --state-grow-homotopy 1.0 \
    --S1 64 --S2 96 --iters1 500 --iters2 800 \
    --lr1 0.04 --lr2 0.025 --amp 2.0 --hard-amp 2.0 --l2 0.0 \
    --compiler-mode ea_target_continuation_project \
    --target-continuation-policy manual --target-continuation-init product \
    --target-continuation-iters 80,80,40,80,120,120,120,120 \
    --target-continuation-lrs 0.05,0.04,0.05,0.05,0.04,0.03,0.025,0.02 \
    --target-continuation-weights 0,0.1,0.5,0.9,0.9,0.7,0.4,0.1 \
    --target-continuation-checkpoints final,late2,late2,all,all,late4,late2,final \
    --target-continuation-max-stages 8 \
    --accept-mode soft --accept-drop 0.02 --backtracks 5 --clip 5.0 \
    --stall-enable 1 --stall-gnorm 1e-6 --stall-max-kicks 12 --stall-kick-sigma 0.04 \
    --threshold 0.99 --seed "$seed" --verbose 0 --progress off \
    --save-traces 0 --save-compiler-diagnostics 0 --save-metadata 1 \
    --outdir "reproduction/runs/n5_state_grow/seed${seed2}" \
    --tag "n5_t8_seed${seed2}" \
    --metadata-path "reproduction/runs/n5_state_grow/seed${seed2}/metadata.json"
done

python scripts/aggregate_state_growing.py \
  --root reproduction/runs/n5_state_grow \
  --rows-out reproduction/data/n5_t8_state_grow_seed25_rows.csv \
  --summary-out reproduction/data/n5_t8_state_grow_seed25_summary.json \
  --threshold 0.99
```

### N=6

```bash
for seed in $(seq 1 25); do
  seed2=$(printf "%02d" "$seed")
  srgrape \
    --state-grow 1 \
    --d 2 --N 6 --Nt 20 --T 8.0 --p 4 --target ghz --drives xy \
    --drift-strength-ea 1.0 --drift-strength-hw 1.0 \
    --state-grow-parent-iters 700 --state-grow-parent-S 64 \
    --state-grow-newqubit-sigma 0.05 --state-grow-newqubit-seed 123 \
    --state-grow-homotopy 1.0 \
    --S1 64 --S2 96 --iters1 500 --iters2 800 \
    --lr1 0.04 --lr2 0.025 --amp 2.0 --hard-amp 2.0 --l2 0.0 \
    --compiler-mode ea_target_continuation_project \
    --target-continuation-policy manual --target-continuation-init product \
    --target-continuation-iters 80,80,40,80,120,120,120,120 \
    --target-continuation-lrs 0.05,0.04,0.05,0.05,0.04,0.03,0.025,0.02 \
    --target-continuation-weights 0,0.1,0.5,0.9,0.9,0.7,0.4,0.1 \
    --target-continuation-checkpoints final,late2,late2,all,all,late4,late2,final \
    --target-continuation-max-stages 8 \
    --accept-mode soft --accept-drop 0.02 --backtracks 5 --clip 5.0 \
    --stall-enable 1 --stall-gnorm 1e-6 --stall-max-kicks 12 --stall-kick-sigma 0.04 \
    --threshold 0.99 --seed "$seed" --verbose 0 --progress off \
    --save-traces 0 --save-compiler-diagnostics 0 --save-metadata 1 \
    --outdir "reproduction/runs/n6_state_grow/seed${seed2}" \
    --tag "n6_t8_seed${seed2}" \
    --metadata-path "reproduction/runs/n6_state_grow/seed${seed2}/metadata.json"
done

python scripts/aggregate_state_growing.py \
  --root reproduction/runs/n6_state_grow \
  --rows-out reproduction/data/n6_t8_state_grow_seed1_25_rows.csv \
  --summary-out reproduction/data/n6_t8_state_grow_seed1_25_aggregate.json \
  --threshold 0.99
```

## Direct-GRAPE baselines

```bash
python scripts/run_n5_direct_grape.py \
  --trials 25 --workers 8 --seed 1 \
  --iters1 500 --iters2 800 \
  --outdir reproduction/runs/n5_t8_direct_grape_baseline
cp reproduction/runs/n5_t8_direct_grape_baseline/n5_t8_direct_grape_baseline_rows.csv \
  reproduction/data/n5_t8_direct_grape_baseline_rows.csv
cp reproduction/runs/n5_t8_direct_grape_baseline/n5_t8_direct_grape_baseline_summary.json \
  reproduction/data/n5_t8_direct_grape_baseline_summary.json

python scripts/run_n6_direct_grape.py \
  --trials 25 --seed 1 \
  --outdir reproduction/runs/n6_direct_grape_baseline
cp reproduction/runs/n6_direct_grape_baseline/n6_direct_grape_baseline_rows.csv \
  reproduction/data/n6_direct_grape_baseline_rows.csv
cp reproduction/runs/n6_direct_grape_baseline/n6_direct_grape_baseline_summary.json \
  reproduction/data/n6_direct_grape_baseline_summary.json
```

## N=7 state-growing extension and baseline

The N=7 calculation consumes the N=6 parent controls generated above. This is
why those controls must be generated before launching the N=7 job.

```bash
python scripts/run_n7_state_growing.py \
  --srgrape src/srgrape/srgrape.py \
  --parent-rows reproduction/data/n6_t8_state_grow_seed1_25_rows.csv \
  --outdir reproduction/runs/n7_amp3_state_growing25 \
  --seeds 1-25 --max-workers 4 --skip-completed 1

cp reproduction/runs/n7_amp3_state_growing25/n7_amp3_state_growing_rows.csv \
  reproduction/data/n7_amp3_state_growing_rows.csv
cp reproduction/runs/n7_amp3_state_growing25/n7_amp3_state_growing_summary.json \
  reproduction/data/n7_amp3_state_growing_summary.json

python scripts/run_n7_direct_grape.py \
  --sr-controls-root reproduction/runs/n7_amp3_state_growing25 \
  --outdir reproduction/runs/n7_amp3_grape_baseline25 \
  --seeds 1-25 --max-workers 4 --skip-completed 1
cp reproduction/runs/n7_amp3_grape_baseline25/direct_grape_rows.csv \
  reproduction/data/n7_amp3_direct_grape_rows.csv
cp reproduction/runs/n7_amp3_grape_baseline25/direct_grape_summary.json \
  reproduction/data/n7_amp3_direct_grape_summary.json
```

If a fresh N=7 run has below-threshold seeds, they may be polished with the
same physical GRAPE helper and merged into the summary. The historical run
polished seeds 13 and 20; fresh runs should inspect their own summary first.

```bash
python scripts/polish_n7_controls.py \
  --input-controls reproduction/runs/n7_amp3_state_growing25/seed13/state_grow_final_controls_n7_amp3_seed13.csv \
  --outdir reproduction/runs/n7_amp3_state_growing25/polish_seed13 \
  --tag seed13 --seed 13 \
  --stages 128:260:0.012,160:320:0.009 \
  --accept-drop 0.005 --progress 20

python scripts/polish_n7_controls.py \
  --input-controls reproduction/runs/n7_amp3_state_growing25/seed20/state_grow_final_controls_n7_amp3_seed20.csv \
  --outdir reproduction/runs/n7_amp3_state_growing25/polish_seed20 \
  --tag seed20 --seed 20 \
  --stages 128:260:0.012,160:320:0.009 \
  --accept-drop 0.005 --progress 20

python scripts/apply_n7_polish_summary.py \
  --first-pass-rows reproduction/runs/n7_amp3_state_growing25/n7_amp3_state_growing_rows.csv \
  --polish-summary reproduction/runs/n7_amp3_state_growing25/polish_seed13/n7_polish_summary_seed13.json \
  --polish-summary reproduction/runs/n7_amp3_state_growing25/polish_seed20/n7_polish_summary_seed20.json \
  --rows-out reproduction/data/n7_amp3_state_growing_rows.csv \
  --summary-out reproduction/data/n7_amp3_state_growing_summary.json \
  --threshold 0.99
```

## Degree-cap diagnostic and Figure 2

Run the three degree caps over the same 25 seeds. This writes raw runs,
cumulative CSV/JSON, and representative p=2/p=3 traces under
`reproduction/degree_cap/`:

```bash
python scripts/run_degree_cap.py \
  --outdir reproduction/degree_cap \
  --seeds 1-25 --degrees 2,3,4

cp reproduction/degree_cap/n5_cumulative_degree_seed25_rows.csv \
  reproduction/data/n5_cumulative_degree_seed25_rows.csv
cp reproduction/degree_cap/n5_cumulative_degree_seed25_summary.json \
  reproduction/data/n5_cumulative_degree_seed25_summary.json
```

The accepted manuscript describes the cumulative statistic but does not print
the complete degree-cap CLI invocation. `run_degree_cap.py` therefore records
all defaults in its output summary and should be treated as the reproducible
source-level protocol for a fresh diagnostic rerun.

## Aggregate, print, and plot

After the benchmark jobs finish:

```bash
python scripts/assemble_scaling_summary.py \
  --data-dir reproduction/data \
  --out reproduction/data/scaling_summary_n2_n7.csv

python scripts/print_tables.py \
  --scaling reproduction/data/scaling_summary_n2_n7.csv \
  --degree-summary reproduction/data/n5_cumulative_degree_seed25_summary.json

python scripts/make_figure1.py \
  --summary reproduction/data/scaling_summary_n2_n7.csv \
  --outdir reproduction/figures

python scripts/make_figure2.py \
  --figure fig3 \
  --cumulative-summary reproduction/data/n5_cumulative_degree_seed25_summary.json \
  --cumulative-rows reproduction/data/n5_cumulative_degree_seed25_rows.csv \
  --trace-p2 reproduction/degree_cap/traces/ea_p2_nested_trace.csv \
  --trace-p3 reproduction/degree_cap/traces/ea_p3_continuation_trace.csv \
  --outdir reproduction/figures

ls -lh reproduction/figures/figure1_scaling.{pdf,png}
ls -lh reproduction/figures/figure2_cumulative_degree.{pdf,png}
```

The generated files are not committed. To preserve a particular rerun for
later review, archive `reproduction/` separately and record the Git commit,
Python/package versions, machine, and command logs.

## License

The software is available for non-commercial research, educational, and
nonprofit use under the terms in [LICENSE](LICENSE). It is provided as is,
without warranty or an obligation to provide support.

## Citation

If you use this software, cite the article and:

Ryan Choi, Vwani Roychowdhury, and Louis-S. Bouchard, *SRGRAPE: Reference
implementation for sub-Riemannian lift quantum-control calculations*, GitHub
repository (2026), <https://github.com/lsbouchard/SRGRAPE>.
