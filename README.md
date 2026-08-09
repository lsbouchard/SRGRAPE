# SR-GRAPE

Reference implementation of sub-Riemannian gradient ascent pulse engineering
(SR-GRAPE) for quantum-control optimization.

Authors: Ryan Choi, Vwani Roychowdhury, and Louis-S. Bouchard.

## Overview

SR-GRAPE uses an extended-algebra optimization to construct a structured
initialization, projects that initialization to physically available controls,
and refines it with GRAPE. The implementation also provides direct-GRAPE
baselines for matched comparisons.

This repository contains software only. It does not contain manuscript data,
numerical result archives, or generated figures.

## Repository layout

```text
.
├── LICENSE
├── README.md
├── pyproject.toml
└── src/
    └── srgrape/
        ├── __init__.py
        ├── __main__.py
        └── srgrape.py
```

The implementation is in `src/srgrape/srgrape.py`. The package can be run
through the `srgrape` command or as `python -m srgrape`.

## Requirements

- Python 3.10 or newer
- NumPy 2.3 or newer
- SciPy 1.13 or newer
- QuTiP 5.2 or newer

## Installation

```bash
git clone https://github.com/lsbouchard/SRGRAPE.git
cd SRGRAPE
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Validation

Run the built-in numerical and implementation checks:

```bash
srgrape --self-test
```

Inspect the available presets and validate a configuration without launching
an optimization:

```bash
srgrape --list-presets
srgrape --preset smoke --dry-run
```

The same commands can be run without installing the package:

```bash
PYTHONPATH=src python -m srgrape --self-test
PYTHONPATH=src python -m srgrape --preset smoke --dry-run
```

Use `srgrape --help` for the complete command-line interface. Output files are
created only when a calculation is run and are written to the directory chosen
with `--outdir`.

## License

The software is available for non-commercial research, educational, and
nonprofit use under the terms in [LICENSE](LICENSE). It is provided as is,
without warranty or an obligation to provide support.

## Citation

If you use this software, please cite:

Ryan Choi, Vwani Roychowdhury, and Louis-S. Bouchard, “Sub-Riemannian lift
advantage in quantum control,” *Physical Review A* (2026),
<https://doi.org/10.1103/cvhq-kvhx>.

Repository citation:

Ryan Choi, Vwani Roychowdhury, and Louis-S. Bouchard, *SRGRAPE: Reference
implementation for sub-Riemannian lift quantum-control calculations*, GitHub
repository (2026), <https://github.com/lsbouchard/SRGRAPE>.
