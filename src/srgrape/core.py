"""SR-GRAPE reference implementation.

Extended-algebra (EA) warm-start + GRAPE refinement with matched baseline comparisons.

This code is designed for controlled, apples-to-apples comparisons between:

  (A) METHOD: EA synthesis + compilation to physical controls + GRAPE refinement.
  (B) BASELINE: standard GRAPE from a random initialization.

Comparison rules enforced by construction:
  1) The physical model is defined by (d, N, drives, drift_strength_hw, T, Nt, S).
     Both METHOD and BASELINE are optimized on the same model and constraints.

  2) The optimizer (Adam ascent + optional backtracking/acceptance rules) is shared.

  3) The only intended difference is the initialization: compiled EA seed vs random.

Important note on drift homotopy:
  - Homotopy can be used as a continuation strategy.
  - For a conservative comparison, you can either:
      --baseline-mode matched : baseline uses the same homotopy schedule as METHOD
      --baseline-mode budget  : baseline gets the same total iteration budget but
                                runs a single GRAPE stage on the physical model.

Dependencies:
  - numpy
  - scipy
  - qutip

"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import shlex
import sys
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from itertools import combinations_with_replacement, product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import qutip as qt
from scipy.linalg import expm
from scipy.optimize import minimize, nnls


PRESETS: dict[str, dict[str, Any]] = {
    "smoke": {
        "d": 2,
        "N": 5,
        "Nt": 20,
        "T": 5.0,
        "p": 4,
        "drives": "xy",
        "drift_strength_ea": 1.0,
        "drift_strength_hw": 1.0,
        "homotopy": "2.0,1.5,1.0",
        "homotopy_mode": "mult",
        "ea_iters": 600,
        "S1": 28,
        "S2": 48,
        "iters1": 80,
        "iters2": 160,
        "amp": 2.0,
        "hard_amp": 2.0,
        "compiler_eps": 0.01,
        "compiler_budget_frac": 0.8,
        "compiler_max_terms": 24,
        "accept_mode": "soft",
        "accept_drop": 2e-3,
        "backtracks": 5,
        "clip": 5.0,
        "stall_enable": 1,
        "stall_gnorm": 1e-6,
        "stall_max_kicks": 12,
        "stall_kick_sigma": 0.05,
        "baseline_mode": "matched",
        "baseline_init": "rmsmatch",
        "baseline_sigma": 0.2,
        "compare": 1,
        "trials": 1,
        "threshold": 0.95,
    },
    "flagship": {
        "d": 2,
        "N": 5,
        "Nt": 20,
        "T": 5.0,
        "p": 4,
        "drives": "xy",
        "drift_strength_ea": 1.0,
        "drift_strength_hw": 1.0,
        "homotopy": "2.0,1.5,1.0",
        "homotopy_mode": "mult",
        "S1": 28,
        "S2": 48,
        "iters1": 800,
        "iters2": 1200,
        "amp": 2.0,
        "hard_amp": 2.0,
        "compiler_eps": 0.01,
        "compiler_budget_frac": 0.8,
        "compiler_max_terms": 24,
        "accept_mode": "soft",
        "accept_drop": 2e-3,
        "backtracks": 5,
        "clip": 5.0,
        "stall_enable": 1,
        "stall_gnorm": 1e-6,
        "stall_max_kicks": 12,
        "stall_kick_sigma": 0.05,
        "baseline_mode": "matched",
        "baseline_init": "rmsmatch",
        "baseline_sigma": 0.2,
        "compare": 1,
        "trials": 10,
        "threshold": 0.99,
    },
    "flagship_safe": {
        "d": 2,
        "N": 5,
        "Nt": 20,
        "T": 5.0,
        "p": 4,
        "drives": "xy",
        "drift_strength_ea": 1.0,
        "drift_strength_hw": 1.0,
        "homotopy": "2.0,1.5,1.0",
        "homotopy_mode": "mult",
        "S1": 28,
        "S2": 48,
        "iters1": 800,
        "iters2": 1200,
        "amp": 2.0,
        "hard_amp": 2.0,
        "compiler_eps": 0.01,
        "compiler_budget_frac": 0.8,
        "compiler_max_terms": 24,
        "accept_mode": "soft",
        "accept_drop": 2e-3,
        "backtracks": 5,
        "clip": 5.0,
        "stall_enable": 1,
        "stall_gnorm": 1e-6,
        "stall_max_kicks": 12,
        "stall_kick_sigma": 0.05,
        "baseline_mode": "matched",
        "baseline_init": "rmsmatch",
        "baseline_sigma": 0.2,
        "compare": 1,
        "trials": 5,
        "threshold": 0.95,
    },
}


DENSE_BACKEND_MAX_ENTRIES = 12_000_000
DENSE_TOL = 1e-12


def format_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:0.1f}s"
    minutes, sec = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes:d}m{sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours:d}h{minutes:02d}m{sec:02d}s"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_parent(path)
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def command_as_shell(argv: list[str]) -> str:
    return " ".join(shlex.quote(x) for x in argv)


def commutator_word_length(degree: int) -> int:
    if degree <= 0:
        return 0
    length = 1
    for _ in range(2, degree + 1):
        length = 2 + 2 * length
    return length


def compiler_feasibility_info(T: float, Nt: int, S: int, compiler_eps: float, hard_amp: float, budget_frac: float, max_degree: int) -> dict[str, Any]:
    M = Nt * S
    dt_micro = float(T) / max(1, M)
    dur = float(compiler_eps) / max(1e-30, float(hard_amp))
    steps_per_primitive = max(1, int(np.ceil(dur / dt_micro)))
    slice_budget = max(1, int(np.floor(float(budget_frac) * S)))
    degree_info = {}
    for degree in range(1, max_degree + 1):
        pulses = commutator_word_length(degree)
        word_steps = pulses * steps_per_primitive
        degree_info[degree] = {
            "word_pulses": pulses,
            "steps_per_primitive": steps_per_primitive,
            "word_steps": word_steps,
            "fits_in_budget": bool(word_steps <= slice_budget),
        }
    return {
        "slice_budget": slice_budget,
        "dt_micro": dt_micro,
        "steps_per_primitive": steps_per_primitive,
        "degree_info": degree_info,
    }


def control_label_degree(label: str) -> int:
    if "Re(" in label or "Im(" in label:
        inner = label[label.find("(") + 1 : label.rfind(")")]
        return inner.count("*") + 1 if inner else 1
    return label.count("*") + 1


def metadata_friendly(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [metadata_friendly(x) for x in value]
    if isinstance(value, dict):
        return {str(k): metadata_friendly(v) for k, v in value.items()}
    return value


class ConsoleReporter:
    def __init__(self, mode: str = "auto"):
        self.mode = mode
        self.enabled = mode != "off"
        self.interactive = self.enabled and sys.stdout.isatty()
        self._active_len = 0

    def _clear_active(self) -> None:
        if self.interactive and self._active_len > 0:
            print("\r" + (" " * self._active_len) + "\r", end="", flush=True)
            self._active_len = 0

    def info(self, message: str = "") -> None:
        self._clear_active()
        print(message, flush=True)

    def progress(self, label: str, current: int, total: int, start_time: float, extra: str = "", force: bool = False) -> None:
        if not self.enabled:
            return
        total = max(1, int(total))
        current = min(max(0, int(current)), total)
        elapsed = time.time() - start_time
        rate = current / elapsed if elapsed > 1e-12 else 0.0
        remaining = max(0, total - current)
        eta = remaining / rate if rate > 1e-12 else float("inf")
        frac = current / total
        suffix = f" {extra}" if extra else ""
        if self.interactive:
            width = 24
            filled = min(width, int(round(width * frac)))
            bar = "#" * filled + "-" * (width - filled)
            eta_text = "eta --" if not np.isfinite(eta) else f"eta {format_seconds(eta)}"
            line = f"[PROGRESS] {label} |{bar}| {current}/{total} {frac:6.1%} elapsed {format_seconds(elapsed)} {eta_text}{suffix}"
            padded = line.ljust(max(self._active_len, len(line)))
            print("\r" + padded, end="", flush=True)
            self._active_len = len(padded)
            if force or current >= total:
                print()
                self._active_len = 0
        else:
            pct = 100.0 * frac
            eta_text = "--" if not np.isfinite(eta) else format_seconds(eta)
            print(
                f"[PROGRESS] {label}: {current}/{total} ({pct:0.1f}%) "
                f"elapsed={format_seconds(elapsed)} eta={eta_text}{suffix}",
                flush=True,
            )


class CheckpointManager:
    def __init__(self, root: str, run_stub: str):
        self.enabled = bool(root)
        self.root = Path(root) if root else None
        self.run_stub = run_stub
        if self.enabled and self.root is not None:
            self.root.mkdir(parents=True, exist_ok=True)

    def path(self, label: str, suffix: str) -> Optional[Path]:
        if not self.enabled or self.root is None:
            return None
        return self.root / f"{self.run_stub}_{label}{suffix}"

    def save_npz(self, label: str, **arrays: Any) -> None:
        path = self.path(label, ".npz")
        if path is None:
            return
        ensure_parent(path)
        np.savez_compressed(path, **arrays)

    def save_json(self, label: str, payload: dict[str, Any]) -> None:
        path = self.path(label, ".json")
        if path is None:
            return
        write_json(path, payload)


def su_d_generators(d: int) -> List[qt.Qobj]:
    """Generate d²-1 traceless Hermitian generators of su(d)."""
    kets = [qt.basis(d, j) for j in range(d)]
    gens = []
    
                         
    for m in range(d):
        for n in range(m + 1, d):
            X = kets[m] * kets[n].dag() + kets[n] * kets[m].dag()
            gens.append(X)
    
                         
    for m in range(d):
        for n in range(m + 1, d):
            Y = -1j * (kets[m] * kets[n].dag() - kets[n] * kets[m].dag())
            gens.append(Y)
    
                     
    for k in range(1, d):
        diag = np.zeros(d)
        for j in range(k):
            diag[j] = 1.0
        diag[k] = -k
        Z = qt.Qobj(np.diag(diag) / np.sqrt(k * (k + 1)))
        gens.append(Z)
    
    return gens

def embed_operator(op: qt.Qobj, site: int, N: int, d: int) -> qt.Qobj:
    """Embed single-site operator into N-site system."""
    ops = [qt.qeye(d) for _ in range(N)]
    ops[site] = op
    return qt.tensor(*ops)

def qobj_sparse_nnz(op: qt.Qobj, tol: float = 0.0) -> int:
    """Cheap nonzero test for sparse QuTiP operators.

    QuTiP's ``Qobj.norm()`` computes a trace norm by default and can invoke
    sparse eigensolvers.  During Pauli-product algebra construction we only
    need to know whether the Hermitian or anti-Hermitian part vanished.
    """
    data = op.data.as_scipy()
    if tol <= 0.0:
        return int(data.nnz)
    if data.nnz == 0:
        return 0
    return int(np.count_nonzero(np.abs(data.data) > tol))

def qobj_frobenius_norm_sparse(op: qt.Qobj) -> float:
    """Frobenius norm from sparse data without densifying or eigensolving."""
    data = op.data.as_scipy()
    if data.nnz == 0:
        return 0.0
    return float(np.sqrt(np.vdot(data.data, data.data).real))

def qobj_scaled_pauli_trace_norm(op: qt.Qobj) -> float:
    """Trace norm for the scaled Pauli products generated by d=2 controls.

    A scaled Pauli product has all singular values equal to the same scale.
    Therefore ``||A||_1 = sqrt(D) ||A||_F``.  This preserves the intended
    trace-norm normalization while avoiding ARPACK calls from ``Qobj.norm()``.
    """
    frob = qobj_frobenius_norm_sparse(op)
    if frob <= 0.0:
        return 0.0
    return float(np.sqrt(op.shape[0]) * frob)

def build_control_algebra(d: int, N: int, max_p: int) -> Dict[int, List[Tuple[qt.Qobj, str]]]:
    """Build polynomial control operators organized by degree."""
    base_gens = su_d_generators(d)
    
                      
    L = []
    L_labels = []
    for site in range(N):
        for j, gen in enumerate(base_gens):
            op = embed_operator(gen, site, N, d)
            L.append(op)
            L_labels.append(f"H{j}_s{site}")
    
    controls_by_degree = {1: [(L[i], L_labels[i]) for i in range(len(L))]}
    
                               
    for p in range(2, max_p + 1):
        controls_by_degree[p] = []
        
        for indices in combinations_with_replacement(range(len(L)), p):
            op = L[indices[0]]
            label_parts = [L_labels[indices[0]]]
            
            for idx in indices[1:]:
                op = op * L[idx]
                label_parts.append(L_labels[idx])
            
                            
            op_herm = (op + op.dag()) / 2
            op_anti = 1j * (op - op.dag()) / 2
            
            label = "*".join(label_parts)
            
            if qobj_sparse_nnz(op_herm, tol=1e-12) > 0:
                controls_by_degree[p].append((op_herm, f"Re({label})"))
            if qobj_sparse_nnz(op_anti, tol=1e-12) > 0:
                controls_by_degree[p].append((op_anti, f"Im({label})"))
    
    return controls_by_degree

def select_polynomial_basis_improved(controls_by_degree: Dict[int, List[Tuple[qt.Qobj, str]]], 
                                   target_p: int, max_total: int,
                                   include_linear: bool = False,
                                   use_pauli_key_selection: bool = False) -> List[Tuple[qt.Qobj, str]]:
    """
    Improved selection strategy for polynomial operators.
    
    Key improvements:
    - Better degree distribution for p=4
    - Option to include some linear terms for stability
    - Prioritize operators that couple different sites
    """
    selected = []
    selected_vecs = []
    selected_pauli_keys: set[tuple[tuple[int, str], ...]] = set()
    
                            
    if target_p == 1:
        for op, label in controls_by_degree[1]:
            selected.append((op, label))
        return selected
    
                               
    if target_p == 2:
        if include_linear:
            degree_fractions = {1: 1.0, 2: 1.0}
        else:
            degree_fractions = {2: 1.0}
    elif target_p == 3:
        if include_linear:
            degree_fractions = {1: 0.1, 2: 0.35, 3: 0.55}
        else:
            degree_fractions = {2: 0.4, 3: 0.6}
    elif target_p == 4:
                                                             
        if include_linear:
            degree_fractions = {1: 0.05, 2: 0.35, 3: 0.40, 4: 0.20}
        else:
            degree_fractions = {2: 0.40, 3: 0.45, 4: 0.15}
    else:
                                                                   
        degree_fractions = {k: 1.0 for k in range(1, target_p + 1)}

                                              
    def prioritize_operators(operators, degree):
        """Sort operators to prioritize cross-site couplings."""
        cross_site = []
        single_site = []
        
        for op, label in operators:
                                                      
            sites_involved = set()
            for part in label.split('*'):
                if '_s' in part:
                    site_num = part[part.find('_s')+2]
                    sites_involved.add(site_num)
            
            if len(sites_involved) > 1:
                cross_site.append((op, label))
            else:
                single_site.append((op, label))
        
                                           
        return cross_site + single_site
    
                                             
    for degree in sorted(degree_fractions.keys()):
        if degree not in controls_by_degree:
            continue
            
        target_count = int(degree_fractions[degree] * max_total)
        if (target_p == 3 and degree == 2) or (target_p == 4 and degree in (2, 3)):
                                                                       
                                                                            
                                                                              
                                          
            target_count = len(controls_by_degree[degree])
        operators = controls_by_degree[degree]
        
                                                          
        if degree >= 2:
            operators = prioritize_operators(operators, degree)
        
        selected_from_degree = 0
        for op, label in operators:
            if len(selected) >= max_total:
                break

            if use_pauli_key_selection:
                parsed = parse_ea_label_to_pauli_string(label)
                if parsed is not None:
                    _, axes_by_site = parsed
                    key = canonical_pauli_key(axes_by_site)
                    if key in selected_pauli_keys:
                        continue
                    op_norm = qobj_scaled_pauli_trace_norm(op)
                    if op_norm < 1e-12:
                        continue
                    selected_pauli_keys.add(key)
                    selected.append((op / op_norm, label))
                    selected_from_degree += 1

                    if selected_from_degree >= target_count:
                        break
                    continue
                
            vec = op.full().flatten()
            vec_norm = np.linalg.norm(vec)
            if vec_norm < 1e-10:
                continue
            vec = vec / vec_norm
            
                                       
            vec_orth = vec.copy()
            for basis_vec in selected_vecs:
                vec_orth -= np.vdot(basis_vec, vec_orth) * basis_vec
            
            if np.linalg.norm(vec_orth) > 1e-6:
                vec_orth = vec_orth / np.linalg.norm(vec_orth)
                selected_vecs.append(vec_orth)
                op_norm = qobj_scaled_pauli_trace_norm(op)
                if op_norm < 1e-12:
                    continue
                selected.append((op / op_norm, label))
                selected_from_degree += 1
                
                if selected_from_degree >= target_count:
                    break
    
    return selected

class ImprovedPolynomialPontryagin:
    """Improved Pontryagin optimization with better p=4 handling."""
    
    def __init__(self, d: int, N: int, Nt: int, T: float, p: int,
                 drift_strength: float = 0.1, seed: int = -1, 
                 verbose: bool = True, warm_start_from: Optional['ImprovedPolynomialPontryagin'] = None):
        if seed >= 0:
            np.random.seed(seed)
            
        self.d = d
        self.N = N 
        self.Nt = Nt
        self.dt = T / Nt
        self.p = p
        self.verbose = verbose
        self.dim = d**N
        
        max_dim = d**(2*N) - 1
        
        if verbose:
            print(f"\nBuilding controls for p={p}")
            print(f"  System dimension: {d}^{N} = {self.dim}")
            print(f"  Maximum operators (su({self.dim})): {max_dim}")
        
                                    
        controls_by_degree = build_control_algebra(d, N, p)
        
        if verbose:
            for deg in sorted(controls_by_degree.keys()):
                print(f"  Degree {deg}: {len(controls_by_degree[deg])} operators generated")
        
                                                        
                                                                            
                                                                            
                                                                           
        include_linear = (p >= 2)
        controls_with_labels = select_polynomial_basis_improved(
            controls_by_degree,
            p,
            max_dim,
            include_linear=include_linear,
            use_pauli_key_selection=(d == 2),
        )
        
        self.controls = [op for op, _ in controls_with_labels]
        self.control_labels = [label for _, label in controls_with_labels]
        self.control_degrees = [control_label_degree(label) for label in self.control_labels]
        
                         
        degree_counts = {}
        for label in self.control_labels:
            if "Re(" in label or "Im(" in label:
                inner = label[label.find("(")+1:label.find(")")]
                deg = inner.count("*") + 1
            else:
                deg = label.count("*") + 1
            degree_counts[deg] = degree_counts.get(deg, 0) + 1
        
        if verbose:
            print(f"\nSelected operators by degree:")
            for deg in sorted(degree_counts.keys()):
                print(f"  Degree {deg}: {degree_counts[deg]} operators")
            print(f"  Total: {len(self.controls)} operators")
        
                     
        self.H_drift = self._build_drift(drift_strength)
        self._prepare_dense_backend()
        
                               
        if warm_start_from is not None and p == 4:
                                          
            self.u = self._warm_start_initialization(warm_start_from)
            if verbose:
                print("  Using warm start from p=3 solution")
        else:
            scale = 0.1
            self.u = scale * np.random.randn(Nt, len(self.controls))

    def _prepare_dense_backend(self) -> None:
        self.use_dense_backend = False
        self.H_drift_dense: Optional[np.ndarray] = None
        self.control_stack_dense: Optional[np.ndarray] = None
        if not should_use_dense_backend(len(self.controls), self.dim):
            return
        control_stack = build_dense_operator_stack(self.controls)
        if control_stack is None:
            return
        self.H_drift_dense = qobj_to_dense_matrix(self.H_drift)
        self.control_stack_dense = control_stack
        self.use_dense_backend = True
            
    def _warm_start_initialization(self, prev_optimizer):
        """Initialize from previous optimization result."""
                                        
        u_new = 0.001 * np.random.randn(self.Nt, len(self.controls))
        
                                                  
        for i, (ctrl, label) in enumerate(zip(self.controls, self.control_labels)):
            for j, (prev_ctrl, prev_label) in enumerate(zip(prev_optimizer.controls, prev_optimizer.control_labels)):
                                                              
                if self._operators_similar(ctrl, prev_ctrl):
                                                                 
                    u_new[:, i] = 0.8 * prev_optimizer.u[:, j]
                    break
        
        return u_new
    
    def _operators_similar(self, op1, op2, tol=0.95):
        """Check if two operators are similar."""
        overlap = abs((op1.dag() * op2).tr()) / (op1.norm() * op2.norm())
        return overlap > tol
    
    def _build_drift(self, strength: float) -> qt.Qobj:
        """Build drift Hamiltonian."""
        dims = [[self.d] * self.N, [self.d] * self.N]
        H0 = qt.Qobj(np.zeros((self.dim, self.dim)), dims=dims)
        
        if self.N > 1:
            z_vals = np.arange(self.d) - (self.d - 1) / 2
            Z = qt.Qobj(np.diag(z_vals))
            embedded_z = [embed_operator(Z, i, self.N, self.d) for i in range(self.N)]
            for i in range(self.N):
                for j in range(i + 1, self.N):
                    H0 += strength * embedded_z[i] * embedded_z[j]
        
        return H0
    
    def H(self, k: int) -> qt.Qobj:
        """Total Hamiltonian at slice k."""
        H_ctrl = sum(self.u[k, j] * self.controls[j] for j in range(len(self.controls)))
        return self.H_drift + H_ctrl
    
    def compute_fidelity_and_gradient(self, psi0: qt.Qobj, target: qt.Qobj, 
                                    regularization: float = 0.0):
        """Forward-backward sweep with optional regularization."""
        if self.use_dense_backend and self.H_drift_dense is not None and self.control_stack_dense is not None:
            psi0_vec = qobj_to_dense_vector(psi0)
            target_vec = qobj_to_dense_vector(target)
            states, unitaries = propagate_dense(
                self.H_drift_dense,
                self.control_stack_dense,
                self.u,
                self.dt,
                psi0_vec,
                store_steps=True,
            )
            alpha = complex(np.vdot(target_vec, states[-1]))
            fidelity = float(abs(alpha) ** 2)

            costates = np.empty_like(states)
            costates[self.Nt] = target_vec
            for k in range(self.Nt - 1, -1, -1):
                if unitaries is None:
                    raise ValueError("Dense backend requires stored unitaries")
                costates[k] = unitaries[k].conj().T @ costates[k + 1]

            grad = np.zeros_like(self.u)
            for k in range(self.Nt):
                mat_els = np.einsum(
                    "i,aij,j->a",
                    np.conj(costates[k + 1]),
                    self.control_stack_dense,
                    states[k + 1],
                    optimize=True,
                )
                grad[k] = 2.0 * np.real(np.conj(alpha) * ((-1j * self.dt) * mat_els))

            if regularization > 0:
                grad -= regularization * self.u
            return fidelity, grad

                                                 
        states = [psi0]
        unitaries = []

        for k in range(self.Nt):
            U = (-1j * self.H(k) * self.dt).expm()
            states.append(U * states[-1])
            unitaries.append(U)

        alpha = overlap(target, states[-1])
        fidelity = float(abs(alpha) ** 2)

        costates = [target]
        for U in reversed(unitaries):
            costates.append(U.dag() * costates[-1])
        costates = costates[::-1]

        grad = np.zeros_like(self.u)
        for k in range(self.Nt):
            for j in range(len(self.controls)):
                mat_el = _scalar_from(costates[k + 1].dag() * self.controls[j] * states[k + 1])
                grad[k, j] = 2.0 * np.real(np.conj(alpha) * ((-1j * self.dt) * mat_el))

        if regularization > 0:
            grad -= regularization * self.u

        return fidelity, grad
    
    def optimize_improved(
        self,
        psi0: qt.Qobj,
        target: qt.Qobj,
        max_iter: int = 500,
        tol: float = 1e-8,
        reporter: Optional[ConsoleReporter] = None,
        progress_every: int = 20,
        trace_rows: Optional[list[dict[str, Any]]] = None,
        progress_label: str = "EA",
    ) -> float:
        """Improved optimization with adaptive strategies for p>=4."""
        if self.verbose:
            if reporter is None:
                print(f"\nOptimizing...")
        
                                        
        if self.p == 1:
            step = 0.1
            momentum = 0.8
            regularization = 0.0
        elif self.p == 2:
            step = 0.05
            momentum = 0.9
            regularization = 0.0
        elif self.p == 3:
            step = 0.02
            momentum = 0.95
            regularization = 0.0
        else:          
                                       
            step = 0.005                        
            momentum = 0.98                                 
            regularization = 1e-6                           
        initial_step = float(step)
        
        velocity = np.zeros_like(self.u)
        best_fidelity = 0
        best_u = self.u.copy()
        patience = 50                      
        no_improvement_count = 0
        stop_reason = "max_iter"
        trace = trace_rows if trace_rows is not None else []
        t_start = time.time()
        prev_best_eligible = bool(getattr(self, "projectability_best_eligible", True))
        
        if self.verbose:
            if reporter is None:
                print(f"{'Iter':>5} {'Fidelity':>12} {'||grad||':>12} {'Step':>12}")
                print("-" * 50)
            else:
                reporter.info(f"{'Iter':>5} {'Fidelity':>12} {'||grad||':>12} {'Step':>12}")
                reporter.info("-" * 50)
        
        for it in range(max_iter):
                                                  
            fidelity, grad = self.compute_fidelity_and_gradient(psi0, target, regularization)
            control_l2 = float(getattr(self, "ea_control_l2", 0.0))
            smooth_l2 = float(getattr(self, "ea_smooth_l2", 0.0))
            control_penalty = 0.0
            smooth_penalty = 0.0
            if control_l2 > 0:
                control_penalty = control_l2 * float(np.sum(self.u * self.u))
                grad -= 2.0 * control_l2 * self.u
            if smooth_l2 > 0 and self.u.shape[0] > 1:
                du = np.diff(self.u, axis=0)
                smooth_penalty = smooth_l2 * float(np.sum(du * du))
                smooth_grad = np.zeros_like(self.u)
                smooth_grad[0] = 2.0 * (self.u[0] - self.u[1])
                smooth_grad[-1] = 2.0 * (self.u[-1] - self.u[-2])
                if self.u.shape[0] > 2:
                    smooth_grad[1:-1] = 2.0 * (2.0 * self.u[1:-1] - self.u[:-2] - self.u[2:])
                grad -= smooth_l2 * smooth_grad
            grad_norm = np.linalg.norm(grad)
            
                                                                        
                                                                         
                                                                      
                                               
            best_eligible = bool(getattr(self, "projectability_best_eligible", True))
            if best_eligible and fidelity > best_fidelity:
                best_fidelity = fidelity
                best_u = self.u.copy()
                no_improvement_count = 0
            elif best_eligible:
                no_improvement_count += 1
            
                            
            if trace_rows is not None:
                trace.append(
                    {
                        "iter": int(it),
                        "fidelity": float(fidelity),
                        "best_fidelity": float(best_fidelity),
                        "grad_norm": float(grad_norm),
                        "step": float(step),
                        "ea_control_l2": float(control_l2),
                        "ea_smooth_l2": float(smooth_l2),
                        "ea_control_penalty": float(control_penalty),
                        "ea_smooth_penalty": float(smooth_penalty),
                        "no_improvement_count": int(no_improvement_count),
                        "projectability_best_eligible": bool(best_eligible),
                        "elapsed_s": float(time.time() - t_start),
                    }
                )

            emit_progress = (
                it == 0
                or fidelity > 0.999
                or grad_norm < tol
                or it == max_iter - 1
                or (it % max(1, progress_every) == 0)
            )

            if self.verbose and emit_progress and reporter is None:
                print(f"{it:5d} {fidelity:12.8f} {grad_norm:12.4e} {step:12.4e}")
            elif reporter is not None and emit_progress:
                reporter.progress(
                    progress_label,
                    it + 1,
                    max_iter,
                    t_start,
                    extra=f"best={best_fidelity:.6f} F={fidelity:.6f} |g|={grad_norm:.3e} step={step:.3e}",
                    force=(fidelity > 0.9995 or grad_norm < tol or it == max_iter - 1),
                )
            
                               
            if best_eligible and (fidelity > 0.9995 or grad_norm < tol):
                stop_reason = "target" if fidelity > 0.9995 else "grad_tol"
                break
            
                                     
            if self.p >= 4 and no_improvement_count > patience and it > 200:
                if self.verbose:
                    if reporter is None:
                        print("Early stopping due to no improvement")
                    else:
                        reporter.info("Early stopping due to no improvement")
                stop_reason = "no_improvement"
                break
            
                                  
            velocity = momentum * velocity + step * grad
            
                                        
            if self.p >= 4:
                max_grad_norm = 10.0
                if np.linalg.norm(velocity) > max_grad_norm:
                    velocity = velocity * max_grad_norm / np.linalg.norm(velocity)
            
            self.u += velocity

            post_update_projector = getattr(self, "post_update_projector", None)
            if post_update_projector is not None:
                if getattr(post_update_projector, "accepts_gradient", False):
                    post_update_projector(self, grad)
                else:
                    post_update_projector(self)
            
                                                               
            if it > 0:
                if best_eligible and not prev_best_eligible:
                                                                              
                                                                            
                                                                               
                                                 
                    velocity *= 0.0
                    step = max(step, initial_step)
                elif (not best_eligible) and prev_best_eligible:
                    pass
                                                                                
                elif fidelity < prev_fidelity and it > 200:
                    step *= 0.8
                    if self.p >= 3:
                        velocity *= 0.5                                
                else:
                    if self.p >= 4:
                        step *= 1.01                                 
                    else:
                        step *= 1.02
                        
                                                                           
                                                                          
            if self.p >= 3:
                step = min(step, 0.1)               
                step = max(step, 1e-8)               
                
            prev_fidelity = fidelity
            prev_best_eligible = bool(best_eligible)
        
        self.u = best_u
        self.last_trace = trace
        self.last_stop_reason = stop_reason
        if self.verbose:
            if reporter is None:
                print(f"\nFinal fidelity: {best_fidelity:.8f}")
            else:
                reporter.info(f"\nFinal fidelity: {best_fidelity:.8f}")
        return best_fidelity


                        


def _scalar_from(x) -> complex:
    if isinstance(x, qt.Qobj):
        arr = x.full()
        flat = np.asarray(arr).reshape(-1)
        return complex(flat[0])
    if isinstance(x, np.ndarray):
        flat = np.asarray(x).reshape(-1)
        return complex(flat[0])
    return complex(x)


def qobj_to_dense_matrix(op: qt.Qobj) -> np.ndarray:
    return np.asarray(op.full(), dtype=np.complex128)


def qobj_to_dense_vector(psi: qt.Qobj) -> np.ndarray:
    return np.asarray(psi.full(), dtype=np.complex128).reshape(-1)


def should_use_dense_backend(n_ops: int, dim: int, max_entries: int = DENSE_BACKEND_MAX_ENTRIES) -> bool:
    return bool(dim > 0 and n_ops >= 0 and (dim * dim * max(1, n_ops)) <= max_entries)


def build_dense_operator_stack(ops: Sequence[qt.Qobj], max_entries: int = DENSE_BACKEND_MAX_ENTRIES) -> Optional[np.ndarray]:
    if len(ops) == 0:
        return np.zeros((0, 0, 0), dtype=np.complex128)
    dim = int(ops[0].shape[0])
    if not should_use_dense_backend(len(ops), dim, max_entries=max_entries):
        return None
    return np.stack([qobj_to_dense_matrix(op) for op in ops], axis=0)


def assemble_dense_hamiltonian(H0_dense: np.ndarray, control_stack: np.ndarray, u_row: np.ndarray) -> np.ndarray:
    H = np.array(H0_dense, copy=True)
    if control_stack.size == 0:
        return H
    active = np.flatnonzero(np.abs(u_row) > DENSE_TOL)
    if active.size == 0:
        return H
    if active.size == u_row.size:
        H += np.tensordot(u_row, control_stack, axes=(0, 0))
    else:
        H += np.tensordot(u_row[active], control_stack[active], axes=(0, 0))
    return H


def expm_dense_hamiltonian(H_dense: np.ndarray, dt: float) -> np.ndarray:
    return expm((-1j * dt) * H_dense)


def propagate_dense(
    H0_dense: np.ndarray,
    control_stack: np.ndarray,
    u: np.ndarray,
    dt_eff: float,
    psi0_vec: np.ndarray,
    store_steps: bool,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    M = int(u.shape[0])
    dim = int(psi0_vec.size)
    states = np.empty((M + 1, dim), dtype=np.complex128)
    states[0] = psi0_vec
    unitaries = np.empty((M, dim, dim), dtype=np.complex128) if store_steps else None

    for k in range(M):
        H_dense = assemble_dense_hamiltonian(H0_dense, control_stack, u[k])
        Uk = expm_dense_hamiltonian(H_dense, dt_eff)
        if unitaries is not None:
            unitaries[k] = Uk
        states[k + 1] = Uk @ states[k]

    return states, unitaries


def forward_bundle_dense(
    H0_dense: np.ndarray,
    control_stack: np.ndarray,
    u: np.ndarray,
    dt: float,
    psi0_vec: np.ndarray,
    target_vec: np.ndarray,
    jitters: Sequence[float],
    store_steps: bool,
) -> tuple[float, list[complex], list[np.ndarray], list[Optional[np.ndarray]]]:
    fidelities: list[float] = []
    alphas: list[complex] = []
    states_list: list[np.ndarray] = []
    unitaries_list: list[Optional[np.ndarray]] = []

    for eps in jitters:
        dt_eff = dt * (1.0 + float(eps))
        states, unitaries = propagate_dense(H0_dense, control_stack, u, dt_eff, psi0_vec, store_steps=store_steps)
        alpha = complex(np.vdot(target_vec, states[-1]))
        fidelities.append(float(abs(alpha) ** 2))
        alphas.append(alpha)
        states_list.append(states)
        unitaries_list.append(unitaries)

    return float(np.mean(fidelities)), alphas, states_list, unitaries_list


def backward_gradient_dense(
    alphas: Sequence[complex],
    states_list: Sequence[np.ndarray],
    unitaries_list: Sequence[Optional[np.ndarray]],
    control_stack: np.ndarray,
    target_vec: np.ndarray,
    dt: float,
    u_curr: np.ndarray,
    mask: np.ndarray,
    l2_coeff: float,
    clip: float,
) -> np.ndarray:
    grad = np.zeros_like(u_curr, dtype=float)
    M, m = u_curr.shape
    active_mask = ~mask
    active_idx = np.flatnonzero(active_mask)
    active_stack = control_stack[active_idx] if active_idx.size != m else control_stack

    for alpha, states, unitaries in zip(alphas, states_list, unitaries_list):
        if unitaries is None:
            raise ValueError("unitaries are required for backward propagation")
        lam = np.empty_like(states)
        lam[M] = target_vec
        for k in range(M - 1, -1, -1):
            lam[k] = unitaries[k].conj().T @ lam[k + 1]

        alpha_conj = np.conj(alpha)
        for k in range(M):
            if active_idx.size == m:
                mat_els = np.einsum("i,aij,j->a", np.conj(lam[k + 1]), control_stack, states[k + 1], optimize=True)
                grad[k] += 2.0 * np.real(alpha_conj * ((-1j * dt) * mat_els))
            else:
                mat_els = np.einsum("i,aij,j->a", np.conj(lam[k + 1]), active_stack, states[k + 1], optimize=True)
                grad[k, active_idx] += 2.0 * np.real(alpha_conj * ((-1j * dt) * mat_els))

    grad /= max(1, len(alphas))

    if l2_coeff > 0:
        grad -= 2.0 * l2_coeff * u_curr

    if active_idx.size != m:
        grad[:, mask] = 0.0

    if clip > 0:
        gnorm = float(np.linalg.norm(grad))
        if gnorm > clip:
            grad *= (clip / max(1e-30, gnorm))

    return grad


def overlap(phi: qt.Qobj, psi: qt.Qobj) -> complex:
    return _scalar_from(phi.dag() * psi)


def expmH(H: qt.Qobj, t: float) -> qt.Qobj:
    return (-1j * H * t).expm()


def hs_inner(A: qt.Qobj, B: qt.Qobj) -> float:
    return float(np.real((A.dag() * B).tr()))


def hs_norm(A: qt.Qobj) -> float:
    return float(np.sqrt(max(1e-30, hs_inner(A, A))))


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, dtype=float) ** 2)))


def apply_seed_gain_dither_clip(
    u_seed: np.ndarray,
    *,
    seed_gain: float,
    seed_dither: float,
    amp: float,
    seed: int,
) -> np.ndarray:
    """Apply deterministic seed post-processing used before physical GRAPE."""
    out = np.asarray(u_seed, dtype=float).copy()
    if float(seed_gain) != 1.0:
        out = float(seed_gain) * out
    if float(seed_dither) > 0:
        rng0 = np.random.RandomState(int(seed))
        out = out + float(seed_dither) * rng0.randn(*out.shape)
    if float(amp) > 0:
        np.clip(out, -float(amp), float(amp), out=out)
    return out



         


def create_ghz(d: int, N: int) -> qt.Qobj:
    st = sum(qt.tensor(*[qt.basis(d, j) for _ in range(N)]) for j in range(d))
    return st.unit()


def create_w(d: int, N: int) -> qt.Qobj:
                                                                                      
                                                                            
    if d < 2:
        raise ValueError("W state requires d >= 2")
    st = sum(
        qt.tensor(*[qt.basis(d, 1) if k == j else qt.basis(d, 0) for k in range(N)])
        for j in range(N)
    )
    return st.unit()



                    


def build_drift_base(d: int, N: int) -> qt.Qobj:
    """Build a drift Hamiltonian with *unit* coupling strength.

    Returned operator is the sum of pairwise ZZ couplings:
      H0_base = sum_{i<j} Z_i Z_j

    The physical drift is then H0 = drift_strength * H0_base.
    """
    z_diag = np.arange(d) - (d - 1) / 2.0
    Z = qt.Qobj(np.diag(z_diag))

    dims = [[d] * N, [d] * N]
    H0 = qt.Qobj(np.zeros((d**N, d**N), dtype=complex), dims=dims)

    if N > 1:
        embedded_z = [embed_operator(Z, i, N, d) for i in range(N)]
        for i in range(N):
            for j in range(i + 1, N):
                H0 += embedded_z[i] * embedded_z[j]

    return H0


def build_single_site_axes(
    d: int,
    N: int,
    mode: str,
    normalize: bool = True,
) -> tuple[list[qt.Qobj], list[str]]:
    """Single-site su(d) controls embedded in an N-qudit register.

    mode:
      - 'su': all su(d) generators
      - 'xy': only off-diagonal generators (a qubit-like XY subset)

    Normalization:
      - Each axis is normalized to unit Hilbert-Schmidt norm.
    """
    gens = su_d_generators(d)
    axes: list[qt.Qobj] = []
    labels: list[str] = []

    def is_offdiag(M: np.ndarray, tol: float = 1e-12) -> bool:
        return np.linalg.norm(M - np.diag(np.diag(M))) > tol

    offdiag_flags = [is_offdiag(g.full()) for g in gens]
    mode_l = mode.lower()
    for s in range(N):
        for idx, (g, is_xy_axis) in enumerate(zip(gens, offdiag_flags)):
            if mode_l == "xy" and not is_xy_axis:
                continue
            op = embed_operator(g, s, N, d)
            if normalize:
                op = op / max(1e-12, hs_norm(op))
            tag = "XY" if mode_l == "xy" else "SU"
            axes.append(op)
            labels.append(f"{tag}{idx}_s{s}")

    return axes, labels


def pauli_product_qobj(d: int, N: int, axes_by_site: dict[int, str], scale_by_dim: bool = True) -> qt.Qobj:
    """Build a qubit Pauli-product operator, optionally scaled as P/D.

    This is used only for diagnostic effective-product controls.  It is not a
    hardware control alphabet unless the hardware can directly drive those
    many-body products.
    """
    if d != 2:
        raise ValueError("effective Pauli-product controls are implemented for qubits only")
    mats = {
        "I": qt.qeye(2),
        "X": qt.sigmax(),
        "Y": qt.sigmay(),
        "Z": qt.sigmaz(),
    }
    op = qt.tensor(*[mats[axes_by_site.get(q, "I")] for q in range(N)])
    if scale_by_dim:
        op = op / float(d**N)
    return op


def build_effective_d2_axes_from_opt(
    opt: "ImprovedPolynomialPontryagin",
    physical_axes: list[qt.Qobj],
    physical_labels: list[str],
) -> tuple[list[qt.Qobj], list[str], dict[str, int]]:
    """Append unique degree-2 Pauli-product controls present in the EA basis."""
    axes = list(physical_axes)
    labels = list(physical_labels)
    key_to_index: dict[str, int] = {}
    for label in opt.control_labels:
        parsed = parse_ea_label_to_pauli_string(label)
        if parsed is None:
            continue
        _phase, axes_by_site = parsed
        if len(axes_by_site) != 2:
            continue
        key = pauli_axes_to_string(axes_by_site)
        if key in key_to_index:
            continue
        key_to_index[key] = len(axes)
        axes.append(pauli_product_qobj(opt.d, opt.N, axes_by_site, scale_by_dim=True))
        labels.append(f"EFF2:{key}")
    return axes, labels, key_to_index



                                               


def build_ctrl_label_lookup(ctrl_labels: Sequence[str]) -> dict[str, int]:
    lookup: dict[str, int] = {}
    for i, phys_label in enumerate(ctrl_labels):
        std = phys_label.replace("H", "SU").replace("XY", "SU")
        lookup[phys_label] = i
        lookup[std] = i
    return lookup


def parse_label_to_indices(
    label: str,
    ctrl_labels: Sequence[str],
    ctrl_label_lookup: Optional[dict[str, int]] = None,
) -> list[int] | None:
    """Map an EA control label onto a list of physical control indices.

    EA labels may look like: 'SU3_s0' or products like 'SU3_s0*SU4_s1'.

    Returns None if any component is not found among ctrl_labels.
    """
    clean = label.replace("Re(", "").replace("Im(", "").replace(")", "")
    parts = clean.split("*")
    indices: list[int] = []
    lookup = ctrl_label_lookup if ctrl_label_lookup is not None else build_ctrl_label_lookup(ctrl_labels)

    for part in parts:
        part_std = part.replace("H", "SU").replace("XY", "SU")
        idx = lookup.get(part_std)
        if idx is None:
            return None
        indices.append(int(idx))

    return indices


PAULI_AXIS_BY_GEN = {0: "X", 1: "Y", 2: "Z"}
PAULI_GEN_BY_AXIS = {"X": 0, "Y": 1, "Z": 2}
PAULI_MUL = {
    ("I", "I"): (1, "I"),
    ("I", "X"): (1, "X"),
    ("I", "Y"): (1, "Y"),
    ("I", "Z"): (1, "Z"),
    ("X", "I"): (1, "X"),
    ("Y", "I"): (1, "Y"),
    ("Z", "I"): (1, "Z"),
    ("X", "X"): (1, "I"),
    ("Y", "Y"): (1, "I"),
    ("Z", "Z"): (1, "I"),
    ("X", "Y"): (1j, "Z"),
    ("Y", "Z"): (1j, "X"),
    ("Z", "X"): (1j, "Y"),
    ("Y", "X"): (-1j, "Z"),
    ("Z", "Y"): (-1j, "X"),
    ("X", "Z"): (-1j, "Y"),
}


def parse_ea_label_to_pauli_string(label: str) -> tuple[complex, dict[int, str]] | None:
    """Canonicalize a qubit EA product label into phase * Pauli string.

    The EA basis labels are products of local generators, e.g.
    ``Re(H0_s0*H1_s0*H2_s3)``.  For qubits, H0/H1/H2 correspond to X/Y/Z.
    The Re/Im wrapper tells us which Hermitian part was selected.
    """
    is_re = label.startswith("Re(")
    is_im = label.startswith("Im(")
    clean = label.replace("Re(", "").replace("Im(", "").replace(")", "")
    if clean == label and (label.startswith("H") or label.startswith("SU") or label.startswith("XY")):
        is_re = True

    phase: complex = 1.0 + 0.0j
    axes_by_site: dict[int, str] = {}
    for part in clean.split("*"):
        if "_s" not in part:
            return None
        try:
            gen_txt, site_txt = part.split("_s", 1)
            site = int(site_txt)
        except ValueError:
            return None
        gen_digits = "".join(ch for ch in gen_txt if ch.isdigit())
        if gen_digits == "":
            return None
        axis = PAULI_AXIS_BY_GEN.get(int(gen_digits))
        if axis is None:
            return None
        old_axis = axes_by_site.get(site, "I")
        ph, new_axis = PAULI_MUL[(old_axis, axis)]
        phase *= ph
        if new_axis == "I":
            axes_by_site.pop(site, None)
        else:
            axes_by_site[site] = new_axis

                                                                
                                                                     
    if is_im:
        herm_coeff = -float(np.imag(phase))
    else:
        herm_coeff = float(np.real(phase))
    if abs(herm_coeff) < 1e-12:
        return None
    return complex(herm_coeff), axes_by_site


def physical_axis_index(ctrl_labels: Sequence[str], site: int, axis: str) -> int | None:
    gen = PAULI_GEN_BY_AXIS[axis]
    candidates = [f"XY{gen}_s{site}", f"SU{gen}_s{site}", f"H{gen}_s{site}"]
    lookup = build_ctrl_label_lookup(ctrl_labels)
    for cand in candidates:
        if cand in lookup:
            return int(lookup[cand])
    return None


def emit_control_area(
    u_slice: np.ndarray,
    start: int,
    ctrl_idx: int,
    area: float,
    dt_micro: float,
    amp_bound: float,
) -> int:
    """Emit a bounded local-control pulse with a requested normalized area."""
    if ctrl_idx is None or abs(area) < 1e-14 or dt_micro <= 0:
        return 0
    max_amp = float(amp_bound)
    if max_amp <= 0:
        return 0
    steps = max(1, int(np.ceil(abs(area) / (max_amp * dt_micro))))
    if start + steps > u_slice.shape[0]:
        return 0
    amp = float(area) / (steps * dt_micro)
    if abs(amp) > max_amp:
        amp = np.sign(amp) * max_amp
    u_slice[start:start + steps, ctrl_idx] += amp
    return steps


def estimate_control_area_steps(area: float, dt_micro: float, amp_bound: float) -> int:
    """Estimate microsteps needed for a bounded pulse area."""
    if abs(area) < 1e-14 or dt_micro <= 0:
        return 0
    max_amp = float(amp_bound)
    if max_amp <= 0:
        return 10**12
    return max(1, int(np.ceil(abs(area) / (max_amp * dt_micro))))


def emit_local_z_bch_block(
    u_slice: np.ndarray,
    start: int,
    ctrl_labels: Sequence[str],
    site: int,
    theta: float,
    dt_micro: float,
    amp_bound: float,
) -> tuple[int, str]:
    """Emit a small local Z rotation from XY controls using a BCH loop.

    With normalized physical controls X/sqrt(D), Y/sqrt(D),

        exp(-i a Xn) exp(-i b Yn) exp(+i a Xn) exp(+i b Yn)

    has leading logarithm -i 2ab Z/D.  Therefore ab=theta/2 targets
    exp(-i theta Z/D).  This is much cheaper than fixed pi/4 conjugation for
    small local-Z coefficients, and constructive audits decide whether the
    approximation is good enough.
    """
    if abs(theta) < 1e-14:
        return 0, "zero_theta"
    ctrl_x = physical_axis_index(ctrl_labels, site, "X")
    ctrl_y = physical_axis_index(ctrl_labels, site, "Y")
    if ctrl_x is None or ctrl_y is None:
        return 0, "missing_z_synthesis"
    mag = float(np.sqrt(abs(theta) / 2.0))
    beta = float(np.sign(theta) * mag)
    if mag <= 0.0 or beta == 0.0:
        return 0, "zero_theta"
    original = u_slice.copy()
    pos = int(start)
    for ctrl, area in [
        (ctrl_x, +mag),
        (ctrl_y, +beta),
        (ctrl_x, -mag),
        (ctrl_y, -beta),
    ]:
        used = emit_control_area(u_slice, pos, int(ctrl), area, dt_micro, amp_bound)
        if used <= 0:
            u_slice[:, :] = original
            return 0, "insufficient_budget_local_z_bch"
        pos += used
    return pos - start, "bch_local_z"


def estimate_local_z_bch_steps(theta: float, dt_micro: float, amp_bound: float) -> int:
    if abs(theta) < 1e-14:
        return 0
    mag = float(np.sqrt(abs(theta) / 2.0))
    return 4 * estimate_control_area_steps(mag, dt_micro, amp_bound)


def emit_one_body_product_block(
    u_slice: np.ndarray,
    start: int,
    ctrl_labels: Sequence[str],
    site: int,
    axis: str,
    theta: float,
    dt_micro: float,
    amp_bound: float,
    dim: int,
) -> tuple[int, str]:
    """Emit exp(-i theta sigma_axis(site) / D) on normalized physical controls."""
    if abs(theta) < 1e-14:
        return 0, "zero_theta"
    sqrtD = float(np.sqrt(dim))
    ctrl = physical_axis_index(ctrl_labels, site, axis)
    if ctrl is not None:
        used = emit_control_area(u_slice, start, ctrl, theta / sqrtD, dt_micro, amp_bound)
        return (used, "ok") if used > 0 else (0, "insufficient_budget_linear")

    if axis != "Z":
        return 0, "missing_linear_axis"

                                                                              
                                                                               
                                                                  
    ctrl_x = physical_axis_index(ctrl_labels, site, "X")
    ctrl_y = physical_axis_index(ctrl_labels, site, "Y")
    if ctrl_x is None or ctrl_y is None:
        return 0, "missing_z_synthesis"
    exact_steps = (
        2 * estimate_control_area_steps(np.pi / 4.0 * sqrtD, dt_micro, amp_bound)
        + estimate_control_area_steps(theta / sqrtD, dt_micro, amp_bound)
    )
    bch_steps = estimate_local_z_bch_steps(theta, dt_micro, amp_bound)
    if bch_steps > 0 and bch_steps <= exact_steps and abs(theta) <= 0.5:
        used_bch, status_bch = emit_local_z_bch_block(
            u_slice,
            start,
            ctrl_labels,
            site,
            theta,
            dt_micro,
            amp_bound,
        )
        if used_bch > 0:
            return used_bch, status_bch
    original = u_slice.copy()
    pos = int(start)
    for c, area in [
        (ctrl_x, +np.pi / 4.0 * sqrtD),
        (ctrl_y, theta / sqrtD),
        (ctrl_x, -np.pi / 4.0 * sqrtD),
    ]:
        used = emit_control_area(u_slice, pos, c, area, dt_micro, amp_bound)
        if used <= 0:
            u_slice[:, :] = original
            return 0, "insufficient_budget_linear_z"
        pos += used
    return pos - start, "ok"


def emit_basis_rotation_to_z(
    u_slice: np.ndarray,
    start: int,
    ctrl_labels: Sequence[str],
    site: int,
    axis: str,
    inverse: bool,
    dt_micro: float,
    amp_bound: float,
    dim: int,
) -> int:
    """Emit local rotations that map a requested Pauli axis to Z during drift.

    Uses normalized physical axes, so an unnormalized Pauli rotation angle phi
    requires normalized pulse area phi*sqrt(D).
    """
    if axis == "Z":
        return 0
    sqrtD = float(np.sqrt(dim))
    if axis == "X":
                                                                         
        ctrl = physical_axis_index(ctrl_labels, site, "Y")
        phi = (-np.pi / 4.0 if not inverse else +np.pi / 4.0) * sqrtD
    elif axis == "Y":
                                      
        ctrl = physical_axis_index(ctrl_labels, site, "X")
        phi = (+np.pi / 4.0 if not inverse else -np.pi / 4.0) * sqrtD
    else:
        return 0
    if ctrl is None:
        return 0
    return emit_control_area(u_slice, start, ctrl, phi, dt_micro, amp_bound)


@lru_cache(maxsize=None)
def hadamard_sign_matrix(n_cols: int) -> np.ndarray:
    """Return a Hadamard sign matrix with at least n_cols orthogonal columns/rows."""
    size = 1
    while size < max(1, int(n_cols)):
        size *= 2
    H = np.array([[1]], dtype=int)
    while H.shape[0] < size:
        H = np.block([[H, H], [H, -H]])
    H.setflags(write=False)
    return H


def emit_drift_echo_product_block(
    u_slice: np.ndarray,
    start: int,
    ctrl_labels: Sequence[str],
    target_sites: tuple[int, int],
    theta: float,
    dt_micro: float,
    amp_bound: float,
    dim: int,
    drift_strength: float,
) -> tuple[int, str]:
    """Emit a refocused ZZ block isolating one pair from the all-to-all ZZ drift.

    The always-on drift is H0=sum_{a<b} Z_a Z_b / 4 in unnormalized Pauli units.
    A Hadamard toggling frame assigns identical/opposite sign rows to the target
    pair and orthogonal rows to spectator qubits, averaging spectator couplings
    to zero.  Local X pi pulses flip a qubit's Z sign between intervals.
    """
    original = u_slice.copy()

    def fail(status: str) -> tuple[int, str]:
        u_slice[:, :] = original
        return 0, status

    if dim <= 0 or (dim & (dim - 1)) != 0:
        return fail("non_qubit_dim")
    n_qubits = int(round(np.log2(dim)))
    i, j = target_sites
    if i == j or i < 0 or j < 0 or i >= n_qubits or j >= n_qubits:
        return fail("bad_pair")
    if abs(float(drift_strength)) <= 1e-14:
        return fail("zero_drift_strength")

    H = hadamard_sign_matrix(n_qubits)
    L = H.shape[0]
    patterns: dict[int, np.ndarray] = {}
    theta_over_drift = float(theta) / float(drift_strength)
    sign_pair = 1 if theta_over_drift >= 0 else -1
    patterns[i] = H[0].copy()
    patterns[j] = sign_pair * H[0].copy()
    row_idx = 1
    for q in range(n_qubits):
        if q in (i, j):
            continue
        if row_idx >= L:
            return fail("hadamard_too_small")
        patterns[q] = H[row_idx].copy()
        row_idx += 1

    total_drift_time = abs(4.0 * theta_over_drift / max(1, dim))
    if total_drift_time <= 1e-14:
        return 0, "zero_theta"
    interval_time = total_drift_time / L
    interval_steps = max(1, int(np.ceil(interval_time / dt_micro)))
    flip_area = (np.pi / 2.0) * float(np.sqrt(dim))
    pos = int(start)
    current = np.ones(n_qubits, dtype=int)
    ctrl_x = [physical_axis_index(ctrl_labels, q, "X") for q in range(n_qubits)]

    for ell in range(L):
        desired = np.array([patterns[q][ell] for q in range(n_qubits)], dtype=int)
        for q in range(n_qubits):
            if desired[q] == current[q]:
                continue
            if ctrl_x[q] is None:
                return fail("missing_x_refocus")
            used = emit_control_area(u_slice, pos, int(ctrl_x[q]), flip_area, dt_micro, amp_bound)
            if used <= 0:
                return fail("insufficient_budget_refocus")
            pos += used
            current[q] = desired[q]
        if pos + interval_steps > u_slice.shape[0]:
            return fail("insufficient_budget_drift")
        pos += interval_steps

    for q in range(n_qubits):
        if current[q] == 1:
            continue
        if ctrl_x[q] is None:
            return fail("missing_x_restore")
        used = emit_control_area(u_slice, pos, int(ctrl_x[q]), flip_area, dt_micro, amp_bound)
        if used <= 0:
            return fail("insufficient_budget_restore")
        pos += used
        current[q] = 1
    return pos - start, "ok"


def pauli_factor_pair_for_axis(axis: str) -> tuple[str, str]:
    """Return A,B axes with A*B = i*axis."""
    if axis == "X":
        return "Y", "Z"
    if axis == "Y":
        return "Z", "X"
    if axis == "Z":
        return "X", "Y"
    raise ValueError(f"unsupported Pauli axis {axis!r}")


def emit_pauli_product_bch_block(
    u_slice: np.ndarray,
    start: int,
    ctrl_labels: Sequence[str],
    axes_by_site: dict[int, str],
    theta: float,
    dt_micro: float,
    amp_bound: float,
    dim: int,
    *,
    max_degree: int = 4,
    bch_reps: int = 1,
    drift_strength: float = 1.0,
) -> tuple[int, str]:
    """Emit a Pauli-product exponential using product-aware recursive BCH words.

    For degree r>=3, choose a factorization A*B=iC where C is the requested
    Pauli string, then use the group commutator
    exp(-i aA/D) exp(-i bB/D) exp(+i aA/D) exp(+i bB/D).
    Since [A,B]=2iC, each repetition contributes approximately
    exp[-i (2ab/D^2) C].  Setting ab=theta*D/(2*n_reps) targets
    exp[-i theta C/D].
    """
    clean_axes = {int(s): str(a) for s, a in axes_by_site.items() if str(a) != "I"}
    degree = len(clean_axes)
    if degree == 0 or abs(theta) < 1e-14:
        return 0, "zero_theta"
    if degree == 1:
        site, axis = next(iter(clean_axes.items()))
        return emit_one_body_product_block(u_slice, start, ctrl_labels, site, axis, theta, dt_micro, amp_bound, dim)
    if degree == 2:
        return emit_two_body_product_block(
            u_slice,
            start,
            ctrl_labels,
            clean_axes,
            theta,
            dt_micro,
            amp_bound,
            dim,
            drift_strength,
        )
    if degree > max_degree:
        return 0, f"unsupported_degree{degree}"

    original = u_slice.copy()

    def fail(status: str) -> tuple[int, str]:
        u_slice[:, :] = original
        return 0, status

    sites = sorted(clean_axes)
    last = sites[-1]
    overlap = sites[-2]
    left_sites = sites[:-1]
    a_axis, b_axis = pauli_factor_pair_for_axis(clean_axes[overlap])

    axes_A = {s: clean_axes[s] for s in left_sites}
    axes_A[overlap] = a_axis
    axes_B = {overlap: b_axis, last: clean_axes[last]}

    reps = max(1, int(bch_reps))
    mag = float(np.sqrt(abs(theta) * float(dim) / (2.0 * reps)))
    beta = float(np.sign(theta) * mag)
    if beta == 0.0:
        return 0, "zero_theta"

    pos = int(start)
    for _ in range(reps):
        for axes_term, angle in [
            (axes_A, +mag),
            (axes_B, +beta),
            (axes_A, -mag),
            (axes_B, -beta),
        ]:
            used, status = emit_pauli_product_bch_block(
                u_slice,
                pos,
                ctrl_labels,
                axes_term,
                angle,
                dt_micro,
                amp_bound,
                dim,
                max_degree=max_degree,
                bch_reps=bch_reps,
                drift_strength=drift_strength,
            )
            if used <= 0:
                return fail(f"bch_{status}")
            pos += used
    return pos - start, f"bch_degree{degree}"


def emit_two_body_product_block(
    u_slice: np.ndarray,
    start: int,
    ctrl_labels: Sequence[str],
    axes_by_site: dict[int, str],
    theta: float,
    dt_micro: float,
    amp_bound: float,
    dim: int,
    drift_strength: float,
) -> tuple[int, str]:
    """Emit a refocused product block for a two-body Pauli product.

    The block maps the requested local axes to Z, uses a Hadamard echo pattern
    to isolate the selected pair inside the always-on all-to-all ZZ drift, then
    maps the axes back.  Unlike the legacy dictionary compiler, this treats
    cross-site EA products as physical Pauli products rather than as
    commutators of mutually commuting single-site controls.
    """
    if len(axes_by_site) != 2:
        return 0, "unsupported_degree"
    original = u_slice.copy()

    def fail(status: str) -> tuple[int, str]:
        u_slice[:, :] = original
        return 0, status

    sites = sorted(axes_by_site)
    pos = int(start)
    used_pre: list[tuple[int, str]] = []
    for site in sites:
        if axes_by_site[site] == "X" and physical_axis_index(ctrl_labels, site, "Y") is None:
            return fail("missing_local_axis")
        if axes_by_site[site] == "Y" and physical_axis_index(ctrl_labels, site, "X") is None:
            return fail("missing_local_axis")
        used = emit_basis_rotation_to_z(
            u_slice, pos, ctrl_labels, site, axes_by_site[site], False, dt_micro, amp_bound, dim
        )
        if used == 0 and axes_by_site[site] != "Z":
            return fail("insufficient_budget_basis_rotation")
        pos += used
        used_pre.append((site, axes_by_site[site]))

    used_echo, echo_status = emit_drift_echo_product_block(
        u_slice,
        pos,
        ctrl_labels,
        (sites[0], sites[1]),
        theta,
        dt_micro,
        amp_bound,
        dim,
        drift_strength,
    )
    if used_echo <= 0:
        return fail(echo_status)
    pos += used_echo

    for site, axis in reversed(used_pre):
        if axis == "X" and physical_axis_index(ctrl_labels, site, "Y") is None:
            return fail("missing_inverse_axis")
        if axis == "Y" and physical_axis_index(ctrl_labels, site, "X") is None:
            return fail("missing_inverse_axis")
        used = emit_basis_rotation_to_z(
            u_slice, pos, ctrl_labels, site, axis, True, dt_micro, amp_bound, dim
        )
        if used == 0 and axis != "Z":
            return fail("insufficient_budget_inverse_basis")
        pos += used
    if pos > u_slice.shape[0]:
        return fail("insufficient_budget")
    return pos - start, "ok"


def estimate_basis_rotation_steps(axis: str, dt_micro: float, amp_bound: float, dim: int, instant_overhead: bool) -> int:
    if axis == "Z" or instant_overhead:
        return 0
    return estimate_control_area_steps((np.pi / 4.0) * float(np.sqrt(dim)), dt_micro, amp_bound)


def estimate_drift_echo_steps(theta: float, dt_micro: float, amp_bound: float, dim: int, instant_overhead: bool, drift_strength: float = 1.0) -> tuple[int, str]:
    if dim <= 0 or (dim & (dim - 1)) != 0:
        return 0, "non_qubit_dim"
    if abs(float(drift_strength)) <= 1e-14:
        return 0, "zero_drift_strength"
    n_qubits = int(round(np.log2(dim)))
    L = hadamard_sign_matrix(n_qubits).shape[0]
    total_drift_time = abs(4.0 * float(theta) / (float(drift_strength) * max(1, dim)))
    if total_drift_time <= 1e-14:
        return 0, "zero_theta"
    interval_steps = max(1, int(np.ceil((total_drift_time / L) / dt_micro)))
    drift_steps = L * interval_steps
    if instant_overhead:
        return drift_steps, "instant_cost"

    flip_steps = estimate_control_area_steps((np.pi / 2.0) * float(np.sqrt(dim)), dt_micro, amp_bound)
                                                                                  
                                                                                
                                                                                
    max_flips = max(0, (L - 1) * n_qubits + n_qubits)
    return drift_steps + max_flips * flip_steps, "finite_cost"


def estimate_two_body_product_steps(
    axes_by_site: dict[int, str],
    theta: float,
    dt_micro: float,
    amp_bound: float,
    dim: int,
    instant_overhead: bool,
    drift_strength: float = 1.0,
) -> tuple[int, str]:
    if len(axes_by_site) != 2:
        return 0, "unsupported_degree"
    steps = 0
    for axis in axes_by_site.values():
        steps += estimate_basis_rotation_steps(axis, dt_micro, amp_bound, dim, instant_overhead)
    drift_steps, status = estimate_drift_echo_steps(theta, dt_micro, amp_bound, dim, instant_overhead, drift_strength)
    if drift_steps <= 0:
        return 0, status
    steps += drift_steps
    for axis in axes_by_site.values():
        steps += estimate_basis_rotation_steps(axis, dt_micro, amp_bound, dim, instant_overhead)
    return steps, status


def estimate_one_body_product_steps(
    axis: str,
    theta: float,
    dt_micro: float,
    amp_bound: float,
    dim: int,
    instant_overhead: bool,
) -> tuple[int, str]:
    if abs(theta) < 1e-14:
        return 0, "zero_theta"
    sqrtD = float(np.sqrt(dim))
    if axis in ("X", "Y"):
        return estimate_control_area_steps(theta / sqrtD, dt_micro, amp_bound), "finite_cost"
    if axis == "Z":
        bch_steps = estimate_local_z_bch_steps(theta, dt_micro, amp_bound)
        if instant_overhead:
            return estimate_control_area_steps(theta / sqrtD, dt_micro, amp_bound), "instant_z_cost"
        steps = 2 * estimate_control_area_steps((np.pi / 4.0) * sqrtD, dt_micro, amp_bound)
        steps += estimate_control_area_steps(theta / sqrtD, dt_micro, amp_bound)
        if bch_steps > 0 and bch_steps <= steps and abs(theta) <= 0.5:
            return bch_steps, "finite_z_bch_cost"
        return steps, "finite_z_cost"
    return 0, "missing_linear_axis"


def estimate_pauli_product_steps(
    axes_by_site: dict[int, str],
    theta: float,
    dt_micro: float,
    amp_bound: float,
    dim: int,
    *,
    max_degree: int = 4,
    bch_reps: int = 1,
    instant_overhead: bool = False,
    drift_strength: float = 1.0,
) -> tuple[int, str]:
    clean_axes = {int(s): str(a) for s, a in axes_by_site.items() if str(a) != "I"}
    degree = len(clean_axes)
    if degree == 0 or abs(theta) < 1e-14:
        return 0, "zero_theta"
    if degree == 1:
        _site, axis = next(iter(clean_axes.items()))
        return estimate_one_body_product_steps(axis, theta, dt_micro, amp_bound, dim, instant_overhead)
    if degree == 2:
        return estimate_two_body_product_steps(clean_axes, theta, dt_micro, amp_bound, dim, instant_overhead, drift_strength)
    if degree > max_degree:
        return 0, f"unsupported_degree{degree}"

    sites = sorted(clean_axes)
    last = sites[-1]
    overlap = sites[-2]
    left_sites = sites[:-1]
    a_axis, b_axis = pauli_factor_pair_for_axis(clean_axes[overlap])
    axes_A = {s: clean_axes[s] for s in left_sites}
    axes_A[overlap] = a_axis
    axes_B = {overlap: b_axis, last: clean_axes[last]}

    reps = max(1, int(bch_reps))
    mag = float(np.sqrt(abs(theta) * float(dim) / (2.0 * reps)))
    beta = float(np.sign(theta) * mag)
    total = 0
    for _ in range(reps):
        for axes_term, angle in [
            (axes_A, +mag),
            (axes_B, +beta),
            (axes_A, -mag),
            (axes_B, -beta),
        ]:
            used, status = estimate_pauli_product_steps(
                axes_term,
                angle,
                dt_micro,
                amp_bound,
                dim,
                max_degree=max_degree,
                bch_reps=bch_reps,
                instant_overhead=instant_overhead,
                drift_strength=drift_strength,
            )
            if used <= 0:
                return 0, f"cost_{status}"
            total += used
    return total, f"{'instant' if instant_overhead else 'finite'}_bch_degree{degree}"


def get_commutator_sequence(indices: list[int], area: float) -> list[tuple[int, float]]:
    """Recursive commutator decomposition into a bang-bang sequence.

    Returns a list of (control_index, signed_area) segments.

    This is a heuristic decomposition used only to generate an initial seed.
    """
    if len(indices) == 1:
        return [(indices[0], float(area))]

                                                                                    
    A = indices[:1]
    B = indices[1:]

    sgn = 1.0 if area >= 0 else -1.0
    sub = float(np.sqrt(abs(area)))

                                                            
    seq: list[tuple[int, float]] = []
    seq += get_commutator_sequence(B, -sub * sgn)
    seq += get_commutator_sequence(A, -sub)
    seq += get_commutator_sequence(B, +sub * sgn)
    seq += get_commutator_sequence(A, +sub)
    return seq


def resample_controls(u: np.ndarray, M_new: int) -> np.ndarray:
    """Nearest-neighbor resampling along the time axis."""
    u = np.asarray(u, dtype=float)
    if u.shape[0] == M_new:
        return u.copy()
    idx = (np.arange(M_new) * u.shape[0] / M_new).astype(int)
    return u[idx, :].copy()



def _simplify_indices(indices: list[int]) -> list[int]:
    """Simple reduction: cancel even multiplicities of identical indices (Pauli-involution heuristic).

    This is exact for involutions with A^2=I and for products reduced modulo 2.
    For general su(d) generators it is only a heuristic; we apply it conservatively
    by cancelling only consecutive duplicates.
    """
    out: list[int] = []
    for idx in indices:
        if out and out[-1] == idx:
            out.pop()
        else:
            out.append(idx)
    return out


def _word_sign_left_nest(r: int) -> int:
    """Sign of the left-nested commutator template built from group commutators.

    With template C(A,B)=e^{-i eps A}e^{-i eps B}e^{+i eps A}e^{+i eps B},
    log C = -eps^2 [A,B] + O(eps^3). For left-nesting, the sign alternates.
    """
    return -1 if (r % 2 == 0) else +1


def _build_word_pulses_leftnest(indices: list[int], eps: float, hard_amp: float) -> tuple[list[int], int]:
    """Return a pulse *index* word (no durations yet) implementing a left-nested commutator.

    For indices [a,b,c,...], we interpret target as W_r = [H_a, [H_b, [H_c, ...]]].
    One application of the constructed word yields log ≈ s * eps^r W_r, with s given by
    _word_sign_left_nest(r).

    Output:
      - seq: list of signed indices, where sign encodes +/− pulse (positive means +A, negative means −A)
      - r: degree

    We implement recursion:
      S_1(a) = exp(-i eps H_a)
      S_r(a, rest) = C( S_1(a), S_{r-1}(rest) )
    where C(X,Y)=XYX^{-1}Y^{-1}. At the pulse level we represent inverses by reversing and flipping signs.
    """
    idxs = _simplify_indices(indices)
    r = len(idxs)
    if r == 0:
        return [], 0
    if r == 1:
        return [idxs[0]+1], 1                               

                           
    tail_seq, tail_r = _build_word_pulses_leftnest(idxs[1:], eps, hard_amp)

                         
    head = [idxs[0]+1]

    def inv(seq: list[int]) -> list[int]:
                                               
        return [-(x) for x in reversed(seq)]

                                                             
    seq = head + tail_seq + inv(head) + inv(tail_seq)
    return seq, r


def _emit_word_to_u(u_slice: np.ndarray, seq: list[int], eps: float, hard_amp: float, dt_micro: float, max_steps: int) -> int:
    """Emit a primitive pulse word into a slice micro-grid.

    Each token in seq is a signed physical control index (1-based).

    We want each primitive pulse to realize area eps:
        \\int u(t) dt = eps
    while respecting a fixed micro-grid step dt_micro.

    If the ideal duration dur = eps/hard_amp is smaller than dt_micro, we keep the
    pulse within one microstep but reduce the amplitude so that area is still eps.

    Returns number of microsteps consumed.
    """
    used = 0
    if hard_amp <= 0 or dt_micro <= 0:
        return 0

                                              
    dur = float(eps) / float(hard_amp)
    dur = max(dur, 0.0)

                                                                  
    steps = int(np.ceil(dur / dt_micro))
    steps = max(1, steps)
    total_steps = steps * len(seq)
    if total_steps > max_steps:
        return 0

    for token in seq:
                                                                        
                                                           
        amp = float(hard_amp) * (dur / (steps * dt_micro)) if dur > 0 else 0.0
        amp = min(float(hard_amp), amp)

        idx = abs(token) - 1
        sgn = 1.0 if token > 0 else -1.0
        u_slice[used:used+steps, idx] = sgn * amp
        used += steps

    return used


def compile_ea_to_controls(
    opt: ImprovedPolynomialPontryagin,
    ctrl_labels: list[str],
    T: float,
    Nt: int,
    S: int,
    amp_bound: float,
    hard_amp: float,
    linear_split: bool = True,
    verbose: bool = True,
    compiler_eps: float = 0.02,
    budget_frac: float = 1.0,
    max_terms_per_slice: int = 32,
    diagnostics: Optional[list[dict[str, Any]]] = None,
    reporter: Optional[ConsoleReporter] = None,
    progress_label: str = "Compiler",
) -> np.ndarray:
    """Compile the EA path into a time-grid of physical control amplitudes (M=Nt*S).

    This version is *time-budgeted* per slice. It does NOT time-warp a longer word into T.

    Strategy:
      1) For each EA slice k, build a prioritized list of EA terms.
      2) For degree-1 terms, allocate residual slice time as constant controls.
      3) For degree>=2 terms, compile as left-nested commutator words using group commutators.
      4) Emit primitive pulses at amplitude hard_amp with duration eps/hard_amp, discretized to microsteps.
      5) Enforce per-slice microstep budget = floor(budget_frac * S).

    Accuracy:
      - A degree-r word uses a fixed eps and yields log ≈ s * eps^r W_r + O(eps^{r+1}).
      - Repetition count n approximates the requested angle theta by n*eps^r.
      - This is a BCH/Trotter short-time approximation; it is meaningful only when eps is small.

    Limitations:
      - EA labels are parsed as products and compiled as left-nested commutators. This matches a Lie-word
        interpretation but may not match all associative polynomial directions. For qubits with Pauli axes,
        many products collapse to Lie elements under identities.
      - If the slice time budget is insufficient, we truncate smaller terms.

    Returns:
      u_out: (Nt*S, n_phys)
    """
    if S <= 0:
        raise ValueError('S must be positive')

    n_ea_steps = int(opt.u.shape[0])
    if n_ea_steps != Nt:
        Nt = n_ea_steps

    n_phys = len(ctrl_labels)
    M = Nt * S
    dt_ea = T / Nt
    dt_micro = T / M

    HARD_AMP = float(min(max(1e-12, hard_amp), amp_bound if amp_bound > 0 else hard_amp))
    if amp_bound > 0:
        HARD_AMP = min(HARD_AMP, amp_bound)

    if verbose:
        msg1 = '    [Compiler] Translating EA controls to physical controls (budgeted)...'
        msg2 = f'    [Compiler] Nt={Nt}, S={S}, M={M}, dt_ea={dt_ea:.6g}, dt_micro={dt_micro:.6g}'
        msg3 = f'    [Compiler] HARD_AMP={HARD_AMP:.3g} (amp_bound={amp_bound}) eps={compiler_eps} budget_frac={budget_frac}'
        if reporter is None:
            print(msg1)
            print(msg2)
            print(msg3)
        else:
            reporter.info(msg1)
            reporter.info(msg2)
            reporter.info(msg3)

                 
    u_out = np.zeros((M, n_phys), dtype=float)

                                    
    slice_steps = S
    slice_budget = max(1, int(np.floor(budget_frac * slice_steps)))
    steps_per_primitive = max(1, int(np.ceil((float(compiler_eps) / max(1e-30, HARD_AMP)) / dt_micro)))

                                     
    ctrl_label_lookup = build_ctrl_label_lookup(ctrl_labels)
    cached_indices: list[list[int] | None] = []
    cached_degrees: list[int] = []
    for lab in opt.control_labels:
        inds = parse_label_to_indices(lab, ctrl_labels, ctrl_label_lookup=ctrl_label_lookup)
        cached_indices.append(inds)
        cached_degrees.append(len(_simplify_indices(inds)) if inds is not None else 0)

    total_used_steps = 0
    t_start = time.time()

    for k in range(Nt):
                               
        u_slice = u_out[k*S:(k+1)*S, :]
        used = 0
        higher_degree_used = 0

        active = np.where(np.abs(opt.u[k]) > 1e-8)[0]
        if active.size == 0:
            if diagnostics is not None:
                diagnostics.append(
                    {
                        "slice": int(k),
                        "active_terms": 0,
                        "retained_terms": 0,
                        "used_steps": 0,
                        "higher_degree_steps": 0,
                        "remaining_linear_steps": int(slice_steps),
                        "emitted_words": "{}",
                        "skipped_words": "{}",
                    }
                )
            if reporter is not None:
                reporter.progress(progress_label, k + 1, Nt, t_start, extra=f"used={total_used_steps + used}/{(k + 1) * slice_steps}", force=(k == Nt - 1))
            continue

                                       
        items = []
        for idx in active.tolist():
            amp = float(opt.u[k, idx])
            lab = opt.control_labels[idx]
            inds = cached_indices[idx]
            if inds is None:
                continue
            deg = cached_degrees[idx]
            if deg == 0:
                continue
            theta = amp * dt_ea
            items.append((deg, -abs(theta), idx, lab, inds, theta))
        items.sort()
        if len(items) > max_terms_per_slice:
            items = items[:max_terms_per_slice]

        emitted_words: dict[int, int] = {}
        skipped_words: dict[int, int] = {}

                                                                  
        for deg, _negmag, idx, lab, inds, theta in items:
            if deg <= 1:
                continue
            if used >= slice_budget:
                break

                                     
            seq, r = _build_word_pulses_leftnest(inds, compiler_eps, HARD_AMP)
            if r != deg or r == 0:
                continue

                           
            s_template = _word_sign_left_nest(r)

                                                                                         
            unit = (compiler_eps ** r)
            if unit <= 0:
                continue

            n_req = int(round(abs(theta) / unit))
            n_req = max(1, n_req)

                                                                                          
            want_sign = 1 if theta >= 0 else -1
            eff_sign = s_template
            if want_sign * eff_sign < 0:
                seq_eff = [-(x) for x in seq]
            else:
                seq_eff = seq

            word_steps = steps_per_primitive * len(seq_eff)

                                             
            for _ in range(n_req):
                if used >= slice_budget:
                    break
                used_now = _emit_word_to_u(u_slice, seq_eff, compiler_eps, HARD_AMP, dt_micro, slice_budget - used)
                if used_now <= 0:
                    skipped_words[deg] = skipped_words.get(deg, 0) + 1
                    break
                used += used_now
                higher_degree_used += used_now
                emitted_words[deg] = emitted_words.get(deg, 0) + 1
                if word_steps > slice_budget - (used - used_now):
                    skipped_words[deg] = skipped_words.get(deg, 0) + max(0, n_req - emitted_words.get(deg, 0))
                    break

                                                                                      
                                                                                                  
        remaining = slice_steps - used
        if remaining > 0:
                                                                     
            vec = np.zeros(n_phys, dtype=float)
            for deg, _negmag, idx, lab, inds, theta in items:
                if deg != 1:
                    continue
                vec[inds[0]] += theta / (remaining * dt_micro)
            if amp_bound > 0:
                np.clip(vec, -amp_bound, amp_bound, out=vec)
            u_slice[used:used+remaining, :] = vec
            used += remaining

        total_used_steps += min(used, slice_steps)

        if diagnostics is not None:
            diagnostics.append(
                {
                    "slice": int(k),
                    "active_terms": int(active.size),
                    "retained_terms": int(len(items)),
                    "used_steps": int(min(used, slice_steps)),
                    "higher_degree_steps": int(higher_degree_used),
                    "remaining_linear_steps": int(max(0, slice_steps - higher_degree_used)),
                    "emitted_words": json.dumps(emitted_words, sort_keys=True),
                    "skipped_words": json.dumps(skipped_words, sort_keys=True),
                }
            )

        if reporter is not None:
            reporter.progress(
                progress_label,
                k + 1,
                Nt,
                t_start,
                extra=f"avg_used={total_used_steps / max(1, k + 1):.1f}/{S}",
                force=(k == Nt - 1),
            )

    if amp_bound > 0:
        np.clip(u_out, -amp_bound, amp_bound, out=u_out)

    if verbose:
        msg1 = f'    [Compiler] Total used microsteps: {total_used_steps} / {M} (avg per slice {total_used_steps/Nt:.2f} / {S})'
        msg2 = f'    [Compiler] Seed RMS={rms(u_out):.6g}, max|u|={float(np.max(np.abs(u_out))):.6g}'
        if reporter is None:
            print(msg1)
            print(msg2)
        else:
            reporter.info(msg1)
            reporter.info(msg2)

    return u_out


def compile_ea_to_controls_product_aware(
    opt: ImprovedPolynomialPontryagin,
    ctrl_labels: list[str],
    T: float,
    Nt: int,
    S: int,
    amp_bound: float,
    hard_amp: float,
    drift_strength_hw: float = 1.0,
    compiler_eps: float = 0.02,
    budget_frac: float = 1.0,
    max_terms_per_slice: int = 32,
    product_bch_reps: int = 1,
    product_max_degree: int = 4,
    compiler_sort_mode: str = "legacy",
    compiler_operation_mode: str = "product",
    product_time_scale: float = 1.0,
    term_theta_threshold: float = 0.0,
    aggregate_terms: bool = False,
    diagnostics: Optional[list[dict[str, Any]]] = None,
    term_diagnostics: Optional[list[dict[str, Any]]] = None,
    reporter: Optional[ConsoleReporter] = None,
    progress_label: str = "ProductCompiler",
) -> np.ndarray:
    """Compile EA controls by treating qubit EA labels as Pauli products.

    This urgent replacement fixes the most serious error in the dictionary
    compiler: cross-site products must not be compiled as commutators of local
    controls, since those controls commute.  Degree-1 terms are emitted as
    local rotations, degree-2 terms as refocused two-body drift products, and
    degree-3/4 terms as recursive BCH words built from noncommuting Pauli
    products.
    """
    if S <= 0:
        raise ValueError("S must be positive")
    n_ea_steps = int(opt.u.shape[0])
    if n_ea_steps != Nt:
        Nt = n_ea_steps
    n_phys = len(ctrl_labels)
    M = Nt * S
    dt_ea = T / Nt
    dt_micro = T / M
    dim = int(opt.dim)
    u_out = np.zeros((M, n_phys), dtype=float)
    slice_budget = max(1, int(np.floor(float(budget_frac) * S)))
    pulse_amp = float(max(1e-12, hard_amp))
    if amp_bound > 0:
        pulse_amp = min(pulse_amp, float(amp_bound))
    t_start = time.time()

    parsed: list[tuple[complex, dict[int, str]] | None] = [
        parse_ea_label_to_pauli_string(label) for label in opt.control_labels
    ]

    def axes_key(axes_by_site: dict[int, str]) -> tuple[tuple[int, str], ...]:
        return tuple(sorted((int(site), str(axis)) for site, axis in axes_by_site.items() if str(axis) != "I"))

    total_used_steps = 0
    total_terms = 0
    total_emitted = 0
    total_skipped = 0

    def record_term(
        *,
        slice_index: int,
        idx: int,
        label: str,
        degree: int,
        axes_by_site: dict[int, str],
        theta: float,
        retained: bool,
        emitted_flag: bool,
        status: str,
        used_steps: int,
        budget_before: int,
        budget_after: int,
        rank: int,
        required_steps: int | None = None,
    ) -> None:
        if term_diagnostics is None:
            return
        term_diagnostics.append(
            {
                "slice": int(slice_index),
                "rank": int(rank),
                "control_index": int(idx),
                "label": str(label),
                "degree": int(degree),
                "pauli_string": pauli_axes_to_string(axes_by_site),
                "theta": float(theta),
                "requested_weight": float(abs(theta)),
                "retained": int(bool(retained)),
                "emitted": int(bool(emitted_flag)),
                "status": str(status),
                "used_steps": int(used_steps),
                "required_steps": int(required_steps if required_steps is not None else used_steps),
                "available_steps": int(max(0, slice_budget - budget_before)),
                "budget_before": int(budget_before),
                "budget_after": int(budget_after),
            }
        )

    for k in range(Nt):
        u_slice = u_out[k * S : (k + 1) * S, :]
        u_budget = u_slice[:slice_budget, :]
        used = 0
        emitted: dict[str, int] = {}
        skipped: dict[str, int] = {}
        active = np.where(np.abs(opt.u[k]) > 1e-8)[0]
        items: list[tuple[int, float, int, str, float, dict[int, str]]] = []
        aggregate: dict[tuple[tuple[int, str], ...], dict[str, Any]] = {}
        for idx in active.tolist():
            label = opt.control_labels[idx]
            parsed_item = parsed[idx]
            if parsed_item is None:
                skipped["unparsed"] = skipped.get("unparsed", 0) + 1
                record_term(
                    slice_index=k,
                    idx=idx,
                    label=label,
                    degree=0,
                    axes_by_site={},
                    theta=0.0,
                    retained=False,
                    emitted_flag=False,
                    status="unparsed",
                    used_steps=0,
                    budget_before=used,
                    budget_after=used,
                    rank=-1,
                )
                continue
            phase, axes_by_site = parsed_item
            if len(axes_by_site) == 0:
                skipped["identity"] = skipped.get("identity", 0) + 1
                record_term(
                    slice_index=k,
                    idx=idx,
                    label=label,
                    degree=0,
                    axes_by_site={},
                    theta=0.0,
                    retained=False,
                    emitted_flag=False,
                    status="identity",
                    used_steps=0,
                    budget_before=used,
                    budget_after=used,
                    rank=-1,
                )
                continue
            theta = float(product_time_scale) * float(np.real(phase) * opt.u[k, idx] * dt_ea)
            if abs(theta) < float(term_theta_threshold):
                continue
            if aggregate_terms:
                key = axes_key(axes_by_site)
                entry = aggregate.setdefault(
                    key,
                    {
                        "theta": 0.0,
                        "axes_by_site": dict(axes_by_site),
                        "source_count": 0,
                        "labels": [],
                    },
                )
                entry["theta"] = float(entry["theta"]) + float(theta)
                entry["source_count"] = int(entry["source_count"]) + 1
                if len(entry["labels"]) < 4:
                    entry["labels"].append(str(label))
            else:
                items.append((len(axes_by_site), -abs(theta), idx, label, theta, axes_by_site))
        if aggregate_terms:
            for key, entry in aggregate.items():
                theta_agg = float(entry["theta"])
                if abs(theta_agg) < float(term_theta_threshold):
                    continue
                axes_agg = dict(entry["axes_by_site"])
                label_agg = "AGG:{}:n={}:{}".format(
                    pauli_axes_to_string(axes_agg),
                    int(entry["source_count"]),
                    "|".join(entry["labels"]),
                )
                items.append((len(key), -abs(theta_agg), -1, label_agg, theta_agg, axes_agg))

                                                                             
                                                                             
                                                                              
        sort_mode = str(compiler_sort_mode)
        if sort_mode == "weight":
            items.sort(key=lambda x: x[1])
        elif sort_mode == "high_degree":
            items.sort(key=lambda x: (-x[0], x[1]))
        elif sort_mode == "low_degree":
            items.sort(key=lambda x: (x[0], x[1]))
        else:
            items.sort(key=lambda x: (x[0] != 2, x[0], x[1]))
        if len(items) > max_terms_per_slice:
            skipped["term_cap"] = skipped.get("term_cap", 0) + len(items) - max_terms_per_slice
            for rank, (degree, _neg_abs, idx, label, theta, axes_by_site) in enumerate(items[max_terms_per_slice:], start=max_terms_per_slice):
                record_term(
                    slice_index=k,
                    idx=idx,
                    label=label,
                    degree=degree,
                    axes_by_site=axes_by_site,
                    theta=theta,
                    retained=False,
                    emitted_flag=False,
                    status="term_cap",
                    used_steps=0,
                    budget_before=used,
                    budget_after=used,
                    rank=rank,
                )
            items = items[:max_terms_per_slice]

        total_terms += len(items)
        for rank, (degree, _neg_abs, idx, label, theta, axes_by_site) in enumerate(items):
            budget_before = used
            if used >= slice_budget:
                skipped["slice_budget"] = skipped.get("slice_budget", 0) + 1
                record_term(
                    slice_index=k,
                    idx=idx,
                    label=label,
                    degree=degree,
                    axes_by_site=axes_by_site,
                    theta=theta,
                    retained=True,
                    emitted_flag=False,
                    status="slice_budget",
                    used_steps=0,
                    budget_before=budget_before,
                    budget_after=used,
                    rank=rank,
                )
                continue
            if 1 <= degree <= int(product_max_degree):
                if str(compiler_operation_mode) in ("product_cost", "product_instant"):
                    instant_overhead = str(compiler_operation_mode) == "product_instant"
                    required_steps, cost_status = estimate_pauli_product_steps(
                        axes_by_site,
                        theta,
                        dt_micro,
                        pulse_amp,
                        dim,
                        max_degree=int(product_max_degree),
                        bch_reps=product_bch_reps,
                        instant_overhead=instant_overhead,
                        drift_strength=float(drift_strength_hw),
                    )
                    fits = required_steps > 0 and used + required_steps <= slice_budget
                    status = f"{cost_status}_{'fit' if fits else 'no_fit'}"
                    if not fits:
                        skipped[status] = skipped.get(status, 0) + 1
                        record_term(
                            slice_index=k,
                            idx=idx,
                            label=label,
                            degree=degree,
                            axes_by_site=axes_by_site,
                            theta=theta,
                            retained=True,
                            emitted_flag=False,
                            status=status,
                            used_steps=0,
                            required_steps=required_steps,
                            budget_before=budget_before,
                            budget_after=used,
                            rank=rank,
                        )
                        continue
                    used += required_steps
                    emitted[f"degree{degree}_{status}"] = emitted.get(f"degree{degree}_{status}", 0) + 1
                    total_emitted += 1
                    record_term(
                        slice_index=k,
                        idx=idx,
                        label=label,
                        degree=degree,
                        axes_by_site=axes_by_site,
                        theta=theta,
                        retained=True,
                        emitted_flag=True,
                        status=status,
                        used_steps=required_steps,
                        required_steps=required_steps,
                        budget_before=budget_before,
                        budget_after=used,
                        rank=rank,
                    )
                else:
                    estimated_steps, estimated_status = estimate_pauli_product_steps(
                        axes_by_site,
                        theta,
                        dt_micro,
                        pulse_amp,
                        dim,
                        max_degree=int(product_max_degree),
                        bch_reps=product_bch_reps,
                        instant_overhead=False,
                        drift_strength=float(drift_strength_hw),
                    )
                    used_now, status = emit_pauli_product_bch_block(
                        u_budget,
                        used,
                        ctrl_labels,
                        axes_by_site,
                        theta,
                        dt_micro,
                        pulse_amp,
                        dim,
                        max_degree=int(product_max_degree),
                        bch_reps=product_bch_reps,
                        drift_strength=float(drift_strength_hw),
                    )
                    if used_now <= 0:
                        skipped[status] = skipped.get(status, 0) + 1
                        record_term(
                            slice_index=k,
                            idx=idx,
                            label=label,
                            degree=degree,
                            axes_by_site=axes_by_site,
                            theta=theta,
                            retained=True,
                            emitted_flag=False,
                            status=status,
                            used_steps=0,
                            required_steps=estimated_steps,
                            budget_before=budget_before,
                            budget_after=used,
                            rank=rank,
                        )
                        continue
                    used += used_now
                    emitted[f"degree{degree}_{status}"] = emitted.get(f"degree{degree}_{status}", 0) + 1
                    total_emitted += 1
                    record_term(
                        slice_index=k,
                        idx=idx,
                        label=label,
                        degree=degree,
                        axes_by_site=axes_by_site,
                        theta=theta,
                        retained=True,
                        emitted_flag=True,
                        status=status,
                        used_steps=used_now,
                        required_steps=estimated_steps if estimated_steps > 0 else used_now,
                        budget_before=budget_before,
                        budget_after=used,
                        rank=rank,
                    )
            else:
                status = f"unsupported_degree{degree}"
                skipped[status] = skipped.get(status, 0) + 1
                record_term(
                    slice_index=k,
                    idx=idx,
                    label=label,
                    degree=degree,
                    axes_by_site=axes_by_site,
                    theta=theta,
                    retained=True,
                    emitted_flag=False,
                    status=status,
                    used_steps=0,
                    budget_before=budget_before,
                    budget_after=used,
                    rank=rank,
                )

        total_used_steps += min(used, slice_budget)
        total_skipped += sum(skipped.values())
        if diagnostics is not None:
            diagnostics.append(
                {
                    "slice": int(k),
                    "active_terms": int(active.size),
                    "retained_terms": int(len(items)),
                    "used_steps": int(min(used, slice_budget)),
                    "emitted_terms": int(sum(emitted.values())),
                    "skipped_terms": int(sum(skipped.values())),
                    "emitted_words": json.dumps(emitted, sort_keys=True),
                    "skipped_words": json.dumps(skipped, sort_keys=True),
                }
            )
        if reporter is not None:
            reporter.progress(
                progress_label,
                k + 1,
                Nt,
                t_start,
                extra=f"avg_used={total_used_steps / max(1, k + 1):.1f}/{S}",
                force=(k == Nt - 1),
            )

    if amp_bound > 0:
        np.clip(u_out, -amp_bound, amp_bound, out=u_out)
    if reporter is not None:
        reporter.info(
            f"    [ProductCompiler] emitted={total_emitted} retained={total_terms} "
            f"skipped={total_skipped} used={total_used_steps}/{M} RMS={rms(u_out):.6g}"
        )
    else:
        print(
            f"    [ProductCompiler] emitted={total_emitted} retained={total_terms} "
            f"skipped={total_skipped} used={total_used_steps}/{M} RMS={rms(u_out):.6g}"
        )
    return u_out


PAULI_AXES = ("X", "Y", "Z")
PAULI_AXIS_VECTORS = {
    "X": np.array([1.0, 0.0, 0.0]),
    "Y": np.array([0.0, 1.0, 0.0]),
    "Z": np.array([0.0, 0.0, 1.0]),
}


def canonical_pauli_key(axes_by_site: dict[int, str] | Sequence[tuple[int, str]]) -> tuple[tuple[int, str], ...]:
    """Canonical sparse key for a non-identity Pauli product."""
    if isinstance(axes_by_site, dict):
        iterable = axes_by_site.items()
    else:
        iterable = axes_by_site
    return tuple(sorted((int(site), str(axis)) for site, axis in iterable if str(axis) != "I"))


def pauli_key_degree(key: tuple[tuple[int, str], ...]) -> int:
    return len(key)


def pauli_key_to_axes_dict(key: tuple[tuple[int, str], ...]) -> dict[int, str]:
    return {int(site): str(axis) for site, axis in key}


def pauli_key_to_string(key: tuple[tuple[int, str], ...]) -> str:
    return pauli_axes_to_string(pauli_key_to_axes_dict(key))


def all_sparse_pauli_keys(N: int) -> list[tuple[tuple[int, str], ...]]:
    keys: list[tuple[tuple[int, str], ...]] = []
    for axes in product(("I", "X", "Y", "Z"), repeat=int(N)):
        key = tuple((site, axis) for site, axis in enumerate(axes) if axis != "I")
        keys.append(key)
    return keys


def pauli_product_dense_from_key(N: int, key: tuple[tuple[int, str], ...]) -> np.ndarray:
    axes_by_site = pauli_key_to_axes_dict(key)
    mats = {
        "I": qt.qeye(2),
        "X": qt.sigmax(),
        "Y": qt.sigmay(),
        "Z": qt.sigmaz(),
    }
    return qt.tensor(*[mats[axes_by_site.get(site, "I")] for site in range(int(N))]).full()


def exact_pauli_expansion(
    op: qt.Qobj,
    N: int,
    *,
    tol: float = 1e-10,
) -> list[tuple[tuple[tuple[int, str], ...], complex]]:
    """Expand a qubit operator in the normalized EA product basis P/D.

    The EA controls in this code are normalized with QuTiP's operator norm,
    which gives Pauli products as P/D for qubits.  We therefore compute
    coefficients in the basis B_P=P/D using Hilbert-Schmidt overlaps:

        op = sum_P c_P B_P.
    """
    D = 2 ** int(N)
    arr = op.full()
    out: list[tuple[tuple[tuple[int, str], ...], complex]] = []
    for key in all_sparse_pauli_keys(int(N)):
        P = pauli_product_dense_from_key(int(N), key)
        B = P / float(D)
        denom = np.trace(B.conj().T @ B)
        if abs(denom) <= 1e-30:
            continue
        coeff = np.trace(B.conj().T @ arr) / denom
        if abs(coeff) > float(tol):
            if abs(float(np.imag(coeff))) < 10.0 * float(tol):
                coeff = complex(float(np.real(coeff)), 0.0)
            out.append((key, complex(coeff)))
    out.sort(key=lambda item: (pauli_key_degree(item[0]), pauli_key_to_string(item[0])))
    return out


def project_nonnegative_to_l1_ball(values: np.ndarray, radius: float) -> np.ndarray:
    """Euclidean projection of nonnegative values onto sum(values)<=radius."""
    v = np.maximum(np.asarray(values, dtype=float), 0.0)
    r = float(radius)
    if r < 0.0:
        r = 0.0
    if float(np.sum(v)) <= r:
        return v.copy()
    if r <= 0.0:
        return np.zeros_like(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(u) + 1) > (cssv - r))[0]
    if rho.size == 0:
        theta = cssv[-1] / len(u)
    else:
        idx = int(rho[-1])
        theta = (cssv[idx] - r) / float(idx + 1)
    return np.maximum(v - theta, 0.0)


def project_signed_to_l1_ball(values: np.ndarray, radius: float) -> np.ndarray:
    """Euclidean projection of signed values onto sum(abs(values))<=radius."""
    v = np.asarray(values, dtype=float)
    signs = np.sign(v)
    magnitudes = project_nonnegative_to_l1_ball(np.abs(v), float(radius))
    return signs * magnitudes


def degree2_control_matrix_from_expansions(
    expansions: Sequence[Sequence[tuple[tuple[tuple[int, str], ...], complex]]],
    *,
    keys: Sequence[tuple[tuple[int, str], ...]],
    pauli_tol: float,
) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Return the exact degree-2 coefficient matrix for the selected EA controls."""
    key_to_row = {key: i for i, key in enumerate(keys)}
    d2_indices: list[int] = []
    cols: list[np.ndarray] = []
    for j, expansion in enumerate(expansions):
        col = np.zeros(len(keys), dtype=float)
        for key, coeff in expansion:
            if key in key_to_row and abs(np.imag(coeff)) < 1e-10:
                col[key_to_row[key]] += float(np.real(coeff))
        if np.linalg.norm(col) > float(pauli_tol):
            d2_indices.append(int(j))
            cols.append(col)
    if not cols:
        return [], np.zeros((len(keys), 0), dtype=float), np.zeros((0, len(keys)), dtype=float)
    A = np.stack(cols, axis=1)
    return d2_indices, A, np.linalg.pinv(A, rcond=1e-10)


def sparse_pauli_keys_of_degree(N: int, degree: int) -> list[tuple[tuple[int, str], ...]]:
    """Return all sparse Pauli keys with exactly ``degree`` non-identity sites."""
    keys: list[tuple[tuple[int, str], ...]] = []
    for sites in product(range(int(N)), repeat=int(degree)):
                                                                               
                                                     
        if tuple(sorted(sites)) != tuple(sites) or len(set(sites)) != int(degree):
            continue
        for axes in product(PAULI_AXES, repeat=int(degree)):
            keys.append(tuple((int(site), str(axis)) for site, axis in zip(sites, axes)))
    return keys


def control_matrix_from_expansions_for_keys(
    expansions: Sequence[Sequence[tuple[tuple[tuple[int, str], ...], complex]]],
    *,
    keys: Sequence[tuple[tuple[int, str], ...]],
    pauli_tol: float,
) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Return exact coefficient matrix for the requested sparse-Pauli keys."""
    key_to_row = {key: i for i, key in enumerate(keys)}
    indices: list[int] = []
    cols: list[np.ndarray] = []
    for j, expansion in enumerate(expansions):
        col = np.zeros(len(keys), dtype=float)
        for key, coeff in expansion:
            if key in key_to_row and abs(np.imag(coeff)) < 1e-10:
                col[key_to_row[key]] += float(np.real(coeff))
        if np.linalg.norm(col) > float(pauli_tol):
            indices.append(int(j))
            cols.append(col)
    if not cols:
        return [], np.zeros((len(keys), 0), dtype=float), np.zeros((0, len(keys)), dtype=float)
    A = np.stack(cols, axis=1)
    return indices, A, np.linalg.pinv(A, rcond=1e-10)


def configure_ea_batch2_projectability(
    opt: ImprovedPolynomialPontryagin,
    *,
    drift_strength_hw: float,
    budget_frac: float,
    pauli_tol: float = 1e-10,
) -> dict[str, Any]:
    """Constrain EA degree-2 controls to a physically compilable budget.

    For N=2 the physical ZZ drift, after arbitrary local rotations, can
    synthesize a two-qubit coupling matrix whose nuclear norm is bounded by

        |drift_strength_hw| * (D/4) * dt_EA.

    For N>2 we use a conservative signed-L1 budget over exact degree-2 Pauli
    angles.  This matches the explicit pair-isolating echo construction used
    by the N>2 constructive compiler path: every retained degree-2 angle is a
    product with an actual refocused physical string.
    """
    if int(opt.d) != 2:
        raise ValueError("EA batch2 projectability is implemented for qubits only")

    expansions, _rows, summary = build_exact_pauli_expansion_audit(opt, tol=float(pauli_tol))
    keys = (
        [((0, a0), (1, a1)) for a0 in PAULI_AXES for a1 in PAULI_AXES]
        if int(opt.N) == 2
        else pair_axis_keys_for_N(int(opt.N))
    )
    d2_indices, A, A_pinv = degree2_control_matrix_from_expansions(
        expansions,
        keys=keys,
        pauli_tol=float(pauli_tol),
    )
    if not d2_indices:
        raise ValueError("No degree-2 controls found for projectability constraint")
    drift_rate = float(drift_strength_hw) * float(opt.dim) / 4.0

    if int(opt.N) == 2:
        radius = abs(drift_rate) * float(opt.dt) * float(budget_frac)
        drift_matrix = np.zeros((3, 3), dtype=float)
        drift_matrix[2, 2] = drift_rate * float(opt.dt)

        stats = {
            "mode": "ea_batch2_projectable",
            "projector": "n2_nuclear_norm",
            "enabled": True,
            "N": int(opt.N),
            "d2_control_count": int(len(d2_indices)),
            "budget_frac": float(budget_frac),
            "nuclear_radius": float(radius),
            "theta_l1_radius": float(radius),
            "drift_theta_zz": float(drift_matrix[2, 2]),
            "pauli_expansion_summary": summary,
            "last_max_nuclear_before": 0.0,
            "last_max_nuclear_after": 0.0,
            "last_max_l1_before": 0.0,
            "last_max_l1_after": 0.0,
            "last_projected_slices": 0,
        }

        def projector(opt_obj: ImprovedPolynomialPontryagin) -> None:
            max_before = 0.0
            max_after = 0.0
            projected = 0
            for k in range(int(opt_obj.Nt)):
                coeff_vec = A @ np.asarray(opt_obj.u[k, d2_indices], dtype=float)
                ea_matrix = coeff_vec.reshape(3, 3) * float(opt_obj.dt)
                total_matrix = drift_matrix + ea_matrix
                U, singular_vals, Vh = np.linalg.svd(total_matrix, full_matrices=True)
                nuc_before = float(np.sum(singular_vals))
                max_before = max(max_before, nuc_before)
                if nuc_before <= radius + 1e-12:
                    max_after = max(max_after, nuc_before)
                    continue
                singular_proj = project_nonnegative_to_l1_ball(singular_vals, radius)
                total_proj = U @ np.diag(singular_proj) @ Vh
                ea_proj = total_proj - drift_matrix
                coeff_target = ea_proj.reshape(9) / float(opt_obj.dt)
                opt_obj.u[k, d2_indices] = A_pinv @ coeff_target
                nuc_after = float(np.sum(np.linalg.svd(total_proj, compute_uv=False)))
                max_after = max(max_after, nuc_after)
                projected += 1
            stats["last_max_nuclear_before"] = float(max_before)
            stats["last_max_nuclear_after"] = float(max_after)
            stats["last_max_l1_before"] = float(max_before)
            stats["last_max_l1_after"] = float(max_after)
            stats["last_projected_slices"] = int(projected)

    else:
        theta_radius = abs(drift_rate) * float(opt.dt) * float(budget_frac)
        stats = {
            "mode": "ea_batch2_projectable",
            "projector": "nbody_signed_l1_pair_echo",
            "enabled": True,
            "N": int(opt.N),
            "d2_control_count": int(len(d2_indices)),
            "degree2_key_count": int(len(keys)),
            "budget_frac": float(budget_frac),
            "nuclear_radius": float(theta_radius),
            "theta_l1_radius": float(theta_radius),
            "drift_rate": float(drift_rate),
            "pauli_expansion_summary": summary,
            "last_max_nuclear_before": 0.0,
            "last_max_nuclear_after": 0.0,
            "last_max_l1_before": 0.0,
            "last_max_l1_after": 0.0,
            "last_projected_slices": 0,
        }

        def projector(opt_obj: ImprovedPolynomialPontryagin) -> None:
            max_before = 0.0
            max_after = 0.0
            projected = 0
            for k in range(int(opt_obj.Nt)):
                theta_vec = (A @ np.asarray(opt_obj.u[k, d2_indices], dtype=float)) * float(opt_obj.dt)
                l1_before = float(np.sum(np.abs(theta_vec)))
                max_before = max(max_before, l1_before)
                if l1_before <= theta_radius + 1e-12:
                    max_after = max(max_after, l1_before)
                    continue
                theta_proj = project_signed_to_l1_ball(theta_vec, theta_radius)
                coeff_target = theta_proj / float(opt_obj.dt)
                opt_obj.u[k, d2_indices] = A_pinv @ coeff_target
                l1_after = float(np.sum(np.abs(theta_proj)))
                max_after = max(max_after, l1_after)
                projected += 1
            stats["last_max_nuclear_before"] = float(max_before)
            stats["last_max_nuclear_after"] = float(max_after)
            stats["last_max_l1_before"] = float(max_before)
            stats["last_max_l1_after"] = float(max_after)
            stats["last_projected_slices"] = int(projected)

    opt.post_update_projector = projector
    projector(opt)
    opt.ea_batch2_projectability_summary = stats
    return stats


def configure_ea_native_xy_projectability(
    opt: ImprovedPolynomialPontryagin,
    *,
    amp_bound: float,
    amp_frac: float = 1.0,
    include_z: bool = False,
    z_amp_frac: float = 1.0,
    pauli_tol: float = 1e-10,
) -> dict[str, Any]:
    """Constrain EA one-body controls to the physical local-control envelope."""
    if int(opt.d) != 2:
        raise ValueError("EA native XY projectability is implemented for qubits only")
    expansions, _rows, summary = build_exact_pauli_expansion_audit(opt, tol=float(pauli_tol))
    axes = ("X", "Y", "Z") if bool(include_z) else ("X", "Y")
    keys = tuple((site, axis) for site in range(int(opt.N)) for axis in axes)
    key_to_row = {((int(site), str(axis)),): idx for idx, (site, axis) in enumerate(keys)}
    indices: list[int] = []
    cols: list[np.ndarray] = []
    for j, expansion in enumerate(expansions):
        col = np.zeros(len(keys), dtype=float)
        for key, coeff in expansion:
            if key in key_to_row and abs(np.imag(coeff)) < 1e-10:
                col[key_to_row[key]] += float(np.real(coeff))
        if np.linalg.norm(col) > float(pauli_tol):
            indices.append(int(j))
            cols.append(col)
    if not cols:
        raise ValueError("No native X/Y degree-1 controls found for projectability constraint")
    A = np.stack(cols, axis=1)
    A_pinv = np.linalg.pinv(A, rcond=1e-10)
    xy_coeff_bound = max(0.0, float(amp_bound) * float(amp_frac) * float(np.sqrt(opt.dim)))
    z_coeff_bound = max(0.0, float(amp_bound) * float(z_amp_frac) * float(np.sqrt(opt.dim)))
    bounds = np.array(
        [z_coeff_bound if str(axis) == "Z" else xy_coeff_bound for _site, axis in keys],
        dtype=float,
    )
    previous_projector = getattr(opt, "post_update_projector", None)
    stats = {
        "mode": "ea_local1_projectable",
        "projector": "local1_coeff_clip",
        "enabled": True,
        "control_count": int(len(indices)),
        "local1_key_count": int(len(keys)),
        "include_z": bool(include_z),
        "amp_bound": float(amp_bound),
        "xy_amp_frac": float(amp_frac),
        "z_amp_frac": float(z_amp_frac),
        "xy_coefficient_bound": float(xy_coeff_bound),
        "z_coefficient_bound": float(z_coeff_bound),
        "pauli_expansion_summary": summary,
        "last_max_abs_coeff_before": 0.0,
        "last_max_abs_coeff_after": 0.0,
        "last_projected_slices": 0,
    }

    def projector(opt_obj: ImprovedPolynomialPontryagin, grad: np.ndarray | None = None) -> None:
        if previous_projector is not None:
            if getattr(previous_projector, "accepts_gradient", False):
                previous_projector(opt_obj, grad)
            else:
                previous_projector(opt_obj)
        max_before = 0.0
        max_after = 0.0
        projected = 0
        for k in range(int(opt_obj.Nt)):
            coeff_vec = A @ np.asarray(opt_obj.u[k, indices], dtype=float)
            slice_max_before = float(np.max(np.abs(coeff_vec))) if coeff_vec.size else 0.0
            max_before = max(max_before, slice_max_before)
            coeff_proj = np.clip(coeff_vec, -bounds, bounds)
            slice_max_after = float(np.max(np.abs(coeff_proj))) if coeff_proj.size else 0.0
            max_after = max(max_after, slice_max_after)
            if np.linalg.norm(coeff_proj - coeff_vec) > 1e-12:
                opt_obj.u[k, indices] = A_pinv @ coeff_proj
                projected += 1
        stats["last_max_abs_coeff_before"] = float(max_before)
        stats["last_max_abs_coeff_after"] = float(max_after)
        stats["last_projected_slices"] = int(projected)

    opt.post_update_projector = projector
    setattr(projector, "accepts_gradient", True)
    projector(opt)
    opt.ea_native_xy_projectability_summary = stats
    return stats


def configure_ea_degree3_projectability(
    opt: ImprovedPolynomialPontryagin,
    *,
    theta_radius: float,
    pauli_tol: float = 1e-10,
) -> dict[str, Any]:
    """Constrain EA degree-3 slice angles to a small BCH-compilable L1 ball."""
    if int(opt.d) != 2:
        raise ValueError("EA degree-3 projectability is implemented for qubits only")
    if int(opt.N) < 3:
        raise ValueError("EA degree-3 projectability requires at least three qubits")
    expansions, _rows, summary = build_exact_pauli_expansion_audit(opt, tol=float(pauli_tol))
    keys = sparse_pauli_keys_of_degree(int(opt.N), 3)
    d3_indices, A3, A3_pinv = control_matrix_from_expansions_for_keys(
        expansions,
        keys=keys,
        pauli_tol=float(pauli_tol),
    )
    if not d3_indices:
        raise ValueError("No degree-3 controls found for projectability constraint")
    radius = max(0.0, float(theta_radius))
    previous_projector = getattr(opt, "post_update_projector", None)
    stats = {
        "mode": "ea_degree3_projectable",
        "projector": "degree3_signed_l1_bch",
        "enabled": True,
        "N": int(opt.N),
        "degree3_control_count": int(len(d3_indices)),
        "degree3_key_count": int(len(keys)),
        "theta_l1_radius": float(radius),
        "pauli_expansion_summary": summary,
        "last_max_l1_before": 0.0,
        "last_max_l1_after": 0.0,
        "last_projected_slices": 0,
    }

    def projector(opt_obj: ImprovedPolynomialPontryagin) -> None:
        if previous_projector is not None:
            previous_projector(opt_obj)
        max_before = 0.0
        max_after = 0.0
        projected = 0
        for k in range(int(opt_obj.Nt)):
            theta_vec = (A3 @ np.asarray(opt_obj.u[k, d3_indices], dtype=float)) * float(opt_obj.dt)
            l1_before = float(np.sum(np.abs(theta_vec)))
            max_before = max(max_before, l1_before)
            if l1_before <= radius + 1e-12:
                max_after = max(max_after, l1_before)
                continue
            theta_proj = project_signed_to_l1_ball(theta_vec, radius)
            coeff_target = theta_proj / float(opt_obj.dt)
            opt_obj.u[k, d3_indices] = A3_pinv @ coeff_target
            l1_after = float(np.sum(np.abs(theta_proj)))
            max_after = max(max_after, l1_after)
            projected += 1
        stats["last_max_l1_before"] = float(max_before)
        stats["last_max_l1_after"] = float(max_after)
        stats["last_projected_slices"] = int(projected)

    opt.post_update_projector = projector
    projector(opt)
    opt.ea_degree3_projectability_summary = stats
    return stats


def configure_ea_degree3_cost_projectability(
    opt: ImprovedPolynomialPontryagin,
    *,
    T: float,
    Nt: int,
    S: int,
    hard_amp: float,
    drift_strength_hw: float,
    step_budget_frac: float,
    term_theta_threshold: float = 1e-8,
    product_bch_reps: int = 1,
    max_terms_per_slice: int = 0,
    warmup_calls: int = 0,
    ramp_calls: int = 0,
    pauli_tol: float = 1e-10,
) -> dict[str, Any]:
    """Constrain degree-3 EA angles using the constructive BCH cost model.

    The compiler geometry is not a signed-L1 ball: a third-order Pauli product
    is emitted by nested BCH words whose finite-grid cost is nonlinear in theta
    and includes local-pulse/echo overhead.  This projector therefore chooses a
    sparse set of degree-3 angles per coarse slice whose estimated emitted
    microstep count fits a user-specified budget.
    """
    if int(opt.d) != 2:
        raise ValueError("EA degree-3 cost projectability is implemented for qubits only")
    if int(opt.N) < 3:
        raise ValueError("EA degree-3 cost projectability requires at least three qubits")
    if int(S) <= 0:
        raise ValueError("S must be positive for degree-3 cost projectability")

    expansions, _rows, summary = build_exact_pauli_expansion_audit(opt, tol=float(pauli_tol))
    keys = sparse_pauli_keys_of_degree(int(opt.N), 3)
    d3_indices, A3, A3_pinv = control_matrix_from_expansions_for_keys(
        expansions,
        keys=keys,
        pauli_tol=float(pauli_tol),
    )
    if not d3_indices:
        raise ValueError("No degree-3 controls found for cost projectability constraint")

    dt_micro = float(T) / (max(1, int(Nt)) * max(1, int(S)))
    step_budget = int(max(0, np.floor(float(step_budget_frac) * int(S))))
    threshold = max(0.0, float(term_theta_threshold))
    previous_projector = getattr(opt, "post_update_projector", None)
    cost_cache: dict[tuple[tuple[tuple[int, str], ...], float], int] = {}

    def theta_cache_key(theta: float) -> float:
                                                                            
                                                                          
                                                                           
        return round(abs(float(theta)), 10)

    def term_cost(key: tuple[tuple[int, str], ...], theta: float) -> int:
        if abs(float(theta)) <= threshold:
            return 0
        cache_key = (key, theta_cache_key(theta))
        cached = cost_cache.get(cache_key)
        if cached is not None:
            return int(cached)
        steps, _status = estimate_pauli_product_steps(
            pauli_key_to_axes_dict(key),
            float(theta),
            dt_micro,
            float(hard_amp),
            int(opt.dim),
            max_degree=3,
            bch_reps=int(product_bch_reps),
            instant_overhead=False,
            drift_strength=float(drift_strength_hw),
        )
        out = int(max(0, steps))
        cost_cache[cache_key] = out
        return out

    min_cost_probe = max(2.0 * threshold, 1e-12)
    min_term_costs = {key: term_cost(key, min_cost_probe) for key in keys}
    positive_min_costs = [cost for cost in min_term_costs.values() if cost > 0]
    global_min_term_cost = int(min(positive_min_costs) if positive_min_costs else 0)

    def fit_theta_to_steps(key: tuple[tuple[int, str], ...], theta: float, budget: int) -> float:
        if budget <= 0 or abs(float(theta)) <= threshold:
            return 0.0
        if int(min_term_costs.get(key, 0)) > int(budget):
            return 0.0
        full_cost = term_cost(key, theta)
        if full_cost > 0 and full_cost <= budget:
            return float(theta)

        lo = 0.0
        hi = abs(float(theta))
        sign = 1.0 if float(theta) >= 0.0 else -1.0
        best = 0.0
        for _ in range(32):
            mid = 0.5 * (lo + hi)
            if mid <= threshold:
                break
            cost = term_cost(key, sign * mid)
            if cost > 0 and cost <= budget:
                best = mid
                lo = mid
            else:
                hi = mid
        return sign * best if best > threshold else 0.0

    def default_term_target(max_packable: int) -> int:
        if int(max_terms_per_slice) > 0:
            return max(1, min(int(max_terms_per_slice), int(max_packable)))
                                                                              
                                                                             
        return max(1, min(int(max_packable), int(np.ceil(np.sqrt(max(1, int(max_packable)))))))

    stats = {
        "mode": "ea_degree3_cost_projectable",
        "projector": "degree3_constructive_cost_multiterm",
        "enabled": True,
        "N": int(opt.N),
        "degree3_control_count": int(len(d3_indices)),
        "degree3_key_count": int(len(keys)),
        "step_budget_frac": float(step_budget_frac),
        "step_budget": int(step_budget),
        "global_min_term_cost": int(global_min_term_cost),
        "max_terms_per_slice": int(max_terms_per_slice),
        "S": int(S),
        "dt_micro": float(dt_micro),
        "term_theta_threshold": float(threshold),
        "product_bch_reps": int(product_bch_reps),
        "warmup_calls": int(max(0, warmup_calls)),
        "ramp_calls": int(max(0, ramp_calls)),
        "pauli_expansion_summary": summary,
        "last_call_count": 0,
        "last_relaxation_alpha": 1.0,
        "last_best_eligible": True,
        "cost_cache_entries": 0,
        "last_max_requested_steps": 0,
        "last_max_emitted_steps": 0,
        "last_max_candidate_terms": 0,
        "last_max_requested_terms": 0,
        "last_max_packable_terms": 0,
        "last_max_target_terms": 0,
        "last_min_kept_terms": 0,
        "last_max_kept_terms": 0,
        "last_max_l1_before": 0.0,
        "last_max_l1_after": 0.0,
        "last_projected_slices": 0,
    }

    key_to_idx = {key: i for i, key in enumerate(keys)}
    call_state = {"count": 0}

    def relaxation_alpha() -> float:
        count = int(call_state["count"])
        warm = int(max(0, warmup_calls))
        ramp = int(max(0, ramp_calls))
        if count <= warm:
            return 0.0
        if ramp <= 0:
            return 1.0
        return float(min(1.0, max(0.0, (count - warm) / float(ramp))))

    def build_projection(
        candidates: list[dict[str, Any]],
        target_terms: int,
        grad_theta_vec: np.ndarray | None,
    ) -> tuple[np.ndarray, int, int, float]:
        theta_proj = np.zeros(len(keys), dtype=float)
        if target_terms <= 0 or not candidates:
            return theta_proj, 0, 0, 0.0

        selected = candidates[: int(target_terms)]
        min_total = int(sum(int(c["min_cost"]) for c in selected))
        if min_total > step_budget:
            return theta_proj, 0, 0, 0.0

        remaining = max(0, int(step_budget) - min_total)
        weights = np.array([max(float(c["score"]), 1e-30) for c in selected], dtype=float)
        if not np.isfinite(weights).all() or float(np.sum(weights)) <= 0.0:
            weights = np.ones(len(selected), dtype=float)
        raw_extra = remaining * weights / float(np.sum(weights))
        slots = [int(c["min_cost"]) + int(np.floor(extra)) for c, extra in zip(selected, raw_extra)]

                                                                
        leftover_round = int(step_budget) - int(sum(slots))
        for idx in range(max(0, leftover_round)):
            slots[idx % len(slots)] += 1

        emitted_steps = 0
        kept_terms = 0
        benefit_proxy = 0.0
        current_costs = [0 for _ in selected]

        for i, (candidate, slot) in enumerate(zip(selected, slots)):
            remaining_total = int(step_budget) - emitted_steps
            budget_i = max(0, min(int(slot), remaining_total))
            theta_keep = fit_theta_to_steps(candidate["key"], float(candidate["theta"]), budget_i)
            if abs(theta_keep) <= threshold:
                continue
            keep_cost = term_cost(candidate["key"], theta_keep)
            if keep_cost <= 0 or keep_cost > remaining_total:
                continue
            theta_proj[int(candidate["idx"])] = float(theta_keep)
            emitted_steps += int(keep_cost)
            current_costs[i] = int(keep_cost)
            kept_terms += 1
            benefit_proxy += float(candidate["score"]) * abs(float(theta_keep))

                                                                              
                                                                           
        improved = True
        while improved:
            improved = False
            for i, candidate in enumerate(selected):
                remaining_total = int(step_budget) - emitted_steps
                if remaining_total <= 0:
                    break
                idx = int(candidate["idx"])
                if abs(theta_proj[idx]) <= threshold:
                    continue
                current_cost = int(current_costs[i])
                theta_bigger = fit_theta_to_steps(
                    candidate["key"],
                    float(candidate["theta"]),
                    current_cost + remaining_total,
                )
                if abs(theta_bigger) <= abs(theta_proj[idx]) + threshold:
                    continue
                bigger_cost = term_cost(candidate["key"], theta_bigger)
                if bigger_cost <= current_cost or bigger_cost - current_cost > remaining_total:
                    continue
                benefit_proxy += float(candidate["score"]) * (abs(float(theta_bigger)) - abs(float(theta_proj[idx])))
                theta_proj[idx] = float(theta_bigger)
                emitted_steps += int(bigger_cost - current_cost)
                current_costs[i] = int(bigger_cost)
                improved = True

        return theta_proj, int(emitted_steps), int(kept_terms), float(benefit_proxy)

    def projector(opt_obj: ImprovedPolynomialPontryagin, grad: np.ndarray | None = None) -> None:
        if previous_projector is not None:
            previous_projector(opt_obj)
        call_state["count"] += 1
        alpha = relaxation_alpha()
        best_eligible = alpha >= 1.0 - 1e-15
        opt_obj.projectability_best_eligible = bool(best_eligible)

        max_requested_steps = 0
        max_emitted_steps = 0
        max_candidate_terms = 0
        max_requested_terms = 0
        max_packable_terms = 0
        max_target_terms_seen = 0
        min_kept_terms: int | None = None
        max_kept_terms = 0
        max_l1_before = 0.0
        max_l1_after = 0.0
        projected = 0

        if alpha <= 0.0:
            for k in range(int(opt_obj.Nt)):
                theta_vec = (A3 @ np.asarray(opt_obj.u[k, d3_indices], dtype=float)) * float(opt_obj.dt)
                l1_before = float(np.sum(np.abs(theta_vec)))
                max_l1_before = max(max_l1_before, l1_before)
                max_l1_after = max(max_l1_after, l1_before)
            stats["last_call_count"] = int(call_state["count"])
            stats["last_relaxation_alpha"] = float(alpha)
            stats["last_best_eligible"] = bool(best_eligible)
            stats["cost_cache_entries"] = int(len(cost_cache))
            stats["last_max_requested_steps"] = 0
            stats["last_max_emitted_steps"] = 0
            stats["last_max_candidate_terms"] = 0
            stats["last_max_requested_terms"] = 0
            stats["last_max_packable_terms"] = 0
            stats["last_max_target_terms"] = 0
            stats["last_min_kept_terms"] = 0
            stats["last_max_kept_terms"] = 0
            stats["last_max_l1_before"] = float(max_l1_before)
            stats["last_max_l1_after"] = float(max_l1_after)
            stats["last_projected_slices"] = 0
            return

        for k in range(int(opt_obj.Nt)):
            theta_vec = (A3 @ np.asarray(opt_obj.u[k, d3_indices], dtype=float)) * float(opt_obj.dt)
            l1_before = float(np.sum(np.abs(theta_vec)))
            max_l1_before = max(max_l1_before, l1_before)

            if global_min_term_cost > 0 and step_budget < global_min_term_cost:
                if l1_before > threshold:
                    opt_obj.u[k, d3_indices] = 0.0
                    projected += 1
                min_kept_terms = 0 if min_kept_terms is None else min(min_kept_terms, 0)
                continue

            grad_theta_vec: np.ndarray | None = None
            if grad is not None:
                grad_d3 = np.asarray(grad[k, d3_indices], dtype=float)
                grad_theta_vec = (A3_pinv.T @ grad_d3) / float(opt_obj.dt)

            candidates: list[dict[str, Any]] = []
            requested_steps = 0
            for idx, theta in enumerate(theta_vec):
                theta_f = float(theta)
                if abs(theta_f) <= threshold:
                    continue
                key = keys[idx]
                cost = term_cost(key, theta_f)
                if cost <= 0:
                    continue
                min_cost = int(min_term_costs.get(key, 0))
                requested_steps += int(cost)
                grad_gain = 0.0
                if grad_theta_vec is not None and idx < int(grad_theta_vec.size):
                    grad_gain = max(0.0, float(np.sign(theta_f) * grad_theta_vec[idx]))
                score = abs(theta_f) * (1.0 + grad_gain / (1.0 + abs(grad_gain)))
                candidates.append(
                    {
                        "idx": int(idx),
                        "key": key,
                        "theta": theta_f,
                        "abs_theta": abs(theta_f),
                        "cost": int(cost),
                        "min_cost": int(min_cost),
                        "score": float(score),
                    }
                )

            max_requested_steps = max(max_requested_steps, requested_steps)
            max_candidate_terms = max(max_candidate_terms, len(candidates))
            max_requested_terms = max(max_requested_terms, len(candidates))
            if requested_steps <= step_budget:
                theta_proj = theta_vec.copy()
                emitted_steps = requested_steps
                kept_terms = len(candidates)
            else:
                affordable = [
                    c for c in candidates
                    if int(c["min_cost"]) > 0 and int(c["min_cost"]) <= int(step_budget)
                ]
                affordable.sort(key=lambda c: (float(c["score"]), float(c["abs_theta"])), reverse=True)
                packable_terms = 0
                running_min_cost = 0
                for candidate in affordable:
                    next_cost = running_min_cost + int(candidate["min_cost"])
                    if next_cost > int(step_budget):
                        break
                    running_min_cost = next_cost
                    packable_terms += 1
                target_terms = default_term_target(packable_terms)
                max_packable_terms = max(max_packable_terms, packable_terms)
                max_target_terms_seen = max(max_target_terms_seen, target_terms)

                best_theta = np.zeros(len(keys), dtype=float)
                best_steps = 0
                best_kept = 0
                best_proxy = -1.0
                                                                             
                                                                
                trial_counts = sorted(set([1, target_terms, max(1, target_terms - 1), min(packable_terms, target_terms + 1)]))
                for term_count in trial_counts:
                    if term_count <= 0 or term_count > packable_terms:
                        continue
                    trial_theta, trial_steps, trial_kept, trial_proxy = build_projection(
                        affordable,
                        int(term_count),
                        grad_theta_vec,
                    )
                    if trial_kept <= 0:
                        continue
                    if trial_kept > best_kept or (trial_kept == best_kept and trial_proxy > best_proxy):
                        best_theta = trial_theta
                        best_steps = int(trial_steps)
                        best_kept = int(trial_kept)
                        best_proxy = float(trial_proxy)
                theta_proj = best_theta
                emitted_steps = int(best_steps)
                kept_terms = int(best_kept)
                if alpha < 1.0:
                    theta_proj = (1.0 - alpha) * theta_vec + alpha * theta_proj
                coeff_target = theta_proj / float(opt_obj.dt)
                opt_obj.u[k, d3_indices] = A3_pinv @ coeff_target
                projected += 1

            l1_after = float(np.sum(np.abs(theta_proj)))
            max_l1_after = max(max_l1_after, l1_after)
            max_emitted_steps = max(max_emitted_steps, emitted_steps)
            min_kept_terms = kept_terms if min_kept_terms is None else min(min_kept_terms, kept_terms)
            max_kept_terms = max(max_kept_terms, kept_terms)

        stats["last_call_count"] = int(call_state["count"])
        stats["last_relaxation_alpha"] = float(alpha)
        stats["last_best_eligible"] = bool(best_eligible)
        stats["cost_cache_entries"] = int(len(cost_cache))
        stats["last_max_requested_steps"] = int(max_requested_steps)
        stats["last_max_emitted_steps"] = int(max_emitted_steps)
        stats["last_max_candidate_terms"] = int(max_candidate_terms)
        stats["last_max_requested_terms"] = int(max_requested_terms)
        stats["last_max_packable_terms"] = int(max_packable_terms)
        stats["last_max_target_terms"] = int(max_target_terms_seen)
        stats["last_min_kept_terms"] = int(min_kept_terms or 0)
        stats["last_max_kept_terms"] = int(max_kept_terms)
        stats["last_max_l1_before"] = float(max_l1_before)
        stats["last_max_l1_after"] = float(max_l1_after)
        stats["last_projected_slices"] = int(projected)

    opt.post_update_projector = projector
    setattr(projector, "accepts_gradient", True)
    projector(opt)
    opt.ea_degree3_projectability_summary = stats
    return stats


def operator_hs_overlap(op_a: qt.Qobj, op_b: qt.Qobj) -> complex:
    """Normalized Hilbert-Schmidt overlap between two controls."""
    a = qobj_to_dense_matrix(op_a).reshape(-1)
    b = qobj_to_dense_matrix(op_b).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-30:
        return 0.0 + 0.0j
    return complex(np.vdot(a, b) / denom)


def embed_lower_degree_solution(
    lower: ImprovedPolynomialPontryagin,
    higher: ImprovedPolynomialPontryagin,
    *,
    tol: float = 1e-8,
) -> dict[str, Any]:
    """Embed a lower-degree EA control table into a higher-degree basis.

    This gives p=3 a certified p=2 fallback: if every p=2 generator appears in
    the p=3 basis, the p=3 optimizer starts from exactly the p=2 trajectory with
    all genuine degree-3 coordinates set to zero.
    """
    if lower.Nt != higher.Nt:
        raise ValueError("Cannot embed lower-degree solution with mismatched Nt")
    higher.u[:, :] = 0.0
    used_high: set[int] = set()
    rows: list[dict[str, Any]] = []
    missing = 0
    max_error = 0.0

    for j_low, (op_low, label_low) in enumerate(zip(lower.controls, lower.control_labels)):
        best_j = -1
        best_ov = 0.0 + 0.0j
        best_abs = -1.0
        for j_high, op_high in enumerate(higher.controls):
            if j_high in used_high:
                continue
            ov = operator_hs_overlap(op_high, op_low)
            ov_abs = abs(ov)
            if ov_abs > best_abs:
                best_abs = ov_abs
                best_ov = ov
                best_j = j_high
        err = float(1.0 - best_abs)
        max_error = max(max_error, err)
        if best_j < 0 or err > float(tol):
            missing += 1
            rows.append(
                {
                    "lower_index": int(j_low),
                    "lower_label": str(label_low),
                    "higher_index": "",
                    "higher_label": "",
                    "overlap_abs": float(best_abs),
                    "phase": "",
                    "matched": False,
                }
            )
            continue
        phase = float(np.real(best_ov))
        if abs(float(np.imag(best_ov))) > 1e-7:
            phase = 1.0 if np.real(best_ov) >= 0.0 else -1.0
        phase = 1.0 if phase >= 0.0 else -1.0
        higher.u[:, best_j] = phase * lower.u[:, j_low]
        used_high.add(best_j)
        rows.append(
            {
                "lower_index": int(j_low),
                "lower_label": str(label_low),
                "higher_index": int(best_j),
                "higher_label": str(higher.control_labels[best_j]),
                "overlap_abs": float(best_abs),
                "phase": float(phase),
                "matched": True,
            }
        )

    return {
        "lower_p": int(lower.p),
        "higher_p": int(higher.p),
        "lower_control_count": int(len(lower.controls)),
        "higher_control_count": int(len(higher.controls)),
        "matched_count": int(len(lower.controls) - missing),
        "missing_count": int(missing),
        "max_embedding_error": float(max_error),
        "success": bool(missing == 0),
        "rows": rows,
    }


def configure_ea_degree4_cost_projectability(
    opt: ImprovedPolynomialPontryagin,
    *,
    T: float,
    Nt: int,
    S: int,
    hard_amp: float,
    drift_strength_hw: float,
    step_budget_frac: float,
    term_theta_threshold: float = 1e-10,
    product_bch_reps: int = 1,
    max_terms_per_slice: int = 0,
    warmup_calls: int = 0,
    ramp_calls: int = 0,
    pauli_tol: float = 1e-10,
) -> dict[str, Any]:
    """Constrain degree-4 EA angles by the same constructive BCH cost model.

    This is the p=4 analog of the degree-3 cost projector.  It is intentionally
    conservative: it only keeps degree-4 sparse-Pauli angles whose estimated
    recursive BCH emission fits in the per-slice microstep budget.
    """
    if int(opt.d) != 2:
        raise ValueError("EA degree-4 cost projectability is implemented for qubits only")
    if int(opt.N) < 4:
        raise ValueError("EA degree-4 projectability requires at least four qubits")
    if int(S) <= 0:
        raise ValueError("S must be positive for degree-4 cost projectability")

    degree = 4
    expansions, _rows, summary = build_exact_pauli_expansion_audit(opt, tol=float(pauli_tol))
    keys = sparse_pauli_keys_of_degree(int(opt.N), degree)
    d4_indices, A4, A4_pinv = control_matrix_from_expansions_for_keys(
        expansions,
        keys=keys,
        pauli_tol=float(pauli_tol),
    )
    if not d4_indices:
        raise ValueError("No degree-4 controls found for cost projectability constraint")

    dt_micro = float(T) / (max(1, int(Nt)) * max(1, int(S)))
    step_budget = int(max(0, np.floor(float(step_budget_frac) * int(S))))
    threshold = max(0.0, float(term_theta_threshold))
    previous_projector = getattr(opt, "post_update_projector", None)
    cost_cache: dict[tuple[tuple[tuple[int, str], ...], float], int] = {}

    def theta_cache_key(theta: float) -> float:
        return round(abs(float(theta)), 10)

    def term_cost(key: tuple[tuple[int, str], ...], theta: float) -> int:
        if abs(float(theta)) <= threshold:
            return 0
        cache_key = (key, theta_cache_key(theta))
        cached = cost_cache.get(cache_key)
        if cached is not None:
            return int(cached)
        steps, _status = estimate_pauli_product_steps(
            pauli_key_to_axes_dict(key),
            float(theta),
            dt_micro,
            float(hard_amp),
            int(opt.dim),
            max_degree=degree,
            bch_reps=int(product_bch_reps),
            instant_overhead=False,
            drift_strength=float(drift_strength_hw),
        )
        out = int(max(0, steps))
        cost_cache[cache_key] = out
        return out

    min_cost_probe = max(2.0 * threshold, 1e-14)
    min_term_costs = {key: term_cost(key, min_cost_probe) for key in keys}
    positive_min_costs = [cost for cost in min_term_costs.values() if cost > 0]
    global_min_term_cost = int(min(positive_min_costs) if positive_min_costs else 0)

    def fit_theta_to_steps(key: tuple[tuple[int, str], ...], theta: float, budget: int) -> float:
        if budget <= 0 or abs(float(theta)) <= threshold:
            return 0.0
        if int(min_term_costs.get(key, 0)) > int(budget):
            return 0.0
        full_cost = term_cost(key, theta)
        if full_cost > 0 and full_cost <= budget:
            return float(theta)
        lo = 0.0
        hi = abs(float(theta))
        sign = 1.0 if float(theta) >= 0.0 else -1.0
        best = 0.0
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            if mid <= threshold:
                break
            cost = term_cost(key, sign * mid)
            if cost > 0 and cost <= budget:
                best = mid
                lo = mid
            else:
                hi = mid
        return sign * best if best > threshold else 0.0

    def default_term_target(max_packable: int) -> int:
        if int(max_terms_per_slice) > 0:
            return max(1, min(int(max_terms_per_slice), int(max_packable)))
        return max(1, min(int(max_packable), int(np.ceil(np.sqrt(max(1, int(max_packable)))))))

    stats = {
        "mode": "ea_degree4_cost_projectable",
        "projector": "degree4_constructive_cost_multiterm",
        "enabled": True,
        "N": int(opt.N),
        "degree4_control_count": int(len(d4_indices)),
        "degree4_key_count": int(len(keys)),
        "step_budget_frac": float(step_budget_frac),
        "step_budget": int(step_budget),
        "global_min_term_cost": int(global_min_term_cost),
        "max_terms_per_slice": int(max_terms_per_slice),
        "S": int(S),
        "dt_micro": float(dt_micro),
        "term_theta_threshold": float(threshold),
        "product_bch_reps": int(product_bch_reps),
        "warmup_calls": int(max(0, warmup_calls)),
        "ramp_calls": int(max(0, ramp_calls)),
        "pauli_expansion_summary": summary,
        "last_call_count": 0,
        "last_relaxation_alpha": 1.0,
        "last_best_eligible": True,
        "cost_cache_entries": 0,
        "last_max_requested_steps": 0,
        "last_max_emitted_steps": 0,
        "last_max_candidate_terms": 0,
        "last_max_packable_terms": 0,
        "last_max_target_terms": 0,
        "last_min_kept_terms": 0,
        "last_max_kept_terms": 0,
        "last_max_l1_before": 0.0,
        "last_max_l1_after": 0.0,
        "last_projected_slices": 0,
    }
    call_state = {"count": 0}

    def relaxation_alpha() -> float:
        count = int(call_state["count"])
        warm = int(max(0, warmup_calls))
        ramp = int(max(0, ramp_calls))
        if count <= warm:
            return 0.0
        if ramp <= 0:
            return 1.0
        return float(min(1.0, max(0.0, (count - warm) / float(ramp))))

    def projector(opt_obj: ImprovedPolynomialPontryagin, grad: np.ndarray | None = None) -> None:
        if previous_projector is not None:
            previous_projector(opt_obj)
        call_state["count"] += 1
        alpha = relaxation_alpha()
        best_eligible = alpha >= 1.0 - 1e-15
        opt_obj.projectability_best_eligible = bool(best_eligible)

        max_requested_steps = 0
        max_emitted_steps = 0
        max_candidate_terms = 0
        max_packable_terms = 0
        max_target_terms = 0
        min_kept_terms: int | None = None
        max_kept_terms = 0
        max_l1_before = 0.0
        max_l1_after = 0.0
        projected = 0

        if alpha <= 0.0:
            for k in range(int(opt_obj.Nt)):
                theta_vec = (A4 @ np.asarray(opt_obj.u[k, d4_indices], dtype=float)) * float(opt_obj.dt)
                l1_before = float(np.sum(np.abs(theta_vec)))
                max_l1_before = max(max_l1_before, l1_before)
                max_l1_after = max(max_l1_after, l1_before)
            stats["last_call_count"] = int(call_state["count"])
            stats["last_relaxation_alpha"] = float(alpha)
            stats["last_best_eligible"] = bool(best_eligible)
            stats["cost_cache_entries"] = int(len(cost_cache))
            stats["last_max_requested_steps"] = 0
            stats["last_max_emitted_steps"] = 0
            stats["last_max_candidate_terms"] = 0
            stats["last_max_packable_terms"] = 0
            stats["last_max_target_terms"] = 0
            stats["last_min_kept_terms"] = 0
            stats["last_max_kept_terms"] = 0
            stats["last_max_l1_before"] = float(max_l1_before)
            stats["last_max_l1_after"] = float(max_l1_after)
            stats["last_projected_slices"] = 0
            return

        for k in range(int(opt_obj.Nt)):
            theta_vec = (A4 @ np.asarray(opt_obj.u[k, d4_indices], dtype=float)) * float(opt_obj.dt)
            l1_before = float(np.sum(np.abs(theta_vec)))
            max_l1_before = max(max_l1_before, l1_before)
            if global_min_term_cost > 0 and step_budget < global_min_term_cost:
                if l1_before > threshold:
                    opt_obj.u[k, d4_indices] = 0.0
                    projected += 1
                min_kept_terms = 0 if min_kept_terms is None else min(min_kept_terms, 0)
                continue

            grad_theta_vec: np.ndarray | None = None
            if grad is not None:
                grad_d4 = np.asarray(grad[k, d4_indices], dtype=float)
                grad_theta_vec = (A4_pinv.T @ grad_d4) / float(opt_obj.dt)

            candidates: list[dict[str, Any]] = []
            requested_steps = 0
            for idx, theta in enumerate(theta_vec):
                theta_f = float(theta)
                if abs(theta_f) <= threshold:
                    continue
                key = keys[idx]
                cost = term_cost(key, theta_f)
                if cost <= 0:
                    continue
                min_cost = int(min_term_costs.get(key, 0))
                requested_steps += int(cost)
                grad_gain = 0.0
                if grad_theta_vec is not None and idx < int(grad_theta_vec.size):
                    grad_gain = max(0.0, float(np.sign(theta_f) * grad_theta_vec[idx]))
                score = abs(theta_f) * (1.0 + grad_gain / (1.0 + abs(grad_gain)))
                candidates.append(
                    {
                        "idx": int(idx),
                        "key": key,
                        "theta": theta_f,
                        "abs_theta": abs(theta_f),
                        "cost": int(cost),
                        "min_cost": int(min_cost),
                        "score": float(score),
                    }
                )

            max_requested_steps = max(max_requested_steps, requested_steps)
            max_candidate_terms = max(max_candidate_terms, len(candidates))
            if requested_steps <= step_budget:
                theta_proj = theta_vec.copy()
                emitted_steps = requested_steps
                kept_terms = len(candidates)
            else:
                affordable = [
                    c for c in candidates
                    if int(c["min_cost"]) > 0 and int(c["min_cost"]) <= int(step_budget)
                ]
                affordable.sort(key=lambda c: (float(c["score"]), float(c["abs_theta"])), reverse=True)
                packable_terms = 0
                running_min_cost = 0
                for candidate in affordable:
                    next_cost = running_min_cost + int(candidate["min_cost"])
                    if next_cost > int(step_budget):
                        break
                    running_min_cost = next_cost
                    packable_terms += 1
                target_terms = default_term_target(packable_terms)
                max_packable_terms = max(max_packable_terms, packable_terms)
                max_target_terms = max(max_target_terms, target_terms)
                selected = affordable[:target_terms]
                theta_proj = np.zeros(len(keys), dtype=float)
                emitted_steps = 0
                kept_terms = 0
                weights = np.array([max(float(c["score"]), 1e-30) for c in selected], dtype=float)
                if selected and (not np.isfinite(weights).all() or float(np.sum(weights)) <= 0.0):
                    weights = np.ones(len(selected), dtype=float)
                for idx_sel, candidate in enumerate(selected):
                    remaining = int(step_budget) - int(emitted_steps)
                    remaining_terms = max(1, len(selected) - idx_sel)
                    slot = remaining if idx_sel == len(selected) - 1 else max(
                        int(candidate["min_cost"]),
                        int(np.floor(remaining * float(weights[idx_sel]) / float(np.sum(weights[idx_sel:])))),
                    )
                    slot = min(slot, remaining)
                    theta_keep = fit_theta_to_steps(candidate["key"], float(candidate["theta"]), slot)
                    if abs(theta_keep) <= threshold:
                        continue
                    keep_cost = term_cost(candidate["key"], theta_keep)
                    if keep_cost <= 0 or keep_cost > remaining:
                        continue
                    theta_proj[int(candidate["idx"])] = float(theta_keep)
                    emitted_steps += int(keep_cost)
                    kept_terms += 1
                if alpha < 1.0:
                    theta_proj = (1.0 - alpha) * theta_vec + alpha * theta_proj
                coeff_target = theta_proj / float(opt_obj.dt)
                opt_obj.u[k, d4_indices] = A4_pinv @ coeff_target
                projected += 1

            l1_after = float(np.sum(np.abs(theta_proj)))
            max_l1_after = max(max_l1_after, l1_after)
            max_emitted_steps = max(max_emitted_steps, emitted_steps)
            min_kept_terms = kept_terms if min_kept_terms is None else min(min_kept_terms, kept_terms)
            max_kept_terms = max(max_kept_terms, kept_terms)

        stats["last_call_count"] = int(call_state["count"])
        stats["last_relaxation_alpha"] = float(alpha)
        stats["last_best_eligible"] = bool(best_eligible)
        stats["cost_cache_entries"] = int(len(cost_cache))
        stats["last_max_requested_steps"] = int(max_requested_steps)
        stats["last_max_emitted_steps"] = int(max_emitted_steps)
        stats["last_max_candidate_terms"] = int(max_candidate_terms)
        stats["last_max_packable_terms"] = int(max_packable_terms)
        stats["last_max_target_terms"] = int(max_target_terms)
        stats["last_min_kept_terms"] = int(min_kept_terms or 0)
        stats["last_max_kept_terms"] = int(max_kept_terms)
        stats["last_max_l1_before"] = float(max_l1_before)
        stats["last_max_l1_after"] = float(max_l1_after)
        stats["last_projected_slices"] = int(projected)

    opt.post_update_projector = projector
    setattr(projector, "accepts_gradient", True)
    projector(opt)
    opt.ea_degree4_projectability_summary = stats
    return stats


def configure_ea_shared_product_scheduler(
    opt: ImprovedPolynomialPontryagin,
    *,
    T: float,
    Nt: int,
    S: int,
    hard_amp: float,
    drift_strength_hw: float,
    product_time_scale: float = 1.0,
    product_bch_reps: int = 1,
    total_budget_frac: float = 0.95,
    degree4_budget_frac: float = 0.30,
    degree4_min_terms_per_slice: int = 1,
    trim_degree2: bool = True,
    degree3_threshold: float = 1e-8,
    degree4_threshold: float = 1e-10,
    pauli_tol: float = 1e-10,
) -> dict[str, Any]:
    """Final shared per-slice scheduler for constructive product projection.

    The separate degree projectors make each layer small, but the physical
    compiler emits one serial microstep string.  This final projector mirrors
    that ordering at the EA level: degree-1 terms are treated as fixed local
    work, degree-2 terms are optionally trimmed only if they overrun the shared
    budget, and degree-3/degree-4 terms are selected together so the final
    p=4 content is schedulable after lower-degree work has been accounted for.
    """
    if int(opt.d) != 2:
        raise ValueError("Shared product scheduler is implemented for qubits only")
    if int(S) <= 0:
        raise ValueError("S must be positive for shared product scheduling")
    if int(opt.N) < 2:
        raise ValueError("Shared product scheduler requires at least two qubits")

    expansions, _rows, expansion_summary = build_exact_pauli_expansion_audit(opt, tol=float(pauli_tol))
    dt_micro = float(T) / (max(1, int(Nt)) * max(1, int(S)))
    total_budget = int(max(0, np.floor(float(total_budget_frac) * int(S))))
    d4_budget_cap = int(max(0, np.floor(float(degree4_budget_frac) * int(S))))
    scale = float(product_time_scale)
    previous_projector = getattr(opt, "post_update_projector", None)

    degree_data: dict[int, dict[str, Any]] = {}
    for degree in (2, 3, 4):
        if int(opt.N) < degree:
            continue
        keys = pair_axis_keys_for_N(int(opt.N)) if degree == 2 else sparse_pauli_keys_of_degree(int(opt.N), degree)
        indices, A, A_pinv = control_matrix_from_expansions_for_keys(
            expansions,
            keys=keys,
            pauli_tol=float(pauli_tol),
        )
        if indices:
            degree_data[int(degree)] = {
                "keys": keys,
                "indices": indices,
                "A": A,
                "A_pinv": A_pinv,
                "threshold": float(degree3_threshold if degree == 3 else degree4_threshold if degree == 4 else 1e-10),
            }

    if 2 not in degree_data:
        raise ValueError("Shared scheduler requires degree-2 Pauli controls")

    def term_threshold(degree: int) -> float:
        if degree == 3:
            return max(0.0, float(degree3_threshold))
        if degree == 4:
            return max(0.0, float(degree4_threshold))
        return 1e-10

    def term_cost(degree: int, key: tuple[tuple[int, str], ...], theta: float) -> int:
        if abs(float(theta)) <= term_threshold(degree):
            return 0
        steps, _status = estimate_pauli_product_steps(
            pauli_key_to_axes_dict(key),
            float(theta),
            dt_micro,
            float(hard_amp),
            int(opt.dim),
            max_degree=max(1, int(degree)),
            bch_reps=int(product_bch_reps),
            instant_overhead=False,
            drift_strength=float(drift_strength_hw),
        )
        return int(max(0, steps))

    min_term_costs: dict[int, dict[tuple[tuple[int, str], ...], int]] = {}
    for degree, data in degree_data.items():
        probe = max(2.0 * term_threshold(degree), 1e-14)
        min_term_costs[degree] = {
            key: term_cost(degree, key, probe)
            for key in data["keys"]
        }

    def theta_vec_for_degree(opt_obj: ImprovedPolynomialPontryagin, k: int, degree: int) -> np.ndarray:
        data = degree_data[degree]
        return (data["A"] @ np.asarray(opt_obj.u[k, data["indices"]], dtype=float)) * float(opt_obj.dt) * scale

    def set_theta_vec_for_degree(opt_obj: ImprovedPolynomialPontryagin, k: int, degree: int, theta_vec: np.ndarray) -> None:
        data = degree_data[degree]
        denom = max(1e-30, float(opt_obj.dt) * scale)
        coeff_target = np.asarray(theta_vec, dtype=float) / denom
        opt_obj.u[k, data["indices"]] = data["A_pinv"] @ coeff_target

    def one_body_terms(opt_obj: ImprovedPolynomialPontryagin, k: int) -> list[tuple[tuple[tuple[int, str], ...], float]]:
        terms: dict[tuple[tuple[int, str], ...], float] = {}
        for j, amp in enumerate(np.asarray(opt_obj.u[k, :], dtype=float)):
            if abs(float(amp)) < 1e-12:
                continue
            for key, coeff in expansions[j]:
                if pauli_key_degree(key) != 1 or abs(np.imag(coeff)) > 1e-8:
                    continue
                theta = scale * float(amp) * float(opt_obj.dt) * float(np.real(coeff))
                if abs(theta) <= term_threshold(1):
                    continue
                terms[key] = terms.get(key, 0.0) + theta
        return sorted(terms.items(), key=lambda item: -abs(float(item[1])))

    def candidate_list(
        degree: int,
        theta_vec: np.ndarray,
        grad_theta_vec: np.ndarray | None = None,
    ) -> list[dict[str, Any]]:
        data = degree_data[degree]
        out: list[dict[str, Any]] = []
        for idx, theta in enumerate(np.asarray(theta_vec, dtype=float)):
            theta_f = float(theta)
            if abs(theta_f) <= term_threshold(degree):
                continue
            key = data["keys"][idx]
            cost = term_cost(degree, key, theta_f)
            if cost <= 0:
                continue
            min_cost = int(min_term_costs.get(degree, {}).get(key, 0))
            grad_gain = 0.0
            if grad_theta_vec is not None and idx < int(grad_theta_vec.size):
                grad_gain = max(0.0, float(np.sign(theta_f) * grad_theta_vec[idx]))
            score = abs(theta_f) * (1.0 + grad_gain / (1.0 + abs(grad_gain)))
            out.append(
                {
                    "idx": int(idx),
                    "key": key,
                    "theta": theta_f,
                    "abs_theta": abs(theta_f),
                    "cost": int(cost),
                    "min_cost": int(min_cost),
                    "score": float(score),
                    "efficiency": float(score) / float(max(1, cost)),
                    "degree": int(degree),
                }
            )
        return out

    def fit_theta_to_steps(degree: int, key: tuple[tuple[int, str], ...], theta: float, budget: int) -> float:
        threshold = term_threshold(degree)
        if int(budget) <= 0 or abs(float(theta)) <= threshold:
            return 0.0
        min_cost = int(min_term_costs.get(degree, {}).get(key, 0))
        if min_cost > int(budget):
            return 0.0
        full_cost = term_cost(degree, key, theta)
        if full_cost > 0 and full_cost <= int(budget):
            return float(theta)
        lo = 0.0
        hi = abs(float(theta))
        sign = 1.0 if float(theta) >= 0.0 else -1.0
        best = 0.0
        for _ in range(32):
            mid = 0.5 * (lo + hi)
            if mid <= threshold:
                break
            cost = term_cost(degree, key, sign * mid)
            if cost > 0 and cost <= int(budget):
                best = mid
                lo = mid
            else:
                hi = mid
        return sign * best if best > threshold else 0.0

    def full_cost(candidates: Sequence[dict[str, Any]]) -> int:
        return int(sum(int(c["cost"]) for c in candidates))

    def project_candidates(
        degree: int,
        candidates: list[dict[str, Any]],
        budget: int,
        *,
        max_terms: int = 0,
        prefer_efficiency: bool = True,
    ) -> tuple[np.ndarray, int, int, float]:
        keys = degree_data[degree]["keys"]
        theta_proj = np.zeros(len(keys), dtype=float)
        if int(budget) <= 0 or not candidates:
            return theta_proj, 0, 0, 0.0

        if full_cost(candidates) <= int(budget) and (int(max_terms) <= 0 or len(candidates) <= int(max_terms)):
            for c in candidates:
                theta_proj[int(c["idx"])] = float(c["theta"])
            return theta_proj, full_cost(candidates), len(candidates), float(sum(float(c["score"]) for c in candidates))

        affordable = [
            c for c in candidates
            if int(c["min_cost"]) > 0 and int(c["min_cost"]) <= int(budget)
        ]
        if prefer_efficiency:
            affordable.sort(key=lambda c: (float(c["efficiency"]), float(c["score"]), float(c["abs_theta"])), reverse=True)
        else:
            affordable.sort(key=lambda c: (float(c["score"]), float(c["abs_theta"]), -float(c["cost"])), reverse=True)
        if not affordable:
            return theta_proj, 0, 0, 0.0

        if int(max_terms) > 0:
            target_terms = min(int(max_terms), len(affordable))
        else:
            packable = 0
            running = 0
            for c in affordable:
                nxt = running + int(c["min_cost"])
                if nxt > int(budget):
                    break
                running = nxt
                packable += 1
            target_terms = max(1, min(packable, int(np.ceil(np.sqrt(max(1, packable))))))
        selected = affordable[:target_terms]
        if not selected:
            return theta_proj, 0, 0, 0.0

        weights = np.array([max(float(c["score"]), 1e-30) for c in selected], dtype=float)
        if not np.isfinite(weights).all() or float(np.sum(weights)) <= 0.0:
            weights = np.ones(len(selected), dtype=float)

        emitted_steps = 0
        kept_terms = 0
        benefit = 0.0
        for i, c in enumerate(selected):
            remaining = int(budget) - int(emitted_steps)
            if remaining <= 0:
                break
                                                                           
                                                             
            if i == len(selected) - 1:
                slot = remaining
            else:
                proportional = int(np.floor(remaining * float(weights[i]) / float(np.sum(weights[i:]))))
                slot = max(int(c["min_cost"]), proportional)
                slot = min(slot, remaining)
            theta_keep = fit_theta_to_steps(degree, c["key"], float(c["theta"]), slot)
            if abs(theta_keep) <= term_threshold(degree):
                continue
            cost_keep = term_cost(degree, c["key"], theta_keep)
            if cost_keep <= 0 or cost_keep > remaining:
                continue
            theta_proj[int(c["idx"])] = float(theta_keep)
            emitted_steps += int(cost_keep)
            kept_terms += 1
            benefit += float(c["score"]) * abs(float(theta_keep))
        return theta_proj, int(emitted_steps), int(kept_terms), float(benefit)

    stats = {
        "mode": "ea_shared_product_scheduler",
        "projector": "shared_constructive_product_budget",
        "enabled": True,
        "N": int(opt.N),
        "S": int(S),
        "total_budget_frac": float(total_budget_frac),
        "total_budget": int(total_budget),
        "degree4_budget_frac": float(degree4_budget_frac),
        "degree4_budget_cap": int(d4_budget_cap),
        "degree4_min_terms_per_slice": int(degree4_min_terms_per_slice),
        "trim_degree2": bool(trim_degree2),
        "product_bch_reps": int(product_bch_reps),
        "pauli_expansion_summary": expansion_summary,
        "degree_control_counts": {str(d): int(len(data["indices"])) for d, data in degree_data.items()},
        "last_projected_slices": 0,
        "last_max_estimated_used_steps": 0,
        "last_max_fixed_lower_steps": 0,
        "last_min_remaining_after_lower": int(total_budget),
        "last_min_remaining_final": int(total_budget),
        "last_max_d2_steps": 0,
        "last_max_d3_steps": 0,
        "last_max_d4_steps": 0,
        "last_min_d4_kept_terms": 0,
        "last_max_d4_kept_terms": 0,
        "last_min_d3_kept_terms": 0,
        "last_max_d3_kept_terms": 0,
    }

    def projector(opt_obj: ImprovedPolynomialPontryagin, grad: np.ndarray | None = None) -> None:
        if previous_projector is not None:
            if getattr(previous_projector, "accepts_gradient", False):
                previous_projector(opt_obj, grad)
            else:
                previous_projector(opt_obj)

        projected = 0
        max_estimated_used = 0
        max_fixed_lower = 0
        min_remaining_after_lower: int | None = None
        min_remaining_final: int | None = None
        max_d2_steps = 0
        max_d3_steps = 0
        max_d4_steps = 0
        min_d4_kept: int | None = None
        max_d4_kept = 0
        min_d3_kept: int | None = None
        max_d3_kept = 0

        for k in range(int(opt_obj.Nt)):
            one_terms = one_body_terms(opt_obj, k)
            one_cost = sum(term_cost(1, key, theta) for key, theta in one_terms)
            remaining = int(total_budget) - int(one_cost)

            theta2 = theta_vec_for_degree(opt_obj, k, 2)
            d2_candidates = candidate_list(2, theta2)
            d2_cost = full_cost(d2_candidates)
            if bool(trim_degree2) and d2_cost > max(0, remaining):
                theta2_proj, d2_cost, _d2_kept, _benefit = project_candidates(
                    2,
                    d2_candidates,
                    max(0, remaining),
                    max_terms=0,
                    prefer_efficiency=False,
                )
                set_theta_vec_for_degree(opt_obj, k, 2, theta2_proj)
                projected += 1
            remaining -= int(d2_cost)
            max_d2_steps = max(max_d2_steps, int(d2_cost))
            fixed_lower = int(one_cost) + int(d2_cost)
            max_fixed_lower = max(max_fixed_lower, fixed_lower)
            min_remaining_after_lower = remaining if min_remaining_after_lower is None else min(min_remaining_after_lower, remaining)

            theta3 = theta_vec_for_degree(opt_obj, k, 3) if 3 in degree_data else np.zeros(0)
            theta4 = theta_vec_for_degree(opt_obj, k, 4) if 4 in degree_data else np.zeros(0)

            grad3 = None
            grad4 = None
            if grad is not None:
                if 3 in degree_data:
                    data3 = degree_data[3]
                    grad3 = (data3["A_pinv"].T @ np.asarray(grad[k, data3["indices"]], dtype=float)) / max(1e-30, float(opt_obj.dt) * scale)
                if 4 in degree_data:
                    data4 = degree_data[4]
                    grad4 = (data4["A_pinv"].T @ np.asarray(grad[k, data4["indices"]], dtype=float)) / max(1e-30, float(opt_obj.dt) * scale)

            d3_candidates = candidate_list(3, theta3, grad3) if 3 in degree_data else []
            d4_candidates = candidate_list(4, theta4, grad4) if 4 in degree_data else []

            theta4_proj = np.zeros_like(theta4)
            d4_steps = 0
            d4_kept = 0
            if remaining > 0 and d4_candidates and int(degree4_min_terms_per_slice) > 0:
                d4_budget = min(int(remaining), int(d4_budget_cap) if int(d4_budget_cap) > 0 else int(remaining))
                theta4_proj, d4_steps, d4_kept, _benefit = project_candidates(
                    4,
                    d4_candidates,
                    d4_budget,
                    max_terms=int(degree4_min_terms_per_slice),
                    prefer_efficiency=True,
                )
                remaining -= int(d4_steps)
                set_theta_vec_for_degree(opt_obj, k, 4, theta4_proj)
                projected += 1
            elif 4 in degree_data:
                                                                            
                                                                       
                set_theta_vec_for_degree(opt_obj, k, 4, theta4_proj)
                if d4_candidates:
                    projected += 1

            theta3_proj = np.zeros_like(theta3)
            d3_steps = 0
            d3_kept = 0
            if remaining > 0 and d3_candidates:
                theta3_proj, d3_steps, d3_kept, _benefit = project_candidates(
                    3,
                    d3_candidates,
                    int(remaining),
                    max_terms=0,
                    prefer_efficiency=True,
                )
                set_theta_vec_for_degree(opt_obj, k, 3, theta3_proj)
                remaining -= int(d3_steps)
                projected += 1
            elif 3 in degree_data:
                set_theta_vec_for_degree(opt_obj, k, 3, theta3_proj)
                if d3_candidates:
                    projected += 1

            if 4 in degree_data and int(degree4_min_terms_per_slice) <= 0 and remaining > 0 and d4_candidates:
                theta4_proj, d4_steps, d4_kept, _benefit = project_candidates(
                    4,
                    d4_candidates,
                    int(remaining),
                    max_terms=0,
                    prefer_efficiency=True,
                )
                set_theta_vec_for_degree(opt_obj, k, 4, theta4_proj)
                remaining -= int(d4_steps)
                projected += 1

            max_d3_steps = max(max_d3_steps, int(d3_steps))
            max_d4_steps = max(max_d4_steps, int(d4_steps))
            min_d3_kept = d3_kept if min_d3_kept is None else min(min_d3_kept, d3_kept)
            max_d3_kept = max(max_d3_kept, int(d3_kept))
            min_d4_kept = d4_kept if min_d4_kept is None else min(min_d4_kept, d4_kept)
            max_d4_kept = max(max_d4_kept, int(d4_kept))
            used = int(total_budget) - int(remaining)
            max_estimated_used = max(max_estimated_used, used)
            min_remaining_final = remaining if min_remaining_final is None else min(min_remaining_final, remaining)

        stats["last_projected_slices"] = int(projected)
        stats["last_max_estimated_used_steps"] = int(max_estimated_used)
        stats["last_max_fixed_lower_steps"] = int(max_fixed_lower)
        stats["last_min_remaining_after_lower"] = int(min_remaining_after_lower if min_remaining_after_lower is not None else total_budget)
        stats["last_min_remaining_final"] = int(min_remaining_final if min_remaining_final is not None else total_budget)
        stats["last_max_d2_steps"] = int(max_d2_steps)
        stats["last_max_d3_steps"] = int(max_d3_steps)
        stats["last_max_d4_steps"] = int(max_d4_steps)
        stats["last_min_d4_kept_terms"] = int(min_d4_kept or 0)
        stats["last_max_d4_kept_terms"] = int(max_d4_kept)
        stats["last_min_d3_kept_terms"] = int(min_d3_kept or 0)
        stats["last_max_d3_kept_terms"] = int(max_d3_kept)

    opt.post_update_projector = projector
    setattr(projector, "accepts_gradient", True)
    projector(opt)
    opt.ea_shared_scheduler_summary = stats
    return stats


def build_exact_pauli_expansion_audit(
    opt: ImprovedPolynomialPontryagin,
    *,
    tol: float = 1e-10,
) -> tuple[list[list[tuple[tuple[tuple[int, str], ...], complex]]], list[dict[str, Any]], dict[str, Any]]:
    """Build Pauli expansions for all EA controls and parser cross-check rows.

    The generated qubit EA controls are sparse Pauli products whose labels
    encode the product exactly, for example ``Re(H0_s0*H1_s2)``.  Parsing those
    labels is orders of magnitude cheaper than dense Hilbert-Schmidt expansion
    over all 4**N Pauli strings.  We therefore use the label parser as the
    default path and fall back to the exact dense expansion only for labels that
    cannot be parsed.
    """
    if int(opt.d) != 2:
        raise ValueError("constructive_batch2 exact Pauli expansion is implemented for qubits only")
    expansions: list[list[tuple[tuple[tuple[int, str], ...], complex]]] = []
    rows: list[dict[str, Any]] = []
    mismatches = 0
    max_residual = 0.0
    parser_rows = 0
    exact_fallback_rows = 0
    D = int(opt.dim)
    for idx, (op, label) in enumerate(zip(opt.controls, opt.control_labels)):
        parsed = parse_ea_label_to_pauli_string(str(label))
        parser_key: tuple[tuple[int, str], ...] | None = None
        parser_coeff: complex | None = None
        if parsed is not None:
            phase, axes_by_site = parsed
            parser_key = canonical_pauli_key(axes_by_site)
            parser_coeff = complex(phase)
            expansion = [(parser_key, complex(parser_coeff))]
            expansions.append(expansion)
            top_key = parser_key
            top_coeff = complex(parser_coeff)
            parser_match = True
            residual = 0.0
            parser_rows += 1
        else:
            expansion = exact_pauli_expansion(op, int(opt.N), tol=tol)
            expansions.append(expansion)

            recon = np.zeros((D, D), dtype=complex)
            for key, coeff in expansion:
                recon += coeff * pauli_product_dense_from_key(int(opt.N), key) / float(D)
            residual = float(np.linalg.norm(op.full() - recon))
            max_residual = max(max_residual, residual)

            top_key: tuple[tuple[int, str], ...] = tuple()
            top_coeff: complex = 0.0 + 0.0j
            if expansion:
                top_key, top_coeff = max(expansion, key=lambda item: abs(item[1]))
            parser_match = len(top_key) == 0
            exact_fallback_rows += 1
        if not parser_match:
            mismatches += 1
        rows.append(
            {
                "control_index": int(idx),
                "label": str(label),
                "terms": int(len(expansion)),
                "top_pauli": pauli_key_to_string(top_key),
                "top_coeff_re": float(np.real(top_coeff)),
                "top_coeff_im": float(np.imag(top_coeff)),
                "parser_pauli": pauli_key_to_string(parser_key or tuple()),
                "parser_coeff_re": float(np.real(parser_coeff)) if parser_coeff is not None else 0.0,
                "parser_coeff_im": float(np.imag(parser_coeff)) if parser_coeff is not None else 0.0,
                "parser_match": int(bool(parser_match)),
                "reconstruction_residual_fro": residual,
            }
        )
    summary = {
        "mode": "label_fast_exact_fallback",
        "control_count": int(len(opt.controls)),
        "parser_rows": int(parser_rows),
        "exact_fallback_rows": int(exact_fallback_rows),
        "parser_mismatches": int(mismatches),
        "max_reconstruction_residual_fro": float(max_residual),
        "tol": float(tol),
    }
    return expansions, rows, summary


def emit_parallel_control_areas(
    u_slice: np.ndarray,
    start: int,
    control_areas: Sequence[tuple[int | None, float]],
    dt_micro: float,
    amp_bound: float,
) -> int:
    """Emit commuting local-control pulse areas in parallel."""
    merged: dict[int, float] = {}
    for ctrl, area in control_areas:
        if ctrl is None or abs(float(area)) < 1e-14:
            continue
        merged[int(ctrl)] = merged.get(int(ctrl), 0.0) + float(area)
    if not merged:
        return 0
    if dt_micro <= 0 or amp_bound <= 0:
        return 0
    steps = max(1, int(np.ceil(max(abs(area) for area in merged.values()) / (float(amp_bound) * float(dt_micro)))))
    if start + steps > u_slice.shape[0]:
        return 0
    for ctrl, area in merged.items():
        amp = float(area) / (steps * float(dt_micro))
        if abs(amp) > float(amp_bound) + 1e-12:
            return 0
        u_slice[start : start + steps, ctrl] += amp
    return steps


def rotation_angles_for_pre_z_to_vector(vec: np.ndarray) -> tuple[float, float]:
    """Return X/Y pulse angles for P^dag Z P = vec.sigma.

    The pre-pulse unitary is P=R_y(beta) R_x(alpha).  Emitting X(alpha)
    followed by Y(beta) realizes this P in chronological order.
    """
    n = np.asarray(vec, dtype=float).reshape(3)
    norm = float(np.linalg.norm(n))
    if norm <= 1e-14:
        raise ValueError("zero frame vector")
    n = n / norm
    nx, ny, nz = [float(x) for x in n]
    nx = max(-1.0, min(1.0, nx))
    beta = -float(np.arcsin(nx))
    cb = float(np.cos(beta))
    alpha = 0.0 if abs(cb) < 1e-12 else float(np.arctan2(ny, nz))
    return alpha, beta


def emit_frame_rotations(
    u_slice: np.ndarray,
    start: int,
    ctrl_labels: Sequence[str],
    vectors_by_site: dict[int, np.ndarray],
    *,
    inverse: bool,
    dt_micro: float,
    amp_bound: float,
    dim: int,
) -> tuple[int, str]:
    """Emit parallel local XY rotations into or out of a drift frame."""
    sqrtD = float(np.sqrt(dim))
    pos = int(start)
    x_areas: list[tuple[int | None, float]] = []
    y_areas: list[tuple[int | None, float]] = []
    for site, vec in vectors_by_site.items():
        alpha, beta = rotation_angles_for_pre_z_to_vector(vec)
        ctrl_x = physical_axis_index(ctrl_labels, int(site), "X")
        ctrl_y = physical_axis_index(ctrl_labels, int(site), "Y")
        if ctrl_x is None or ctrl_y is None:
            return 0, "missing_xy_frame_rotation"
        x_areas.append((ctrl_x, alpha * sqrtD / 2.0))
        y_areas.append((ctrl_y, beta * sqrtD / 2.0))
    if inverse:
        used = emit_parallel_control_areas(u_slice, pos, [(ctrl, -area) for ctrl, area in y_areas], dt_micro, amp_bound)
        if used == 0 and any(abs(area) > 1e-14 for _ctrl, area in y_areas):
            return 0, "insufficient_budget_frame_y_inverse"
        pos += used
        used = emit_parallel_control_areas(u_slice, pos, [(ctrl, -area) for ctrl, area in x_areas], dt_micro, amp_bound)
        if used == 0 and any(abs(area) > 1e-14 for _ctrl, area in x_areas):
            return 0, "insufficient_budget_frame_x_inverse"
        pos += used
    else:
        used = emit_parallel_control_areas(u_slice, pos, x_areas, dt_micro, amp_bound)
        if used == 0 and any(abs(area) > 1e-14 for _ctrl, area in x_areas):
            return 0, "insufficient_budget_frame_x"
        pos += used
        used = emit_parallel_control_areas(u_slice, pos, y_areas, dt_micro, amp_bound)
        if used == 0 and any(abs(area) > 1e-14 for _ctrl, area in y_areas):
            return 0, "insufficient_budget_frame_y"
        pos += used
    return pos - int(start), "ok"


def emit_drift_frame(
    u_slice: np.ndarray,
    start: int,
    ctrl_labels: Sequence[str],
    vectors_by_site: dict[int, np.ndarray],
    duration: float,
    dt_micro: float,
    amp_bound: float,
    dim: int,
) -> tuple[int, float, str]:
    """Emit one rotated-drift frame and return used steps plus actual drift duration."""
    if duration <= 1e-14:
        return 0, 0.0, "zero_duration"
    original = u_slice.copy()
    pos = int(start)
    used, status = emit_frame_rotations(
        u_slice,
        pos,
        ctrl_labels,
        vectors_by_site,
        inverse=False,
        dt_micro=dt_micro,
        amp_bound=amp_bound,
        dim=dim,
    )
    if used == 0 and status != "ok":
        u_slice[:, :] = original
        return 0, 0.0, status
    pos += used
    drift_steps = max(1, int(round(float(duration) / float(dt_micro))))
    if pos + drift_steps > u_slice.shape[0]:
        u_slice[:, :] = original
        return 0, 0.0, "insufficient_budget_drift_frame"
    pos += drift_steps
    actual_duration = drift_steps * float(dt_micro)
    used, status = emit_frame_rotations(
        u_slice,
        pos,
        ctrl_labels,
        vectors_by_site,
        inverse=True,
        dt_micro=dt_micro,
        amp_bound=amp_bound,
        dim=dim,
    )
    if used == 0 and status != "ok":
        u_slice[:, :] = original
        return 0, 0.0, status
    pos += used
    return pos - int(start), actual_duration, "ok"


def n2_theta_matrix_from_terms(theta_by_key: dict[tuple[tuple[int, str], ...], float]) -> np.ndarray:
    matrix = np.zeros((3, 3), dtype=float)
    axis_index = {axis: i for i, axis in enumerate(PAULI_AXES)}
    for key, theta in theta_by_key.items():
        if len(key) != 2:
            continue
        (s0, a0), (s1, a1) = key
        if {int(s0), int(s1)} != {0, 1}:
            continue
        if int(s0) == 0:
            left_axis, right_axis = str(a0), str(a1)
        else:
            left_axis, right_axis = str(a1), str(a0)
        matrix[axis_index[left_axis], axis_index[right_axis]] += float(theta)
    return matrix


def theta_terms_from_n2_matrix(matrix: np.ndarray) -> dict[tuple[tuple[int, str], ...], float]:
    out: dict[tuple[tuple[int, str], ...], float] = {}
    for i, a0 in enumerate(PAULI_AXES):
        for j, a1 in enumerate(PAULI_AXES):
            val = float(matrix[i, j])
            if abs(val) > 1e-14:
                out[((0, a0), (1, a1))] = val
    return out


def emit_n2_batch2_svd(
    u_slice: np.ndarray,
    start: int,
    ctrl_labels: Sequence[str],
    theta_matrix: np.ndarray,
    *,
    drift_strength_hw: float,
    dt_micro: float,
    amp_bound: float,
    dim: int,
    max_frames: int,
    frame_tol: float,
) -> tuple[int, np.ndarray, dict[str, Any]]:
    """Synthesize the full two-qubit coupling matrix by SVD rank-one frames."""
    emitted = np.zeros((3, 3), dtype=float)
    drift_rate = float(drift_strength_hw) * float(dim) / 4.0
    if abs(drift_rate) <= 1e-14:
        return 0, emitted, {"status": "zero_drift_strength", "required_drift_time": float("inf"), "frames": 0}
    U, singular_vals, Vh = np.linalg.svd(np.asarray(theta_matrix, dtype=float), full_matrices=True)
    frames: list[tuple[float, np.ndarray, np.ndarray]] = []
    for idx, sigma in enumerate(singular_vals):
        if float(sigma) <= float(frame_tol):
            continue
        left = np.array(U[:, idx], dtype=float)
        right = np.array(Vh[idx, :], dtype=float)
        if drift_rate < 0:
            left = -left
        frames.append((float(sigma) / abs(drift_rate), left, right))
    frames = frames[: max(0, int(max_frames))]
    required_drift_time = float(sum(duration for duration, _left, _right in frames))
    pos = int(start)
    statuses: dict[str, int] = {}
    for duration, left, right in frames:
        used, actual_duration, status = emit_drift_frame(
            u_slice,
            pos,
            ctrl_labels,
            {0: left, 1: right},
            duration,
            dt_micro,
            amp_bound,
            dim,
        )
        if used <= 0:
            statuses[status] = statuses.get(status, 0) + 1
            return pos - int(start), emitted, {
                "status": status,
                "required_drift_time": required_drift_time,
                "frames": len(frames),
                "status_counts": statuses,
            }
        pos += used
        emitted += drift_rate * actual_duration * np.outer(left, right)
    return pos - int(start), emitted, {
        "status": "ok",
        "required_drift_time": required_drift_time,
        "frames": len(frames),
        "status_counts": statuses,
    }


def pair_axis_keys_for_N(N: int) -> list[tuple[tuple[int, str], ...]]:
    keys: list[tuple[tuple[int, str], ...]] = []
    for i in range(int(N)):
        for j in range(i + 1, int(N)):
            for ai in PAULI_AXES:
                for aj in PAULI_AXES:
                    keys.append(((i, ai), (j, aj)))
    return keys


def build_toggling_frame_dictionary(N: int) -> tuple[list[dict[int, np.ndarray]], np.ndarray, list[tuple[tuple[int, str], ...]]]:
    """Build global rotated-drift frames for all pair-axis degree-2 terms."""
    keys = pair_axis_keys_for_N(int(N))
    key_index = {key: idx for idx, key in enumerate(keys)}
    frames: list[dict[int, np.ndarray]] = []
    columns: list[np.ndarray] = []
                                                                
    for axes in product(PAULI_AXES, repeat=int(N)):
        for tail_signs in product((-1.0, 1.0), repeat=max(0, int(N) - 1)):
            signs = (1.0, *tail_signs)
            frame: dict[int, np.ndarray] = {}
            col = np.zeros(len(keys), dtype=float)
            for site, (axis, sign) in enumerate(zip(axes, signs)):
                frame[site] = float(sign) * PAULI_AXIS_VECTORS[str(axis)]
            for i in range(int(N)):
                for j in range(i + 1, int(N)):
                    key = ((i, str(axes[i])), (j, str(axes[j])))
                    col[key_index[key]] = float(signs[i] * signs[j])
            frames.append(frame)
            columns.append(col)
    return frames, np.stack(columns, axis=1), keys


def emit_nn_batch2_frames(
    u_slice: np.ndarray,
    start: int,
    ctrl_labels: Sequence[str],
    theta_by_key: dict[tuple[tuple[int, str], ...], float],
    *,
    N: int,
    drift_strength_hw: float,
    dt_micro: float,
    amp_bound: float,
    dim: int,
    max_frames: int,
    frame_tol: float,
) -> tuple[int, dict[tuple[tuple[int, str], ...], float], dict[str, Any]]:
    """Approximate all N-qubit degree-2 couplings by NNLS over toggling frames."""
    if abs(float(drift_strength_hw)) <= 1e-14:
        return 0, {}, {"status": "zero_drift_strength", "required_drift_time": float("inf"), "frames": 0}
    frames, A0, keys = build_toggling_frame_dictionary(int(N))
    b = np.array([float(theta_by_key.get(key, 0.0)) for key in keys], dtype=float)
    drift_rate = float(drift_strength_hw) * float(dim) / 4.0
    A = drift_rate * A0
    durations, residual_norm = nnls(A, b)
    selected = [(idx, float(duration)) for idx, duration in enumerate(durations) if float(duration) > float(frame_tol)]
    selected.sort(key=lambda item: item[1], reverse=True)
    selected = selected[: max(0, int(max_frames))]
    pos = int(start)
    emitted_vec = np.zeros_like(b)
    statuses: dict[str, int] = {}
    used_duration = 0.0
    emitted_frames = 0
    skipped_frames = 0
    for idx, duration in selected:
        used, actual_duration, status = emit_drift_frame(
            u_slice,
            pos,
            ctrl_labels,
            frames[idx],
            duration,
            dt_micro,
            amp_bound,
            dim,
        )
        if used <= 0:
            statuses[status] = statuses.get(status, 0) + 1
            skipped_frames += 1
            continue
        pos += used
        used_duration += float(actual_duration)
        emitted_frames += 1
        emitted_vec += drift_rate * actual_duration * A0[:, idx]
    emitted = {key: float(emitted_vec[i]) for i, key in enumerate(keys) if abs(float(emitted_vec[i])) > 1e-14}
    emitted_residual_norm = float(np.linalg.norm(b - emitted_vec))
    requested_norm = float(np.linalg.norm(b))
    return pos - int(start), emitted, {
        "status": "ok" if (not statuses and emitted_residual_norm <= max(float(frame_tol), 1e-10 * max(1.0, requested_norm))) else "partial",
        "required_drift_time": float(sum(duration for _idx, duration in selected)),
        "actual_drift_time": float(used_duration),
        "nnls_residual_norm": float(residual_norm),
        "emitted_residual_norm": float(emitted_residual_norm),
        "emitted_relative_residual": float(emitted_residual_norm / max(1e-30, requested_norm)),
        "requested_norm": float(requested_norm),
        "emitted_norm": float(np.linalg.norm(emitted_vec)),
        "frames": len(selected),
        "emitted_frames": int(emitted_frames),
        "skipped_frames": int(skipped_frames),
        "status_counts": statuses,
    }


def emit_nn_batch2_delta_frames(
    u_slice: np.ndarray,
    start: int,
    ctrl_labels: Sequence[str],
    theta_by_key: dict[tuple[tuple[int, str], ...], float],
    *,
    N: int,
    drift_strength_hw: float,
    dt_micro: float,
    amp_bound: float,
    dim: int,
    max_frames: int,
    frame_tol: float,
) -> tuple[int, dict[tuple[tuple[int, str], ...], float], dict[str, Any]]:
    """Synthesize N-qubit degree-2 EA corrections relative to free ZZ drift.

    The physical drift is always on.  If a rotated frame is used for duration
    tau, it occupies the interval that would otherwise use the free all-Z frame.
    Therefore the
    controllable degree-2 correction is proportional to

        tau * (rotated_frame - free_ZZ_frame),

    not to the rotated frame alone.  This baseline-relative model is the
    correct average-Hamiltonian target for compiling EA degree-2 controls on top
    of the native drift.
    """
    if abs(float(drift_strength_hw)) <= 1e-14:
        return 0, {}, {"status": "zero_drift_strength", "required_drift_time": float("inf"), "frames": 0}
    frames, A0, keys = build_toggling_frame_dictionary(int(N))
    b = np.array([float(theta_by_key.get(key, 0.0)) for key in keys], dtype=float)
    drift_rate = float(drift_strength_hw) * float(dim) / 4.0

    base_vec = np.zeros(len(keys), dtype=float)
    key_index = {key: idx for idx, key in enumerate(keys)}
    for i in range(int(N)):
        for j in range(i + 1, int(N)):
            idx = key_index.get(((i, "Z"), (j, "Z")))
            if idx is not None:
                base_vec[idx] = 1.0

    A_delta = drift_rate * (A0 - base_vec[:, None])
    durations, residual_norm = nnls(A_delta, b)
    selected = [(idx, float(duration)) for idx, duration in enumerate(durations) if float(duration) > float(frame_tol)]
    selected.sort(key=lambda item: item[1], reverse=True)
    selected = selected[: max(0, int(max_frames))]

    pos = int(start)
    emitted_vec = np.zeros_like(b)
    statuses: dict[str, int] = {}
    used_duration = 0.0
    emitted_frames = 0
    skipped_frames = 0
    for idx, duration in selected:
        used, actual_duration, status = emit_drift_frame(
            u_slice,
            pos,
            ctrl_labels,
            frames[idx],
            duration,
            dt_micro,
            amp_bound,
            dim,
        )
        if used <= 0:
            statuses[status] = statuses.get(status, 0) + 1
            skipped_frames += 1
            continue
        pos += used
        used_duration += float(actual_duration)
        emitted_frames += 1
        emitted_vec += drift_rate * actual_duration * (A0[:, idx] - base_vec)
    emitted = {key: float(emitted_vec[i]) for i, key in enumerate(keys) if abs(float(emitted_vec[i])) > 1e-14}
    emitted_residual_norm = float(np.linalg.norm(b - emitted_vec))
    requested_norm = float(np.linalg.norm(b))
    return pos - int(start), emitted, {
        "status": "ok" if (not statuses and emitted_residual_norm <= max(float(frame_tol), 1e-10 * max(1.0, requested_norm))) else "partial",
        "required_drift_time": float(sum(duration for _idx, duration in selected)),
        "actual_drift_time": float(used_duration),
        "nnls_residual_norm": float(residual_norm),
        "emitted_residual_norm": float(emitted_residual_norm),
        "emitted_relative_residual": float(emitted_residual_norm / max(1e-30, requested_norm)),
        "requested_norm": float(requested_norm),
        "emitted_norm": float(np.linalg.norm(emitted_vec)),
        "frames": len(selected),
        "emitted_frames": int(emitted_frames),
        "skipped_frames": int(skipped_frames),
        "status_counts": statuses,
        "mode": "baseline_delta_frames",
    }


def compile_ea_to_controls_constructive_batch2(
    opt: ImprovedPolynomialPontryagin,
    ctrl_labels: list[str],
    T: float,
    Nt: int,
    S: int,
    amp_bound: float,
    hard_amp: float,
    drift_strength_hw: float,
    product_time_scale: float = 1.0,
    term_theta_threshold: float = 1e-10,
    pauli_tol: float = 1e-10,
    residual_tol: float = 1e-6,
    max_frames: int = 64,
    frame_tol: float = 1e-10,
    product_bch_reps: int = 1,
    trotter_reps: int = 1,
    diagnostics: Optional[list[dict[str, Any]]] = None,
    term_diagnostics: Optional[list[dict[str, Any]]] = None,
    pauli_audit_rows: Optional[list[dict[str, Any]]] = None,
    pauli_expansion_cache: Optional[tuple[Any, list[dict[str, Any]], dict[str, Any]]] = None,
    reporter: Optional[ConsoleReporter] = None,
    progress_label: str = "Batch2Compiler",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Whole-slice constructive projection for degree-2 EA products.

    Degree-2 terms are aggregated over the entire slice and synthesized as a
    rotated-drift average Hamiltonian.
    """
    if int(opt.d) != 2:
        raise ValueError("constructive_batch2 is currently implemented for qubits only")
    if S <= 0:
        raise ValueError("S must be positive")
    n_ea_steps = int(opt.u.shape[0])
    if n_ea_steps != Nt:
        Nt = n_ea_steps
    n_phys = len(ctrl_labels)
    M = int(Nt) * int(S)
    dt_ea = float(T) / int(Nt)
    dt_micro = float(T) / M
    dim = int(opt.dim)
    u_out = np.zeros((M, n_phys), dtype=float)
    pulse_amp = float(max(1e-12, hard_amp))
    if amp_bound > 0:
        pulse_amp = min(pulse_amp, float(amp_bound))

    if pauli_expansion_cache is None:
        expansions, audit_rows, pauli_summary = build_exact_pauli_expansion_audit(opt, tol=float(pauli_tol))
    else:
        expansions, audit_rows, pauli_summary = pauli_expansion_cache
    if pauli_audit_rows is not None:
        pauli_audit_rows.extend(audit_rows)

    total_used = 0
    total_requested_weight = 0.0
    total_emitted_weight = 0.0
    t_start = time.time()
    serial_reps = max(1, int(trotter_reps))
    degree2_frame_rows: list[dict[str, Any]] = []

    def record_term(
        *,
        slice_index: int,
        key: tuple[tuple[int, str], ...],
        theta: float,
        emitted_theta: float,
        status: str,
        used_steps: int = 0,
    ) -> None:
        nonlocal total_requested_weight, total_emitted_weight
        requested = abs(float(theta))
        residual = abs(float(theta) - float(emitted_theta))
                                                                              
                                                                              
                                                                                
        term_tol = max(1e-14, float(residual_tol) * max(requested, 1e-14))
        emitted_flag = residual <= term_tol
        emitted_weight = max(0.0, requested - residual)
        total_requested_weight += requested
        total_emitted_weight += emitted_weight
        if term_diagnostics is None:
            return
        term_diagnostics.append(
            {
                "slice": int(slice_index),
                "rank": 0,
                "control_index": -1,
                "label": f"BATCH2:{pauli_key_to_string(key)}",
                "degree": int(pauli_key_degree(key)),
                "pauli_string": pauli_key_to_string(key),
                "theta": float(theta),
                "emitted_theta": float(emitted_theta),
                "residual_theta": float(float(theta) - float(emitted_theta)),
                "requested_weight": requested,
                "emitted_weight": float(emitted_weight),
                "retained": 1,
                "emitted": int(bool(emitted_flag)),
                "status": str(status if emitted_flag else f"{status}_residual"),
                "used_steps": int(used_steps),
                "required_steps": int(used_steps),
                "available_steps": int(S),
                "budget_before": 0,
                "budget_after": int(used_steps),
            }
        )

    for k in range(int(Nt)):
        u_slice = u_out[k * int(S) : (k + 1) * int(S), :]
        theta_by_key: dict[tuple[tuple[int, str], ...], float] = {}
        for j in range(len(opt.controls)):
            amp = float(opt.u[k, j])
            if abs(amp) < 1e-12:
                continue
            for key, coeff in expansions[j]:
                if abs(np.imag(coeff)) > 1e-8:
                    continue
                theta = float(product_time_scale) * amp * dt_ea * float(np.real(coeff))
                if abs(theta) < float(term_theta_threshold):
                    continue
                theta_by_key[key] = theta_by_key.get(key, 0.0) + theta

        degree1 = {key: theta for key, theta in theta_by_key.items() if pauli_key_degree(key) == 1}
        degree2 = {key: theta for key, theta in theta_by_key.items() if pauli_key_degree(key) == 2}
        degree3 = {key: theta for key, theta in theta_by_key.items() if pauli_key_degree(key) == 3}
        degree4 = {key: theta for key, theta in theta_by_key.items() if pauli_key_degree(key) == 4}
        unsupported = {key: theta for key, theta in theta_by_key.items() if pauli_key_degree(key) > 4}
        drift_degree2: dict[tuple[tuple[int, str], ...], float] = {}
        drift_theta = float(drift_strength_hw) * dt_ea * float(dim) / 4.0
        if abs(drift_theta) > 1e-14:
            for i_site in range(int(opt.N)):
                for j_site in range(i_site + 1, int(opt.N)):
                    drift_degree2[((i_site, "Z"), (j_site, "Z"))] = drift_theta
        total_degree2 = dict(drift_degree2)
        for key, theta in degree2.items():
            total_degree2[key] = total_degree2.get(key, 0.0) + float(theta)

        pos = 0
        degree1_status_counts: dict[str, int] = {}
        emitted_degree1_terms: dict[tuple[tuple[int, str], ...], tuple[float, int, str]] = {}
        deferred_degree1_bch: dict[tuple[tuple[int, str], ...], float] = {}
        if int(opt.N) > 2:
            for key, theta in sorted(degree1.items(), key=lambda item: -abs(item[1])):
                axes = pauli_key_to_axes_dict(key)
                used = 0
                status = "not_emitted"
                emitted_theta = 0.0
                if len(axes) == 1:
                    site, axis = next(iter(axes.items()))
                    ctrl = physical_axis_index(ctrl_labels, int(site), str(axis))
                    if ctrl is not None:
                                                                                
                                                                                 
                                                                                
                                                               
                        desired_amp = float(theta) / (float(np.sqrt(dim)) * dt_ea)
                        actual_amp = desired_amp
                        if pulse_amp > 0 and abs(actual_amp) > pulse_amp:
                            actual_amp = float(np.sign(actual_amp) * pulse_amp)
                            status = "constant_clipped"
                        else:
                            status = "constant_ok"
                        u_slice[:, int(ctrl)] += actual_amp
                        emitted_theta = float(actual_amp) * dt_ea * float(np.sqrt(dim))
                        used = 0
                    else:
                                                                              
                                                                              
                                                                 
                        deferred_degree1_bch[key] = float(theta)
                        continue
                degree1_status_counts[status] = degree1_status_counts.get(status, 0) + 1
                emitted_degree1_terms[key] = (float(emitted_theta), int(used), str(status))

        emitted_total_degree2: dict[tuple[tuple[int, str], ...], float] = {}
        batch_status: dict[str, Any] = {"status": "no_degree2", "required_drift_time": 0.0, "frames": 0}
        if total_degree2:
            if int(opt.N) == 2:
                theta_matrix = n2_theta_matrix_from_terms(total_degree2)
                used, emitted_matrix, batch_status = emit_n2_batch2_svd(
                    u_slice,
                    pos,
                    ctrl_labels,
                    theta_matrix,
                    drift_strength_hw=float(drift_strength_hw),
                    dt_micro=dt_micro,
                    amp_bound=pulse_amp,
                    dim=dim,
                    max_frames=int(max_frames),
                    frame_tol=float(frame_tol),
                )
                pos += max(0, int(used))
                emitted_total_degree2 = theta_terms_from_n2_matrix(emitted_matrix)
            else:
                sparse_serial_cutoff = max(1, int(max_frames) // 4)
                if int(opt.N) <= 3 and len(degree2) <= sparse_serial_cutoff:
                    emitted_total_degree2 = dict(drift_degree2)
                    status_counts: dict[str, int] = {}
                    drift_rate = float(drift_strength_hw) * float(dim) / 4.0
                    required_drift_time = 0.0
                    for _rep in range(serial_reps):
                        for key, theta in sorted(degree2.items(), key=lambda item: -abs(float(item[1]))):
                            theta_rep = float(theta) / float(serial_reps)
                            axes = pauli_key_to_axes_dict(key)
                            required_drift_time += abs(float(theta_rep)) / max(1e-30, abs(drift_rate))
                            before = pos
                            used, status = emit_pauli_product_bch_block(
                                u_slice,
                                pos,
                                ctrl_labels,
                                axes,
                                theta_rep,
                                dt_micro,
                                pulse_amp,
                                dim,
                                max_degree=2,
                                bch_reps=1,
                                drift_strength=float(drift_strength_hw),
                            )
                            status_counts[status] = status_counts.get(status, 0) + 1
                            if used <= 0:
                                pos = before
                                continue
                            pos += int(used)
                            emitted_total_degree2[key] = emitted_total_degree2.get(key, 0.0) + float(theta_rep)
                    batch_status = {
                        "status": "serial_pair_echo",
                        "required_drift_time": float(required_drift_time),
                        "frames": int(len(degree2)),
                        "status_counts": status_counts,
                        "mode": "sparse_serial_pair_echo",
                    }
                else:
                                                                            
                                                                         
                                                                           
                                                                            
                                              
                    used, emitted_delta_degree2, batch_status = emit_nn_batch2_delta_frames(
                        u_slice,
                        pos,
                        ctrl_labels,
                        degree2,
                        N=int(opt.N),
                        drift_strength_hw=float(drift_strength_hw),
                        dt_micro=dt_micro,
                        amp_bound=pulse_amp,
                        dim=dim,
                        max_frames=int(max_frames),
                        frame_tol=float(frame_tol),
                    )
                    pos += max(0, int(used))
                    emitted_total_degree2 = dict(drift_degree2)
                    for key, theta_delta in emitted_delta_degree2.items():
                        emitted_total_degree2[key] = emitted_total_degree2.get(key, 0.0) + float(theta_delta)
                emitted_degree2_corrections = {
                    key: float(emitted_total_degree2.get(key, 0.0)) - float(drift_degree2.get(key, 0.0))
                    for key in set(degree2) | set(emitted_total_degree2) | set(drift_degree2)
                }
                emitted_pair_count = sum(
                    1
                    for key, theta in degree2.items()
                    if abs(float(theta) - float(emitted_degree2_corrections.get(key, 0.0))) <= max(1e-14, float(residual_tol) * max(abs(float(theta)), 1e-14))
                )
                batch_status = dict(batch_status)
                if str(batch_status.get("mode", "")) == "sparse_serial_pair_echo":
                    batch_status["status"] = "pair_echo_ok" if emitted_pair_count == len(degree2) else "pair_echo_partial"
                else:
                    batch_status["status"] = "batch2_delta_ok" if emitted_pair_count == len(degree2) else "batch2_delta_partial"
                batch_status["emitted_pair_count"] = int(emitted_pair_count)
                batch_status["candidate_pair_count"] = int(len(degree2))
        emitted_degree2 = dict(emitted_total_degree2)
        for key, theta in drift_degree2.items():
            emitted_degree2[key] = emitted_degree2.get(key, 0.0) - float(theta)

        for key, theta in sorted(deferred_degree1_bch.items(), key=lambda item: -abs(item[1])):
            axes = pauli_key_to_axes_dict(key)
            total_used_z = 0
            emitted_z = 0.0
            status = "bch_local_z_partial"
            for _rep in range(serial_reps):
                before = pos
                used_rep, status_rep = emit_pauli_product_bch_block(
                    u_slice,
                    pos,
                    ctrl_labels,
                    axes,
                    float(theta) / float(serial_reps),
                    dt_micro,
                    pulse_amp,
                    dim,
                    max_degree=1,
                    bch_reps=1,
                    drift_strength=float(drift_strength_hw),
                )
                if used_rep <= 0:
                    pos = before
                    status = status_rep if emitted_z == 0.0 else "bch_local_z_partial"
                    break
                pos += int(used_rep)
                total_used_z += int(used_rep)
                emitted_z += float(theta) / float(serial_reps)
                status = status_rep
            final_status = (
                "bch_local_z_trotter"
                if abs(float(emitted_z) - float(theta)) <= max(1e-14, float(residual_tol) * max(abs(float(theta)), 1e-14))
                else str(status if emitted_z == 0.0 else "bch_local_z_partial")
            )
            degree1_status_counts[final_status] = degree1_status_counts.get(final_status, 0) + 1
            emitted_degree1_terms[key] = (float(emitted_z), int(total_used_z), str(final_status))

        degree3_status_counts: dict[str, int] = {}
        emitted_degree3: dict[tuple[tuple[int, str], ...], float] = {}
        degree3_term_status: dict[tuple[tuple[int, str], ...], str] = {}
        degree3_term_used: dict[tuple[tuple[int, str], ...], int] = {}
        degree3_used_steps = 0
        for key, theta in sorted(degree3.items(), key=lambda item: -abs(float(item[1]))):
            axes = pauli_key_to_axes_dict(key)
            emitted_theta = 0.0
            used = 0
            status = "zero_theta"
            for _rep in range(serial_reps):
                before = pos
                used_rep, status_rep = emit_pauli_product_bch_block(
                    u_slice,
                    pos,
                    ctrl_labels,
                    axes,
                    float(theta) / float(serial_reps),
                    dt_micro,
                    pulse_amp,
                    dim,
                    max_degree=3,
                    bch_reps=int(product_bch_reps),
                    drift_strength=float(drift_strength_hw),
                )
                degree3_status_counts[status_rep] = degree3_status_counts.get(status_rep, 0) + 1
                status = status_rep
                if used_rep <= 0:
                    pos = before
                    break
                pos += int(used_rep)
                used += int(used_rep)
                degree3_used_steps += int(used_rep)
                emitted_theta += float(theta) / float(serial_reps)
            emitted_degree3[key] = float(emitted_theta)
            degree3_term_status[key] = str(status)
            degree3_term_used[key] = int(used)

        degree4_status_counts: dict[str, int] = {}
        emitted_degree4: dict[tuple[tuple[int, str], ...], float] = {}
        degree4_term_status: dict[tuple[tuple[int, str], ...], str] = {}
        degree4_term_used: dict[tuple[tuple[int, str], ...], int] = {}
        degree4_used_steps = 0
        for key, theta in sorted(degree4.items(), key=lambda item: -abs(float(item[1]))):
            axes = pauli_key_to_axes_dict(key)
            emitted_theta = 0.0
            used = 0
            status = "zero_theta"
            for _rep in range(serial_reps):
                before = pos
                used_rep, status_rep = emit_pauli_product_bch_block(
                    u_slice,
                    pos,
                    ctrl_labels,
                    axes,
                    float(theta) / float(serial_reps),
                    dt_micro,
                    pulse_amp,
                    dim,
                    max_degree=4,
                    bch_reps=int(product_bch_reps),
                    drift_strength=float(drift_strength_hw),
                )
                degree4_status_counts[status_rep] = degree4_status_counts.get(status_rep, 0) + 1
                status = status_rep
                if used_rep <= 0:
                    pos = before
                    break
                pos += int(used_rep)
                used += int(used_rep)
                degree4_used_steps += int(used_rep)
                emitted_theta += float(theta) / float(serial_reps)
            emitted_degree4[key] = float(emitted_theta)
            degree4_term_status[key] = str(status)
            degree4_term_used[key] = int(used)

        if int(opt.N) <= 2:
            for key, theta in sorted(degree1.items(), key=lambda item: -abs(item[1])):
                axes = pauli_key_to_axes_dict(key)
                before = pos
                used, status = emit_pauli_product_bch_block(
                    u_slice,
                    pos,
                    ctrl_labels,
                    axes,
                    theta,
                    dt_micro,
                    pulse_amp,
                    dim,
                    max_degree=1,
                    bch_reps=1,
                    drift_strength=float(drift_strength_hw),
                )
                if used > 0:
                    pos += used
                    emitted_theta = theta
                else:
                    emitted_theta = 0.0
                degree1_status_counts[status] = degree1_status_counts.get(status, 0) + 1
                record_term(slice_index=k, key=key, theta=theta, emitted_theta=emitted_theta, status=status, used_steps=pos - before)
        else:
            for key, theta in sorted(degree1.items(), key=lambda item: -abs(item[1])):
                emitted_theta, used_steps, status = emitted_degree1_terms.get(key, (0.0, 0, "not_emitted"))
                record_term(slice_index=k, key=key, theta=theta, emitted_theta=emitted_theta, status=status, used_steps=used_steps)

        for key, theta in sorted(degree2.items(), key=lambda item: pauli_key_to_string(item[0])):
            record_term(
                slice_index=k,
                key=key,
                theta=theta,
                emitted_theta=float(emitted_degree2.get(key, 0.0)),
                status=str(batch_status.get("status", "batch2")),
                used_steps=pos,
            )
        for key, theta in sorted(degree3.items(), key=lambda item: pauli_key_to_string(item[0])):
            record_term(
                slice_index=k,
                key=key,
                theta=theta,
                emitted_theta=float(emitted_degree3.get(key, 0.0)),
                status=str(degree3_term_status.get(key, "degree3_bch")),
                used_steps=int(degree3_term_used.get(key, 0)),
            )
        for key, theta in sorted(degree4.items(), key=lambda item: pauli_key_to_string(item[0])):
            record_term(
                slice_index=k,
                key=key,
                theta=theta,
                emitted_theta=float(emitted_degree4.get(key, 0.0)),
                status=str(degree4_term_status.get(key, "degree4_bch")),
                used_steps=int(degree4_term_used.get(key, 0)),
            )
        for key, theta in sorted(unsupported.items(), key=lambda item: pauli_key_to_string(item[0])):
            record_term(slice_index=k, key=key, theta=theta, emitted_theta=0.0, status=f"unsupported_degree{pauli_key_degree(key)}")

        requested_vec = np.array(list(degree2.values()), dtype=float) if degree2 else np.zeros(0)
        residual_vec = np.array([float(theta) - float(emitted_degree2.get(key, 0.0)) for key, theta in degree2.items()], dtype=float) if degree2 else np.zeros(0)
        emitted_vec = np.array([float(emitted_degree2.get(key, 0.0)) for key in degree2.keys()], dtype=float) if degree2 else np.zeros(0)
        if str(batch_status.get("mode", "")) == "baseline_delta_frames" or str(batch_status.get("status", "")).startswith("batch2_delta"):
            degree2_frame_rows.append(
                {
                    "slice": int(k),
                    "degree2_terms": int(len(degree2)),
                    "status": str(batch_status.get("status", "unknown")),
                    "mode": str(batch_status.get("mode", "")),
                    "requested_norm": float(batch_status.get("requested_norm", np.linalg.norm(requested_vec))),
                    "emitted_norm": float(batch_status.get("emitted_norm", np.linalg.norm(emitted_vec))),
                    "emitted_residual_norm": float(batch_status.get("emitted_residual_norm", np.linalg.norm(residual_vec))),
                    "emitted_relative_residual": float(
                        batch_status.get(
                            "emitted_relative_residual",
                            float(np.linalg.norm(residual_vec)) / max(1e-30, float(np.linalg.norm(requested_vec))),
                        )
                    ),
                    "nnls_residual_norm": float(batch_status.get("nnls_residual_norm", 0.0)),
                    "required_drift_time": float(batch_status.get("required_drift_time", 0.0)),
                    "actual_drift_time": float(batch_status.get("actual_drift_time", 0.0)),
                    "available_slice_time": float(S) * dt_micro,
                    "frames": int(batch_status.get("frames", 0)),
                    "emitted_frames": int(batch_status.get("emitted_frames", 0)),
                    "skipped_frames": int(batch_status.get("skipped_frames", 0)),
                    "status_counts": dict(batch_status.get("status_counts", {})),
                }
            )
        requested_d3_vec = np.array(list(degree3.values()), dtype=float) if degree3 else np.zeros(0)
        residual_d3_vec = np.array([float(theta) - float(emitted_degree3.get(key, 0.0)) for key, theta in degree3.items()], dtype=float) if degree3 else np.zeros(0)
        emitted_d3_vec = np.array([float(emitted_degree3.get(key, 0.0)) for key in degree3.keys()], dtype=float) if degree3 else np.zeros(0)
        requested_d4_vec = np.array(list(degree4.values()), dtype=float) if degree4 else np.zeros(0)
        residual_d4_vec = np.array([float(theta) - float(emitted_degree4.get(key, 0.0)) for key, theta in degree4.items()], dtype=float) if degree4 else np.zeros(0)
        emitted_d4_vec = np.array([float(emitted_degree4.get(key, 0.0)) for key in degree4.keys()], dtype=float) if degree4 else np.zeros(0)
        total_used += min(pos, int(S))
        if diagnostics is not None:
            diagnostics.append(
                {
                    "slice": int(k),
                    "active_terms": int(sum(1 for theta in theta_by_key.values() if abs(theta) >= float(term_theta_threshold))),
                    "retained_terms": int(len(degree1) + len(degree2) + len(degree3) + len(degree4) + len(unsupported)),
                    "used_steps": int(min(pos, int(S))),
                    "emitted_terms": int(
                        sum(1 for key, theta in degree1.items() if abs(theta - theta) <= float(residual_tol))
                        + sum(1 for key, theta in degree2.items() if abs(theta - emitted_degree2.get(key, 0.0)) <= float(residual_tol))
                        + sum(1 for key, theta in degree3.items() if abs(theta - emitted_degree3.get(key, 0.0)) <= float(residual_tol))
                        + sum(1 for key, theta in degree4.items() if abs(theta - emitted_degree4.get(key, 0.0)) <= float(residual_tol))
                    ),
                    "skipped_terms": int(len(unsupported)),
                    "degree1_terms": int(len(degree1)),
                    "degree2_terms": int(len(degree2)),
                    "degree3_terms": int(len(degree3)),
                    "degree4_terms": int(len(degree4)),
                    "unsupported_terms": int(len(unsupported)),
                    "requested_degree2_norm": float(np.linalg.norm(requested_vec)),
                    "emitted_degree2_norm": float(np.linalg.norm(emitted_vec)),
                    "residual_degree2_norm": float(np.linalg.norm(residual_vec)),
                    "requested_degree3_norm": float(np.linalg.norm(requested_d3_vec)),
                    "emitted_degree3_norm": float(np.linalg.norm(emitted_d3_vec)),
                    "residual_degree3_norm": float(np.linalg.norm(residual_d3_vec)),
                    "requested_degree4_norm": float(np.linalg.norm(requested_d4_vec)),
                    "emitted_degree4_norm": float(np.linalg.norm(emitted_d4_vec)),
                    "residual_degree4_norm": float(np.linalg.norm(residual_d4_vec)),
                    "required_drift_time": float(batch_status.get("required_drift_time", 0.0)),
                    "available_slice_time": float(S) * dt_micro,
                    "batch2_frames": int(batch_status.get("frames", 0)),
                    "batch2_emitted_frames": int(batch_status.get("emitted_frames", 0)),
                    "batch2_skipped_frames": int(batch_status.get("skipped_frames", 0)),
                    "batch2_actual_drift_time": float(batch_status.get("actual_drift_time", 0.0)),
                    "batch2_frame_requested_norm": float(batch_status.get("requested_norm", np.linalg.norm(requested_vec))),
                    "batch2_frame_emitted_norm": float(batch_status.get("emitted_norm", np.linalg.norm(emitted_vec))),
                    "batch2_frame_residual_norm": float(batch_status.get("emitted_residual_norm", np.linalg.norm(residual_vec))),
                    "batch2_frame_relative_residual": float(
                        batch_status.get(
                            "emitted_relative_residual",
                            float(np.linalg.norm(residual_vec)) / max(1e-30, float(np.linalg.norm(requested_vec))),
                        )
                    ),
                    "batch2_status": str(batch_status.get("status", "unknown")),
                    "batch2_status_counts": json.dumps(batch_status.get("status_counts", {}), sort_keys=True),
                    "degree1_status_counts": json.dumps(degree1_status_counts, sort_keys=True),
                    "degree3_status_counts": json.dumps(degree3_status_counts, sort_keys=True),
                    "degree3_used_steps": int(degree3_used_steps),
                    "degree4_status_counts": json.dumps(degree4_status_counts, sort_keys=True),
                    "degree4_used_steps": int(degree4_used_steps),
                }
            )
        if reporter is not None:
            reporter.progress(
                progress_label,
                k + 1,
                int(Nt),
                t_start,
                extra=f"avg_used={total_used / max(1, k + 1):.1f}/{S}",
                force=(k == int(Nt) - 1),
            )

    if amp_bound > 0:
        np.clip(u_out, -float(amp_bound), float(amp_bound), out=u_out)
    emitted_fraction = total_emitted_weight / total_requested_weight if total_requested_weight > 0 else 0.0
    degree2_frame_summary: dict[str, Any] = {
        "enabled": bool(degree2_frame_rows),
        "slice_count": int(len(degree2_frame_rows)),
        "max_relative_residual": 0.0,
        "mean_relative_residual": 0.0,
        "max_residual_norm": 0.0,
        "mean_residual_norm": 0.0,
        "total_emitted_frames": 0,
        "total_skipped_frames": 0,
        "total_actual_drift_time": 0.0,
        "total_required_drift_time": 0.0,
        "status_counts": {},
    }
    if degree2_frame_rows:
        rels = [float(row.get("emitted_relative_residual", 0.0)) for row in degree2_frame_rows]
        residuals = [float(row.get("emitted_residual_norm", 0.0)) for row in degree2_frame_rows]
        status_counts: dict[str, int] = {}
        for row in degree2_frame_rows:
            status = str(row.get("status", "unknown"))
            status_counts[status] = int(status_counts.get(status, 0)) + 1
        degree2_frame_summary.update(
            {
                "max_relative_residual": float(max(rels) if rels else 0.0),
                "mean_relative_residual": float(np.mean(rels) if rels else 0.0),
                "max_residual_norm": float(max(residuals) if residuals else 0.0),
                "mean_residual_norm": float(np.mean(residuals) if residuals else 0.0),
                "total_emitted_frames": int(sum(int(row.get("emitted_frames", 0)) for row in degree2_frame_rows)),
                "total_skipped_frames": int(sum(int(row.get("skipped_frames", 0)) for row in degree2_frame_rows)),
                "total_actual_drift_time": float(sum(float(row.get("actual_drift_time", 0.0)) for row in degree2_frame_rows)),
                "total_required_drift_time": float(sum(float(row.get("required_drift_time", 0.0)) for row in degree2_frame_rows)),
                "status_counts": status_counts,
            }
        )
    summary = {
        "mode": "constructive_batch2",
        "pauli_expansion": pauli_summary,
        "requested_weight": float(total_requested_weight),
        "emitted_weight": float(total_emitted_weight),
        "emitted_weight_fraction": float(emitted_fraction),
        "degree2_frame_summary": degree2_frame_summary,
        "degree2_frame_rows": degree2_frame_rows,
        "used_steps": int(total_used),
        "total_steps": int(M),
        "max_frames": int(max_frames),
        "frame_tol": float(frame_tol),
        "residual_tol": float(residual_tol),
    }
    msg = (
        f"    [Batch2Compiler] requested_weight={total_requested_weight:.6g} "
        f"emitted_weight={total_emitted_weight:.6g} emitted_fraction={emitted_fraction:.3f} "
        f"used={total_used}/{M} RMS={rms(u_out):.6g}"
    )
    if reporter is None:
        print(msg)
    else:
        reporter.info(msg)
    return u_out, summary


def compile_ea_to_effective_d2_controls(
    opt: ImprovedPolynomialPontryagin,
    effective_labels: list[str],
    T: float,
    Nt: int,
    S: int,
    amp_bound: float,
    budget_frac: float = 1.0,
    max_terms_per_slice: int = 48,
    compiler_sort_mode: str = "legacy",
    product_time_scale: float = 1.0,
    diagnostics: Optional[list[dict[str, Any]]] = None,
    term_diagnostics: Optional[list[dict[str, Any]]] = None,
    reporter: Optional[ConsoleReporter] = None,
    progress_label: str = "EffectiveD2Compiler",
) -> np.ndarray:
    """Diagnostic effective degree-2 product seed.

    This is not a hardware pulse compiler.  It assumes selected degree-2 Pauli
    products are directly available as effective controls labeled ``EFF2:*``.
    The schedule uses an instant-overhead degree-2 cost model and emits
    amplitudes on effective product axes scaled as P/D.
    """
    if S <= 0:
        raise ValueError("S must be positive")
    n_ea_steps = int(opt.u.shape[0])
    if n_ea_steps != Nt:
        Nt = n_ea_steps
    M = Nt * S
    dt_ea = T / Nt
    dt_micro = T / M
    dim = int(opt.dim)
    n_eff = len(effective_labels)
    u_out = np.zeros((M, n_eff), dtype=float)
    slice_budget = max(1, int(np.floor(float(budget_frac) * S)))
    amp = float(max(1e-12, amp_bound))
    label_lookup = {label: j for j, label in enumerate(effective_labels)}
    t_start = time.time()

    parsed: list[tuple[complex, dict[int, str]] | None] = [
        parse_ea_label_to_pauli_string(label) for label in opt.control_labels
    ]

    total_used_steps = 0
    total_terms = 0
    total_emitted = 0
    total_skipped = 0

    def record_term(
        *,
        slice_index: int,
        idx: int,
        label: str,
        degree: int,
        axes_by_site: dict[int, str],
        theta: float,
        retained: bool,
        emitted_flag: bool,
        status: str,
        used_steps: int,
        required_steps: int,
        budget_before: int,
        budget_after: int,
        rank: int,
    ) -> None:
        if term_diagnostics is None:
            return
        term_diagnostics.append(
            {
                "slice": int(slice_index),
                "rank": int(rank),
                "control_index": int(idx),
                "label": str(label),
                "degree": int(degree),
                "pauli_string": pauli_axes_to_string(axes_by_site),
                "theta": float(theta),
                "requested_weight": float(abs(theta)),
                "retained": int(bool(retained)),
                "emitted": int(bool(emitted_flag)),
                "status": str(status),
                "used_steps": int(used_steps),
                "required_steps": int(required_steps),
                "available_steps": int(max(0, slice_budget - budget_before)),
                "budget_before": int(budget_before),
                "budget_after": int(budget_after),
            }
        )

    for k in range(Nt):
        u_slice = u_out[k * S : (k + 1) * S, :]
        used = 0
        emitted: dict[str, int] = {}
        skipped: dict[str, int] = {}
        active = np.where(np.abs(opt.u[k]) > 1e-8)[0]
        degree2_items: list[tuple[int, float, int, str, float, dict[int, str]]] = []

        for idx in active.tolist():
            label = opt.control_labels[idx]
            parsed_item = parsed[idx]
            if parsed_item is None:
                skipped["unparsed"] = skipped.get("unparsed", 0) + 1
                record_term(
                    slice_index=k,
                    idx=idx,
                    label=label,
                    degree=0,
                    axes_by_site={},
                    theta=0.0,
                    retained=False,
                    emitted_flag=False,
                    status="unparsed",
                    used_steps=0,
                    required_steps=0,
                    budget_before=used,
                    budget_after=used,
                    rank=-1,
                )
                continue
            phase, axes_by_site = parsed_item
            theta = float(product_time_scale) * float(np.real(phase) * opt.u[k, idx] * dt_ea)
            degree = len(axes_by_site)
            if degree == 2:
                degree2_items.append((degree, -abs(theta), idx, label, theta, axes_by_site))
            else:
                status = "identity" if degree == 0 else f"unsupported_degree{degree}"
                skipped[status] = skipped.get(status, 0) + 1
                record_term(
                    slice_index=k,
                    idx=idx,
                    label=label,
                    degree=degree,
                    axes_by_site=axes_by_site,
                    theta=theta,
                    retained=False,
                    emitted_flag=False,
                    status=status,
                    used_steps=0,
                    required_steps=0,
                    budget_before=used,
                    budget_after=used,
                    rank=-1,
                )

        sort_mode = str(compiler_sort_mode)
        if sort_mode == "weight":
            degree2_items.sort(key=lambda x: x[1])
        elif sort_mode == "low_degree":
            degree2_items.sort(key=lambda x: (x[0], x[1]))
        elif sort_mode == "high_degree":
            degree2_items.sort(key=lambda x: (-x[0], x[1]))
        else:
            degree2_items.sort(key=lambda x: (x[0] != 2, x[0], x[1]))

        if len(degree2_items) > max_terms_per_slice:
            skipped["term_cap"] = skipped.get("term_cap", 0) + len(degree2_items) - max_terms_per_slice
            for rank, (degree, _neg_abs, idx, label, theta, axes_by_site) in enumerate(
                degree2_items[max_terms_per_slice:], start=max_terms_per_slice
            ):
                record_term(
                    slice_index=k,
                    idx=idx,
                    label=label,
                    degree=degree,
                    axes_by_site=axes_by_site,
                    theta=theta,
                    retained=False,
                    emitted_flag=False,
                    status="term_cap",
                    used_steps=0,
                    required_steps=0,
                    budget_before=used,
                    budget_after=used,
                    rank=rank,
                )
            degree2_items = degree2_items[:max_terms_per_slice]

        total_terms += len(degree2_items)
        for rank, (degree, _neg_abs, idx, label, theta, axes_by_site) in enumerate(degree2_items):
            budget_before = used
            if used >= slice_budget:
                skipped["slice_budget"] = skipped.get("slice_budget", 0) + 1
                record_term(
                    slice_index=k,
                    idx=idx,
                    label=label,
                    degree=degree,
                    axes_by_site=axes_by_site,
                    theta=theta,
                    retained=True,
                    emitted_flag=False,
                    status="slice_budget",
                    used_steps=0,
                    required_steps=0,
                    budget_before=budget_before,
                    budget_after=used,
                    rank=rank,
                )
                continue

            key = pauli_axes_to_string(axes_by_site)
            eff_label = f"EFF2:{key}"
            ctrl_idx = label_lookup.get(eff_label)
            if ctrl_idx is None:
                skipped["missing_effective_axis"] = skipped.get("missing_effective_axis", 0) + 1
                record_term(
                    slice_index=k,
                    idx=idx,
                    label=label,
                    degree=degree,
                    axes_by_site=axes_by_site,
                    theta=theta,
                    retained=True,
                    emitted_flag=False,
                    status="missing_effective_axis",
                    used_steps=0,
                    required_steps=0,
                    budget_before=budget_before,
                    budget_after=used,
                    rank=rank,
                )
                continue

            instant_steps, instant_status = estimate_pauli_product_steps(
                axes_by_site,
                theta,
                dt_micro,
                amp,
                dim,
                max_degree=2,
                bch_reps=1,
                instant_overhead=True,
            )
            amp_steps = estimate_control_area_steps(theta, dt_micro, amp)
            required_steps = max(instant_steps, amp_steps, 1)
            if required_steps <= 0 or used + required_steps > slice_budget:
                status = f"effective_d2_{instant_status}_no_fit"
                skipped[status] = skipped.get(status, 0) + 1
                record_term(
                    slice_index=k,
                    idx=idx,
                    label=label,
                    degree=degree,
                    axes_by_site=axes_by_site,
                    theta=theta,
                    retained=True,
                    emitted_flag=False,
                    status=status,
                    used_steps=0,
                    required_steps=required_steps,
                    budget_before=budget_before,
                    budget_after=used,
                    rank=rank,
                )
                continue

            amp_eff = float(theta) / (required_steps * dt_micro)
            if abs(amp_eff) > amp:
                amp_eff = float(np.sign(amp_eff) * amp)
            u_slice[used:used + required_steps, ctrl_idx] += amp_eff
            used += required_steps
            status = f"effective_d2_{instant_status}_fit"
            emitted[status] = emitted.get(status, 0) + 1
            total_emitted += 1
            record_term(
                slice_index=k,
                idx=idx,
                label=label,
                degree=degree,
                axes_by_site=axes_by_site,
                theta=theta,
                retained=True,
                emitted_flag=True,
                status=status,
                used_steps=required_steps,
                required_steps=required_steps,
                budget_before=budget_before,
                budget_after=used,
                rank=rank,
            )

        total_used_steps += min(used, slice_budget)
        total_skipped += sum(skipped.values())
        if diagnostics is not None:
            diagnostics.append(
                {
                    "slice": int(k),
                    "active_terms": int(active.size),
                    "retained_terms": int(len(degree2_items)),
                    "used_steps": int(min(used, slice_budget)),
                    "emitted_terms": int(sum(emitted.values())),
                    "skipped_terms": int(sum(skipped.values())),
                    "emitted_words": json.dumps(emitted, sort_keys=True),
                    "skipped_words": json.dumps(skipped, sort_keys=True),
                }
            )
        if reporter is not None:
            reporter.progress(
                progress_label,
                k + 1,
                Nt,
                t_start,
                extra=f"avg_used={total_used_steps / max(1, k + 1):.1f}/{S}",
                force=(k == Nt - 1),
            )

    if amp_bound > 0:
        np.clip(u_out, -float(amp_bound), float(amp_bound), out=u_out)
    msg = (
        f"    [EffectiveD2Compiler] emitted={total_emitted} retained={total_terms} "
        f"skipped={total_skipped} used={total_used_steps}/{M} RMS={rms(u_out):.6g}"
    )
    if reporter is None:
        print(msg)
    else:
        reporter.info(msg)
    return u_out


def final_state_after_controls(
    psi0: qt.Qobj,
    H0: qt.Qobj,
    ctrl_axes: list[qt.Qobj],
    T: float,
    u: np.ndarray,
    H0_dense: Optional[np.ndarray] = None,
    ctrl_stack_dense: Optional[np.ndarray] = None,
    psi0_vec: Optional[np.ndarray] = None,
) -> qt.Qobj:
    """Propagate a state and return the final ket."""
    u_arr = np.asarray(u, dtype=float)
    M = int(u_arr.shape[0])
    if M <= 0:
        return psi0.copy()
    dt = float(T) / M
    if ctrl_stack_dense is None:
        ctrl_stack_dense = build_dense_operator_stack(ctrl_axes)
    if ctrl_stack_dense is not None:
        H0_local = np.array(H0_dense if H0_dense is not None else qobj_to_dense_matrix(H0), copy=False)
        psi = np.array(psi0_vec if psi0_vec is not None else qobj_to_dense_vector(psi0), dtype=np.complex128, copy=True)
        for k in range(M):
            H = assemble_dense_hamiltonian(H0_local, ctrl_stack_dense, u_arr[k])
            psi = expm(-1j * dt * H) @ psi
        return qt.Qobj(psi.reshape((-1, 1)), dims=psi0.dims)

    psi = psi0
    for k in range(M):
        H = H0
        for j, uj in enumerate(u_arr[k]):
            if abs(float(uj)) > 1e-12:
                H = H + float(uj) * ctrl_axes[j]
        psi = expmH(H, dt) * psi
    return psi


def compile_ea_to_d2_hardware_project(
    opt: ImprovedPolynomialPontryagin,
    ctrl_axes: list[qt.Qobj],
    ctrl_labels: list[str],
    H0_base: qt.Qobj,
    psi0: qt.Qobj,
    target: qt.Qobj,
    T: float,
    Nt: int,
    S: int,
    drift_strength_hw: float,
    amp_bound: float,
    hard_amp: float,
    budget_frac: float = 1.0,
    max_terms_per_slice: int = 48,
    compiler_sort_mode: str = "legacy",
    product_time_scale: float = 1.0,
    projection_iters: int = 40,
    projection_lr: float = 0.06,
    projection_threshold: float = 0.999,
    projection_init: str = "product_d2",
    projection_clip: float = 5.0,
    projection_backtracks: int = 3,
    projection_accept_mode: str = "soft",
    projection_accept_drop: float = 2e-3,
    diagnostics: Optional[list[dict[str, Any]]] = None,
    term_diagnostics: Optional[list[dict[str, Any]]] = None,
    reporter: Optional[ConsoleReporter] = None,
    progress_every: int = 10,
    seed: int = 1,
    H0_base_dense: Optional[np.ndarray] = None,
    ctrl_stack_dense: Optional[np.ndarray] = None,
    psi0_vec: Optional[np.ndarray] = None,
    target_vec: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Project an effective degree-2 reference into hardware controls.

    The returned controls use only the physical control labels passed in
    ``ctrl_labels``.  The effective degree-2 controls are used only to define a
    reference final state and an upper-bound diagnostic.
    """
    if S <= 0:
        raise ValueError("S must be positive")

    H0_phys = float(drift_strength_hw) * H0_base
    H0_phys_dense = None if H0_base_dense is None else (float(drift_strength_hw) * H0_base_dense)

    eff_axes, eff_labels, eff_map = build_effective_d2_axes_from_opt(opt, ctrl_axes, ctrl_labels)
    eff_stack_dense = build_dense_operator_stack(eff_axes)
    eff_diagnostics: list[dict[str, Any]] = []
    eff_term_diagnostics: list[dict[str, Any]] = []
    if reporter is not None:
        reporter.info(
            f"[D2PROJECT] effective reference has {len(eff_map)} direct degree-2 controls "
            "used only as a projection target."
        )
    u_eff = compile_ea_to_effective_d2_controls(
        opt=opt,
        effective_labels=eff_labels,
        T=T,
        Nt=Nt,
        S=S,
        amp_bound=amp_bound,
        budget_frac=budget_frac,
        max_terms_per_slice=max_terms_per_slice,
        compiler_sort_mode=compiler_sort_mode,
        product_time_scale=product_time_scale,
        diagnostics=eff_diagnostics,
        term_diagnostics=eff_term_diagnostics,
        reporter=reporter,
        progress_label="D2EffectiveReference",
    )

    effective_reference_state = final_state_after_controls(
        psi0,
        H0_phys,
        eff_axes,
        T,
        u_eff,
        H0_dense=H0_phys_dense,
        ctrl_stack_dense=eff_stack_dense,
        psi0_vec=psi0_vec,
    )
    effective_reference_vec = qobj_to_dense_vector(effective_reference_state)
    effective_target_fidelity = simulate_state_fidelity(
        psi0,
        target,
        H0_phys,
        eff_axes,
        T,
        u_eff,
        [0.0],
        H0_dense=H0_phys_dense,
        ctrl_stack_dense=eff_stack_dense,
        psi0_vec=psi0_vec,
        target_vec=target_vec,
    )

    hw_diagnostics: list[dict[str, Any]] = []
    hw_term_diagnostics: list[dict[str, Any]] = []
    if str(projection_init) == "zero":
        u_hw0 = np.zeros((Nt * S, len(ctrl_labels)), dtype=float)
    else:
        u_hw0 = compile_ea_to_controls_product_aware(
            opt=opt,
            ctrl_labels=ctrl_labels,
            T=T,
            Nt=Nt,
            S=S,
            amp_bound=amp_bound,
            hard_amp=hard_amp,
            drift_strength_hw=drift_strength_hw,
            budget_frac=budget_frac,
            max_terms_per_slice=max_terms_per_slice,
            product_bch_reps=1,
            product_max_degree=2,
            compiler_sort_mode=compiler_sort_mode,
            compiler_operation_mode="product",
            product_time_scale=product_time_scale,
            diagnostics=hw_diagnostics,
            term_diagnostics=hw_term_diagnostics,
            reporter=reporter,
            progress_label="D2HardwareBlocks",
        )

    if diagnostics is not None:
        diagnostics.extend(hw_diagnostics)
    if term_diagnostics is not None:
        term_diagnostics.extend(hw_term_diagnostics)

    ctrl_stack_phys = ctrl_stack_dense if ctrl_stack_dense is not None else build_dense_operator_stack(ctrl_axes)
    ref_fidelity_before = simulate_state_fidelity(
        psi0,
        effective_reference_state,
        H0_phys,
        ctrl_axes,
        T,
        u_hw0,
        [0.0],
        H0_dense=H0_phys_dense,
        ctrl_stack_dense=ctrl_stack_phys,
        psi0_vec=psi0_vec,
        target_vec=effective_reference_vec,
    )
    target_fidelity_before = simulate_state_fidelity(
        psi0,
        target,
        H0_phys,
        ctrl_axes,
        T,
        u_hw0,
        [0.0],
        H0_dense=H0_phys_dense,
        ctrl_stack_dense=ctrl_stack_phys,
        psi0_vec=psi0_vec,
        target_vec=target_vec,
    )

    u_hw = u_hw0.copy()
    projection_fidelity = float(ref_fidelity_before)
    projection_iters_used = 0
    if int(projection_iters) > 0:
        if reporter is not None:
            reporter.info(
                f"[D2PROJECT] projecting effective degree-2 reference into {len(ctrl_labels)} hardware controls "
                f"for {int(projection_iters)} iteration(s)."
            )
        proj_cfg = OptConfig(
            lr=float(projection_lr),
            l2=0.0,
            amp=float(amp_bound),
            clip=float(projection_clip),
            backtracks=int(projection_backtracks),
            accept_mode=str(projection_accept_mode),
            accept_drop=float(projection_accept_drop),
            threshold=float(projection_threshold),
            verbose=0,
        )
        projection_fidelity, u_hw, _it_thresh, projection_iters_used = grape_state_iters(
            psi0=psi0,
            target=effective_reference_state,
            H0=H0_phys,
            ctrl_axes=ctrl_axes,
            T=T,
            u_init=u_hw0,
            opt_cfg=proj_cfg,
            iters=int(projection_iters),
            seed=int(seed) + 50_000,
            jitter_list=[0.0],
            label="D2HardwareProject",
            reporter=reporter,
            progress_every=max(1, int(progress_every)),
            H0_dense=H0_phys_dense,
            ctrl_stack_dense=ctrl_stack_phys,
            psi0_vec=psi0_vec,
            target_vec=effective_reference_vec,
        )

    target_fidelity_after = simulate_state_fidelity(
        psi0,
        target,
        H0_phys,
        ctrl_axes,
        T,
        u_hw,
        [0.0],
        H0_dense=H0_phys_dense,
        ctrl_stack_dense=ctrl_stack_phys,
        psi0_vec=psi0_vec,
        target_vec=target_vec,
    )

    summary = {
        "mode": "product_d2_hardware_project",
        "hardware_executable": True,
        "reference_note": "effective degree-2 controls are used only to define the projection reference",
        "effective_d2_control_count": int(len(eff_map)),
        "physical_control_count": int(len(ctrl_labels)),
        "seed_shape": list(u_hw.shape),
        "effective_seed_shape": list(u_eff.shape),
        "effective_seed_rms": float(rms(u_eff)),
        "hardware_seed_rms_before_projection": float(rms(u_hw0)),
        "hardware_seed_rms_after_projection": float(rms(u_hw)),
        "hardware_seed_max_abs_after_projection": float(np.max(np.abs(u_hw))) if u_hw.size else 0.0,
        "effective_reference_target_fidelity": float(effective_target_fidelity),
        "hardware_reference_fidelity_before_projection": float(ref_fidelity_before),
        "hardware_reference_fidelity_after_projection": float(projection_fidelity),
        "hardware_target_fidelity_before_projection": float(target_fidelity_before),
        "hardware_target_fidelity_after_projection": float(target_fidelity_after),
        "projection_iters_requested": int(projection_iters),
        "projection_iters_used": int(projection_iters_used),
        "projection_init": str(projection_init),
        "effective_reference_audit": summarize_compiler_term_diagnostics(eff_term_diagnostics),
        "hardware_block_audit": summarize_compiler_term_diagnostics(hw_term_diagnostics),
    }
    if reporter is not None:
        reporter.info(
            "[D2PROJECT] hardware reference F: before={:.6f} after={:.6f}; target F: before={:.6f} after={:.6f}".format(
                float(ref_fidelity_before),
                float(projection_fidelity),
                float(target_fidelity_before),
                float(target_fidelity_after),
            )
        )
    return u_hw, summary


def ea_reference_degree_limit(reference_mode: str, p: int) -> int | None:
    mode = str(reference_mode).strip().lower()
    if mode == "full":
        return None
    if mode.startswith("degree"):
        try:
            return max(1, min(int(p), int(mode.replace("degree", ""))))
        except ValueError:
            return None
    return None


def build_ea_reference_trajectory(
    opt: ImprovedPolynomialPontryagin,
    psi0: qt.Qobj,
    target: qt.Qobj,
    reference_mode: str = "full",
) -> dict[str, Any]:
    """Propagate the EA solution and return slice-boundary reference states."""
    degree_limit = ea_reference_degree_limit(reference_mode, opt.p)
    dim = int(opt.dim)
    psi0_vec = qobj_to_dense_vector(psi0)
    target_vec = qobj_to_dense_vector(target)
    H0 = qobj_to_dense_matrix(opt.H_drift)
    ctrl_stack_all = build_dense_operator_stack(opt.controls)
    if ctrl_stack_all is None:
        ctrl_stack_all = np.stack([qobj_to_dense_matrix(op) for op in opt.controls], axis=0)

    if degree_limit is None:
        keep_idx = np.arange(len(opt.controls), dtype=int)
    else:
        keep_idx = np.array([i for i, deg in enumerate(opt.control_degrees) if int(deg) <= int(degree_limit)], dtype=int)
    H_ctrl_stack = ctrl_stack_all[keep_idx] if keep_idx.size else np.zeros((0, dim, dim), dtype=np.complex128)

    states = np.empty((int(opt.Nt) + 1, dim), dtype=np.complex128)
    states[0] = psi0_vec
    for k in range(int(opt.Nt)):
        H = np.array(H0, copy=True)
        if keep_idx.size:
            H += np.tensordot(opt.u[k, keep_idx], H_ctrl_stack, axes=(0, 0))
        states[k + 1] = expm(-1j * float(opt.dt) * H) @ states[k]
        norm = float(np.linalg.norm(states[k + 1]))
        if norm > 1e-30:
            states[k + 1] /= norm

    final_alpha = complex(np.vdot(target_vec, states[-1]))
    checkpoint_qobjs = [qt.Qobj(states[k].reshape((-1, 1)), dims=psi0.dims) for k in range(states.shape[0])]
    return {
        "states": states,
        "qobjs": checkpoint_qobjs,
        "target_fidelity": float(abs(final_alpha) ** 2),
        "reference_mode": str(reference_mode),
        "degree_limit": None if degree_limit is None else int(degree_limit),
        "controls_used": int(keep_idx.size),
        "controls_available": int(len(opt.controls)),
    }


def trajectory_checkpoint_indices(Nt: int, mode: str) -> list[int]:
    mode_l = str(mode).strip().lower()
    Nt = int(Nt)
    if Nt <= 0:
        return []
    explicit_range = re.fullmatch(r"(?:range|window|slice)[:_](\d+)[:_-](\d+)(?:[:_-](\d+))?", mode_l)
    if explicit_range is not None:
        start = int(explicit_range.group(1))
        stop = int(explicit_range.group(2))
        stride = int(explicit_range.group(3) or 1)
        if stride <= 0:
            raise ValueError(f"checkpoint mode {mode!r} has nonpositive stride")
        lo = max(1, min(start, stop))
        hi = min(Nt, max(start, stop))
        indices = list(range(lo, hi + 1, stride))
        if Nt not in indices:
            indices.append(Nt)
        return sorted(set(int(i) for i in indices if 0 < int(i) <= Nt))
    if mode_l == "final":
        return [Nt]
    if mode_l == "late2":
        return sorted(set(i for i in (Nt - 1, Nt) if i > 0))
    if mode_l == "late4":
        return sorted(set(i for i in range(max(1, Nt - 3), Nt + 1)))
    if mode_l in ("latehalf", "secondhalf"):
        return list(range(max(1, Nt // 2), Nt + 1))
    if mode_l in ("middle", "mid"):
        lo = max(1, int(round(0.45 * Nt)))
        hi = min(Nt, int(round(0.70 * Nt)))
        indices = list(range(lo, hi + 1))
        if Nt not in indices:
            indices.append(Nt)
        return sorted(set(int(i) for i in indices if 0 < int(i) <= Nt))
    if mode_l == "every2":
        indices = list(range(2, Nt + 1, 2))
    elif mode_l == "every4":
        indices = list(range(4, Nt + 1, 4))
    else:
        indices = list(range(1, Nt + 1))
    if Nt not in indices:
        indices.append(Nt)
    return sorted(set(int(i) for i in indices if 0 < int(i) <= Nt))


def parse_csv_ints(text: str, default: list[int]) -> list[int]:
    vals: list[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(int(float(part)))
    return vals if vals else list(default)


def parse_csv_floats(text: str, default: list[float]) -> list[float]:
    vals: list[float] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    return vals if vals else list(default)


def parse_csv_strings(text: str, default: list[str]) -> list[str]:
    vals = [part.strip() for part in str(text).split(",") if part.strip()]
    return vals if vals else list(default)


def extend_to_length(values: list[Any], n: int) -> list[Any]:
    if n <= 0:
        return []
    if not values:
        return [None] * n
    out = list(values)
    while len(out) < n:
        out.append(out[-1])
    return out[:n]


def stable_softmin_weights(values: Sequence[float], tau: float) -> tuple[float, np.ndarray]:
    """Return soft-min value and derivative weights for max-friendly objectives."""
    vals = np.asarray([float(v) for v in values], dtype=float)
    if vals.size == 0:
        return 0.0, np.zeros(0, dtype=float)
    tau = max(1e-12, float(tau))
    shifted = -(vals - float(np.min(vals))) / tau
    shifted = np.clip(shifted, -700.0, 700.0)
    weights = np.exp(shifted)
    denom = float(np.sum(weights))
    if denom <= 0.0 or not np.isfinite(denom):
        weights = np.ones_like(vals) / float(vals.size)
    else:
        weights = weights / denom
    value = float(np.min(vals)) - tau * float(np.log(float(np.sum(np.exp(shifted)))))
    return value, weights


def resample_reference_states(reference_states: np.ndarray, Nt: int, S: int) -> dict[int, np.ndarray]:
    """Map EA slice-boundary states to microstep indices."""
    Nt = int(Nt)
    S = int(S)
    out: dict[int, np.ndarray] = {}
    for k in range(Nt + 1):
        out[k * S] = reference_states[k]
    return out


def trajectory_objective_gradient_dense(
    H0_dense: np.ndarray,
    ctrl_stack_dense: np.ndarray,
    u: np.ndarray,
    dt: float,
    psi0_vec: np.ndarray,
    target_vec: np.ndarray,
    checkpoint_targets: dict[int, np.ndarray],
    checkpoint_weight: float,
    final_weight: float,
    l2_coeff: float = 0.0,
    clip: float = 0.0,
    objective_mode: str = "weighted_sum",
    objective_softmin_tau: float = 0.02,
) -> tuple[float, np.ndarray, dict[str, Any]]:
    """Objective and gradient for matching EA checkpoints plus final target."""
    states, unitaries = propagate_dense(H0_dense, ctrl_stack_dense, u, dt, psi0_vec, store_steps=True)
    if unitaries is None:
        raise ValueError("trajectory optimizer requires stored unitaries")
    M, m = u.shape
    dim = int(psi0_vec.size)
    sources = np.zeros((M + 1, dim), dtype=np.complex128)

    checkpoint_steps: list[int] = []
    checkpoint_fidelities: list[float] = []
    checkpoint_alphas: list[complex] = []
    checkpoint_refs: list[np.ndarray] = []
    checkpoint_terms = sorted((int(k), v) for k, v in checkpoint_targets.items() if 0 < int(k) <= M)
    for step_idx, ref_vec in checkpoint_terms:
        alpha = complex(np.vdot(ref_vec, states[step_idx]))
        checkpoint_steps.append(int(step_idx))
        checkpoint_fidelities.append(float(abs(alpha) ** 2))
        checkpoint_alphas.append(alpha)
        checkpoint_refs.append(ref_vec)

    alpha_final = complex(np.vdot(target_vec, states[-1]))
    final_fidelity = float(abs(alpha_final) ** 2)
    checkpoint_mean = float(np.mean(checkpoint_fidelities)) if checkpoint_fidelities else 0.0

    mode = str(objective_mode).strip().lower()
    ck_count = max(1, len(checkpoint_terms))
    final_coeff = 0.0
    checkpoint_coeffs: list[float] = [0.0 for _ in checkpoint_terms]

    if mode in ("geomean", "log_geomean", "balanced_log"):
        eps = 1e-12
        fw = max(0.0, float(final_weight))
        cw = max(0.0, float(checkpoint_weight))
        if checkpoint_fidelities:
            norm = max(eps, fw + cw)
            fw /= norm
            cw /= norm
            objective = fw * float(np.log(final_fidelity + eps)) + cw * float(np.log(checkpoint_mean + eps))
            final_coeff = fw / max(eps, final_fidelity + eps)
            per_ck = cw / max(eps, checkpoint_mean + eps) / float(ck_count)
            checkpoint_coeffs = [per_ck for _ in checkpoint_terms]
        else:
            objective = float(np.log(final_fidelity + eps))
            final_coeff = 1.0 / max(eps, final_fidelity + eps)
    elif mode in ("softmin", "gate_softmin"):
        if checkpoint_fidelities:
            objective, coeffs = stable_softmin_weights(
                [final_fidelity, checkpoint_mean],
                float(objective_softmin_tau),
            )
            final_coeff = float(coeffs[0])
            per_ck = float(coeffs[1]) / float(ck_count)
            checkpoint_coeffs = [per_ck for _ in checkpoint_terms]
        else:
            objective = float(final_fidelity)
            final_coeff = 1.0
    elif mode in ("softmin_all", "gate_softmin_all"):
        if checkpoint_fidelities:
            objective, coeffs = stable_softmin_weights(
                [final_fidelity] + checkpoint_fidelities,
                float(objective_softmin_tau),
            )
            final_coeff = float(coeffs[0])
            checkpoint_coeffs = [float(x) for x in coeffs[1:]]
        else:
            objective = float(final_fidelity)
            final_coeff = 1.0
    else:
        objective = float(final_weight) * final_fidelity
        if checkpoint_fidelities:
            objective += float(checkpoint_weight) * checkpoint_mean
        final_coeff = float(final_weight)
        per_ck = float(checkpoint_weight) / float(ck_count)
        checkpoint_coeffs = [per_ck for _ in checkpoint_terms]

    sources[M] += float(final_coeff) * alpha_final * target_vec
    for step_idx, alpha, ref_vec, coeff in zip(checkpoint_steps, checkpoint_alphas, checkpoint_refs, checkpoint_coeffs):
        sources[int(step_idx)] += float(coeff) * alpha * ref_vec

    if l2_coeff > 0:
        objective -= float(l2_coeff) * float(np.sum(u * u))

    grad = np.zeros_like(u, dtype=float)
    lam_next = sources[M].copy()
    for k in range(M - 1, -1, -1):
        mat_els = np.einsum("i,aij,j->a", np.conj(lam_next), ctrl_stack_dense, states[k + 1], optimize=True)
        grad[k] = 2.0 * np.real((-1j * dt) * mat_els)
        lam_next = unitaries[k].conj().T @ lam_next + sources[k]

    if l2_coeff > 0:
        grad -= 2.0 * float(l2_coeff) * u
    if clip > 0:
        gnorm = float(np.linalg.norm(grad))
        if gnorm > clip:
            grad *= float(clip) / max(1e-30, gnorm)

    metrics = {
        "objective": float(objective),
        "objective_mode": str(mode),
        "objective_softmin_tau": float(objective_softmin_tau),
        "target_fidelity": float(final_fidelity),
        "checkpoint_fidelity_mean": float(checkpoint_mean),
        "checkpoint_fidelity_min": float(np.min(checkpoint_fidelities)) if checkpoint_fidelities else 0.0,
        "checkpoint_count": float(len(checkpoint_terms)),
        "checkpoint_steps": checkpoint_steps,
        "checkpoint_fidelities": checkpoint_fidelities,
    }
    return float(objective), grad, metrics


def trajectory_hardware_project(
    psi0: qt.Qobj,
    target: qt.Qobj,
    H0: qt.Qobj,
    ctrl_axes: list[qt.Qobj],
    T: float,
    u_init: np.ndarray,
    reference_states: np.ndarray,
    Nt: int,
    S: int,
    checkpoint_mode: str,
    trajectory_weight: float,
    project_iters: int,
    project_lr: float,
    amp_bound: float,
    clip: float,
    backtracks: int,
    accept_mode: str,
    accept_drop: float,
    threshold: float,
    reporter: Optional[ConsoleReporter],
    progress_every: int,
    label: str,
    optimizer: str = "adam",
    lbfgs_maxls: int = 20,
    lbfgs_ftol: float = 1e-9,
    lbfgs_gtol: float = 1e-5,
    objective_mode: str = "weighted_sum",
    objective_softmin_tau: float = 0.02,
    H0_dense: Optional[np.ndarray] = None,
    ctrl_stack_dense: Optional[np.ndarray] = None,
    psi0_vec: Optional[np.ndarray] = None,
    target_vec: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Optimize physical controls against an EA reference trajectory."""
    u = np.asarray(u_init, dtype=float).copy()
    M = int(Nt) * int(S)
    if u.shape[0] != M:
        u = resample_controls(u, M)
    if float(amp_bound) > 0:
        np.clip(u, -float(amp_bound), float(amp_bound), out=u)
    H0_arr = np.array(H0_dense if H0_dense is not None else qobj_to_dense_matrix(H0), copy=False)
    ctrl_stack = ctrl_stack_dense if ctrl_stack_dense is not None else build_dense_operator_stack(ctrl_axes)
    if ctrl_stack is None:
        ctrl_stack = np.stack([qobj_to_dense_matrix(op) for op in ctrl_axes], axis=0)
    psi0_arr = np.array(psi0_vec if psi0_vec is not None else qobj_to_dense_vector(psi0), copy=False)
    target_arr = np.array(target_vec if target_vec is not None else qobj_to_dense_vector(target), copy=False)
    dt = float(T) / max(1, M)

    ref_by_boundary = resample_reference_states(np.asarray(reference_states, dtype=np.complex128), int(Nt), int(S))
    checkpoint_boundaries = trajectory_checkpoint_indices(int(Nt), checkpoint_mode)
    checkpoint_targets = {int(k) * int(S): ref_by_boundary[int(k) * int(S)] for k in checkpoint_boundaries}
    trajectory_weight = max(0.0, min(1.0, float(trajectory_weight)))
    final_weight = 1.0 - trajectory_weight

    obj, grad, metrics = trajectory_objective_gradient_dense(
        H0_arr,
        ctrl_stack,
        u,
        dt,
        psi0_arr,
        target_arr,
        checkpoint_targets,
        checkpoint_weight=trajectory_weight,
        final_weight=final_weight,
        clip=float(clip),
        objective_mode=str(objective_mode),
        objective_softmin_tau=float(objective_softmin_tau),
    )
    initial_metrics = dict(metrics)
    best_obj = float(obj)
    best_u = u.copy()
    best_metrics = dict(metrics)
    t_start = time.time()
    iters_used = 0
    optimizer_name = str(optimizer).strip().lower()

    if optimizer_name == "lbfgsb":
        shape = u.shape
        bounds = None
        if float(amp_bound) > 0:
            bounds = [(-float(amp_bound), float(amp_bound))] * int(u.size)
        eval_state: dict[str, Any] = {
            "nfev": 0,
            "last_obj": float(obj),
            "last_metrics": dict(metrics),
            "best_obj": float(best_obj),
            "best_u": best_u.copy(),
            "best_metrics": dict(best_metrics),
        }

        def objective_flat(x_flat: np.ndarray) -> tuple[float, np.ndarray]:
            u_eval = np.asarray(x_flat, dtype=float).reshape(shape)
            obj_eval, grad_eval, metrics_eval = trajectory_objective_gradient_dense(
                H0_arr,
                ctrl_stack,
                u_eval,
                dt,
                psi0_arr,
                target_arr,
                checkpoint_targets,
                checkpoint_weight=trajectory_weight,
                final_weight=final_weight,
                clip=float(clip),
                objective_mode=str(objective_mode),
                objective_softmin_tau=float(objective_softmin_tau),
            )
            eval_state["nfev"] = int(eval_state.get("nfev", 0)) + 1
            eval_state["last_obj"] = float(obj_eval)
            eval_state["last_metrics"] = dict(metrics_eval)
            if float(obj_eval) > float(eval_state["best_obj"]):
                eval_state["best_obj"] = float(obj_eval)
                eval_state["best_u"] = u_eval.copy()
                eval_state["best_metrics"] = dict(metrics_eval)
            return -float(obj_eval), -np.asarray(grad_eval, dtype=float).reshape(-1)

        callback_state = {"iter": 0}

        def callback(_xk: np.ndarray) -> None:
            callback_state["iter"] += 1
            it = int(callback_state["iter"])
            metrics_cb = dict(eval_state.get("best_metrics", {}))
            emit_progress = (it % max(1, int(progress_every)) == 0) or it >= int(project_iters)
            if reporter is not None and emit_progress:
                reporter.progress(
                    label,
                    it,
                    int(project_iters),
                    t_start,
                    extra=(
                        f"obj={float(eval_state['best_obj']):.6f} "
                        f"target={float(metrics_cb.get('target_fidelity', 0.0)):.6f} "
                        f"traj={float(metrics_cb.get('checkpoint_fidelity_mean', 0.0)):.6f}"
                    ),
                    force=(it >= int(project_iters)),
                )

        res = minimize(
            objective_flat,
            u.reshape(-1),
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            callback=callback,
            options={
                "maxiter": int(project_iters),
                "maxls": int(lbfgs_maxls),
                "ftol": float(lbfgs_ftol),
                "gtol": float(lbfgs_gtol),
                "disp": False,
            },
        )
        iters_used = int(getattr(res, "nit", callback_state.get("iter", 0)) or 0)
        best_obj = float(eval_state["best_obj"])
        best_u = np.asarray(eval_state["best_u"], dtype=float).copy()
        best_metrics = dict(eval_state["best_metrics"])
        if reporter is not None:
            reporter.progress(
                label,
                max(1, iters_used),
                max(1, int(project_iters)),
                t_start,
                extra=(
                    f"obj={best_obj:.6f} target={float(best_metrics.get('target_fidelity', 0.0)):.6f} "
                    f"traj={float(best_metrics.get('checkpoint_fidelity_mean', 0.0)):.6f} "
                    f"nfev={int(eval_state.get('nfev', 0))} status={int(getattr(res, 'status', -1))}"
                ),
                force=True,
            )
    else:
        opt = AdamAscent(u.shape, lr=float(project_lr))
        bt_max = max(0, int(backtracks))
        backtrack_scales = [1.0] + [0.5**k for k in range(1, bt_max + 1)]

        for it in range(1, int(project_iters) + 1):
            _obj_curr, grad, _metrics_curr = trajectory_objective_gradient_dense(
                H0_arr,
                ctrl_stack,
                u,
                dt,
                psi0_arr,
                target_arr,
                checkpoint_targets,
                checkpoint_weight=trajectory_weight,
                final_weight=final_weight,
                clip=float(clip),
                objective_mode=str(objective_mode),
                objective_softmin_tau=float(objective_softmin_tau),
            )
            step = opt.step(grad)
            accepted = False
            best_local: tuple[float, np.ndarray, dict[str, float]] | None = None
            for scale in backtrack_scales:
                u_try = u + float(scale) * step
                if float(amp_bound) > 0:
                    np.clip(u_try, -float(amp_bound), float(amp_bound), out=u_try)
                obj_try, _grad_try, metrics_try = trajectory_objective_gradient_dense(
                    H0_arr,
                    ctrl_stack,
                    u_try,
                    dt,
                    psi0_arr,
                    target_arr,
                    checkpoint_targets,
                    checkpoint_weight=trajectory_weight,
                    final_weight=final_weight,
                    clip=float(clip),
                    objective_mode=str(objective_mode),
                    objective_softmin_tau=float(objective_softmin_tau),
                )
                if best_local is None or obj_try > best_local[0]:
                    best_local = (float(obj_try), u_try, dict(metrics_try))
                ok = obj_try >= obj if str(accept_mode) == "hard" else obj_try >= (obj - float(accept_drop))
                if ok:
                    u = u_try
                    obj = float(obj_try)
                    metrics = dict(metrics_try)
                    accepted = True
                    break
            if not accepted:
                if best_local is not None and best_local[0] > obj:
                    obj, u, metrics = best_local
                else:
                    opt.lr *= 0.95
            if obj > best_obj:
                best_obj = float(obj)
                best_u = u.copy()
                best_metrics = dict(metrics)
            iters_used = it
            emit_progress = (it % max(1, int(progress_every)) == 0) or it == int(project_iters) or best_metrics["target_fidelity"] >= float(threshold)
            if reporter is not None and emit_progress:
                reporter.progress(
                    label,
                    it,
                    int(project_iters),
                    t_start,
                    extra=(
                        f"obj={best_obj:.6f} target={best_metrics['target_fidelity']:.6f} "
                        f"traj={best_metrics['checkpoint_fidelity_mean']:.6f}"
                    ),
                    force=(it == int(project_iters) or best_metrics["target_fidelity"] >= float(threshold)),
                )
            if best_metrics["target_fidelity"] >= float(threshold):
                break

    summary = {
        "hardware_executable": True,
        "control_count": int(len(ctrl_axes)),
        "seed_shape": list(best_u.shape),
        "checkpoint_mode": str(checkpoint_mode),
        "checkpoint_count": int(len(checkpoint_targets)),
        "checkpoint_steps": [int(x) for x in best_metrics.get("checkpoint_steps", [])],
        "checkpoint_fidelities_before": [float(x) for x in initial_metrics.get("checkpoint_fidelities", [])],
        "checkpoint_fidelities_after": [float(x) for x in best_metrics.get("checkpoint_fidelities", [])],
        "trajectory_weight": float(trajectory_weight),
        "final_weight": float(final_weight),
        "objective_mode": str(objective_mode),
        "objective_softmin_tau": float(objective_softmin_tau),
        "optimizer": str(optimizer_name),
        "project_iters_requested": int(project_iters),
        "project_iters_used": int(iters_used),
        "objective_before": float(initial_metrics["objective"]),
        "objective_after": float(best_obj),
        "trajectory_fidelity_before": float(initial_metrics["checkpoint_fidelity_mean"]),
        "trajectory_fidelity_after": float(best_metrics["checkpoint_fidelity_mean"]),
        "trajectory_fidelity_min_after": float(best_metrics["checkpoint_fidelity_min"]),
        "target_fidelity_before": float(initial_metrics["target_fidelity"]),
        "target_fidelity_after": float(best_metrics["target_fidelity"]),
        "seed_rms_before": float(rms(u_init)),
        "seed_rms_after": float(rms(best_u)),
        "seed_max_abs_after": float(np.max(np.abs(best_u))) if best_u.size else 0.0,
    }
    return best_u, summary


def compile_ea_to_trajectory_hardware_project(
    opt: ImprovedPolynomialPontryagin,
    ctrl_axes: list[qt.Qobj],
    ctrl_labels: list[str],
    H0_base: qt.Qobj,
    psi0: qt.Qobj,
    target: qt.Qobj,
    T: float,
    Nt: int,
    S: int,
    drift_strength_hw: float,
    amp_bound: float,
    hard_amp: float,
    trajectory_reference: str = "full",
    trajectory_weight: float = 0.5,
    trajectory_checkpoints: str = "every2",
    trajectory_project_iters: int = 40,
    trajectory_project_lr: float = 0.05,
    trajectory_stage_weights: str = "",
    trajectory_stage_checkpoints: str = "",
    trajectory_stage_iters: str = "",
    trajectory_stage_lrs: str = "",
    trajectory_stage_select: str = "last",
    trajectory_select_min_target_fidelity: float = 0.95,
    trajectory_select_min_mean_fidelity: float = 0.95,
    trajectory_select_min_worst_fidelity: float = 0.75,
    trajectory_auto_polish: int = 1,
    trajectory_polish_objective: str = "softmin_all",
    trajectory_polish_softmin_tau: float = 0.05,
    trajectory_polish_stage_weights: str = "0.9,0.8,0.7,0.6,0.5",
    trajectory_polish_stage_checkpoints: str = "auto",
    trajectory_polish_stage_iters: str = "40,40,40,40,40",
    trajectory_polish_stage_lrs: str = "0.03,0.03,0.02,0.02,0.01",
    trajectory_init: str = "linear",
    compiler_eps: float = 0.02,
    budget_frac: float = 1.0,
    max_terms_per_slice: int = 48,
    compiler_sort_mode: str = "low_degree",
    product_time_scale: float = 1.0,
    projection_clip: float = 5.0,
    projection_backtracks: int = 3,
    projection_accept_mode: str = "soft",
    projection_accept_drop: float = 2e-3,
    projection_threshold: float = 0.999,
    trajectory_optimizer: str = "adam",
    trajectory_lbfgs_maxls: int = 20,
    trajectory_lbfgs_ftol: float = 1e-9,
    trajectory_lbfgs_gtol: float = 1e-5,
    trajectory_objective: str = "weighted_sum",
    trajectory_softmin_tau: float = 0.02,
    batch2_residual_tol: float = 1e-6,
    batch2_max_frames: int = 64,
    batch2_frame_tol: float = 1e-10,
    constructive_trotter_reps: int = 1,
    pauli_audit_tol: float = 1e-10,
    diagnostics: Optional[list[dict[str, Any]]] = None,
    term_diagnostics: Optional[list[dict[str, Any]]] = None,
    reporter: Optional[ConsoleReporter] = None,
    progress_every: int = 10,
    H0_base_dense: Optional[np.ndarray] = None,
    ctrl_stack_dense: Optional[np.ndarray] = None,
    psi0_vec: Optional[np.ndarray] = None,
    target_vec: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Project an EA state trajectory into physical local controls only."""
    reference = build_ea_reference_trajectory(opt, psi0, target, trajectory_reference)
    init_summary: dict[str, Any] | None = None
    if reporter is not None:
        reporter.info(
            "[EATRAJ] reference={} controls_used={}/{} targetF={:.6f}".format(
                str(trajectory_reference),
                int(reference["controls_used"]),
                int(reference["controls_available"]),
                float(reference["target_fidelity"]),
            )
        )

    init_mode = str(trajectory_init).strip().lower()
    if init_mode == "zero":
        u0 = np.zeros((int(Nt) * int(S), len(ctrl_labels)), dtype=float)
    elif init_mode == "product":
        u0 = compile_ea_to_controls_product_aware(
            opt=opt,
            ctrl_labels=ctrl_labels,
            T=T,
            Nt=Nt,
            S=S,
            amp_bound=amp_bound,
            hard_amp=hard_amp,
            drift_strength_hw=drift_strength_hw,
            compiler_eps=compiler_eps,
            budget_frac=budget_frac,
            max_terms_per_slice=max_terms_per_slice,
            product_bch_reps=1,
            product_max_degree=2,
            compiler_sort_mode=compiler_sort_mode,
            compiler_operation_mode="product",
            product_time_scale=product_time_scale,
            diagnostics=diagnostics,
            term_diagnostics=term_diagnostics,
            reporter=reporter,
            progress_label="EATrajInitialProduct",
        )
    elif init_mode in ("batch2", "constructive_batch2"):
        u0, init_summary = compile_ea_to_controls_constructive_batch2(
            opt=opt,
            ctrl_labels=ctrl_labels,
            T=T,
            Nt=Nt,
            S=S,
            amp_bound=amp_bound,
            hard_amp=hard_amp,
            drift_strength_hw=drift_strength_hw,
            product_time_scale=product_time_scale,
            term_theta_threshold=1e-10,
            pauli_tol=float(pauli_audit_tol),
            residual_tol=float(batch2_residual_tol),
            max_frames=int(batch2_max_frames),
            frame_tol=float(batch2_frame_tol),
            product_bch_reps=1,
            trotter_reps=int(constructive_trotter_reps),
            diagnostics=diagnostics,
            term_diagnostics=term_diagnostics,
            reporter=reporter,
            progress_label="EATrajInitialBatch2",
        )
    else:
        u0 = compile_ea_to_controls(
            opt=opt,
            ctrl_labels=ctrl_labels,
            T=T,
            Nt=Nt,
            S=S,
            amp_bound=amp_bound,
            hard_amp=hard_amp,
            linear_split=True,
            verbose=False,
            compiler_eps=compiler_eps,
            budget_frac=budget_frac,
            max_terms_per_slice=max_terms_per_slice,
            diagnostics=diagnostics,
            reporter=reporter,
            progress_label="EATrajInitialLinear",
        )

    H0_phys = float(drift_strength_hw) * H0_base
    H0_phys_dense = None if H0_base_dense is None else float(drift_strength_hw) * H0_base_dense
    stage_weights = parse_csv_floats(str(trajectory_stage_weights), [])
    if stage_weights:
        stage_n = len(stage_weights)
        stage_checkpoints = extend_to_length(
            parse_csv_strings(str(trajectory_stage_checkpoints), [str(trajectory_checkpoints)]),
            stage_n,
        )
        stage_iters = extend_to_length(
            parse_csv_ints(str(trajectory_stage_iters), [int(trajectory_project_iters)]),
            stage_n,
        )
        stage_lrs = extend_to_length(
            parse_csv_floats(str(trajectory_stage_lrs), [float(trajectory_project_lr)]),
            stage_n,
        )
    else:
        stage_weights = [float(trajectory_weight)]
        stage_checkpoints = [str(trajectory_checkpoints)]
        stage_iters = [int(trajectory_project_iters)]
        stage_lrs = [float(trajectory_project_lr)]

    u_projected = u0
    selected_u = u0
    selected_stage_summary: dict[str, Any] | None = None
    best_stage_score: tuple[float, float, float] | None = None
    stage_summaries: list[dict[str, Any]] = []
    project_summary: dict[str, Any] | None = None
    audit_H0_arr = H0_phys_dense if H0_phys_dense is not None else qobj_to_dense_matrix(H0_phys)
    audit_ctrl_stack = ctrl_stack_dense if ctrl_stack_dense is not None else build_dense_operator_stack(ctrl_axes)
    if audit_ctrl_stack is None:
        audit_ctrl_stack = np.stack([qobj_to_dense_matrix(op) for op in ctrl_axes], axis=0)
    audit_psi0_arr = np.array(psi0_vec if psi0_vec is not None else qobj_to_dense_vector(psi0), copy=False)
    audit_target_arr = np.array(target_vec if target_vec is not None else qobj_to_dense_vector(target), copy=False)
    audit_dt = float(T) / max(1, int(Nt) * int(S))
    audit_ref_by_boundary = resample_reference_states(np.asarray(reference["states"], dtype=np.complex128), int(Nt), int(S))
    audit_boundaries = trajectory_checkpoint_indices(int(Nt), str(trajectory_checkpoints))
    audit_targets = {int(k) * int(S): audit_ref_by_boundary[int(k) * int(S)] for k in audit_boundaries}
    if reporter is not None:
        reporter.info(
            f"[EATRAJ] projecting EA trajectory into {len(ctrl_labels)} physical controls "
            f"over {len(stage_weights)} stage(s)."
        )

    select_mode = str(trajectory_stage_select).strip().lower()

    def score_stage(stage_row: dict[str, Any], stage_index: int) -> tuple[float, ...]:
        if select_mode == "best_target":
            return (
                -float(stage_row.get("audit_target_fidelity_after", 0.0)),
                -float(stage_row.get("audit_trajectory_fidelity_after", 0.0)),
                -float(stage_row.get("audit_trajectory_fidelity_min_after", 0.0)),
            )
        if select_mode == "best_objective":
            return (
                -float(stage_row.get("objective_after", 0.0)),
                -float(stage_row.get("target_fidelity_after", 0.0)),
                -float(stage_row.get("trajectory_fidelity_after", 0.0)),
            )
        if select_mode == "best_gate":
            audit_target = float(stage_row.get("audit_target_fidelity_after", 0.0))
            audit_mean = float(stage_row.get("audit_trajectory_fidelity_after", 0.0))
            audit_min = float(stage_row.get("audit_trajectory_fidelity_min_after", 0.0))
            target_deficit = max(0.0, float(trajectory_select_min_target_fidelity) - audit_target)
            traj_deficit = max(0.0, float(trajectory_select_min_mean_fidelity) - audit_mean)
            traj_min_deficit = max(0.0, float(trajectory_select_min_worst_fidelity) - audit_min)
            return (
                target_deficit + traj_deficit + traj_min_deficit,
                target_deficit,
                traj_deficit,
                traj_min_deficit,
                -audit_target,
                -audit_mean,
                -audit_min,
                float(stage_index),
            )
        return (float(stage_index), 0.0, 0.0)

    def record_projected_stage(
        stage_index: int,
        stage_u: np.ndarray,
        stage_project_summary: dict[str, Any],
        stage_weight: float,
        stage_checkpoint_mode: str,
        stage_iters_used: int,
        stage_lr: float,
        *,
        polish_stage: int = 0,
    ) -> tuple[np.ndarray, dict[str, Any], tuple[float, ...]]:
        stage_row = dict(stage_project_summary)
        _stage_audit_obj, _stage_audit_grad, stage_audit_metrics = trajectory_objective_gradient_dense(
            audit_H0_arr,
            audit_ctrl_stack,
            stage_u,
            audit_dt,
            audit_psi0_arr,
            audit_target_arr,
            audit_targets,
            checkpoint_weight=1.0,
            final_weight=0.0,
            clip=0.0,
        )
        stage_row.update(
            {
                "stage": int(stage_index),
                "polish_stage": int(polish_stage),
                "trajectory_weight": float(stage_weight),
                "checkpoint_mode": str(stage_checkpoint_mode),
                "project_iters_requested": int(stage_iters_used),
                "project_lr": float(stage_lr),
                "audit_checkpoint_mode": str(trajectory_checkpoints),
                "audit_checkpoint_count": int(len(audit_targets)),
                "audit_checkpoint_steps": [int(x) for x in stage_audit_metrics.get("checkpoint_steps", [])],
                "audit_checkpoint_fidelities_after": [float(x) for x in stage_audit_metrics.get("checkpoint_fidelities", [])],
                "audit_target_fidelity_after": float(stage_audit_metrics["target_fidelity"]),
                "audit_trajectory_fidelity_after": float(stage_audit_metrics["checkpoint_fidelity_mean"]),
                "audit_trajectory_fidelity_min_after": float(stage_audit_metrics["checkpoint_fidelity_min"]),
            }
        )
        stage_score = score_stage(stage_row, stage_index)
        stage_row["selection_score"] = [float(x) for x in stage_score]
        stage_summaries.append(stage_row)
        return stage_u, stage_row, stage_score

    for s_idx, (w, ckpt, iters, lr) in enumerate(zip(stage_weights, stage_checkpoints, stage_iters, stage_lrs), start=1):
        if reporter is not None:
            reporter.info(
                "[EATRAJ] stage {} start: weight={} checkpoints={} iters={} lr={}".format(
                    s_idx, float(w), str(ckpt), int(iters), float(lr)
                )
            )
        u_projected, project_summary = trajectory_hardware_project(
            psi0=psi0,
            target=target,
            H0=H0_phys,
            ctrl_axes=ctrl_axes,
            T=T,
            u_init=u_projected,
            reference_states=reference["states"],
            Nt=Nt,
            S=S,
            checkpoint_mode=str(ckpt),
            trajectory_weight=float(w),
            project_iters=int(iters),
            project_lr=float(lr),
            amp_bound=amp_bound,
            clip=projection_clip,
            backtracks=projection_backtracks,
            accept_mode=projection_accept_mode,
            accept_drop=projection_accept_drop,
            threshold=projection_threshold,
            reporter=reporter,
            progress_every=progress_every,
            label=f"EATrajectoryProject{s_idx}",
            optimizer=str(trajectory_optimizer),
            lbfgs_maxls=int(trajectory_lbfgs_maxls),
            lbfgs_ftol=float(trajectory_lbfgs_ftol),
            lbfgs_gtol=float(trajectory_lbfgs_gtol),
            objective_mode=str(trajectory_objective),
            objective_softmin_tau=float(trajectory_softmin_tau),
            H0_dense=H0_phys_dense,
            ctrl_stack_dense=ctrl_stack_dense,
            psi0_vec=psi0_vec,
            target_vec=target_vec,
        )
        _stage_u, stage_row, stage_score = record_projected_stage(
            int(s_idx),
            u_projected,
            project_summary,
            float(w),
            str(ckpt),
            int(iters),
            float(lr),
        )
        if best_stage_score is None or stage_score < best_stage_score:
            best_stage_score = stage_score
            selected_u = u_projected.copy()
            selected_stage_summary = dict(stage_row)
    if project_summary is None:
        raise RuntimeError("EA trajectory projection produced no stage summary")
    polish_used = False
    if (
        int(trajectory_auto_polish)
        and select_mode == "best_gate"
        and best_stage_score is not None
        and float(best_stage_score[0]) > 1e-12
    ):
        polish_used = True
        polish_weights = parse_csv_floats(str(trajectory_polish_stage_weights), [0.9, 0.8, 0.7, 0.6, 0.5])
        polish_n = len(polish_weights)
        polish_checkpoint_default = str(trajectory_checkpoints)
        if str(trajectory_polish_stage_checkpoints).strip().lower() in ("", "auto"):
            polish_checkpoint_default = "every2" if int(Nt) <= 10 else str(trajectory_checkpoints)
            polish_checkpoint_text = ",".join([polish_checkpoint_default] * polish_n)
        else:
            polish_checkpoint_text = str(trajectory_polish_stage_checkpoints)
        polish_checkpoints = extend_to_length(parse_csv_strings(polish_checkpoint_text, [polish_checkpoint_default]), polish_n)
        polish_iters = extend_to_length(parse_csv_ints(str(trajectory_polish_stage_iters), [40]), polish_n)
        polish_lrs = extend_to_length(parse_csv_floats(str(trajectory_polish_stage_lrs), [0.03]), polish_n)
        polish_u = u0.copy()
        if reporter is not None:
            reporter.info(
                "[EATRAJ] best_gate did not pass; running soft-min polish over {} stage(s).".format(
                    int(polish_n)
                )
            )
        for r_idx, (w, ckpt, iters, lr) in enumerate(
            zip(polish_weights, polish_checkpoints, polish_iters, polish_lrs),
            start=1,
        ):
            stage_index = len(stage_summaries) + 1
            if reporter is not None:
                reporter.info(
                    "[EATRAJ] polish stage {} start: weight={} checkpoints={} iters={} lr={} objective={}".format(
                        int(r_idx),
                        float(w),
                        str(ckpt),
                        int(iters),
                        float(lr),
                        str(trajectory_polish_objective),
                    )
                )
            polish_u, polish_summary = trajectory_hardware_project(
                psi0=psi0,
                target=target,
                H0=H0_phys,
                ctrl_axes=ctrl_axes,
                T=T,
                u_init=polish_u,
                reference_states=reference["states"],
                Nt=Nt,
                S=S,
                checkpoint_mode=str(ckpt),
                trajectory_weight=float(w),
                project_iters=int(iters),
                project_lr=float(lr),
                amp_bound=amp_bound,
                clip=projection_clip,
                backtracks=projection_backtracks,
                accept_mode=projection_accept_mode,
                accept_drop=projection_accept_drop,
                threshold=projection_threshold,
                reporter=reporter,
                progress_every=progress_every,
                label=f"EATrajectoryPolish{r_idx}",
                optimizer=str(trajectory_optimizer),
                lbfgs_maxls=int(trajectory_lbfgs_maxls),
                lbfgs_ftol=float(trajectory_lbfgs_ftol),
                lbfgs_gtol=float(trajectory_lbfgs_gtol),
                objective_mode=str(trajectory_polish_objective),
                objective_softmin_tau=float(trajectory_polish_softmin_tau),
                H0_dense=H0_phys_dense,
                ctrl_stack_dense=ctrl_stack_dense,
                psi0_vec=psi0_vec,
                target_vec=target_vec,
            )
            _polish_u, polish_row, polish_score = record_projected_stage(
                int(stage_index),
                polish_u,
                polish_summary,
                float(w),
                str(ckpt),
                int(iters),
                float(lr),
                polish_stage=int(r_idx),
            )
            polish_row["polish_objective"] = str(trajectory_polish_objective)
            polish_row["polish_softmin_tau"] = float(trajectory_polish_softmin_tau)
            if best_stage_score is None or polish_score < best_stage_score:
                best_stage_score = polish_score
                selected_u = polish_u.copy()
                selected_stage_summary = dict(polish_row)
    if select_mode in ("best_target", "best_objective", "best_gate") and selected_stage_summary is not None:
        u_projected = selected_u
        project_summary = dict(selected_stage_summary)
        project_summary["selected_by"] = select_mode
        project_summary["best_stage_score"] = [float(x) for x in best_stage_score] if best_stage_score is not None else []
    else:
        project_summary = dict(project_summary)
        project_summary["selected_by"] = "last"
    audit_initial_obj, _audit_initial_grad, audit_initial_metrics = trajectory_objective_gradient_dense(
        audit_H0_arr,
        audit_ctrl_stack,
        u0,
        audit_dt,
        audit_psi0_arr,
        audit_target_arr,
        audit_targets,
        checkpoint_weight=1.0,
        final_weight=0.0,
        clip=0.0,
    )
    audit_final_obj, _audit_final_grad, audit_final_metrics = trajectory_objective_gradient_dense(
        audit_H0_arr,
        audit_ctrl_stack,
        u_projected,
        audit_dt,
        audit_psi0_arr,
        audit_target_arr,
        audit_targets,
        checkpoint_weight=1.0,
        final_weight=0.0,
        clip=0.0,
    )
    project_summary.update(
        {
            "audit_checkpoint_mode": str(trajectory_checkpoints),
            "checkpoint_mode": str(trajectory_checkpoints),
            "checkpoint_count": int(len(audit_targets)),
            "checkpoint_steps": [int(x) for x in audit_final_metrics.get("checkpoint_steps", [])],
            "checkpoint_fidelities_before": [float(x) for x in audit_initial_metrics.get("checkpoint_fidelities", [])],
            "checkpoint_fidelities_after": [float(x) for x in audit_final_metrics.get("checkpoint_fidelities", [])],
            "audit_objective_before": float(audit_initial_obj),
            "audit_objective_after": float(audit_final_obj),
            "target_fidelity_before": float(audit_initial_metrics["target_fidelity"]),
            "target_fidelity_after": float(audit_final_metrics["target_fidelity"]),
            "trajectory_fidelity_before": float(audit_initial_metrics["checkpoint_fidelity_mean"]),
            "trajectory_fidelity_after": float(audit_final_metrics["checkpoint_fidelity_mean"]),
            "trajectory_fidelity_min_after": float(audit_final_metrics["checkpoint_fidelity_min"]),
        }
    )
    summary = {
        "mode": "ea_trajectory_hardware_project",
        "hardware_executable": True,
        "effective_control_mode": False,
        "reference": {
            "mode": str(trajectory_reference),
            "degree_limit": reference["degree_limit"],
            "controls_used": int(reference["controls_used"]),
            "controls_available": int(reference["controls_available"]),
            "target_fidelity": float(reference["target_fidelity"]),
        },
        "projection": project_summary,
        "projection_stages": stage_summaries,
        "projection_stage_select": str(trajectory_stage_select),
        "trajectory_objective": str(trajectory_objective),
        "trajectory_softmin_tau": float(trajectory_softmin_tau),
        "trajectory_auto_polish": bool(int(trajectory_auto_polish)),
        "trajectory_polish_used": bool(polish_used),
        "trajectory_polish_objective": str(trajectory_polish_objective),
        "trajectory_polish_softmin_tau": float(trajectory_polish_softmin_tau),
        "physical_control_count": int(len(ctrl_labels)),
        "seed_shape": list(u_projected.shape),
        "seed_rms": float(rms(u_projected)),
        "seed_max_abs": float(np.max(np.abs(u_projected))) if u_projected.size else 0.0,
        "initialization": init_mode,
        "initialization_summary": init_summary,
    }
    if reporter is not None:
        reporter.info(
            "[EATRAJ] hardware target F before/after={:.6f}/{:.6f}; trajectory F before/after={:.6f}/{:.6f}".format(
                float(project_summary["target_fidelity_before"]),
                float(project_summary["target_fidelity_after"]),
                float(project_summary["trajectory_fidelity_before"]),
                float(project_summary["trajectory_fidelity_after"]),
            )
        )
    return u_projected, summary


def compile_ea_to_target_continuation_project(
    opt: ImprovedPolynomialPontryagin,
    ctrl_axes: list[qt.Qobj],
    ctrl_labels: list[str],
    H0_base: qt.Qobj,
    psi0: qt.Qobj,
    target: qt.Qobj,
    T: float,
    Nt: int,
    S: int,
    drift_strength_hw: float,
    amp_bound: float,
    hard_amp: float,
    trajectory_reference: str = "full",
    continuation_iters: str = "30,20,20",
    continuation_lrs: str = "0.05,0.04,0.03",
    continuation_weights: str = "0,0.1,0.2",
    continuation_checkpoints: str = "final,late2,late4",
    continuation_min_improve: float = 1e-4,
    continuation_init: str = "linear",
    continuation_policy: str = "adaptive",
    continuation_escape_weights: str = "0.5,0.9",
    continuation_escape_checkpoints: str = "late2,all",
    continuation_escape_iters: str = "30,80",
    continuation_target_stall_tol: Optional[float] = None,
    continuation_max_stages: int = 4,
    compiler_eps: float = 0.02,
    budget_frac: float = 1.0,
    max_terms_per_slice: int = 48,
    compiler_sort_mode: str = "low_degree",
    product_time_scale: float = 1.0,
    projection_clip: float = 5.0,
    projection_backtracks: int = 3,
    projection_accept_mode: str = "soft",
    projection_accept_drop: float = 2e-3,
    projection_threshold: float = 0.999,
    diagnostics: Optional[list[dict[str, Any]]] = None,
    term_diagnostics: Optional[list[dict[str, Any]]] = None,
    reporter: Optional[ConsoleReporter] = None,
    progress_every: int = 10,
    H0_base_dense: Optional[np.ndarray] = None,
    ctrl_stack_dense: Optional[np.ndarray] = None,
    psi0_vec: Optional[np.ndarray] = None,
    target_vec: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Target-first EA continuation projected into physical controls only."""
    reference = build_ea_reference_trajectory(opt, psi0, target, trajectory_reference)
    if reporter is not None:
        reporter.info(
            "[TARGETCONT] reference={} controls_used={}/{} targetF={:.6f}".format(
                str(trajectory_reference),
                int(reference["controls_used"]),
                int(reference["controls_available"]),
                float(reference["target_fidelity"]),
            )
        )

    init_mode = str(continuation_init).strip().lower()
    if init_mode == "zero":
        current_u = np.zeros((int(Nt) * int(S), len(ctrl_labels)), dtype=float)
    elif init_mode == "product":
        current_u = compile_ea_to_controls_product_aware(
            opt=opt,
            ctrl_labels=ctrl_labels,
            T=T,
            Nt=Nt,
            S=S,
            amp_bound=amp_bound,
            hard_amp=hard_amp,
            drift_strength_hw=drift_strength_hw,
            compiler_eps=compiler_eps,
            budget_frac=budget_frac,
            max_terms_per_slice=max_terms_per_slice,
            product_bch_reps=1,
            product_max_degree=2,
            compiler_sort_mode=compiler_sort_mode,
            compiler_operation_mode="product",
            product_time_scale=product_time_scale,
            diagnostics=diagnostics,
            term_diagnostics=term_diagnostics,
            reporter=reporter,
            progress_label="TargetContInitialProduct",
        )
    else:
        current_u = compile_ea_to_controls(
            opt=opt,
            ctrl_labels=ctrl_labels,
            T=T,
            Nt=Nt,
            S=S,
            amp_bound=amp_bound,
            hard_amp=hard_amp,
            linear_split=True,
            verbose=False,
            compiler_eps=compiler_eps,
            budget_frac=budget_frac,
            max_terms_per_slice=max_terms_per_slice,
            diagnostics=diagnostics,
            reporter=reporter,
            progress_label="TargetContInitialLinear",
        )

    policy = str(continuation_policy).strip().lower()
    if policy not in ("manual", "adaptive"):
        policy = "manual"
    manual_iters = parse_csv_ints(continuation_iters, [30, 20, 20])
    manual_n = len(manual_iters)
    manual_lrs = extend_to_length(parse_csv_floats(continuation_lrs, [0.05, 0.04, 0.03]), manual_n)
    manual_weights = extend_to_length(parse_csv_floats(continuation_weights, [0.0, 0.1, 0.2]), manual_n)
    manual_checkpoints = extend_to_length(parse_csv_strings(continuation_checkpoints, ["final", "late2", "late4"]), manual_n)
    max_stages = int(continuation_max_stages)
    stall_tol = float(continuation_target_stall_tol) if continuation_target_stall_tol is not None else float(continuation_min_improve)

    stage_plan: list[dict[str, Any]] = []
    if policy == "adaptive":
        stage_plan.append(
            {
                "stage_kind": "target-only",
                "iters": int(manual_iters[0]),
                "lr": float(manual_lrs[0]),
                "weight": 0.0,
                "checkpoints": "final",
                "source": "adaptive_base",
            }
        )
        light_iters = int(manual_iters[1] if manual_n > 1 else manual_iters[0])
        light_lr = float(manual_lrs[1] if manual_n > 1 else manual_lrs[0])
        light_weight = float(manual_weights[1] if manual_n > 1 else 0.1)
        light_checkpoint = str(manual_checkpoints[1] if manual_n > 1 else "late2")
        if light_weight > 0.0:
            stage_plan.append(
                {
                    "stage_kind": "light-guided",
                    "iters": light_iters,
                    "lr": light_lr,
                    "weight": light_weight,
                    "checkpoints": light_checkpoint,
                    "source": "adaptive_light",
                }
            )

        escape_weights = parse_csv_floats(continuation_escape_weights, [0.5, 0.9])
        escape_iters = extend_to_length(parse_csv_ints(continuation_escape_iters, [30, 80]), len(escape_weights))
        escape_checkpoints = extend_to_length(parse_csv_strings(continuation_escape_checkpoints, ["late2", "all"]), len(escape_weights))
        fallback_escape_lr = max(float(x) for x in manual_lrs) if manual_lrs else 0.05
        for ei, escape_weight in enumerate(escape_weights):
            stage_plan.append(
                {
                    "stage_kind": "escape-guided",
                    "iters": int(escape_iters[ei]),
                    "lr": float(fallback_escape_lr),
                    "weight": float(escape_weight),
                    "checkpoints": str(escape_checkpoints[ei]),
                    "source": "adaptive_escape",
                }
            )
        if max_stages > 0:
            stage_plan = stage_plan[:max_stages]
    else:
        for si in range(manual_n):
            stage_plan.append(
                {
                    "stage_kind": "target-only" if float(manual_weights[si]) <= 0.0 else "manual-guided",
                    "iters": int(manual_iters[si]),
                    "lr": float(manual_lrs[si]),
                    "weight": float(manual_weights[si]),
                    "checkpoints": str(manual_checkpoints[si]),
                    "source": "manual",
                }
            )
        if max_stages > 0:
            stage_plan = stage_plan[:max_stages]

    H0_phys = float(drift_strength_hw) * H0_base
    H0_phys_dense = None if H0_base_dense is None else float(drift_strength_hw) * H0_base_dense
    ctrl_stack = ctrl_stack_dense if ctrl_stack_dense is not None else build_dense_operator_stack(ctrl_axes)
    psi0_arr = np.array(psi0_vec if psi0_vec is not None else qobj_to_dense_vector(psi0), copy=False)
    target_arr = np.array(target_vec if target_vec is not None else qobj_to_dense_vector(target), copy=False)

    target_before = simulate_state_fidelity(
        psi0,
        target,
        H0_phys,
        ctrl_axes,
        T,
        current_u,
        [0.0],
        H0_dense=H0_phys_dense,
        ctrl_stack_dense=ctrl_stack,
        psi0_vec=psi0_arr,
        target_vec=target_arr,
    )
    best_u = current_u.copy()
    best_target = float(target_before)
    current_target = float(target_before)
    stage_rows: list[dict[str, Any]] = []
    previous_stage_improvement = float("inf")
    adaptive_stop_reason = "completed"

    for si, plan_row in enumerate(stage_plan):
        if policy == "adaptive" and si > 0 and best_target >= 0.95 and str(plan_row["stage_kind"]) == "escape-guided":
            adaptive_stop_reason = "target_sufficient_before_escape"
            break
        if best_target >= float(projection_threshold):
            adaptive_stop_reason = "projection_threshold"
            break

        started_from_best_target_seed = False
        if (
            policy == "adaptive"
            and str(plan_row["stage_kind"]) == "escape-guided"
            and current_target < best_target + max(0.0, stall_tol)
        ):
            current_u = best_u.copy()
            current_target = best_target
            started_from_best_target_seed = True

        requested_weight = float(plan_row["weight"])
        used_weight = max(0.0, min(1.0, requested_weight))
        adapted = False
        adaptation_note = ""
        if policy == "manual" and si > 0 and used_weight > 0.0 and previous_stage_improvement < float(continuation_min_improve):
            used_weight = 0.25 * used_weight
            adapted = True
            adaptation_note = "reduced_after_target_stall"
        ckpt = str(plan_row["checkpoints"])
        if reporter is not None:
            reporter.info(
                "[TARGETCONT] stage {} start: kind={} iters={} lr={} weight={} checkpoints={}{}".format(
                    si + 1,
                    str(plan_row["stage_kind"]),
                    int(plan_row["iters"]),
                    float(plan_row["lr"]),
                    used_weight,
                    ckpt,
                    (" reset_to_best_target_seed" if started_from_best_target_seed else "")
                    + (f" {adaptation_note}" if adapted else ""),
                )
            )

        u_stage, stage_summary = trajectory_hardware_project(
            psi0=psi0,
            target=target,
            H0=H0_phys,
            ctrl_axes=ctrl_axes,
            T=T,
            u_init=current_u,
            reference_states=reference["states"],
            Nt=Nt,
            S=S,
            checkpoint_mode=ckpt,
            trajectory_weight=used_weight,
            project_iters=int(plan_row["iters"]),
            project_lr=float(plan_row["lr"]),
            amp_bound=amp_bound,
            clip=projection_clip,
            backtracks=projection_backtracks,
            accept_mode=projection_accept_mode,
            accept_drop=projection_accept_drop,
            threshold=projection_threshold,
            reporter=reporter,
            progress_every=progress_every,
            label=f"TargetContStage{si + 1}",
            H0_dense=H0_phys_dense,
            ctrl_stack_dense=ctrl_stack,
            psi0_vec=psi0_arr,
            target_vec=target_arr,
        )
        target_after = float(stage_summary.get("target_fidelity_after", 0.0))
        target_stage_before = float(stage_summary.get("target_fidelity_before", current_target))
        improvement = target_after - target_stage_before
        accepted = target_after + float(projection_accept_drop) >= current_target
        if accepted:
            current_u = u_stage.copy()
            current_target = target_after
        else:
            current_u = best_u.copy()
            current_target = best_target
        if target_after > best_target:
            best_target = target_after
            best_u = u_stage.copy()
        previous_stage_improvement = improvement
        target_stalled = improvement < stall_tol

        row = {
            "stage": int(si + 1),
            "policy": policy,
            "stage_kind": str(plan_row["stage_kind"]),
            "source": str(plan_row["source"]),
            "started_from_best_target_seed": bool(started_from_best_target_seed),
            "iters": int(plan_row["iters"]),
            "lr": float(plan_row["lr"]),
            "requested_weight": float(requested_weight),
            "used_weight": float(used_weight),
            "checkpoints": ckpt,
            "adapted_after_target_stall": bool(adapted),
            "adaptation_note": adaptation_note,
            "adapted_to_target_only": bool(adapted and used_weight <= 0.0),
            "accepted_for_next_stage": bool(accepted),
            "target_before": target_stage_before,
            "target_after": target_after,
            "target_improvement": float(improvement),
            "target_stalled": bool(target_stalled),
            "best_target_after_stage": float(best_target),
            "trajectory_before": float(stage_summary.get("trajectory_fidelity_before", 0.0)),
            "trajectory_after": float(stage_summary.get("trajectory_fidelity_after", 0.0)),
            "objective_before": float(stage_summary.get("objective_before", 0.0)),
            "objective_after": float(stage_summary.get("objective_after", 0.0)),
            "checkpoint_count": int(stage_summary.get("checkpoint_count", 0)),
            "seed_rms_after": float(stage_summary.get("seed_rms_after", 0.0)),
            "seed_max_abs_after": float(stage_summary.get("seed_max_abs_after", 0.0)),
        }
        stage_rows.append(row)
        if reporter is not None:
            reporter.info(
                "[TARGETCONT] stage {} done: target {:.6f}->{:.6f}; traj {:.6f}->{:.6f}; "
                "best={:.6f}; accepted={}".format(
                    si + 1,
                    row["target_before"],
                    row["target_after"],
                    row["trajectory_before"],
                    row["trajectory_after"],
                    row["best_target_after_stage"],
                    int(accepted),
                )
            )

    final_u = best_u.copy()
    summary = {
        "mode": "ea_target_continuation_project",
        "hardware_executable": True,
        "effective_control_mode": False,
        "reference": {
            "mode": str(trajectory_reference),
            "degree_limit": reference["degree_limit"],
            "controls_used": int(reference["controls_used"]),
            "controls_available": int(reference["controls_available"]),
            "target_fidelity": float(reference["target_fidelity"]),
        },
        "physical_control_count": int(len(ctrl_labels)),
        "seed_shape": list(final_u.shape),
        "seed_rms": float(rms(final_u)),
        "seed_max_abs": float(np.max(np.abs(final_u))) if final_u.size else 0.0,
        "initialization": init_mode,
        "policy": policy,
        "stage_plan_requested": stage_plan,
        "adaptive_stop_reason": adaptive_stop_reason,
        "target_fidelity_initial": float(target_before),
        "target_fidelity_best": float(best_target),
        "target_fidelity_final_seed": float(best_target),
        "min_improve": float(continuation_min_improve),
        "target_stall_tol": float(stall_tol),
        "stages": stage_rows,
    }
    if reporter is not None:
        reporter.info(
            "[TARGETCONT] best hardware target F={:.6f} from initial {:.6f}; physical_controls={}".format(
                float(best_target),
                float(target_before),
                len(ctrl_labels),
            )
        )
    return final_u, summary


def build_physical_control_reference_trajectory(
    u_phys: np.ndarray,
    H0_base: qt.Qobj,
    ctrl_axes: Sequence[qt.Qobj],
    psi0: qt.Qobj,
    target: qt.Qobj,
    T: float,
    Nt: int,
    S: int,
    drift_strength_hw: float,
) -> dict[str, Any]:
    """Return slice-boundary states generated by already-compiled physical controls."""
    slices = _dense_physical_slice_unitaries(
        u_phys,
        H0_base,
        ctrl_axes,
        T,
        Nt,
        S,
        drift_strength_hw,
    )
    psi0_vec = qobj_to_dense_vector(psi0)
    target_vec = qobj_to_dense_vector(target)
    states = np.empty((int(Nt) + 1, psi0_vec.size), dtype=np.complex128)
    states[0] = psi0_vec
    for k, U_slice in enumerate(slices):
        states[k + 1] = U_slice @ states[k]
        norm = float(np.linalg.norm(states[k + 1]))
        if norm > 1e-30:
            states[k + 1] /= norm
    final_alpha = complex(np.vdot(target_vec, states[-1]))
    return {
        "states": states,
        "qobjs": [qt.Qobj(states[k].reshape((-1, 1)), dims=psi0.dims) for k in range(states.shape[0])],
        "target_fidelity": float(abs(final_alpha) ** 2),
        "reference_mode": "constructive_physical_controls",
        "source_shape": list(np.asarray(u_phys).shape),
        "Nt": int(Nt),
        "S": int(S),
    }


def compile_ea_to_constructive_batch2_compress_project(
    opt: ImprovedPolynomialPontryagin,
    ctrl_axes: list[qt.Qobj],
    ctrl_labels: list[str],
    H0_base: qt.Qobj,
    psi0: qt.Qobj,
    target: qt.Qobj,
    T: float,
    Nt: int,
    S: int,
    reference_S: int,
    drift_strength_hw: float,
    amp_bound: float,
    hard_amp: float,
    product_time_scale: float,
    term_theta_threshold: float,
    pauli_tol: float,
    residual_tol: float,
    max_frames: int,
    frame_tol: float,
    product_bch_reps: int,
    trotter_reps: int,
    trajectory_weight: float,
    trajectory_checkpoints: str,
    trajectory_project_iters: int,
    trajectory_project_lr: float,
    projection_clip: float,
    projection_backtracks: int,
    projection_accept_mode: str,
    projection_accept_drop: float,
    projection_threshold: float,
    compression_target_guard: bool = True,
    compression_target_guard_tol: float = 0.0,
    diagnostics: Optional[list[dict[str, Any]]] = None,
    term_diagnostics: Optional[list[dict[str, Any]]] = None,
    pauli_audit_rows: Optional[list[dict[str, Any]]] = None,
    pauli_expansion_cache: Optional[tuple[Any, list[dict[str, Any]], dict[str, Any]]] = None,
    reporter: Optional[ConsoleReporter] = None,
    progress_every: int = 10,
    H0_base_dense: Optional[np.ndarray] = None,
    ctrl_stack_dense: Optional[np.ndarray] = None,
    psi0_vec: Optional[np.ndarray] = None,
    target_vec: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, dict[str, Any], np.ndarray, int]:
    """Audit a constructive reference, then compress it to the requested GRAPE grid."""
    ref_S = int(reference_S) if int(reference_S) > 0 else int(S)
    if reporter is not None:
        reporter.info(
            f"[BATCH2COMPRESS] building constructive reference with S_ref={ref_S}; "
            f"compressing to S={int(S)}."
        )
    reference_amp_bound = float(hard_amp) if float(hard_amp) > 0.0 else float(amp_bound)
    u_ref, batch2_summary = compile_ea_to_controls_constructive_batch2(
        opt=opt,
        ctrl_labels=ctrl_labels,
        T=T,
        Nt=Nt,
        S=ref_S,
        amp_bound=reference_amp_bound,
        hard_amp=hard_amp,
        drift_strength_hw=drift_strength_hw,
        product_time_scale=product_time_scale,
        term_theta_threshold=term_theta_threshold,
        pauli_tol=pauli_tol,
        residual_tol=residual_tol,
        max_frames=max_frames,
        frame_tol=frame_tol,
        product_bch_reps=product_bch_reps,
        trotter_reps=trotter_reps,
        diagnostics=diagnostics,
        term_diagnostics=term_diagnostics,
        pauli_audit_rows=pauli_audit_rows,
        pauli_expansion_cache=pauli_expansion_cache,
        reporter=reporter,
        progress_label="Batch2Reference",
    )
    reference = build_physical_control_reference_trajectory(
        u_ref,
        H0_base,
        ctrl_axes,
        psi0,
        target,
        T,
        Nt,
        ref_S,
        drift_strength_hw,
    )
    u0 = resample_controls(u_ref, int(Nt) * int(S))
    if float(amp_bound) > 0:
        np.clip(u0, -float(amp_bound), float(amp_bound), out=u0)
    H0_phys = float(drift_strength_hw) * H0_base
    H0_phys_dense = None if H0_base_dense is None else float(drift_strength_hw) * H0_base_dense
    if reporter is not None:
        reporter.info(
            "[BATCH2COMPRESS] reference targetF={:.6f}; projecting to {} physical controls for {} iteration(s).".format(
                float(reference["target_fidelity"]),
                len(ctrl_labels),
                int(trajectory_project_iters),
            )
        )
    u_projected, project_summary = trajectory_hardware_project(
        psi0=psi0,
        target=target,
        H0=H0_phys,
        ctrl_axes=ctrl_axes,
        T=T,
        u_init=u0,
        reference_states=reference["states"],
        Nt=Nt,
        S=S,
        checkpoint_mode=trajectory_checkpoints,
        trajectory_weight=trajectory_weight,
        project_iters=trajectory_project_iters,
        project_lr=trajectory_project_lr,
        amp_bound=amp_bound,
        clip=projection_clip,
        backtracks=projection_backtracks,
        accept_mode=projection_accept_mode,
        accept_drop=projection_accept_drop,
        threshold=projection_threshold,
        reporter=reporter,
        progress_every=progress_every,
        label="Batch2CompressProject",
        H0_dense=H0_phys_dense,
        ctrl_stack_dense=ctrl_stack_dense,
        psi0_vec=psi0_vec,
        target_vec=target_vec,
    )
    selected_u = u_projected
    guard_applied = False
    target_before = float(project_summary.get("target_fidelity_before", 0.0))
    target_after = float(project_summary.get("target_fidelity_after", 0.0))
    trajectory_before = float(project_summary.get("trajectory_fidelity_before", 0.0))
    trajectory_after = float(project_summary.get("trajectory_fidelity_after", 0.0))
    if bool(compression_target_guard) and target_after < target_before - float(compression_target_guard_tol):
        selected_u = u0.copy()
        guard_applied = True
        if reporter is not None:
            reporter.info(
                "[BATCH2COMPRESS] target guard kept pre-projection compressed seed: "
                "target {:.6f}->{:.6f} would decrease.".format(target_before, target_after)
            )
    summary = {
        "mode": "constructive_batch2_compress",
        "hardware_executable": True,
        "reference_S": int(ref_S),
        "compressed_S": int(S),
        "reference": {
            "target_fidelity": float(reference["target_fidelity"]),
            "source_shape": list(reference["source_shape"]),
        },
        "batch2_reference": batch2_summary,
        "compression": project_summary,
        "compression_target_guard": {
            "enabled": bool(compression_target_guard),
            "applied": bool(guard_applied),
            "tol": float(compression_target_guard_tol),
            "selected_seed": "pre_projection" if guard_applied else "projected",
            "selected_target_fidelity": float(target_before if guard_applied else target_after),
            "selected_trajectory_fidelity": float(trajectory_before if guard_applied else trajectory_after),
        },
        "seed_shape": list(selected_u.shape),
        "seed_rms": float(rms(selected_u)),
        "seed_max_abs": float(np.max(np.abs(selected_u))) if selected_u.size else 0.0,
    }
    if reporter is not None:
        reporter.info(
            "[BATCH2COMPRESS] target F before/after={:.6f}/{:.6f}; trajectory F before/after={:.6f}/{:.6f}".format(
                float(project_summary["target_fidelity_before"]),
                float(project_summary["target_fidelity_after"]),
                float(project_summary["trajectory_fidelity_before"]),
                float(project_summary["trajectory_fidelity_after"]),
            )
        )
    return selected_u, summary, u_ref, ref_S



                 


@dataclass
class OptConfig:
    lr: float = 0.06
    l2: float = 0.0
    amp: float = 1.0
    clip: float = 0.0
    backtracks: int = 0
    accept_mode: str = "hard"                    
    accept_drop: float = 0.0
    threshold: float = 0.99
    verbose: int = 0

    stall_enable: bool = False
    stall_gnorm: float = 1e-10
    stall_max_kicks: int = 0
    stall_kick_sigma: float = 0.0


class AdamAscent:
    def __init__(self, shape: tuple[int, int], lr: float, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
        self.lr = float(lr)
        self.b1 = float(b1)
        self.b2 = float(b2)
        self.eps = float(eps)
        self.t = 0
        self.m = np.zeros(shape, dtype=float)
        self.v = np.zeros(shape, dtype=float)

    def step(self, g: np.ndarray) -> np.ndarray:
        self.t += 1
        self.m = self.b1 * self.m + (1.0 - self.b1) * g
        self.v = self.b2 * self.v + (1.0 - self.b2) * (g * g)
        mhat = self.m / (1.0 - self.b1**self.t)
        vhat = self.v / (1.0 - self.b2**self.t)
        return self.lr * mhat / (np.sqrt(vhat) + self.eps)


def simulate_state_fidelity(
    psi0: qt.Qobj,
    target: qt.Qobj,
    H0: qt.Qobj,
    ctrl_axes: list[qt.Qobj],
    T: float,
    u: np.ndarray,
    jitter_list: Iterable[float] | None = None,
    H0_dense: Optional[np.ndarray] = None,
    ctrl_stack_dense: Optional[np.ndarray] = None,
    psi0_vec: Optional[np.ndarray] = None,
    target_vec: Optional[np.ndarray] = None,
) -> float:
    """Average state-transfer fidelity over dt-jitter values.

    jitter_list defines dt scaling factors epsilon such that dt_eff = dt * (1 + epsilon).
    """
    u = np.asarray(u, dtype=float)
    M = int(u.shape[0])
    dt = float(T) / M

    if jitter_list is None:
        jitters = [0.0]
    else:
        jitters = [float(eps) for eps in jitter_list]

    control_stack = ctrl_stack_dense if ctrl_stack_dense is not None else build_dense_operator_stack(ctrl_axes)
    if control_stack is not None:
        H0_dense_local = np.array(H0_dense if H0_dense is not None else qobj_to_dense_matrix(H0), copy=False)
        psi0_vec_local = np.array(psi0_vec if psi0_vec is not None else qobj_to_dense_vector(psi0), copy=False)
        target_vec_local = np.array(target_vec if target_vec is not None else qobj_to_dense_vector(target), copy=False)
        fidelity, _alphas, _states, _unitaries = forward_bundle_dense(
            H0_dense_local,
            control_stack,
            u,
            dt,
            psi0_vec_local,
            target_vec_local,
            jitters,
            store_steps=False,
        )
        return float(fidelity)

    F_accum = 0.0
    m = len(ctrl_axes)
    for eps in jitters:
        dt_eff = dt * (1.0 + eps)
        psi = psi0
        for k in range(M):
            H = H0
                          
            for j in range(m):
                uj = float(u[k, j])
                if abs(uj) > 1e-12:
                    H = H + uj * ctrl_axes[j]
            psi = expmH(H, dt_eff) * psi
        alpha = overlap(target, psi)
        F_accum += float(abs(alpha) ** 2)

    return float(F_accum / max(1, len(jitters)))


def grape_state_iters(
    psi0: qt.Qobj,
    target: qt.Qobj,
    H0: qt.Qobj,
    ctrl_axes: list[qt.Qobj],
    T: float,
    u_init: np.ndarray,
    opt_cfg: OptConfig,
    iters: int,
    seed: int,
    mask: np.ndarray | None = None,
    jitter_list: list[float] | None = None,
    label: str = "GRAPE",
    reporter: Optional[ConsoleReporter] = None,
    progress_every: int = 50,
    trace_rows: Optional[list[dict[str, Any]]] = None,
    H0_dense: Optional[np.ndarray] = None,
    ctrl_stack_dense: Optional[np.ndarray] = None,
    psi0_vec: Optional[np.ndarray] = None,
    target_vec: Optional[np.ndarray] = None,
) -> tuple[float, np.ndarray, int | None, int]:
    """Same as grape_state, but with an explicit iteration budget."""
    rng = np.random.RandomState(int(seed))

    u = np.asarray(u_init, dtype=float).copy()
    M, m = u.shape
    dt = float(T) / M

    if mask is None:
        mask = np.zeros(m, dtype=bool)

    if jitter_list is None:
        jitter_list = [0.0]
    else:
        jitter_list = [float(eps) for eps in jitter_list]

    if opt_cfg.amp > 0:
        np.clip(u, -opt_cfg.amp, opt_cfg.amp, out=u)
    if np.any(mask):
        u[:, mask] = 0.0

    opt = AdamAscent(u.shape, lr=opt_cfg.lr)
    bt_max = int(max(0, opt_cfg.backtracks))
    backtrack_scales = [1.0] + [0.5**k for k in range(1, bt_max + 1)]

    if ctrl_stack_dense is None:
        ctrl_stack_dense = build_dense_operator_stack(ctrl_axes)
    use_dense_backend = ctrl_stack_dense is not None
    if use_dense_backend:
        H0_dense_local = np.array(H0_dense if H0_dense is not None else qobj_to_dense_matrix(H0), copy=False)
        psi0_vec_local = np.array(psi0_vec if psi0_vec is not None else qobj_to_dense_vector(psi0), copy=False)
        target_vec_local = np.array(target_vec if target_vec is not None else qobj_to_dense_vector(target), copy=False)

    def fwd(u_curr: np.ndarray):
        if use_dense_backend:
            return forward_bundle_dense(
                H0_dense_local,
                ctrl_stack_dense,
                u_curr,
                dt,
                psi0_vec_local,
                target_vec_local,
                jitter_list,
                store_steps=True,
            )

        Fs: list[float] = []
        alphas: list[complex] = []
        psis: list[list[qt.Qobj]] = []
        Us: list[list[qt.Qobj]] = []

        for eps in jitter_list:
            dt_eff = dt * (1.0 + float(eps))
            psi: list[qt.Qobj] = [None] * (M + 1)                
            Usteps: list[qt.Qobj] = [None] * M                
            psi[0] = psi0
            for k in range(M):
                H = H0
                for j in range(m):
                    uj = float(u_curr[k, j])
                    if abs(uj) > 1e-12:
                        H = H + uj * ctrl_axes[j]
                Uk = expmH(H, dt_eff)
                Usteps[k] = Uk
                psi[k + 1] = Uk * psi[k]
            alpha = overlap(target, psi[M])
            Fs.append(float(abs(alpha) ** 2))
            alphas.append(alpha)
            psis.append(psi)
            Us.append(Usteps)

        return float(np.mean(Fs)), alphas, psis, Us

    def bwd(alphas: list[complex], psis: list[list[qt.Qobj]], Us: list[list[qt.Qobj]], u_curr: np.ndarray) -> np.ndarray:
        if use_dense_backend:
            return backward_gradient_dense(
                alphas,
                psis,                          
                Us,                            
                ctrl_stack_dense,
                target_vec_local,
                dt,
                u_curr,
                mask,
                opt_cfg.l2,
                opt_cfg.clip,
            )

        grad = np.zeros_like(u_curr)

        for r in range(len(jitter_list)):
            alpha = alphas[r]
            psi = psis[r]
            Usteps = Us[r]

            lam: list[qt.Qobj] = [None] * (M + 1)                
            lam[M] = target
            for k in range(M - 1, -1, -1):
                lam[k] = Usteps[k].dag() * lam[k + 1]

            alpha_conj = np.conj(alpha)
            for k in range(M):
                lam_k1 = lam[k + 1]
                psi_k1 = psi[k + 1]
                for j in range(m):
                    if mask[j]:
                        continue
                    mat_el = _scalar_from(lam_k1.dag() * (ctrl_axes[j] * psi_k1))
                    term = (-1j * dt) * mat_el
                    grad[k, j] += 2.0 * float(np.real(alpha_conj * term))

        grad /= max(1, len(jitter_list))

        if opt_cfg.l2 > 0:
            grad -= 2.0 * opt_cfg.l2 * u_curr

        if np.any(mask):
            grad[:, mask] = 0.0

        if opt_cfg.clip and opt_cfg.clip > 0:
            gnorm = float(np.linalg.norm(grad))
            if gnorm > opt_cfg.clip:
                grad *= (opt_cfg.clip / max(1e-30, gnorm))

        return grad

    F_curr, a_curr, p_curr, U_curr = fwd(u)
    bestF = float(F_curr)
    bestU = u.copy()
    iters_to_thresh: int | None = None
    trace = trace_rows if trace_rows is not None else []
    stop_reason = "max_iter"

    if opt_cfg.verbose:
        if reporter is None:
            print(f"[{label}] it=   0  F={bestF:.9f}")
        else:
            reporter.info(f"[{label}] it=   0  F={bestF:.9f}")

    kicks_used = 0
    t_start = time.time()

    for it in range(1, int(iters) + 1):
        g = bwd(a_curr, p_curr, U_curr, u)
        gnorm = float(np.linalg.norm(g))

                                                            
        if opt_cfg.stall_enable and (gnorm <= opt_cfg.stall_gnorm) and (kicks_used < opt_cfg.stall_max_kicks):
            kicks_used += 1
            u = u + float(opt_cfg.stall_kick_sigma) * rng.randn(*u.shape)
            if opt_cfg.amp > 0:
                np.clip(u, -opt_cfg.amp, opt_cfg.amp, out=u)
            if np.any(mask):
                u[:, mask] = 0.0
            F_curr, a_curr, p_curr, U_curr = fwd(u)
            if F_curr > bestF:
                bestF = float(F_curr)
                bestU = u.copy()
            if opt_cfg.verbose:
                if reporter is None:
                    print(f"[{label}] stall-kick #{kicks_used}: gnorm={gnorm:.3e}, F={F_curr:.9f}")
                else:
                    reporter.info(f"[{label}] stall-kick #{kicks_used}: gnorm={gnorm:.3e}, F={F_curr:.9f}")
            if trace_rows is not None:
                trace.append(
                    {
                        "iter": int(it),
                        "current_fidelity": float(F_curr),
                        "best_fidelity": float(bestF),
                        "grad_norm": float(gnorm),
                        "lr": float(opt.lr),
                        "accepted": 0,
                        "stall_kick": 1,
                        "kicks_used": int(kicks_used),
                        "elapsed_s": float(time.time() - t_start),
                    }
                )
            continue

        du_base = opt.step(g)

                                    
        accepted = False
        best_local_F = -1.0
        best_local_u = None
        best_local_pack = None

        for s in backtrack_scales:
            u_try = u + s * du_base
            if opt_cfg.amp > 0:
                np.clip(u_try, -opt_cfg.amp, opt_cfg.amp, out=u_try)
            if np.any(mask):
                u_try[:, mask] = 0.0

            F_try, a_try, p_try, U_try = fwd(u_try)

            if F_try > best_local_F:
                best_local_F = float(F_try)
                best_local_u = u_try
                best_local_pack = (a_try, p_try, U_try)

            if opt_cfg.accept_mode == "hard":
                ok = F_try >= F_curr
            else:
                ok = F_try >= (F_curr - float(opt_cfg.accept_drop))

            if ok:
                accepted = True
                u = u_try
                F_curr = float(F_try)
                a_curr, p_curr, U_curr = a_try, p_try, U_try
                break

        if not accepted:
                                                                                          
                                             
            if best_local_F > F_curr and best_local_u is not None and best_local_pack is not None:
                u = best_local_u
                F_curr = float(best_local_F)
                a_curr, p_curr, U_curr = best_local_pack
            else:
                opt.lr *= 0.95

        if F_curr > bestF:
            bestF = float(F_curr)
            bestU = u.copy()

        if iters_to_thresh is None and bestF >= opt_cfg.threshold:
            iters_to_thresh = int(it)

        if trace_rows is not None:
            trace.append(
                {
                    "iter": int(it),
                    "current_fidelity": float(F_curr),
                    "best_fidelity": float(bestF),
                    "grad_norm": float(gnorm),
                    "lr": float(opt.lr),
                    "accepted": int(accepted),
                    "stall_kick": 0,
                    "kicks_used": int(kicks_used),
                    "elapsed_s": float(time.time() - t_start),
                }
            )

        emit_progress = (it % max(1, progress_every) == 0) or (it == iters) or (bestF >= opt_cfg.threshold)
        if opt_cfg.verbose and emit_progress and reporter is None:
            print(f"[{label}] it={it:4d}  F={bestF:.9f}  (lr={opt.lr:.3g}, ||g||={gnorm:.3e})")
        elif reporter is not None and emit_progress:
            reporter.progress(
                label,
                it,
                int(iters),
                t_start,
                extra=f"best={bestF:.6f} F={F_curr:.6f} lr={opt.lr:.3g} |g|={gnorm:.3e}",
                force=(it == iters or bestF >= opt_cfg.threshold),
            )

        if bestF >= opt_cfg.threshold:
                                           
            stop_reason = "threshold"
            break

    wall = time.time() - t_start
    if opt_cfg.verbose:
        msg = f"[{label}] stage done: iters={it}, bestF={bestF:.9f}, wall={wall:.2f}s, reason={stop_reason}"
        if reporter is None:
            print(msg)
        else:
            reporter.info(msg)

    return bestF, bestU, iters_to_thresh, int(it)



                               


@dataclass
class Stage:
    scale: float
    S: int
    iters: int
    lr: float


def build_stages(
    scales: list[float],
    S1: int,
    S2: int,
    iters1: int,
    iters2: int,
    lr1: float,
    lr2: float,
) -> list[Stage]:
    """Stage schedule: one coarse stage per scale + optional final refinement."""
    if len(scales) == 0:
        scales = [1.0]

    stages: list[Stage] = []
    for i, sc in enumerate(scales):
        stages.append(Stage(scale=float(sc), S=int(S1), iters=int(iters1), lr=float(lr1)))

                                                  
        if i == len(scales) - 1 and (S2 != S1 or iters2 != iters1):
            if iters2 > 0:
                stages.append(Stage(scale=float(sc), S=int(S2), iters=int(iters2), lr=float(lr2)))

                                     
    stages = [st for st in stages if st.iters > 0]
    return stages


def drift_for_stage(
    H0_base: qt.Qobj,
    drift_strength_hw: float,
    stage_scale: float,
    homotopy_mode: str,
) -> qt.Qobj:
    """Compute the drift used in a stage.

    homotopy_mode:
      - 'mult': stage_scale is a multiplier of the physical drift strength.
                H0_stage = (stage_scale * drift_strength_hw) * H0_base

      - 'abs' : stage_scale is an absolute drift strength (same units as drift_strength_hw).
                H0_stage = stage_scale * H0_base

    The physical drift used for final evaluation is always:
      H0_phys = drift_strength_hw * H0_base
    """
    mode = homotopy_mode.lower().strip()
    if mode == "abs":
        coeff = float(stage_scale)
    else:
        coeff = float(stage_scale) * float(drift_strength_hw)
    return coeff * H0_base


def random_controls(
    M: int,
    m: int,
    rng: np.random.RandomState,
    amp_bound: float,
    init_mode: str,
    sigma: float,
    target_rms: float | None,
) -> np.ndarray:
    mode = init_mode.lower().strip()
    if mode == "uniform":
        u = rng.uniform(-amp_bound, amp_bound, size=(M, m))
        return u.astype(float)

    if mode == "rmsmatch":
        if target_rms is None:
            target_rms = 0.1 * amp_bound
        z = rng.randn(M, m)
        z_rms = rms(z)
        if z_rms < 1e-30:
            z_rms = 1.0
        u = (float(target_rms) / z_rms) * z
        if amp_bound > 0:
            np.clip(u, -amp_bound, amp_bound, out=u)
        return u.astype(float)

                       
    u = float(sigma) * rng.randn(M, m)
    if amp_bound > 0:
        np.clip(u, -amp_bound, amp_bound, out=u)
    return u.astype(float)


@dataclass
class RunResult:
    name: str
    seed: int
    initF_phys: float
    finalF_phys: float
    bestF_last_stage: float
    iters_to_threshold: int | None
    total_stage_iters: int
    wall_s: float


def run_method_stages(
    name: str,
    psi0: qt.Qobj,
    target: qt.Qobj,
    H0_base: qt.Qobj,
    drift_strength_hw: float,
    ctrl_axes: list[qt.Qobj],
    T: float,
    Nt: int,
    stages: list[Stage],
    homotopy_mode: str,
    u0: np.ndarray,
    opt_cfg_template: OptConfig,
    threshold: float,
    seed: int,
    jitter_list: list[float] | None,
    verbose: int,
    reporter: Optional[ConsoleReporter] = None,
    progress_every: int = 50,
    trace_dir: Optional[Path] = None,
    trace_prefix: str = "",
    checkpoint_manager: Optional[CheckpointManager] = None,
    H0_base_dense: Optional[np.ndarray] = None,
    ctrl_stack_dense: Optional[np.ndarray] = None,
    psi0_vec: Optional[np.ndarray] = None,
    target_vec: Optional[np.ndarray] = None,
) -> tuple[RunResult, np.ndarray]:
    """Run a multi-stage GRAPE refinement and return final physical fidelity."""
    t0 = time.time()

    H0_phys = drift_strength_hw * H0_base
    H0_phys_dense = None if H0_base_dense is None else (float(drift_strength_hw) * H0_base_dense)

                                     
    if len(stages) == 0:
        raise ValueError("No stages")

    M0 = Nt * stages[0].S
    u = resample_controls(u0, M0)

    if opt_cfg_template.amp > 0:
        np.clip(u, -opt_cfg_template.amp, opt_cfg_template.amp, out=u)

    initF_phys = simulate_state_fidelity(
        psi0,
        target,
        H0_phys,
        ctrl_axes,
        T,
        resample_controls(u, Nt * stages[-1].S),
        jitter_list,
        H0_dense=H0_phys_dense,
        ctrl_stack_dense=ctrl_stack_dense,
        psi0_vec=psi0_vec,
        target_vec=target_vec,
    )

    bestF_last_stage = -1.0
    iters_to_threshold: int | None = None
    total_iters = 0

    for si, st in enumerate(stages):
        H0_stage = drift_for_stage(H0_base, drift_strength_hw, st.scale, homotopy_mode)
        H0_stage_dense = None
        if H0_base_dense is not None:
            coeff = float(st.scale) if homotopy_mode.lower().strip() == "abs" else float(st.scale) * float(drift_strength_hw)
            H0_stage_dense = coeff * H0_base_dense

        M = Nt * st.S
        u = resample_controls(u, M)

        opt_cfg = OptConfig(**{**opt_cfg_template.__dict__})
        opt_cfg.lr = float(st.lr)
        opt_cfg.threshold = float(threshold)
        opt_cfg.verbose = int(verbose)

        stage_label = f"{name}:S{st.S}:scale{st.scale:g}"
        if reporter is not None:
            reporter.info(f"[STAGE] {stage_label} start")
        stage_trace: list[dict[str, Any]] = []
        F_stage, u, it_thresh, it_used = grape_state_iters(
            psi0=psi0,
            target=target,
            H0=H0_stage,
            ctrl_axes=ctrl_axes,
            T=T,
            u_init=u,
            opt_cfg=opt_cfg,
            iters=int(st.iters),
            seed=int(seed),
            mask=None,
            jitter_list=jitter_list,
            label=stage_label,
            reporter=reporter,
            progress_every=progress_every,
            trace_rows=stage_trace,
            H0_dense=H0_stage_dense,
            ctrl_stack_dense=ctrl_stack_dense,
            psi0_vec=psi0_vec,
            target_vec=target_vec,
        )

        total_iters += int(it_used)
        if trace_dir is not None:
            write_rows_csv(trace_dir / f"{trace_prefix}{name.lower()}_stage{si+1:02d}_trace.csv", stage_trace)
        if checkpoint_manager is not None:
            checkpoint_manager.save_npz(
                f"{trace_prefix}{name.lower()}_stage{si+1:02d}",
                controls=u,
                stage_index=np.array([si + 1], dtype=int),
                best_fidelity=np.array([F_stage], dtype=float),
            )
        if reporter is not None:
            reporter.info(f"[STAGE] {stage_label} done: bestF={F_stage:.6f} iters={it_used}")

        if si == len(stages) - 1:
            bestF_last_stage = float(F_stage)
            iters_to_threshold = it_thresh

                                                                  
    u_final = resample_controls(u, Nt * stages[-1].S)
    finalF_phys = simulate_state_fidelity(
        psi0,
        target,
        H0_phys,
        ctrl_axes,
        T,
        u_final,
        jitter_list,
        H0_dense=H0_phys_dense,
        ctrl_stack_dense=ctrl_stack_dense,
        psi0_vec=psi0_vec,
        target_vec=target_vec,
    )

    wall = time.time() - t0

    rr = RunResult(
        name=name,
        seed=int(seed),
        initF_phys=float(initF_phys),
        finalF_phys=float(finalF_phys),
        bestF_last_stage=float(bestF_last_stage),
        iters_to_threshold=iters_to_threshold,
        total_stage_iters=int(total_iters),
        wall_s=float(wall),
    )
    if checkpoint_manager is not None:
        checkpoint_manager.save_npz(
            f"{trace_prefix}{name.lower()}_final",
            controls=u_final,
            initF_phys=np.array([rr.initF_phys], dtype=float),
            finalF_phys=np.array([rr.finalF_phys], dtype=float),
        )
    return rr, u_final


def load_state_grow_parent_controls(path: str, expected_labels: list[str]) -> tuple[np.ndarray, list[str] | None]:
    """Load optional parent controls from NPZ or AWG-style CSV."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    if p.suffix.lower() == ".npz":
        data = np.load(p)
        for key in ("u_final", "controls", "u", "u_parent", "u5"):
            if key in data:
                return np.asarray(data[key], dtype=float), None
        raise ValueError(f"No recognized control array in {p}")
    with p.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        if len(header) < 2:
            raise ValueError(f"CSV parent control file {p} has no control columns")
        labels = header[1:]
        rows = [[float(x) for x in row[1:]] for row in reader if row]
    u = np.asarray(rows, dtype=float)
    if u.ndim != 2:
        raise ValueError(f"Could not parse control matrix from {p}")
    return u, labels if labels else expected_labels


def embed_parent_controls(
    u_parent: np.ndarray,
    parent_labels: list[str],
    child_labels: list[str],
) -> np.ndarray:
    """Embed N-1 controls into the matching first N-1 sites of the N system."""
    u_parent = np.asarray(u_parent, dtype=float)
    u_child = np.zeros((u_parent.shape[0], len(child_labels)), dtype=float)
    child_lookup = {label: j for j, label in enumerate(child_labels)}
    copied = 0
    for j, label in enumerate(parent_labels):
        idx = child_lookup.get(label)
        if idx is None:
            continue
        u_child[:, idx] = u_parent[:, j]
        copied += 1
    if copied == 0:
        raise ValueError("No parent control labels matched child labels during state-grow embedding")
    return u_child


def add_new_qubit_controls(
    u: np.ndarray,
    child_labels: list[str],
    new_site: int,
    sigma: float,
    seed: int,
    amp_bound: float,
) -> np.ndarray:
    """Add small deterministic controls on the newly appended qubit."""
    out = np.asarray(u, dtype=float).copy()
    sigma = float(sigma)
    if sigma <= 0:
        return out
    rng = np.random.RandomState(int(seed))
    site_suffix = f"_s{int(new_site)}"
    for j, label in enumerate(child_labels):
        if label.endswith(site_suffix):
            out[:, j] = out[:, j] + sigma * rng.randn(out.shape[0])
    if float(amp_bound) > 0:
        np.clip(out, -float(amp_bound), float(amp_bound), out=out)
    return out


def run_state_grow_mode(
    args: argparse.Namespace,
    reporter: ConsoleReporter,
    metadata_path: Optional[Path],
    ts: str,
    tag: str,
    run_slug: str,
    main_wall_start: float,
) -> None:
    """State-growing continuation: solve/embed N-1 controls, then refine N controls."""
    d = int(args.d)
    N = int(args.N)
    if d != 2:
        raise SystemExit("--state-grow is currently implemented for qubits only")
    if N < 2:
        raise SystemExit("--state-grow requires N >= 2")
    if str(args.target).lower() != "ghz":
        raise SystemExit("--state-grow currently supports --target ghz only")

    T = float(args.T)
    Nt = int(args.Nt)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    parent_N = N - 1
    parent_p = int(args.state_grow_parent_p) if int(args.state_grow_parent_p) > 0 else int(args.p)
    parent_iters = int(args.state_grow_parent_iters) if int(args.state_grow_parent_iters) > 0 else resolved_ea_iters(args)
    parent_S = int(args.state_grow_parent_S) if int(args.state_grow_parent_S) > 0 else int(args.S1)
    grow_scales = [float(x) for x in str(args.state_grow_homotopy).split(",") if x.strip()]
    if not grow_scales:
        grow_scales = [1.0]
    method_stages = build_stages(
        grow_scales,
        int(args.S1),
        int(args.S2),
        int(args.iters1),
        int(args.iters2),
        float(args.lr1),
        float(args.lr2),
    )
    if len(method_stages) == 0:
        raise SystemExit("--state-grow requires at least one refinement stage; increase --iters1/--iters2")

    reporter.info("[STATEGROW] mode enabled")
    reporter.info(
        f"[STATEGROW] parent N={parent_N}, child N={N}, T={T}, Nt={Nt}, parent_p={parent_p}, parent_iters={parent_iters}"
    )
    reporter.info(f"[STATEGROW] parent_S={parent_S}, child stages={stage_dicts(method_stages)}")
    if bool(args.dry_run):
        reporter.info("[STATEGROW] dry-run requested; no optimization will be executed.")
        return

    psi_parent, target_parent = build_targets(d, parent_N, "ghz")
    axes_parent, labels_parent = build_single_site_axes(d, parent_N, str(args.drives))
    H0_parent = build_drift_base(d, parent_N)
    H0_parent_dense = qobj_to_dense_matrix(H0_parent)
    stack_parent = build_dense_operator_stack(axes_parent)
    psi_parent_vec = qobj_to_dense_vector(psi_parent)
    target_parent_vec = qobj_to_dense_vector(target_parent)

    parent_ea_fidelity: float | None = None
    parent_projection_summary: dict[str, Any] | None = None
    parent_trace: list[dict[str, Any]] = []
    parent_source = "solve"

    if str(args.state_grow_parent_controls).strip():
        parent_source = str(args.state_grow_parent_controls)
        u_parent, loaded_labels = load_state_grow_parent_controls(parent_source, labels_parent)
        if loaded_labels is not None and len(loaded_labels) == u_parent.shape[1]:
            labels_parent_for_embed = list(loaded_labels)
        else:
            labels_parent_for_embed = labels_parent
        reporter.info(f"[STATEGROW] loaded parent controls from {parent_source}")
    else:
        opt_parent = ImprovedPolynomialPontryagin(
            d,
            parent_N,
            Nt,
            T,
            parent_p,
            drift_strength=float(args.drift_strength_ea),
            seed=int(args.seed),
            verbose=bool(int(args.verbose)),
        )
        parent_ea_fidelity = float(
            opt_parent.optimize_improved(
                psi_parent,
                target_parent,
                max_iter=parent_iters,
                reporter=reporter,
                progress_every=int(args.progress_every_ea),
                trace_rows=parent_trace if bool(int(args.save_traces)) else None,
                progress_label="STATEGROW:parentEA",
            )
        )
        reporter.info(f"[STATEGROW] parent EA fidelity={parent_ea_fidelity:.12f}")
        u_parent, parent_projection_summary = compile_ea_to_target_continuation_project(
            opt=opt_parent,
            ctrl_axes=axes_parent,
            ctrl_labels=labels_parent,
            H0_base=H0_parent,
            psi0=psi_parent,
            target=target_parent,
            T=T,
            Nt=Nt,
            S=parent_S,
            drift_strength_hw=float(args.drift_strength_hw),
            amp_bound=float(args.amp),
            hard_amp=float(args.hard_amp),
            trajectory_reference=str(args.trajectory_reference),
            continuation_iters=str(args.target_continuation_iters),
            continuation_lrs=str(args.target_continuation_lrs),
            continuation_weights=str(args.target_continuation_weights),
            continuation_checkpoints=str(args.target_continuation_checkpoints),
            continuation_min_improve=float(args.target_continuation_min_improve),
            continuation_init=str(args.target_continuation_init),
            continuation_policy=str(args.target_continuation_policy),
            continuation_escape_weights=str(args.target_continuation_escape_weights),
            continuation_escape_checkpoints=str(args.target_continuation_escape_checkpoints),
            continuation_escape_iters=str(args.target_continuation_escape_iters),
            continuation_target_stall_tol=args.target_continuation_target_stall_tol,
            continuation_max_stages=int(args.target_continuation_max_stages),
            compiler_eps=float(args.compiler_eps),
            budget_frac=float(args.compiler_budget_frac),
            max_terms_per_slice=int(args.compiler_max_terms),
            compiler_sort_mode=str(args.compiler_sort_mode),
            product_time_scale=float(args.product_time_scale),
            projection_clip=float(args.clip),
            projection_backtracks=max(1, int(args.backtracks)),
            projection_accept_mode=str(args.accept_mode),
            projection_accept_drop=float(args.accept_drop),
            projection_threshold=float(args.projection_threshold),
            reporter=reporter,
            progress_every=int(args.progress_every_grape),
            H0_base_dense=H0_parent_dense,
            ctrl_stack_dense=stack_parent,
            psi0_vec=psi_parent_vec,
            target_vec=target_parent_vec,
        )
        labels_parent_for_embed = labels_parent

    parent_seed_fidelity = simulate_state_fidelity(
        psi_parent,
        target_parent,
        float(args.drift_strength_hw) * H0_parent,
        axes_parent,
        T,
        u_parent,
        None,
        H0_dense=float(args.drift_strength_hw) * H0_parent_dense,
        ctrl_stack_dense=stack_parent,
        psi0_vec=psi_parent_vec,
        target_vec=target_parent_vec,
    )
    reporter.info(f"[STATEGROW] parent physical seed fidelity={parent_seed_fidelity:.12f}")

    psi_child, target_child = build_targets(d, N, "ghz")
    axes_child, labels_child = build_single_site_axes(d, N, str(args.drives))
    H0_child = build_drift_base(d, N)
    H0_child_dense = qobj_to_dense_matrix(H0_child)
    stack_child = build_dense_operator_stack(axes_child)
    psi_child_vec = qobj_to_dense_vector(psi_child)
    target_child_vec = qobj_to_dense_vector(target_child)

    u_embedded = embed_parent_controls(u_parent, labels_parent_for_embed, labels_child)
    u_dithered = add_new_qubit_controls(
        u_embedded,
        labels_child,
        new_site=N - 1,
        sigma=float(args.state_grow_newqubit_sigma),
        seed=int(args.state_grow_newqubit_seed),
        amp_bound=float(args.amp),
    )
    H0_child_phys_dense = float(args.drift_strength_hw) * H0_child_dense
    H0_child_phys = float(args.drift_strength_hw) * H0_child
    child_embedded_fidelity = simulate_state_fidelity(
        psi_child,
        target_child,
        H0_child_phys,
        axes_child,
        T,
        u_embedded,
        None,
        H0_dense=H0_child_phys_dense,
        ctrl_stack_dense=stack_child,
        psi0_vec=psi_child_vec,
        target_vec=target_child_vec,
    )
    child_dithered_fidelity = simulate_state_fidelity(
        psi_child,
        target_child,
        H0_child_phys,
        axes_child,
        T,
        u_dithered,
        None,
        H0_dense=H0_child_phys_dense,
        ctrl_stack_dense=stack_child,
        psi0_vec=psi_child_vec,
        target_vec=target_child_vec,
    )
    reporter.info(f"[STATEGROW] embedded child fidelity={child_embedded_fidelity:.12f}")
    reporter.info(f"[STATEGROW] dithered child fidelity={child_dithered_fidelity:.12f}")

    opt_cfg = OptConfig(
        lr=float(args.lr1),
        l2=float(args.l2),
        amp=float(args.amp),
        clip=float(args.clip),
        backtracks=int(args.backtracks),
        accept_mode=str(args.accept_mode),
        accept_drop=float(args.accept_drop),
        threshold=float(args.threshold),
        verbose=int(args.verbose),
        stall_enable=bool(int(args.stall_enable)),
        stall_gnorm=float(args.stall_gnorm),
        stall_max_kicks=int(args.stall_max_kicks),
        stall_kick_sigma=float(args.stall_kick_sigma),
    )
    rr, u_final = run_method_stages(
        name="STATEGROW",
        psi0=psi_child,
        target=target_child,
        H0_base=H0_child,
        drift_strength_hw=float(args.drift_strength_hw),
        ctrl_axes=axes_child,
        T=T,
        Nt=Nt,
        stages=method_stages,
        homotopy_mode="mult",
        u0=u_dithered,
        opt_cfg_template=opt_cfg,
        threshold=float(args.threshold),
        seed=int(args.seed),
        jitter_list=None,
        verbose=int(args.verbose),
        reporter=reporter,
        progress_every=int(args.progress_every_grape),
        H0_base_dense=H0_child_dense,
        ctrl_stack_dense=stack_child,
        psi0_vec=psi_child_vec,
        target_vec=target_child_vec,
    )

    parent_path = outdir / f"state_grow_parent_controls{tag}.csv"
    embedded_path = outdir / f"state_grow_embedded_controls{tag}.csv"
    dithered_path = outdir / f"state_grow_dithered_controls{tag}.csv"
    final_path = outdir / f"state_grow_final_controls{tag}.csv"
    write_awg_csv(parent_path, u_parent, T / u_parent.shape[0], labels_parent)
    write_awg_csv(embedded_path, u_embedded, T / u_embedded.shape[0], labels_child)
    write_awg_csv(dithered_path, u_dithered, T / u_dithered.shape[0], labels_child)
    write_awg_csv(final_path, u_final, T / u_final.shape[0], labels_child)

    npz_path = outdir / f"state_grow_controls{tag}.npz"
    np.savez_compressed(
        npz_path,
        u_parent=u_parent,
        u_embedded=u_embedded,
        u_dithered=u_dithered,
        u_final=u_final,
    )

    summary = {
        "version": "srgrape",
        "mode": "state_grow",
        "timestamp": ts,
        "command": command_as_shell([sys.argv[0], *sys.argv[1:]]),
        "parent": {
            "N": int(parent_N),
            "source": parent_source,
            "p": int(parent_p),
            "ea_iters": int(parent_iters),
            "ea_fidelity": parent_ea_fidelity,
            "physical_seed_fidelity": float(parent_seed_fidelity),
            "projection_summary": metadata_friendly(parent_projection_summary),
            "controls_path": str(parent_path),
        },
        "child": {
            "N": int(N),
            "embedded_fidelity": float(child_embedded_fidelity),
            "dithered_fidelity": float(child_dithered_fidelity),
            "final_fidelity": float(rr.finalF_phys),
            "best_stage_fidelity": float(rr.bestF_last_stage),
            "iters_to_threshold": rr.iters_to_threshold,
            "total_stage_iters": int(rr.total_stage_iters),
            "wall_s": float(rr.wall_s),
            "stages": stage_dicts(method_stages),
            "new_qubit_sigma": float(args.state_grow_newqubit_sigma),
            "new_qubit_seed": int(args.state_grow_newqubit_seed),
            "embedded_controls_path": str(embedded_path),
            "dithered_controls_path": str(dithered_path),
            "final_controls_path": str(final_path),
            "npz_path": str(npz_path),
        },
        "args": metadata_friendly(vars(args)),
        "total_wall_s": float(time.time() - main_wall_start),
    }
    summary_path = outdir / f"state_grow_summary{tag}_{ts}.json"
    write_json(summary_path, summary)
    maybe_write_metadata(metadata_path, summary)
    reporter.info(f"[STATEGROW] final child fidelity={rr.finalF_phys:.12f}")
    reporter.info(f"[STATEGROW] wrote {summary_path}")
    reporter.info(f"[STATEGROW] wrote {final_path}")
    reporter.info(f"[TIME] total wall={format_seconds(time.time() - main_wall_start)}")



             


def write_awg_csv(path: Path, u: np.ndarray, dt: float, labels: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    u = np.asarray(u, dtype=float)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t"] + labels)
        for k in range(u.shape[0]):
            w.writerow([f"{k*dt:.12g}"] + [f"{u[k, j]:.12g}" for j in range(u.shape[1])])


def write_results_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(rows) == 0:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)



      


def resolved_ea_iters(args: argparse.Namespace) -> int:
    ea_iters = int(args.ea_iters)
    if ea_iters <= 0:
        ea_iters = 5000 if int(args.N) >= 5 else 1200
    return ea_iters


def apply_ea_path_shape_args(opt: ImprovedPolynomialPontryagin, args: argparse.Namespace) -> None:
    """Attach optional EA path-shaping penalties without changing defaults."""
    opt.ea_control_l2 = max(0.0, float(getattr(args, "ea_control_l2", 0.0)))
    opt.ea_smooth_l2 = max(0.0, float(getattr(args, "ea_smooth_l2", 0.0)))


def success_rate(values: list[float], threshold: float) -> float:
    if not values:
        return 0.0
    return float(np.mean([float(v) >= float(threshold) for v in values]))


def default_output_path(outdir: Path, stem: str, run_slug: str, suffix: str) -> Path:
    return outdir / f"{stem}_{run_slug}{suffix}"


def print_preset_list(reporter: ConsoleReporter) -> None:
    reporter.info("Available presets:")
    for name in sorted(PRESETS):
        preset = PRESETS[name]
        reporter.info(
            f"  {name}: N={preset.get('N')} Nt={preset.get('Nt')} p={preset.get('p')} "
            f"S1={preset.get('S1')} S2={preset.get('S2')} compare={preset.get('compare')} "
            f"trials={preset.get('trials')} threshold={preset.get('threshold')}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )

                      
    ap.add_argument("--preset", type=str, default="", choices=[""] + sorted(PRESETS.keys()), help="Optional named preset")
    ap.add_argument("--list-presets", action="store_true", help="List named presets and exit")
    ap.add_argument("--dry-run", action="store_true", help="Validate settings, print derived schedule, and exit")
    ap.add_argument("--self-test", action="store_true", help="Run built-in quick self-tests and exit")
    ap.add_argument("--progress", type=str, default="auto", choices=["auto", "on", "off"], help="Progress display mode")
    ap.add_argument("--progress-every-ea", type=int, default=10, help="EA progress cadence in iterations")
    ap.add_argument("--progress-every-grape", type=int, default=10, help="GRAPE progress cadence in iterations")

            
    ap.add_argument("--d", type=int, default=2, help="Local dimension (2=qubit, 3=qutrit, ...)")
    ap.add_argument("--N", type=int, default=5, help="Number of qudits")
    ap.add_argument("--T", type=float, default=5.0, help="Total evolution time")
    ap.add_argument("--Nt", type=int, default=20, help="Number of coarse (EA) time slices")
    ap.add_argument("--drives", type=str, default="xy", choices=["su", "xy"], help="Control alphabet")

                     
    ap.add_argument("--drift-strength-ea", type=float, default=0.1, help="Drift used inside EA synthesis")
    ap.add_argument("--drift-strength-hw", type=float, default=0.1, help="Physical drift used for GRAPE/evaluation")

        
    ap.add_argument("--p", type=int, default=4, help="EA polynomial degree")
    ap.add_argument("--ea-iters", type=int, default=0, help="EA optimizer iterations (0=auto)")
    ap.add_argument("--ea-control-l2", type=float, default=0.0, help="Optional EA control-magnitude penalty used to shape the EA reference path")
    ap.add_argument("--ea-smooth-l2", type=float, default=0.0, help="Optional EA adjacent-slice smoothness penalty used to shape the EA reference path")

                 
    ap.add_argument("--S1", type=int, default=8, help="Micro-steps per EA slice for coarse GRAPE")
    ap.add_argument("--S2", type=int, default=12, help="Micro-steps per EA slice for fine GRAPE")
    ap.add_argument("--hard-amp", type=float, default=1.0, help="Bang-bang amplitude used by the dictionary compiler")
    ap.add_argument(
        "--compiler-mode",
        type=str,
        default="dictionary",
        choices=[
            "dictionary",
            "product",
            "product_instant",
            "product_cost",
            "product_effective_d2",
            "product_d2_hardware_project",
            "ea_trajectory_hardware_project",
            "ea_target_continuation_project",
            "constructive_project",
            "constructive_batch2",
            "constructive_batch2_compress",
        ],
        help=(
            "EA-to-control compiler. product_effective_d2 is a diagnostic effective-Hamiltonian "
            "mode with direct degree-2 product controls, not a hardware pulse compiler. "
            "product_d2_hardware_project uses that effective reference but returns physical controls only. "
            "ea_trajectory_hardware_project projects EA reference states into physical controls. "
            "ea_target_continuation_project prioritizes target fidelity first, then late EA checkpoints. "
            "constructive_project emits product-formula physical strings and audits EA-vs-physical "
            "slice agreement before any GRAPE refinement. constructive_batch2 uses a whole-slice "
            "average-Hamiltonian/toggling-frame degree-2 projection before any GRAPE refinement. "
            "constructive_batch2_compress first audits a high-resolution constructive_batch2 reference, "
            "then compresses it onto the requested low-resolution physical-control grid."
        ),
    )
    ap.add_argument("--compiler-eps", type=float, default=0.02, help="Micro-step eps for BCH words")
    ap.add_argument("--compiler-budget-frac", type=float, default=1.0, help="Fraction of each slice micro-grid reserved for BCH words (rest for linear)")
    ap.add_argument("--compiler-max-terms", type=int, default=32, help="Max EA terms per slice to attempt compiling")
    ap.add_argument(
        "--compiler-sort-mode",
        type=str,
        default="legacy",
        choices=["legacy", "weight", "high_degree", "low_degree"],
        help="Term priority before applying compiler-max-terms",
    )
    ap.add_argument("--product-bch-reps", type=int, default=1, help="Repetitions for recursive product-BCH compiler words")
    ap.add_argument("--constructive-trotter-reps", type=int, default=1, help="Subcycles per EA slice for constructive_batch2 serial Z/product blocks")
    ap.add_argument("--product-max-degree", type=int, default=4, help="Maximum Pauli-product degree attempted by the product compiler")
    ap.add_argument("--product-time-scale", type=float, default=1.0, help="Scale EA product angles before product compilation/audit")
    ap.add_argument("--projection-iters", type=int, default=40, help="Iterations for product_d2_hardware_project reference-to-hardware projection")
    ap.add_argument("--projection-lr", type=float, default=0.06, help="Learning rate for product_d2_hardware_project")
    ap.add_argument("--projection-threshold", type=float, default=0.999, help="Reference-fidelity threshold for product_d2_hardware_project")
    ap.add_argument("--projection-init", type=str, default="product_d2", choices=["product_d2", "zero"], help="Initial physical controls for product_d2_hardware_project")
    ap.add_argument("--trajectory-reference", type=str, default="full", choices=["full", "degree2", "degree3", "degree4"], help="EA reference used by ea_trajectory_hardware_project")
    ap.add_argument("--trajectory-weight", type=float, default=0.5, help="Weight on EA checkpoint matching in ea_trajectory_hardware_project")
    ap.add_argument(
        "--trajectory-checkpoints",
        type=str,
        default="every2",
        help="EA checkpoints used by ea_trajectory_hardware_project: all, final, late2, late4, latehalf, every2, every4, or range:start:end[:step]",
    )
    ap.add_argument("--trajectory-project-iters", type=int, default=40, help="Iterations for ea_trajectory_hardware_project")
    ap.add_argument("--trajectory-project-lr", type=float, default=0.05, help="Learning rate for ea_trajectory_hardware_project")
    ap.add_argument("--trajectory-stage-weights", type=str, default="", help="Optional comma-separated trajectory weights for staged ea_trajectory_hardware_project")
    ap.add_argument("--trajectory-stage-checkpoints", type=str, default="", help="Optional comma-separated checkpoint modes for staged ea_trajectory_hardware_project")
    ap.add_argument("--trajectory-stage-iters", type=str, default="", help="Optional comma-separated iteration counts for staged ea_trajectory_hardware_project")
    ap.add_argument("--trajectory-stage-lrs", type=str, default="", help="Optional comma-separated learning rates for staged ea_trajectory_hardware_project")
    ap.add_argument("--trajectory-stage-select", type=str, default="last", choices=["last", "best_gate", "best_target", "best_objective"], help="How staged ea_trajectory_hardware_project chooses the returned physical seed")
    ap.add_argument("--trajectory-optimizer", type=str, default="adam", choices=["adam", "lbfgsb"], help="Optimizer used inside ea_trajectory_hardware_project stages")
    ap.add_argument("--trajectory-lbfgs-maxls", type=int, default=20, help="L-BFGS-B maximum line-search steps per iteration for ea_trajectory_hardware_project")
    ap.add_argument("--trajectory-lbfgs-ftol", type=float, default=1e-9, help="L-BFGS-B ftol for ea_trajectory_hardware_project")
    ap.add_argument("--trajectory-lbfgs-gtol", type=float, default=1e-5, help="L-BFGS-B gtol for ea_trajectory_hardware_project")
    ap.add_argument("--trajectory-objective", type=str, default="weighted_sum", choices=["weighted_sum", "geomean", "log_geomean", "balanced_log", "softmin", "gate_softmin", "softmin_all", "gate_softmin_all"], help="Objective used inside ea_trajectory_hardware_project; weighted_sum preserves the default objective")
    ap.add_argument("--trajectory-softmin-tau", type=float, default=0.02, help="Temperature for softmin trajectory objectives")
    ap.add_argument("--trajectory-auto-polish", type=int, default=1, help="If best_gate has no passing EA-trajectory projection stage, run a robust soft-min polish schedule")
    ap.add_argument("--trajectory-polish-objective", type=str, default="softmin_all", choices=["weighted_sum", "geomean", "log_geomean", "balanced_log", "softmin", "gate_softmin", "softmin_all", "gate_softmin_all"], help="Objective used by automatic ea_trajectory_hardware_project polish stages")
    ap.add_argument("--trajectory-polish-softmin-tau", type=float, default=0.05, help="Temperature for automatic polish soft-min objectives")
    ap.add_argument("--trajectory-polish-stage-weights", type=str, default="0.9,0.8,0.7,0.6,0.5", help="Comma-separated checkpoint weights for automatic polish stages")
    ap.add_argument("--trajectory-polish-stage-checkpoints", type=str, default="auto", help="Comma-separated checkpoint modes for automatic polish stages; auto uses every2 for Nt<=10 and the audit checkpoint mode otherwise")
    ap.add_argument("--trajectory-polish-stage-iters", type=str, default="40,40,40,40,40", help="Comma-separated iteration counts for automatic polish stages")
    ap.add_argument("--trajectory-polish-stage-lrs", type=str, default="0.03,0.03,0.02,0.02,0.01", help="Comma-separated learning rates for automatic polish stages")
    ap.add_argument("--trajectory-init", type=str, default="linear", choices=["linear", "product", "constructive_batch2", "batch2", "zero"], help="Physical seed initialization for ea_trajectory_hardware_project")
    ap.add_argument("--eatraj-min-target-fidelity", type=float, default=0.0, help="Minimum final target fidelity for ea_trajectory_hardware_project pass/fail reporting")
    ap.add_argument("--eatraj-min-trajectory-fidelity", type=float, default=0.0, help="Minimum mean EA checkpoint fidelity for ea_trajectory_hardware_project pass/fail reporting")
    ap.add_argument("--eatraj-min-trajectory-min-fidelity", type=float, default=0.0, help="Minimum worst-checkpoint EA fidelity for ea_trajectory_hardware_project pass/fail reporting")
    ap.add_argument("--eatraj-stop-on-fail", type=int, default=0, help="Stop before physical GRAPE if ea_trajectory_hardware_project pass/fail gates fail")
    ap.add_argument("--target-continuation-iters", type=str, default="30,20,20", help="Comma-separated stage iterations for ea_target_continuation_project")
    ap.add_argument("--target-continuation-lrs", type=str, default="0.05,0.04,0.03", help="Comma-separated stage learning rates for ea_target_continuation_project")
    ap.add_argument("--target-continuation-weights", type=str, default="0,0.1,0.2", help="Comma-separated EA-checkpoint weights for ea_target_continuation_project")
    ap.add_argument("--target-continuation-checkpoints", type=str, default="final,late2,late4", help="Comma-separated checkpoint modes for ea_target_continuation_project")
    ap.add_argument("--target-continuation-min-improve", type=float, default=1e-4, help="Minimum target improvement before allowing a weighted checkpoint stage")
    ap.add_argument("--target-continuation-init", type=str, default="linear", choices=["linear", "product", "zero"], help="Initial physical controls for ea_target_continuation_project")
    ap.add_argument("--target-continuation-policy", type=str, default="adaptive", choices=["manual", "adaptive"], help="Stage policy for ea_target_continuation_project")
    ap.add_argument("--target-continuation-escape-weights", type=str, default="0.5,0.9", help="Adaptive escape-stage EA-checkpoint weights")
    ap.add_argument("--target-continuation-escape-checkpoints", type=str, default="late2,all", help="Adaptive escape-stage checkpoint modes")
    ap.add_argument("--target-continuation-escape-iters", type=str, default="30,80", help="Adaptive escape-stage iteration counts")
    ap.add_argument("--target-continuation-target-stall-tol", type=float, default=None, help="Target-improvement tolerance used to mark a continuation stage as stalled")
    ap.add_argument("--target-continuation-max-stages", type=int, default=4, help="Maximum target-continuation stages to attempt; use 0 for all configured stages")
    ap.add_argument(
        "--n2-auto-target-continuation",
        type=int,
        default=1,
        help=(
            "For N=2, automatically reroute ea_trajectory_hardware_project to "
            "ea_target_continuation_project, the high-fidelity physical projection route. "
            "Set to 0 only for strict trajectory-projector diagnostics."
        ),
    )
    ap.add_argument("--ea-only", type=int, default=0, help="Run EA synthesis, save EA traces/metadata/checkpoints, then exit before compilation")
    ap.add_argument("--compiler-audit-only", type=int, default=0, help="Run EA synthesis and compiler audit, then exit before physical GRAPE")
    ap.add_argument("--constructive-include-all", type=int, default=1, help="In constructive_project mode, attempt every nonzero EA product term instead of applying compiler-max-terms")
    ap.add_argument("--constructive-fail-on-skip", type=int, default=1, help="In constructive_project mode, fail the audit if any retained/candidate term is skipped")
    ap.add_argument("--constructive-term-threshold", type=float, default=1e-10, help="Terms with |theta| below this threshold are ignored in constructive_project mode")
    ap.add_argument("--constructive-min-emitted-weight-fraction", type=float, default=0.999, help="Minimum emitted/requested EA product weight for constructive_project")
    ap.add_argument("--constructive-min-slice-overlap", type=float, default=0.99, help="Minimum EA-vs-physical slice unitary overlap for constructive_project")
    ap.add_argument("--constructive-min-checkpoint-fidelity", type=float, default=0.99, help="Minimum EA-vs-physical checkpoint state fidelity for constructive_project")
    ap.add_argument("--constructive-audit-path", type=str, default="", help="Optional JSON path for constructive_project audit summary")
    ap.add_argument("--constructive-audit-rows-path", type=str, default="", help="Optional CSV path for per-slice constructive_project audit rows")
    ap.add_argument("--constructive-stop-on-fail", type=int, default=1, help="Stop before GRAPE when constructive_project audit fails")
    ap.add_argument("--constructive-repair-enable", type=int, default=1, help="For constructive_batch2, shrink/reselect p3/p4 terms and re-audit before GRAPE when high-degree emission fails")
    ap.add_argument("--constructive-repair-attempts", type=int, default=2, help="Maximum constructive_batch2 audit-repair attempts")
    ap.add_argument("--constructive-repair-total-scale", type=float, default=0.85, help="Per-attempt multiplier applied to ea-shared-total-budget-frac during repair")
    ap.add_argument("--constructive-repair-d4-scale", type=float, default=0.75, help="Per-attempt multiplier applied to ea-shared-d4-budget-frac during repair")
    ap.add_argument("--constructive-repair-high-degree-only", type=int, default=1, help="Only trigger automatic repair for degree-3/4 emitted-term failures")
    ap.add_argument("--pauli-audit-path", type=str, default="", help="Optional JSON path for exact Pauli-expansion audit summary")
    ap.add_argument("--pauli-audit-rows-path", type=str, default="", help="Optional CSV path for exact Pauli-expansion audit rows")
    ap.add_argument("--pauli-audit-tol", type=float, default=1e-10, help="Coefficient tolerance for exact Pauli-expansion audit")
    ap.add_argument("--batch2-residual-tol", type=float, default=1e-6, help="Residual theta tolerance for constructive_batch2 emitted-term accounting")
    ap.add_argument("--batch2-max-frames", type=int, default=64, help="Maximum rotated-drift frames used per slice in constructive_batch2")
    ap.add_argument("--batch2-frame-tol", type=float, default=1e-10, help="Minimum frame duration retained in constructive_batch2")
    ap.add_argument("--constructive-reference-S", type=int, default=0, help="Reference micro-steps per EA slice for constructive_batch2_compress (0=use S1)")
    ap.add_argument("--compression-target-guard", type=int, default=1, help="In constructive_batch2_compress, keep the pre-projection compressed seed if projection lowers target fidelity")
    ap.add_argument("--compression-target-guard-tol", type=float, default=0.0, help="Allowed target-fidelity decrease before compression target guard reverts")
    ap.add_argument("--ea-batch2-projectable", type=int, default=0, help="Constrain EA degree-2 slices to a physical batch2/pair-echo budget during EA optimization")
    ap.add_argument("--ea-batch2-budget-frac", type=float, default=0.98, help="Fraction of the physical degree-2 drift-angle budget allowed during EA projectable optimization")
    ap.add_argument("--ea-native-xy-projectable", type=int, default=0, help="Constrain EA native one-body X/Y controls to the physical amplitude envelope")
    ap.add_argument("--ea-native-xy-amp-frac", type=float, default=1.0, help="Fraction of --amp used as the EA native X/Y amplitude bound")
    ap.add_argument("--ea-native-z-projectable", type=int, default=0, help="Also constrain EA one-body Z controls to a local synthesized-control envelope")
    ap.add_argument("--ea-native-z-amp-frac", type=float, default=1.0, help="Fraction of --amp used as the EA one-body Z coefficient bound")
    ap.add_argument("--ea-degree3-projectable", type=int, default=0, help="Constrain EA degree-3 slice angles to a small physical BCH-product budget during EA optimization")
    ap.add_argument("--ea-degree3-projector", type=str, default="l1", choices=["l1", "cost"], help="Degree-3 projectability geometry: signed-L1 legacy cap or constructive-cost microstep budget")
    ap.add_argument("--ea-degree3-theta-radius", type=float, default=0.0, help="Signed-L1 radius for degree-3 EA angles per coarse slice (0 disables degree-3 content)")
    ap.add_argument("--ea-degree3-step-budget-frac", type=float, default=0.20, help="Fraction of the S1 micro-grid reserved for cost-aware degree-3 BCH terms")
    ap.add_argument("--ea-degree3-cost-term-threshold", type=float, default=1e-8, help="Small degree-3 theta values ignored by the cost-aware projector")
    ap.add_argument("--ea-degree3-max-terms-per-slice", type=int, default=0, help="Maximum degree-3 terms packed per slice by the cost-aware projector (0=auto)")
    ap.add_argument("--ea-degree3-projector-warmup", type=int, default=0, help="EA projector calls before applying the degree-3 cost projector; warmup iterates are not saved as final best")
    ap.add_argument("--ea-degree3-projector-ramp", type=int, default=0, help="EA projector calls over which to blend from warmup to the hard degree-3 cost projection")
    ap.add_argument("--ea-degree4-projectable", type=int, default=0, help="Constrain EA degree-4 slice angles to a constructive BCH-product budget during EA optimization")
    ap.add_argument("--ea-degree4-step-budget-frac", type=float, default=0.20, help="Fraction of the S1 micro-grid reserved for cost-aware degree-4 BCH terms")
    ap.add_argument("--ea-degree4-cost-term-threshold", type=float, default=1e-10, help="Small degree-4 theta values ignored by the cost-aware projector")
    ap.add_argument("--ea-degree4-max-terms-per-slice", type=int, default=0, help="Maximum degree-4 terms packed per slice by the cost-aware projector (0=auto)")
    ap.add_argument("--ea-degree4-projector-warmup", type=int, default=0, help="EA projector calls before applying the degree-4 cost projector; warmup iterates are not saved as final best")
    ap.add_argument("--ea-degree4-projector-ramp", type=int, default=0, help="EA projector calls over which to blend from warmup to the hard degree-4 cost projection")
    ap.add_argument("--ea-shared-scheduler", type=int, default=0, help="Final shared p2/p3/p4 constructive microstep-budget projector")
    ap.add_argument("--ea-shared-total-budget-frac", type=float, default=0.95, help="Fraction of S1 used as the shared schedulable microstep budget")
    ap.add_argument("--ea-shared-d4-budget-frac", type=float, default=0.30, help="Maximum fraction of S1 reserved for mandatory degree-4 terms in the shared scheduler")
    ap.add_argument("--ea-shared-d4-min-terms", type=int, default=1, help="Minimum degree-4 terms per slice attempted by the shared scheduler when p=4 terms exist")
    ap.add_argument("--ea-shared-trim-degree2", type=int, default=1, help="Allow the shared scheduler to trim degree-2 terms if p1+p2 alone exceeds the shared budget")
    ap.add_argument("--ea-nested-p2", type=int, default=0, help="For p>=3, solve lower-degree EA stages first and embed them as certified fallbacks")
    ap.add_argument("--ea-nested-p2-iters", type=int, default=0, help="Iterations for nested p=2 warm-start solve (0=use ea-iters)")
    ap.add_argument("--ea-nested-p3-iters", type=int, default=0, help="Iterations for nested p=3 warm-start solve when p=4 (0=use ea-iters)")
    ap.add_argument("--ea-nested-guard", type=int, default=1, help="For nested p=3, revert to embedded p=2 controls if p=3 fails to improve")
    ap.add_argument("--ea-nested-tol", type=float, default=1e-10, help="Tolerance used by the p=2 fallback guard and basis-nestedness audit")
    ap.add_argument("--seed-gain", type=float, default=1.0, help="Multiply the compiled seed controls by this factor")
    ap.add_argument("--seed-dither", type=float, default=0.0, help="Add Gaussian dither (sigma) to the seed")

              
    ap.add_argument("--homotopy", type=str, default="1.0", help="Comma-separated drift scales")
    ap.add_argument("--homotopy-mode", type=str, default="mult", choices=["mult", "abs"], help="Interpretation of homotopy values")

                             
    ap.add_argument("--iters1", type=int, default=800, help="Iterations per coarse stage")
    ap.add_argument("--iters2", type=int, default=1200, help="Iterations for final refinement stage")
    ap.add_argument("--lr1", type=float, default=0.08, help="Learning rate for coarse stages")
    ap.add_argument("--lr2", type=float, default=0.06, help="Learning rate for refinement stage")
    ap.add_argument("--amp", type=float, default=1.0, help="Control amplitude bound")
    ap.add_argument("--l2", type=float, default=0.0, help="L2 penalty")
    ap.add_argument("--clip", type=float, default=0.0, help="Gradient norm clip (0 disables)")

                             
    ap.add_argument("--backtracks", type=int, default=0, help="Backtracking steps (0 disables)")
    ap.add_argument("--accept-mode", type=str, default="hard", choices=["hard", "soft"], help="Acceptance rule")
    ap.add_argument("--accept-drop", type=float, default=0.0, help="Soft accept allowed drop")

                           
    ap.add_argument("--stall-enable", type=int, default=0, help="Enable stall escape")
    ap.add_argument("--stall-gnorm", type=float, default=1e-10, help="Gradient norm threshold")
    ap.add_argument("--stall-max-kicks", type=int, default=0, help="Maximum kicks")
    ap.add_argument("--stall-kick-sigma", type=float, default=0.0, help="Kick sigma")

          
    ap.add_argument("--target", type=str, default="ghz", choices=["ghz", "transfer", "w"], help="Target state")
    ap.add_argument("--state-grow", type=int, default=0, help="Use state-growing continuation from N-1 to N instead of EA compilation")
    ap.add_argument("--state-grow-parent-p", type=int, default=0, help="EA degree for the N-1 parent solve (0=use --p)")
    ap.add_argument("--state-grow-parent-iters", type=int, default=0, help="EA iterations for the N-1 parent solve (0=use resolved --ea-iters)")
    ap.add_argument("--state-grow-parent-S", type=int, default=0, help="Micro-grid S for the N-1 parent projection (0=use --S1)")
    ap.add_argument("--state-grow-newqubit-sigma", type=float, default=0.05, help="Gaussian seed amplitude on the newly added qubit")
    ap.add_argument("--state-grow-newqubit-seed", type=int, default=123, help="RNG seed for newly added qubit controls")
    ap.add_argument("--state-grow-homotopy", type=str, default="1.0", help="Comma-separated physical refinement drift scales for state-grow mode")
    ap.add_argument("--state-grow-parent-controls", type=str, default="", help="Optional CSV/NPZ parent controls to embed instead of solving N-1")

                
    ap.add_argument("--compare", type=int, default=0, help="Run METHOD vs BASELINE trials")
    ap.add_argument("--trials", type=int, default=25, help="Number of trials")
    ap.add_argument("--baseline-mode", type=str, default="budget", choices=["budget", "matched"], help="Baseline schedule")
    ap.add_argument("--baseline-init", type=str, default="rmsmatch", choices=["gauss", "uniform", "rmsmatch"], help="Baseline init distribution")
    ap.add_argument("--baseline-sigma", type=float, default=0.2, help="Baseline Gaussian sigma (if baseline-init=gauss)")

                
    ap.add_argument("--robust-jitter", type=str, default="0.0", help="Comma-separated dt jitter values")

            
    ap.add_argument("--outdir", type=str, default="results", help="Directory for CSV outputs")
    ap.add_argument("--tag", type=str, default="", help="Extra tag for output filenames")
    ap.add_argument("--save-metadata", type=int, default=1, help="Write run metadata JSON sidecar")
    ap.add_argument("--save-traces", type=int, default=1, help="Write EA/GRAPE convergence traces to CSV")
    ap.add_argument("--save-compiler-diagnostics", type=int, default=1, help="Write compiler diagnostics CSV")
    ap.add_argument("--trace-dir", type=str, default="", help="Optional exact directory for convergence trace CSVs")
    ap.add_argument("--metadata-path", type=str, default="", help="Optional exact path for metadata JSON")
    ap.add_argument("--compiler-diagnostics-path", type=str, default="", help="Optional exact path for compiler diagnostics CSV")
    ap.add_argument("--compiler-term-diagnostics-path", type=str, default="", help="Optional exact path for per-term compiler diagnostics CSV")
    ap.add_argument("--compiler-audit-summary-path", type=str, default="", help="Optional exact path for compiler audit summary JSON")
    ap.add_argument("--checkpoint-dir", type=str, default="", help="Optional directory for lightweight checkpoints")
    ap.add_argument(
        "--results-csv",
        type=str,
        default="",
        help=(
            "Optional: write the per-trial comparison CSV to this exact path. "
            "If omitted, the script writes into --outdir with an autogenerated filename."
        ),
    )

                                 
    ap.add_argument("--seed", type=int, default=1, help="Base RNG seed")
    ap.add_argument("--verbose", type=int, default=0, help="Verbose GRAPE printing")
    ap.add_argument("--threshold", type=float, default=0.99, help="Success threshold")
    return ap


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    argv = list(sys.argv[1:] if argv is None else argv)
    preset_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    preset_parser.add_argument("--preset", type=str, default="")
    preset_parser.add_argument("--list-presets", action="store_true")
    preset_ns, _ = preset_parser.parse_known_args(argv)

    ap = build_arg_parser()
    if preset_ns.preset:
        if preset_ns.preset not in PRESETS:
            raise SystemExit(f"Unknown preset: {preset_ns.preset}")
        ap.set_defaults(**PRESETS[preset_ns.preset])
    return ap.parse_args(argv)


def build_targets(d: int, N: int, target_name: str) -> tuple[qt.Qobj, qt.Qobj]:
    psi0 = qt.tensor(*[qt.basis(d, 0) for _ in range(N)])
    if target_name == "ghz":
        target = create_ghz(d, N)
    elif target_name == "transfer":
        psi0 = qt.tensor(qt.basis(d, 1), *[qt.basis(d, 0) for _ in range(N - 1)])
        target = qt.tensor(*[qt.basis(d, 0) for _ in range(N - 1)], qt.basis(d, 1))
    else:
        target = create_w(d, N)
    return psi0, target


def stage_dicts(stages: list[Stage]) -> list[dict[str, Any]]:
    return [asdict(st) for st in stages]


def maybe_write_metadata(path: Optional[Path], payload: dict[str, Any]) -> None:
    if path is not None:
        write_json(path, payload)


def pauli_axes_to_string(axes_by_site: dict[int, str]) -> str:
    if not axes_by_site:
        return "I"
    return "*".join(f"{axes_by_site[site]}{site}" for site in sorted(axes_by_site))


def summarize_compiler_term_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize per-term compiler diagnostics by product degree."""
                                                                           
                                                                            
                                                                        
                                                                      
    covered_residual_statuses = {"ok_residual"}
    by_degree: dict[int, dict[str, Any]] = {}
    totals = {
        "candidate_terms": 0,
        "retained_terms": 0,
        "emitted_terms": 0,
        "skipped_terms": 0,
        "skipped_nonzero_terms": 0,
        "requested_weight": 0.0,
        "retained_weight": 0.0,
        "emitted_weight": 0.0,
        "skipped_weight": 0.0,
        "used_steps": 0,
    }
    status_counts: dict[str, int] = {}

    for row in rows:
        degree = int(row.get("degree", 0))
        entry = by_degree.setdefault(
            degree,
            {
                "candidate_terms": 0,
                "retained_terms": 0,
                "emitted_terms": 0,
                "skipped_terms": 0,
                "skipped_nonzero_terms": 0,
                "requested_weight": 0.0,
                "retained_weight": 0.0,
                "emitted_weight": 0.0,
                "skipped_weight": 0.0,
                "used_steps": 0,
                "status_counts": {},
            },
        )
        requested_weight = float(row.get("requested_weight", 0.0))
        retained_weight = requested_weight if int(row.get("retained", 0)) else 0.0
        emitted_weight = float(row.get("emitted_weight", requested_weight if int(row.get("emitted", 0)) else 0.0))
        emitted_weight = max(0.0, min(requested_weight, emitted_weight))
        skipped_weight = requested_weight - emitted_weight
        status = str(row.get("status", "unknown"))
        covered_flag = int(row.get("emitted", 0)) or status in covered_residual_statuses

        for target in (entry, totals):
            target["candidate_terms"] += 1
            target["retained_terms"] += int(row.get("retained", 0))
            target["emitted_terms"] += int(bool(covered_flag))
            target["skipped_terms"] += 0 if covered_flag else 1
            target["skipped_nonzero_terms"] += (
                1 if ((not covered_flag) and requested_weight > 0.0) else 0
            )
            target["requested_weight"] += requested_weight
            target["retained_weight"] += retained_weight
            target["emitted_weight"] += emitted_weight
            target["skipped_weight"] += skipped_weight
            target["used_steps"] += int(row.get("used_steps", 0))

        entry["status_counts"][status] = int(entry["status_counts"].get(status, 0)) + 1
        status_counts[status] = int(status_counts.get(status, 0)) + 1

    def finalize(target: dict[str, Any]) -> dict[str, Any]:
        requested = float(target.get("requested_weight", 0.0))
        retained = float(target.get("retained_weight", 0.0))
        emitted = float(target.get("emitted_weight", 0.0))
        out = dict(target)
        out["retained_weight_fraction"] = retained / requested if requested > 0 else 0.0
        out["emitted_weight_fraction"] = emitted / requested if requested > 0 else 0.0
        out["emitted_retained_weight_fraction"] = emitted / retained if retained > 0 else 0.0
        return out

    return {
        "totals": finalize(totals),
        "by_degree": {str(degree): finalize(entry) for degree, entry in sorted(by_degree.items())},
        "status_counts": status_counts,
    }


def unitary_overlap(U_ref: np.ndarray, U_test: np.ndarray) -> float:
    """Return normalized Hilbert-Schmidt overlap between two unitaries."""
    dim = int(U_ref.shape[0])
    if dim <= 0:
        return 0.0
    return float(abs(np.trace(U_ref.conj().T @ U_test)) / dim)


def state_fidelity_vec(psi_ref: np.ndarray, psi_test: np.ndarray) -> float:
    """Return pure-state fidelity for dense state vectors."""
    ref = np.asarray(psi_ref, dtype=complex).reshape(-1)
    test = np.asarray(psi_test, dtype=complex).reshape(-1)
    n_ref = float(np.linalg.norm(ref))
    n_test = float(np.linalg.norm(test))
    if n_ref <= 0.0 or n_test <= 0.0:
        return 0.0
    return float(abs(np.vdot(ref / n_ref, test / n_test)) ** 2)


def _dense_ea_slice_unitaries(opt: ImprovedPolynomialPontryagin) -> list[np.ndarray]:
    """Build exact EA slice propagators from the optimized lifted Hamiltonians."""
    dim = int(opt.dim)
    h_drift = opt.H_drift.full()
    ctrl_dense = [ctrl.full() for ctrl in opt.controls]
    slices: list[np.ndarray] = []
    for k in range(int(opt.u.shape[0])):
        H = h_drift.copy()
        for j, amp in enumerate(np.asarray(opt.u[k], dtype=float)):
            if abs(float(amp)) > 0.0:
                H = H + float(amp) * ctrl_dense[j]
        slices.append(expm(-1j * float(opt.dt) * H))
    if any(U.shape != (dim, dim) for U in slices):
        raise ValueError("EA slice propagator dimension mismatch")
    return slices


def _dense_physical_slice_unitaries(
    u_phys: np.ndarray,
    H0_base: qt.Qobj,
    ctrl_axes: Sequence[qt.Qobj],
    T: float,
    Nt: int,
    S: int,
    drift_strength_hw: float,
) -> list[np.ndarray]:
    """Build physical slice propagators by multiplying the emitted microsteps."""
    if S <= 0:
        raise ValueError("S must be positive")
    u = np.asarray(u_phys, dtype=float)
    if u.shape[0] != int(Nt) * int(S):
        raise ValueError(f"u_phys has {u.shape[0]} rows, expected Nt*S={int(Nt) * int(S)}")
    if u.shape[1] != len(ctrl_axes):
        raise ValueError(f"u_phys has {u.shape[1]} controls, expected {len(ctrl_axes)}")
    H0 = float(drift_strength_hw) * H0_base.full()
    ctrl_dense = [axis.full() for axis in ctrl_axes]
    dim = int(H0.shape[0])
    dt_micro = float(T) / (int(Nt) * int(S))
    slices: list[np.ndarray] = []
    for k in range(int(Nt)):
        U_slice = np.eye(dim, dtype=complex)
        block = u[k * int(S) : (k + 1) * int(S), :]
        m = 0
        while m < int(S):
            row = block[m].copy()
            n = m + 1
            while n < int(S) and np.allclose(block[n], row, atol=1e-14, rtol=0.0):
                n += 1
            H = H0.copy()
            for j, amp in enumerate(row):
                if abs(float(amp)) > 0.0:
                    H = H + float(amp) * ctrl_dense[j]
            U_slice = expm(-1j * dt_micro * (n - m) * H) @ U_slice
            m = n
        slices.append(U_slice)
    return slices


def constructive_projection_audit(
    *,
    opt: ImprovedPolynomialPontryagin,
    u_seed: np.ndarray,
    H0_base: qt.Qobj,
    ctrl_axes: Sequence[qt.Qobj],
    psi0: qt.Qobj,
    T: float,
    Nt: int,
    S: int,
    drift_strength_hw: float,
    compiler_audit_summary: Optional[dict[str, Any]] = None,
    mode: str = "constructive_project",
) -> dict[str, Any]:
    """Audit whether the emitted physical controls reproduce the EA evolution.

    This is deliberately an operator/trajectory audit, not a target optimizer.
    A constructive projection is allowed to proceed only if the physical
    microstep product closely tracks the EA slice propagators.
    """
    ea_slices = _dense_ea_slice_unitaries(opt)
    hw_slices = _dense_physical_slice_unitaries(
        u_seed,
        H0_base,
        ctrl_axes,
        T=float(T),
        Nt=int(Nt),
        S=int(S),
        drift_strength_hw=float(drift_strength_hw),
    )
    dim = int(opt.dim)
    U_ea_total = np.eye(dim, dtype=complex)
    U_hw_total = np.eye(dim, dtype=complex)
    psi_ea = qobj_to_dense_vector(psi0)
    psi_hw = qobj_to_dense_vector(psi0)
    rows: list[dict[str, Any]] = []
    slice_overlaps: list[float] = []
    cumulative_unitary_overlaps: list[float] = []
    checkpoint_fidelities: list[float] = []
    for k, (U_ea, U_hw) in enumerate(zip(ea_slices, hw_slices)):
        U_ea_total = U_ea @ U_ea_total
        U_hw_total = U_hw @ U_hw_total
        psi_ea = U_ea @ psi_ea
        psi_hw = U_hw @ psi_hw
        slice_ov = unitary_overlap(U_ea, U_hw)
        cum_ov = unitary_overlap(U_ea_total, U_hw_total)
        ckpt_fid = state_fidelity_vec(psi_ea, psi_hw)
        slice_overlaps.append(slice_ov)
        cumulative_unitary_overlaps.append(cum_ov)
        checkpoint_fidelities.append(ckpt_fid)
        rows.append(
            {
                "slice": int(k),
                "slice_unitary_overlap": float(slice_ov),
                "cumulative_unitary_overlap": float(cum_ov),
                "checkpoint_state_fidelity": float(ckpt_fid),
            }
        )

    totals = (compiler_audit_summary or {}).get("totals", {})
    return {
        "mode": str(mode),
        "rows": rows,
        "slice_count": int(len(rows)),
        "min_slice_unitary_overlap": float(min(slice_overlaps) if slice_overlaps else 0.0),
        "mean_slice_unitary_overlap": float(np.mean(slice_overlaps) if slice_overlaps else 0.0),
        "final_cumulative_unitary_overlap": float(cumulative_unitary_overlaps[-1] if cumulative_unitary_overlaps else 0.0),
        "min_checkpoint_state_fidelity": float(min(checkpoint_fidelities) if checkpoint_fidelities else 0.0),
        "final_checkpoint_state_fidelity": float(checkpoint_fidelities[-1] if checkpoint_fidelities else 0.0),
        "emitted_weight_fraction": float(totals.get("emitted_weight_fraction", 0.0)),
        "requested_weight": float(totals.get("requested_weight", 0.0)),
        "emitted_weight": float(totals.get("emitted_weight", 0.0)),
        "skipped_terms": int(totals.get("skipped_terms", 0)),
        "skipped_nonzero_terms": int(totals.get("skipped_nonzero_terms", totals.get("skipped_terms", 0))),
        "status_counts": dict((compiler_audit_summary or {}).get("status_counts", {})),
    }


def constructive_projection_passed(summary: dict[str, Any], args: argparse.Namespace) -> tuple[bool, list[str]]:
    """Apply explicit pass/fail gates for constructive projection mode."""
    reasons: list[str] = []
    if int(args.constructive_fail_on_skip) and int(summary.get("skipped_nonzero_terms", 0)) > 0:
        reasons.append(f"skipped_nonzero_terms={summary.get('skipped_nonzero_terms')}")
    if float(summary.get("emitted_weight_fraction", 0.0)) < float(args.constructive_min_emitted_weight_fraction):
        reasons.append(
            "emitted_weight_fraction={:.6g}<{}".format(
                float(summary.get("emitted_weight_fraction", 0.0)),
                float(args.constructive_min_emitted_weight_fraction),
            )
        )
    if float(summary.get("min_slice_unitary_overlap", 0.0)) < float(args.constructive_min_slice_overlap):
        reasons.append(
            "min_slice_unitary_overlap={:.6g}<{}".format(
                float(summary.get("min_slice_unitary_overlap", 0.0)),
                float(args.constructive_min_slice_overlap),
            )
        )
    if float(summary.get("min_checkpoint_state_fidelity", 0.0)) < float(args.constructive_min_checkpoint_fidelity):
        reasons.append(
            "min_checkpoint_state_fidelity={:.6g}<{}".format(
                float(summary.get("min_checkpoint_state_fidelity", 0.0)),
                float(args.constructive_min_checkpoint_fidelity),
            )
        )
    return (len(reasons) == 0), reasons


def ea_trajectory_projection_passed(summary: dict[str, Any], args: argparse.Namespace) -> tuple[bool, list[str]]:
    """Apply explicit pass/fail gates for EA-trajectory hardware projection."""
    reasons: list[str] = []
    projection = dict((summary or {}).get("projection", {}))
    target_f = float(projection.get("target_fidelity_after", 0.0))
    traj_mean = float(projection.get("trajectory_fidelity_after", 0.0))
    traj_min = float(projection.get("trajectory_fidelity_min_after", 0.0))
    if target_f < float(args.eatraj_min_target_fidelity):
        reasons.append(
            "target_fidelity_after={:.6g}<{}".format(
                target_f,
                float(args.eatraj_min_target_fidelity),
            )
        )
    if traj_mean < float(args.eatraj_min_trajectory_fidelity):
        reasons.append(
            "trajectory_fidelity_after={:.6g}<{}".format(
                traj_mean,
                float(args.eatraj_min_trajectory_fidelity),
            )
        )
    if traj_min < float(args.eatraj_min_trajectory_min_fidelity):
        reasons.append(
            "trajectory_fidelity_min_after={:.6g}<{}".format(
                traj_min,
                float(args.eatraj_min_trajectory_min_fidelity),
            )
        )
    return (len(reasons) == 0), reasons


def high_degree_compiler_failures(
    compiler_audit_summary: dict[str, Any] | None,
    *,
    degrees: Sequence[int] = (3, 4),
) -> list[str]:
    """Return p>=3 emitted-term failures that should trigger repair."""
    if not compiler_audit_summary:
        return []
    out: list[str] = []
    by_degree = compiler_audit_summary.get("by_degree", {})
    for degree in degrees:
        entry = by_degree.get(str(degree), {})
        candidates = int(entry.get("candidate_terms", 0) or 0)
        if candidates <= 0:
            continue
        emitted = int(entry.get("emitted_terms", 0) or 0)
        skipped = int(entry.get("skipped_nonzero_terms", max(0, candidates - emitted)) or 0)
        frac = float(entry.get("emitted_weight_fraction", 0.0) or 0.0)
        if skipped > 0 or emitted < candidates or frac < 1.0 - 1e-9:
            out.append(
                "degree{} emitted={}/{} skipped={} weight_fraction={:.6g} statuses={}".format(
                    degree,
                    emitted,
                    candidates,
                    skipped,
                    frac,
                    json.dumps(entry.get("status_counts", {}), sort_keys=True),
                )
            )
    return out


def log_compiler_audit_summary(reporter: ConsoleReporter, compiler_audit_summary: dict[str, Any] | None) -> None:
    if not compiler_audit_summary:
        return
    totals = compiler_audit_summary.get("totals", {})
    reporter.info(
        "[AUDIT] product requested_weight={:.6g} emitted_weight={:.6g} emitted_fraction={:.3f}".format(
            float(totals.get("requested_weight", 0.0)),
            float(totals.get("emitted_weight", 0.0)),
            float(totals.get("emitted_weight_fraction", 0.0)),
        )
    )
    for degree, entry in compiler_audit_summary.get("by_degree", {}).items():
        reporter.info(
            "[AUDIT] degree {}: emitted {}/{} terms | emitted_weight_fraction={:.3f} | statuses={}".format(
                degree,
                int(entry.get("emitted_terms", 0)),
                int(entry.get("candidate_terms", 0)),
                float(entry.get("emitted_weight_fraction", 0.0)),
                json.dumps(entry.get("status_counts", {}), sort_keys=True),
            )
        )


def run_self_tests(reporter: ConsoleReporter) -> int:
    reporter.info("[SELF-TEST] Running quick checks")
    failures: list[str] = []

    def record(name: str, ok: bool, detail: str) -> None:
        status = "PASS" if ok else "FAIL"
        reporter.info(f"[SELF-TEST] {name}: {status} | {detail}")
        if not ok:
            failures.append(name)

                                                                                 
    T, Nt, S = 5.0, 20, 24
    eps, hard_amp = 0.01, 2.0
    budget = int(np.floor(0.8 * S))
    dt_micro = T / (Nt * S)
    expected = {}
    for inds in ([0, 1], [0, 1, 2], [0, 1, 2, 3]):
        seq, degree = _build_word_pulses_leftnest(list(inds), eps, hard_amp)
        u = np.zeros((budget, 10))
        used = _emit_word_to_u(u, seq, eps, hard_amp, dt_micro, budget)
        expected[degree] = used
    ok_word = expected.get(2, 0) > 0 and expected.get(3, 0) > 0 and expected.get(4, -1) == 0
    record("compiler_word_integrity", ok_word, f"used={expected}")

    ck_range = trajectory_checkpoint_indices(20, "range:10:20")
    ck_latehalf = trajectory_checkpoint_indices(20, "latehalf")
    ck_stride = trajectory_checkpoint_indices(20, "range:10:20:2")
    ok_checkpoint_parser = (
        ck_range == list(range(10, 21))
        and ck_latehalf == list(range(10, 21))
        and ck_stride == [10, 12, 14, 16, 18, 20]
    )
    record(
        "trajectory_checkpoint_window_parser",
        ok_checkpoint_parser,
        f"range={ck_range} latehalf={ck_latehalf} stride={ck_stride}",
    )

    parsed_xx = parse_ea_label_to_pauli_string("Re(H0_s0*H0_s1)")
    ok_parse = parsed_xx is not None and parsed_xx[1] == {0: "X", 1: "X"}
    record("product_label_parser", ok_parse, f"parsed={parsed_xx}")

                                                                                
                                                                                
    d_pc, N_pc, Nt_pc, T_pc = 2, 2, 1, 1.0
    ctrl_axes_pc, ctrl_labels_pc = build_single_site_axes(d_pc, N_pc, "xy")
    opt_pc = ImprovedPolynomialPontryagin(d_pc, N_pc, Nt_pc, T_pc, 2, drift_strength=0.5, seed=1, verbose=False)
    opt_pc.u[:] = 0.0
    try:
        xx_idx = opt_pc.control_labels.index("Re(H0_s0*H0_s1)")
    except ValueError:
        xx_idx = -1
    if xx_idx >= 0:
        opt_pc.u[0, xx_idx] = 0.1
    u_pc = compile_ea_to_controls_product_aware(
        opt=opt_pc,
        ctrl_labels=ctrl_labels_pc,
        T=T_pc,
        Nt=Nt_pc,
        S=400,
        amp_bound=20.0,
        hard_amp=20.0,
        drift_strength_hw=0.5,
        budget_frac=1.0,
        max_terms_per_slice=4,
        product_bch_reps=1,
        reporter=None,
    )
    ok_product_seed = xx_idx >= 0 and float(np.linalg.norm(u_pc)) > 1e-10
    record("product_compiler_emits_cross_site", ok_product_seed, f"xx_idx={xx_idx} norm={float(np.linalg.norm(u_pc)):.3e}")

    psi0_pc, target_pc = build_targets(d_pc, N_pc, "ghz")
    H0_base_pc = build_drift_base(d_pc, N_pc)
    constructive_smoke = constructive_projection_audit(
        opt=opt_pc,
        u_seed=u_pc,
        H0_base=H0_base_pc,
        ctrl_axes=ctrl_axes_pc,
        psi0=psi0_pc,
        T=T_pc,
        Nt=Nt_pc,
        S=400,
        drift_strength_hw=0.5,
        compiler_audit_summary={"totals": {"emitted_weight_fraction": 1.0, "requested_weight": 1.0, "emitted_weight": 1.0, "skipped_terms": 0}},
    )
    ok_constructive_smoke = (
        int(constructive_smoke.get("slice_count", 0)) == Nt_pc
        and np.isfinite(float(constructive_smoke.get("min_slice_unitary_overlap", 0.0)))
        and np.isfinite(float(constructive_smoke.get("final_checkpoint_state_fidelity", 0.0)))
    )
    record(
        "constructive_projection_audit_smoke",
        ok_constructive_smoke,
        "min_slice={:.6f} final_checkpoint={:.6f}".format(
            float(constructive_smoke.get("min_slice_unitary_overlap", 0.0)),
            float(constructive_smoke.get("final_checkpoint_state_fidelity", 0.0)),
        ),
    )

    H0_pc_dense = 0.5 * qobj_to_dense_matrix(H0_base_pc)
    ctrl_stack_pc = build_dense_operator_stack(ctrl_axes_pc)
    psi0_pc_vec = qobj_to_dense_vector(psi0_pc)
    target_pc_vec = qobj_to_dense_vector(target_pc)
    rng_traj = np.random.default_rng(123)
    u_traj_grad = 0.03 * rng_traj.standard_normal((6, len(ctrl_axes_pc)))
    checkpoint_targets_grad = {2: target_pc_vec, 4: target_pc_vec, 6: target_pc_vec}
    obj_traj_grad, grad_traj, metrics_traj_grad = trajectory_objective_gradient_dense(
        H0_pc_dense,
        ctrl_stack_pc,
        u_traj_grad,
        0.02,
        psi0_pc_vec,
        target_pc_vec,
        checkpoint_targets_grad,
        checkpoint_weight=0.4,
        final_weight=0.6,
        clip=0.0,
    )
    traj_grad_tests = [(0, 0), (0, 1), (2, 0), (2, 3), (5, 1), (5, 3)]
    traj_grad_abs_errors: list[float] = []
    traj_grad_rel_errors: list[float] = []
    for k, j in traj_grad_tests:
        old = float(u_traj_grad[k, j])
        h = 1e-6
        u_traj_grad[k, j] = old + h
        fp, _grad_p, _metrics_p = trajectory_objective_gradient_dense(
            H0_pc_dense,
            ctrl_stack_pc,
            u_traj_grad,
            0.02,
            psi0_pc_vec,
            target_pc_vec,
            checkpoint_targets_grad,
            checkpoint_weight=0.4,
            final_weight=0.6,
            clip=0.0,
        )
        u_traj_grad[k, j] = old - h
        fm, _grad_m, _metrics_m = trajectory_objective_gradient_dense(
            H0_pc_dense,
            ctrl_stack_pc,
            u_traj_grad,
            0.02,
            psi0_pc_vec,
            target_pc_vec,
            checkpoint_targets_grad,
            checkpoint_weight=0.4,
            final_weight=0.6,
            clip=0.0,
        )
        u_traj_grad[k, j] = old
        fd = (float(fp) - float(fm)) / (2.0 * h)
        an = float(grad_traj[k, j])
        abs_err = abs(an - fd)
        traj_grad_abs_errors.append(abs_err)
        if abs(fd) > 1e-7:
            traj_grad_rel_errors.append(abs_err / max(1e-12, abs(fd)))
    max_traj_grad_abs = max(traj_grad_abs_errors) if traj_grad_abs_errors else 0.0
    max_traj_grad_rel = max(traj_grad_rel_errors) if traj_grad_rel_errors else 0.0
    ok_traj_grad = (
        np.isfinite(float(obj_traj_grad))
        and max_traj_grad_abs < 1e-5
        and max_traj_grad_rel < 5e-2
        and int(metrics_traj_grad.get("checkpoint_count", 0)) == 3
        and len(metrics_traj_grad.get("checkpoint_fidelities", [])) == 3
    )
    record(
        "trajectory_objective_gradient_spotcheck",
        ok_traj_grad,
        "obj={:.6f} max_abs_err={:.3e} max_rel_sig={:.3e} checkpoints={}".format(
            float(obj_traj_grad),
            max_traj_grad_abs,
            max_traj_grad_rel,
            [int(x) for x in metrics_traj_grad.get("checkpoint_steps", [])],
        ),
    )

    gate_objective_errors: dict[str, float] = {}
    gate_objective_ok = True
    for mode in ("geomean", "softmin", "softmin_all"):
        u_mode = 0.03 * rng_traj.standard_normal((6, len(ctrl_axes_pc)))
        obj_mode, grad_mode, metrics_mode = trajectory_objective_gradient_dense(
            H0_pc_dense,
            ctrl_stack_pc,
            u_mode,
            0.02,
            psi0_pc_vec,
            target_pc_vec,
            checkpoint_targets_grad,
            checkpoint_weight=0.5,
            final_weight=0.5,
            clip=0.0,
            objective_mode=mode,
            objective_softmin_tau=0.05,
        )
        max_mode_abs = 0.0
        for k, j in traj_grad_tests[:3]:
            old = float(u_mode[k, j])
            h = 1e-6
            u_mode[k, j] = old + h
            fp, _grad_p, _metrics_p = trajectory_objective_gradient_dense(
                H0_pc_dense,
                ctrl_stack_pc,
                u_mode,
                0.02,
                psi0_pc_vec,
                target_pc_vec,
                checkpoint_targets_grad,
                checkpoint_weight=0.5,
                final_weight=0.5,
                clip=0.0,
                objective_mode=mode,
                objective_softmin_tau=0.05,
            )
            u_mode[k, j] = old - h
            fm, _grad_m, _metrics_m = trajectory_objective_gradient_dense(
                H0_pc_dense,
                ctrl_stack_pc,
                u_mode,
                0.02,
                psi0_pc_vec,
                target_pc_vec,
                checkpoint_targets_grad,
                checkpoint_weight=0.5,
                final_weight=0.5,
                clip=0.0,
                objective_mode=mode,
                objective_softmin_tau=0.05,
            )
            u_mode[k, j] = old
            fd = (float(fp) - float(fm)) / (2.0 * h)
            max_mode_abs = max(max_mode_abs, abs(float(grad_mode[k, j]) - fd))
        gate_objective_errors[mode] = float(max_mode_abs)
        gate_objective_ok = (
            gate_objective_ok
            and np.isfinite(float(obj_mode))
            and str(metrics_mode.get("objective_mode")) == mode
            and max_mode_abs < 1e-5
        )
    record(
        "trajectory_gate_objective_gradient_spotcheck",
        gate_objective_ok,
        f"max_abs_by_mode={gate_objective_errors}",
    )

    expansions_pc, pauli_rows_pc, pauli_summary_pc = build_exact_pauli_expansion_audit(opt_pc, tol=1e-10)
    xx_key = ((0, "X"), (1, "X"))
    xx_coeff = 0.0 + 0.0j
    if xx_idx >= 0:
        for key, coeff in expansions_pc[xx_idx]:
            if key == xx_key:
                xx_coeff = coeff
                break
    ok_pauli_expansion = (
        xx_idx >= 0
        and abs(xx_coeff - 1.0) < 1e-10
        and int(pauli_summary_pc.get("parser_mismatches", 1)) == 0
        and float(pauli_summary_pc.get("max_reconstruction_residual_fro", 1.0)) < 1e-9
    )
    record(
        "exact_pauli_expansion_map",
        ok_pauli_expansion,
        "xx_coeff={} mismatches={} max_resid={:.3e} rows={}".format(
            xx_coeff,
            pauli_summary_pc.get("parser_mismatches"),
            float(pauli_summary_pc.get("max_reconstruction_residual_fro", 0.0)),
            len(pauli_rows_pc),
        ),
    )

    opt_b2 = ImprovedPolynomialPontryagin(d_pc, N_pc, 1, 0.2, 2, drift_strength=0.5, seed=7, verbose=False)
    opt_b2.u[:] = 0.0
    try:
        b2_xx_idx = opt_b2.control_labels.index("Re(H0_s0*H0_s1)")
        b2_zz_idx = opt_b2.control_labels.index("Re(H2_s0*H2_s1)")
    except ValueError:
        b2_xx_idx = -1
        b2_zz_idx = -1
                                                                         
                                                                             
                                                                            
    if b2_xx_idx >= 0 and b2_zz_idx >= 0:
        opt_b2.u[0, b2_xx_idx] = 0.475
        opt_b2.u[0, b2_zz_idx] = -0.5
    b2_diag: list[dict[str, Any]] = []
    b2_terms: list[dict[str, Any]] = []
    b2_pauli: list[dict[str, Any]] = []
    u_b2, b2_summary = compile_ea_to_controls_constructive_batch2(
        opt=opt_b2,
        ctrl_labels=ctrl_labels_pc,
        T=0.2,
        Nt=1,
        S=2000,
        amp_bound=1e6,
        hard_amp=1e6,
        drift_strength_hw=0.5,
        residual_tol=2e-3,
        max_frames=4,
        diagnostics=b2_diag,
        term_diagnostics=b2_terms,
        pauli_audit_rows=b2_pauli,
        reporter=None,
    )
    b2_audit = constructive_projection_audit(
        opt=opt_b2,
        u_seed=u_b2,
        H0_base=H0_base_pc,
        ctrl_axes=ctrl_axes_pc,
        psi0=psi0_pc,
        T=0.2,
        Nt=1,
        S=2000,
        drift_strength_hw=0.5,
        compiler_audit_summary={"totals": {"emitted_weight_fraction": b2_summary.get("emitted_weight_fraction", 0.0), "requested_weight": b2_summary.get("requested_weight", 0.0), "emitted_weight": b2_summary.get("emitted_weight", 0.0), "skipped_terms": 0}},
        mode="constructive_batch2",
    )
    ok_batch2 = (
        b2_xx_idx >= 0
        and b2_zz_idx >= 0
        and float(b2_summary.get("emitted_weight_fraction", 0.0)) > 0.99
        and float(b2_audit.get("min_slice_unitary_overlap", 0.0)) > 0.995
        and u_b2.shape == (2000, len(ctrl_labels_pc))
    )
    record(
        "constructive_batch2_n2_slice",
        ok_batch2,
        "emitted_fraction={:.6f} min_slice={:.6f} final_checkpoint={:.6f} shape={}".format(
            float(b2_summary.get("emitted_weight_fraction", 0.0)),
            float(b2_audit.get("min_slice_unitary_overlap", 0.0)),
            float(b2_audit.get("final_checkpoint_state_fidelity", 0.0)),
            u_b2.shape,
        ),
    )

    forbidden_names = set(compile_ea_to_controls_constructive_batch2.__code__.co_names)
    ok_batch2_no_repair = not (
        "trajectory_hardware_project" in forbidden_names
        or "compile_ea_to_target_continuation_project" in forbidden_names
        or "grape_state_iters" in forbidden_names
    )
    record("constructive_batch2_no_repair_modes", ok_batch2_no_repair, f"forbidden={sorted(forbidden_names & {'trajectory_hardware_project', 'compile_ea_to_target_continuation_project', 'grape_state_iters'})}")

    opt_budget = ImprovedPolynomialPontryagin(d_pc, N_pc, 2, 1.0, 2, drift_strength=0.5, seed=9, verbose=False)
    opt_budget.u[:] = 0.0
    try:
        budget_xx_idx = opt_budget.control_labels.index("Re(H0_s0*H0_s1)")
        budget_yx_idx = opt_budget.control_labels.index("Re(H1_s0*H0_s1)")
    except ValueError:
        budget_xx_idx = -1
        budget_yx_idx = -1
    if budget_xx_idx >= 0 and budget_yx_idx >= 0:
        opt_budget.u[:, budget_xx_idx] = 2.0
        opt_budget.u[:, budget_yx_idx] = 2.0
    budget_summary = configure_ea_batch2_projectability(
        opt_budget,
        drift_strength_hw=0.5,
        budget_frac=0.9,
        pauli_tol=1e-10,
    )
    ok_budget_projector = (
        budget_xx_idx >= 0
        and budget_yx_idx >= 0
        and float(budget_summary.get("last_max_nuclear_after", 1e9)) <= float(budget_summary.get("nuclear_radius", 0.0)) + 1e-9
        and int(budget_summary.get("last_projected_slices", 0)) > 0
    )
    record(
        "ea_batch2_projectability_budget",
        ok_budget_projector,
        "radius={:.6f} before={:.6f} after={:.6f} projected={}".format(
            float(budget_summary.get("nuclear_radius", 0.0)),
            float(budget_summary.get("last_max_nuclear_before", 0.0)),
            float(budget_summary.get("last_max_nuclear_after", 0.0)),
            int(budget_summary.get("last_projected_slices", 0)),
        ),
    )

    opt_budget_n3 = ImprovedPolynomialPontryagin(2, 3, 2, 1.0, 2, drift_strength=0.5, seed=10, verbose=False)
    opt_budget_n3.u[:] = 0.0
    n3_active_indices: list[int] = []
    for label in ["Re(H0_s0*H0_s1)", "Re(H1_s1*H0_s2)", "Re(H2_s0*H2_s2)"]:
        try:
            n3_active_indices.append(opt_budget_n3.control_labels.index(label))
        except ValueError:
            pass
    for idx in n3_active_indices:
        opt_budget_n3.u[:, idx] = 2.0
    budget_n3_summary = configure_ea_batch2_projectability(
        opt_budget_n3,
        drift_strength_hw=0.5,
        budget_frac=0.25,
        pauli_tol=1e-10,
    )
    ok_budget_n3_projector = (
        len(n3_active_indices) >= 2
        and str(budget_n3_summary.get("projector")) == "nbody_signed_l1_pair_echo"
        and float(budget_n3_summary.get("last_max_l1_after", 1e9)) <= float(budget_n3_summary.get("theta_l1_radius", 0.0)) + 1e-9
        and int(budget_n3_summary.get("last_projected_slices", 0)) > 0
    )
    record(
        "ea_batch2_projectability_n3_budget",
        ok_budget_n3_projector,
        "radius={:.6f} before={:.6f} after={:.6f} projected={}".format(
            float(budget_n3_summary.get("theta_l1_radius", 0.0)),
            float(budget_n3_summary.get("last_max_l1_before", 0.0)),
            float(budget_n3_summary.get("last_max_l1_after", 0.0)),
            int(budget_n3_summary.get("last_projected_slices", 0)),
        ),
    )

    ctrl_axes_n3, ctrl_labels_n3 = build_single_site_axes(2, 3, "xy")
    opt_b2_n3 = ImprovedPolynomialPontryagin(2, 3, 1, 0.2, 2, drift_strength=0.5, seed=11, verbose=False)
    opt_b2_n3.u[:] = 0.0
    try:
        n3_xx_idx = opt_b2_n3.control_labels.index("Re(H0_s0*H0_s1)")
    except ValueError:
        n3_xx_idx = -1
    if n3_xx_idx >= 0:
        opt_b2_n3.u[0, n3_xx_idx] = 0.1
    n3_diag: list[dict[str, Any]] = []
    n3_terms: list[dict[str, Any]] = []
    u_b2_n3, b2_n3_summary = compile_ea_to_controls_constructive_batch2(
        opt=opt_b2_n3,
        ctrl_labels=ctrl_labels_n3,
        T=0.2,
        Nt=1,
        S=1000,
        amp_bound=1e6,
        hard_amp=1e6,
        drift_strength_hw=0.5,
        residual_tol=2e-3,
        max_frames=16,
        diagnostics=n3_diag,
        term_diagnostics=n3_terms,
        reporter=None,
    )
    ok_batch2_n3 = (
        n3_xx_idx >= 0
        and u_b2_n3.shape == (1000, len(ctrl_labels_n3))
        and float(b2_n3_summary.get("emitted_weight_fraction", 0.0)) > 0.95
        and n3_diag
        and str(n3_diag[0].get("batch2_status")) == "pair_echo_ok"
    )
    record(
        "constructive_batch2_n3_pair_echo",
        ok_batch2_n3,
        "emitted_fraction={:.6f} status={} used_steps={} shape={}".format(
            float(b2_n3_summary.get("emitted_weight_fraction", 0.0)),
            str(n3_diag[0].get("batch2_status") if n3_diag else "missing"),
            int(n3_diag[0].get("used_steps", 0)) if n3_diag else -1,
            u_b2_n3.shape,
        ),
    )

    opt_d3 = ImprovedPolynomialPontryagin(2, 3, 1, 0.2, 3, drift_strength=0.5, seed=12, verbose=False)
    opt_d3.u[:] = 0.0
    d3_expansions, _d3_rows, _d3_summary = build_exact_pauli_expansion_audit(opt_d3, tol=1e-10)
    d3_idx = -1
    d3_coeff = 0.0
    d3_key: tuple[tuple[int, str], ...] = tuple()
    for j, expansion in enumerate(d3_expansions):
        for key, coeff in expansion:
            if pauli_key_degree(key) == 3 and abs(np.imag(coeff)) < 1e-10 and abs(np.real(coeff)) > 1e-10:
                d3_idx = int(j)
                d3_coeff = float(np.real(coeff))
                d3_key = key
                break
        if d3_idx >= 0:
            break
    if d3_idx >= 0:
        opt_d3.u[0, d3_idx] = 0.004 / (float(opt_d3.dt) * d3_coeff)
    d3_projector_summary = configure_ea_degree3_projectability(
        opt_d3,
        theta_radius=5e-4,
        pauli_tol=1e-10,
    )
    d3_diag: list[dict[str, Any]] = []
    d3_terms: list[dict[str, Any]] = []
    u_d3, d3_summary = compile_ea_to_controls_constructive_batch2(
        opt=opt_d3,
        ctrl_labels=ctrl_labels_n3,
        T=0.2,
        Nt=1,
        S=40000,
        amp_bound=1e6,
        hard_amp=1e6,
        drift_strength_hw=0.5,
        residual_tol=1e-4,
        max_frames=16,
        product_bch_reps=1,
        diagnostics=d3_diag,
        term_diagnostics=d3_terms,
        reporter=None,
    )
    d3_audit = constructive_projection_audit(
        opt=opt_d3,
        u_seed=u_d3,
        H0_base=build_drift_base(2, 3),
        ctrl_axes=ctrl_axes_n3,
        psi0=qt.tensor(qt.basis(2, 0), qt.basis(2, 0), qt.basis(2, 0)),
        T=0.2,
        Nt=1,
        S=40000,
        drift_strength_hw=0.5,
        compiler_audit_summary={"totals": {"emitted_weight_fraction": d3_summary.get("emitted_weight_fraction", 0.0), "requested_weight": d3_summary.get("requested_weight", 0.0), "emitted_weight": d3_summary.get("emitted_weight", 0.0), "skipped_terms": 0}},
        mode="constructive_batch2",
    )
    d3_emitted_terms = [row for row in d3_terms if int(row.get("degree", 0)) == 3 and int(row.get("emitted", 0))]
    ok_d3_projector = (
        d3_idx >= 0
        and str(d3_projector_summary.get("projector")) == "degree3_signed_l1_bch"
        and float(d3_projector_summary.get("last_max_l1_after", 1e9)) <= 5e-4 + 1e-12
        and len(d3_emitted_terms) > 0
        and float(d3_audit.get("min_slice_unitary_overlap", 0.0)) > 0.99
    )
    record(
        "constructive_batch2_degree3_bch",
        ok_d3_projector,
        "key={} radius={:.3e} after={:.3e} emitted={} min_slice={:.6f}".format(
            pauli_key_to_string(d3_key),
            float(d3_projector_summary.get("theta_l1_radius", 0.0)),
            float(d3_projector_summary.get("last_max_l1_after", 0.0)),
            len(d3_emitted_terms),
            float(d3_audit.get("min_slice_unitary_overlap", 0.0)),
        ),
    )

    opt_p2_embed = ImprovedPolynomialPontryagin(2, 3, 2, 0.4, 2, drift_strength=0.5, seed=21, verbose=False)
    opt_p3_embed = ImprovedPolynomialPontryagin(2, 3, 2, 0.4, 3, drift_strength=0.5, seed=22, verbose=False)
    embed_summary = embed_lower_degree_solution(opt_p2_embed, opt_p3_embed, tol=1e-8)
    ok_nested_embed = bool(embed_summary.get("success", False)) and int(embed_summary.get("missing_count", 1)) == 0
    record(
        "nested_p2_to_p3_basis_embedding",
        ok_nested_embed,
        "matched={}/{} max_error={:.3e}".format(
            int(embed_summary.get("matched_count", 0)),
            int(embed_summary.get("lower_control_count", 0)),
            float(embed_summary.get("max_embedding_error", 0.0)),
        ),
    )

    opt_p3_embed4 = ImprovedPolynomialPontryagin(2, 4, 1, 0.2, 3, drift_strength=0.5, seed=24, verbose=False)
    opt_p4_embed = ImprovedPolynomialPontryagin(2, 4, 1, 0.2, 4, drift_strength=0.5, seed=25, verbose=False)
    embed4_summary = embed_lower_degree_solution(opt_p3_embed4, opt_p4_embed, tol=1e-8)
    ok_nested_embed4 = bool(embed4_summary.get("success", False)) and int(embed4_summary.get("missing_count", 1)) == 0
    record(
        "nested_p3_to_p4_basis_embedding",
        ok_nested_embed4,
        "matched={}/{} max_error={:.3e}".format(
            int(embed4_summary.get("matched_count", 0)),
            int(embed4_summary.get("lower_control_count", 0)),
            float(embed4_summary.get("max_embedding_error", 0.0)),
        ),
    )

    opt_cost_d3 = ImprovedPolynomialPontryagin(2, 3, 1, 0.2, 3, drift_strength=0.5, seed=23, verbose=False)
    opt_cost_d3.u[:] = 0.0
    cost_expansions, _cost_rows, _cost_summary = build_exact_pauli_expansion_audit(opt_cost_d3, tol=1e-10)
    cost_idx = -1
    cost_coeff = 0.0
    for j, expansion in enumerate(cost_expansions):
        for key, coeff in expansion:
            if pauli_key_degree(key) == 3 and abs(np.imag(coeff)) < 1e-10 and abs(np.real(coeff)) > 1e-10:
                cost_idx = int(j)
                cost_coeff = float(np.real(coeff))
                break
        if cost_idx >= 0:
            break
    if cost_idx >= 0:
        opt_cost_d3.u[0, cost_idx] = 0.02 / (float(opt_cost_d3.dt) * cost_coeff)
    cost_summary = configure_ea_degree3_cost_projectability(
        opt_cost_d3,
        T=0.2,
        Nt=1,
        S=800,
        hard_amp=1e6,
        drift_strength_hw=0.5,
        step_budget_frac=0.125,
        term_theta_threshold=1e-8,
        product_bch_reps=1,
        pauli_tol=1e-10,
    )
    ok_cost_projector = (
        cost_idx >= 0
        and str(cost_summary.get("projector")) == "degree3_constructive_cost_multiterm"
        and int(cost_summary.get("last_max_emitted_steps", 10**9)) <= int(cost_summary.get("step_budget", -1))
        and int(cost_summary.get("last_max_emitted_steps", 0)) > 0
        and float(cost_summary.get("last_max_l1_after", 0.0)) <= float(cost_summary.get("last_max_l1_before", 0.0)) + 1e-12
    )
    record(
        "degree3_cost_projector_budget",
        ok_cost_projector,
        "budget={} requested={} emitted={} l1 {:.3e}->{:.3e}".format(
            int(cost_summary.get("step_budget", 0)),
            int(cost_summary.get("last_max_requested_steps", 0)),
            int(cost_summary.get("last_max_emitted_steps", 0)),
            float(cost_summary.get("last_max_l1_before", 0.0)),
            float(cost_summary.get("last_max_l1_after", 0.0)),
        ),
    )

    eff_axes_pc, eff_labels_pc, eff_map_pc = build_effective_d2_axes_from_opt(opt_pc, ctrl_axes_pc, ctrl_labels_pc)
    eff_term_diag: list[dict[str, Any]] = []
    u_eff_pc = compile_ea_to_effective_d2_controls(
        opt=opt_pc,
        effective_labels=eff_labels_pc,
        T=T_pc,
        Nt=Nt_pc,
        S=128,
        amp_bound=20.0,
        budget_frac=1.0,
        max_terms_per_slice=4,
        term_diagnostics=eff_term_diag,
        reporter=None,
    )
    eff_tail = u_eff_pc[:, len(ctrl_labels_pc):]
    ok_eff_d2 = (
        len(eff_axes_pc) > len(ctrl_axes_pc)
        and any(label.startswith("EFF2:") for label in eff_labels_pc)
        and float(np.linalg.norm(eff_tail)) > 1e-10
        and any(int(row.get("emitted", 0)) for row in eff_term_diag)
    )
    record(
        "effective_d2_compiler",
        ok_eff_d2,
        f"base={len(ctrl_labels_pc)} total={len(eff_labels_pc)} eff_norm={float(np.linalg.norm(eff_tail)):.3e} emitted_rows={sum(int(row.get('emitted', 0)) for row in eff_term_diag)}",
    )

    hw_project_diag: list[dict[str, Any]] = []
    hw_project_terms: list[dict[str, Any]] = []
    u_hw_project, hw_project_summary = compile_ea_to_d2_hardware_project(
        opt=opt_pc,
        ctrl_axes=ctrl_axes_pc,
        ctrl_labels=ctrl_labels_pc,
        H0_base=H0_base_pc,
        psi0=psi0_pc,
        target=target_pc,
        T=T_pc,
        Nt=Nt_pc,
        S=160,
        drift_strength_hw=0.5,
        amp_bound=20.0,
        hard_amp=20.0,
        budget_frac=1.0,
        max_terms_per_slice=4,
        projection_iters=2,
        projection_lr=0.02,
        projection_threshold=1.1,
        diagnostics=hw_project_diag,
        term_diagnostics=hw_project_terms,
        reporter=None,
        progress_every=1000,
        seed=3,
    )
    ok_hw_project = (
        u_hw_project.shape == (Nt_pc * 160, len(ctrl_labels_pc))
        and float(np.linalg.norm(u_hw_project)) > 1e-10
        and bool(hw_project_summary.get("hardware_executable"))
        and int(hw_project_summary.get("physical_control_count", -1)) == len(ctrl_labels_pc)
    )
    record(
        "d2_hardware_project_compiler",
        ok_hw_project,
        "shape={} norm={:.3e} refF={:.6f} targetF={:.6f}".format(
            u_hw_project.shape,
            float(np.linalg.norm(u_hw_project)),
            float(hw_project_summary.get("hardware_reference_fidelity_after_projection", 0.0)),
            float(hw_project_summary.get("hardware_target_fidelity_after_projection", 0.0)),
        ),
    )
    hw_smoke_cfg = OptConfig(lr=0.02, l2=0.0, amp=20.0, clip=5.0, backtracks=1, accept_mode="soft", accept_drop=1e-3, threshold=2.0, verbose=0)
    hw_smoke_stages = [Stage(scale=1.0, S=160, iters=2, lr=0.02)]
    rr_hw_smoke, u_hw_smoke_final = run_method_stages(
        name="D2HWSELFTEST",
        psi0=psi0_pc,
        target=target_pc,
        H0_base=H0_base_pc,
        drift_strength_hw=0.5,
        ctrl_axes=ctrl_axes_pc,
        T=T_pc,
        Nt=Nt_pc,
        stages=hw_smoke_stages,
        homotopy_mode="mult",
        u0=u_hw_project,
        opt_cfg_template=hw_smoke_cfg,
        threshold=2.0,
        seed=4,
        jitter_list=[0.0],
        verbose=0,
        reporter=None,
        progress_every=1000,
        trace_dir=None,
        trace_prefix="",
        checkpoint_manager=None,
        H0_base_dense=qobj_to_dense_matrix(H0_base_pc),
        ctrl_stack_dense=build_dense_operator_stack(ctrl_axes_pc),
        psi0_vec=qobj_to_dense_vector(psi0_pc),
        target_vec=qobj_to_dense_vector(target_pc),
    )
    ok_hw_smoke = (
        np.isfinite(rr_hw_smoke.finalF_phys)
        and 0.0 <= rr_hw_smoke.finalF_phys <= 1.000001
        and u_hw_smoke_final.shape == (Nt_pc * 160, len(ctrl_labels_pc))
    )
    record("d2_hardware_project_smoke", ok_hw_smoke, f"finalF={rr_hw_smoke.finalF_phys:.6f} shape={u_hw_smoke_final.shape}")

    opt_traj = ImprovedPolynomialPontryagin(d_pc, N_pc, 3, 1.5, 2, drift_strength=0.5, seed=5, verbose=False)
    F_traj_eff = float(opt_traj.optimize_improved(psi0_pc, target_pc, max_iter=20, reporter=None, progress_every=1000, trace_rows=[]))
    u_traj, traj_summary = compile_ea_to_trajectory_hardware_project(
        opt=opt_traj,
        ctrl_axes=ctrl_axes_pc,
        ctrl_labels=ctrl_labels_pc,
        H0_base=H0_base_pc,
        psi0=psi0_pc,
        target=target_pc,
        T=1.5,
        Nt=3,
        S=24,
        drift_strength_hw=0.5,
        amp_bound=5.0,
        hard_amp=5.0,
        trajectory_reference="full",
        trajectory_weight=0.5,
        trajectory_checkpoints="all",
        trajectory_project_iters=4,
        trajectory_project_lr=0.03,
        trajectory_init="linear",
        compiler_eps=0.02,
        budget_frac=1.0,
        max_terms_per_slice=8,
        projection_clip=5.0,
        projection_backtracks=1,
        projection_threshold=2.0,
        reporter=None,
        progress_every=1000,
        H0_base_dense=qobj_to_dense_matrix(H0_base_pc),
        ctrl_stack_dense=build_dense_operator_stack(ctrl_axes_pc),
        psi0_vec=qobj_to_dense_vector(psi0_pc),
        target_vec=qobj_to_dense_vector(target_pc),
    )
    traj_proj = traj_summary.get("projection", {})
    traj_improved = (
        float(traj_proj.get("target_fidelity_after", 0.0)) >= float(traj_proj.get("target_fidelity_before", 0.0)) - 1e-9
        or float(traj_proj.get("trajectory_fidelity_after", 0.0)) >= float(traj_proj.get("trajectory_fidelity_before", 0.0)) - 1e-9
    )
    ok_traj_project = (
        u_traj.shape == (3 * 24, len(ctrl_labels_pc))
        and bool(traj_summary.get("hardware_executable"))
        and int(traj_summary.get("physical_control_count", -1)) == len(ctrl_labels_pc)
        and traj_improved
    )
    record(
        "ea_trajectory_hardware_project",
        ok_traj_project,
        "F_eff={:.6f} shape={} target={:.6f}->{:.6f} traj={:.6f}->{:.6f}".format(
            F_traj_eff,
            u_traj.shape,
            float(traj_proj.get("target_fidelity_before", 0.0)),
            float(traj_proj.get("target_fidelity_after", 0.0)),
            float(traj_proj.get("trajectory_fidelity_before", 0.0)),
            float(traj_proj.get("trajectory_fidelity_after", 0.0)),
        ),
    )

    u_tc, tc_summary = compile_ea_to_target_continuation_project(
        opt=opt_traj,
        ctrl_axes=ctrl_axes_pc,
        ctrl_labels=ctrl_labels_pc,
        H0_base=H0_base_pc,
        psi0=psi0_pc,
        target=target_pc,
        T=1.5,
        Nt=3,
        S=24,
        drift_strength_hw=0.5,
        amp_bound=5.0,
        hard_amp=5.0,
        trajectory_reference="full",
        continuation_iters="3,2",
        continuation_lrs="0.03,0.02",
        continuation_weights="0,0.1",
        continuation_checkpoints="final,late2",
        continuation_min_improve=0.0,
        continuation_init="linear",
        continuation_policy="manual",
        compiler_eps=0.02,
        budget_frac=1.0,
        max_terms_per_slice=8,
        projection_clip=5.0,
        projection_backtracks=1,
        projection_threshold=2.0,
        reporter=None,
        progress_every=1000,
        H0_base_dense=qobj_to_dense_matrix(H0_base_pc),
        ctrl_stack_dense=build_dense_operator_stack(ctrl_axes_pc),
        psi0_vec=qobj_to_dense_vector(psi0_pc),
        target_vec=qobj_to_dense_vector(target_pc),
    )
    tc_stages = tc_summary.get("stages", [])
    tc_tol = 1e-9
    tc_nondegraded = (
        float(tc_summary.get("target_fidelity_best", 0.0)) >= float(tc_summary.get("target_fidelity_initial", 0.0)) - tc_tol
        or any(float(row.get("trajectory_after", 0.0)) >= float(row.get("trajectory_before", 0.0)) - tc_tol for row in tc_stages)
        or any(float(row.get("objective_after", 0.0)) >= float(row.get("objective_before", 0.0)) - tc_tol for row in tc_stages)
    )
    ok_tc_project = (
        u_tc.shape == (3 * 24, len(ctrl_labels_pc))
        and bool(tc_summary.get("hardware_executable"))
        and not bool(tc_summary.get("effective_control_mode"))
        and int(tc_summary.get("physical_control_count", -1)) == len(ctrl_labels_pc)
        and tc_nondegraded
    )
    record(
        "ea_target_continuation_project",
        ok_tc_project,
        "shape={} target={:.6f}->{:.6f} stages={}".format(
            u_tc.shape,
            float(tc_summary.get("target_fidelity_initial", 0.0)),
            float(tc_summary.get("target_fidelity_best", 0.0)),
            len(tc_stages),
        ),
    )

    u_tca, tca_summary = compile_ea_to_target_continuation_project(
        opt=opt_traj,
        ctrl_axes=ctrl_axes_pc,
        ctrl_labels=ctrl_labels_pc,
        H0_base=H0_base_pc,
        psi0=psi0_pc,
        target=target_pc,
        T=1.5,
        Nt=3,
        S=24,
        drift_strength_hw=0.5,
        amp_bound=5.0,
        hard_amp=5.0,
        trajectory_reference="full",
        continuation_iters="2,2",
        continuation_lrs="0.03,0.02",
        continuation_weights="0,0.1",
        continuation_checkpoints="final,late2",
        continuation_min_improve=0.0,
        continuation_init="linear",
        continuation_policy="adaptive",
        continuation_escape_weights="0.2",
        continuation_escape_checkpoints="late2",
        continuation_escape_iters="2",
        continuation_max_stages=3,
        compiler_eps=0.02,
        budget_frac=1.0,
        max_terms_per_slice=8,
        projection_clip=5.0,
        projection_backtracks=1,
        projection_threshold=2.0,
        reporter=None,
        progress_every=1000,
        H0_base_dense=qobj_to_dense_matrix(H0_base_pc),
        ctrl_stack_dense=build_dense_operator_stack(ctrl_axes_pc),
        psi0_vec=qobj_to_dense_vector(psi0_pc),
        target_vec=qobj_to_dense_vector(target_pc),
    )
    tca_stages = tca_summary.get("stages", [])
    tca_targets = [float(tca_summary.get("target_fidelity_initial", 0.0))]
    tca_targets.extend(float(row.get("target_after", 0.0)) for row in tca_stages)
    tca_best_expected = max(tca_targets) if tca_targets else 0.0
    ok_tca_project = (
        u_tca.shape == (3 * 24, len(ctrl_labels_pc))
        and str(tca_summary.get("policy")) == "adaptive"
        and bool(tca_summary.get("hardware_executable"))
        and not bool(tca_summary.get("effective_control_mode"))
        and int(tca_summary.get("physical_control_count", -1)) == len(ctrl_labels_pc)
        and len(tca_stages) >= 2
        and abs(float(tca_summary.get("target_fidelity_best", 0.0)) - tca_best_expected) < 1e-10
    )
    record(
        "ea_target_continuation_adaptive",
        ok_tca_project,
        "shape={} stages={} target={:.6f}->{:.6f}".format(
            u_tca.shape,
            len(tca_stages),
            float(tca_summary.get("target_fidelity_initial", 0.0)),
            float(tca_summary.get("target_fidelity_best", 0.0)),
        ),
    )

    finite_steps, finite_status = estimate_pauli_product_steps(
        {0: "Z", 1: "Z", 2: "Z"},
        0.001,
        10.0 / 40000,
        800.0,
        2**3,
        max_degree=3,
        bch_reps=4,
        instant_overhead=False,
    )
    instant_steps, instant_status = estimate_pauli_product_steps(
        {0: "Z", 1: "Z", 2: "Z"},
        0.001,
        10.0 / 40000,
        800.0,
        2**3,
        max_degree=3,
        bch_reps=4,
        instant_overhead=True,
    )
    ok_cost_estimator = finite_steps > instant_steps > 0
    record(
        "product_cost_estimator",
        ok_cost_estimator,
        f"finite={finite_steps}({finite_status}) instant={instant_steps}({instant_status})",
    )

                                                                             
                                                                            
                                                                    
    def _pauli_product_dense(n_qubits: int, axes_by_site: dict[int, str]) -> np.ndarray:
        mats = {
            "I": qt.qeye(2),
            "X": qt.sigmax(),
            "Y": qt.sigmay(),
            "Z": qt.sigmaz(),
        }
        return qt.tensor(*[mats[axes_by_site.get(q, "I")] for q in range(n_qubits)]).full()

    def _simulate_control_block(n_qubits: int, controls_arr: np.ndarray, used_steps: int, dt_step: float) -> np.ndarray:
        h0_arr = build_drift_base(2, n_qubits).full()
        ctrl_axes_arr, _ = build_single_site_axes(2, n_qubits, "xy")
        ctrl_dense = [axis.full() for axis in ctrl_axes_arr]
        dim_arr = 2**n_qubits
        U_arr = np.eye(dim_arr, dtype=complex)
        m_step = 0
        while m_step < int(used_steps):
            row = controls_arr[m_step].copy()
            n_step = m_step + 1
            while n_step < int(used_steps) and np.allclose(controls_arr[n_step], row, atol=1e-14, rtol=0.0):
                n_step += 1
            H_arr = h0_arr.copy()
            for j_ctrl, axis_arr in enumerate(ctrl_dense):
                amp_val = row[j_ctrl]
                if abs(amp_val) > 0:
                    H_arr = H_arr + amp_val * axis_arr
            U_arr = expm(-1j * dt_step * (n_step - m_step) * H_arr) @ U_arr
            m_step = n_step
        return U_arr

    unitary_overlaps: list[float] = []
    n_unitary = 3
    dim_unitary = 2**n_unitary
    S_unitary = 5000
    dt_unitary = 1.0 / S_unitary
    amp_unitary = 200.0
    _, labels_unitary = build_single_site_axes(2, n_unitary, "xy")
    for theta_unitary in (0.04, -0.04):
        u_unitary = np.zeros((S_unitary, len(labels_unitary)))
        used_unitary, status_unitary = emit_two_body_product_block(
            u_unitary,
            0,
            labels_unitary,
            {0: "Z", 1: "Z"},
            theta_unitary,
            dt_unitary,
            amp_unitary,
            dim_unitary,
            1.0,
        )
        if status_unitary != "ok" or used_unitary <= 0:
            unitary_overlaps.append(0.0)
            continue
        U_block = _simulate_control_block(n_unitary, u_unitary, used_unitary, dt_unitary)
        P_target = _pauli_product_dense(n_unitary, {0: "Z", 1: "Z"})
        U_target = expm(-1j * theta_unitary * P_target / dim_unitary)
        unitary_overlaps.append(float(abs(np.trace(U_target.conj().T @ U_block)) / dim_unitary))
    ok_unitary_block = min(unitary_overlaps) > 0.999
    record("product_block_unitary", ok_unitary_block, f"min_overlap={min(unitary_overlaps):.6f} overlaps={unitary_overlaps}")

    higher_product_overlaps: dict[str, float] = {}
    high_degree_cases = [
        ("degree3_positive", 3, {0: "Z", 1: "Z", 2: "Z"}, +0.001, 8, 40000, 10.0, 800.0, 0.99),
        ("degree3_negative", 3, {0: "X", 1: "Y", 2: "Z"}, -0.001, 8, 40000, 10.0, 800.0, 0.99),
        ("degree4_high_hardpulse", 4, {0: "Z", 1: "Z", 2: "Z", 3: "Z"}, +0.00002, 4, 40000, 8.0, 10000.0, 0.99),
    ]
    for case_name, n_case, axes_case, theta_case, reps_case, S_case, T_case, amp_case, _threshold_case in high_degree_cases:
        dim_case = 2**n_case
        dt_case = T_case / S_case
        _, labels_case = build_single_site_axes(2, n_case, "xy")
        u_case = np.zeros((S_case, len(labels_case)))
        used_case, status_case = emit_pauli_product_bch_block(
            u_case,
            0,
            labels_case,
            axes_case,
            theta_case,
            dt_case,
            amp_case,
            dim_case,
            bch_reps=reps_case,
        )
        if used_case <= 0:
            higher_product_overlaps[case_name] = 0.0
            continue
        U_case = _simulate_control_block(n_case, u_case, used_case, dt_case)
        P_case = _pauli_product_dense(n_case, axes_case)
        U_target_case = expm(-1j * theta_case * P_case / dim_case)
        higher_product_overlaps[case_name] = float(abs(np.trace(U_target_case.conj().T @ U_case)) / dim_case)
    ok_higher_products = all(
        higher_product_overlaps.get(case_name, 0.0) > threshold_case
        for case_name, _n_case, _axes_case, _theta_case, _reps_case, _S_case, _T_case, _amp_case, threshold_case in high_degree_cases
    )
    record("higher_product_bch_unitary", ok_higher_products, f"overlaps={higher_product_overlaps}")

    opt_d4_compile = ImprovedPolynomialPontryagin(2, 4, 1, 8.0, 4, drift_strength=1.0, seed=31, verbose=False)
    opt_d4_compile.u[:] = 0.0
    d4_expansions, _d4_rows, _d4_summary = build_exact_pauli_expansion_audit(opt_d4_compile, tol=1e-10)
    d4_idx = -1
    d4_coeff = 0.0
    d4_key: tuple[tuple[int, str], ...] = tuple()
    for j, expansion in enumerate(d4_expansions):
        for key, coeff in expansion:
            if pauli_key_degree(key) == 4 and abs(np.imag(coeff)) < 1e-10 and abs(np.real(coeff)) > 1e-10:
                d4_idx = int(j)
                d4_coeff = float(np.real(coeff))
                d4_key = key
                break
        if d4_idx >= 0:
            break
    if d4_idx >= 0:
        opt_d4_compile.u[0, d4_idx] = 2e-5 / (float(opt_d4_compile.dt) * d4_coeff)
    d4_diag: list[dict[str, Any]] = []
    d4_terms: list[dict[str, Any]] = []
    _u_d4_compile, _d4_compile_summary = compile_ea_to_controls_constructive_batch2(
        opt=opt_d4_compile,
        ctrl_labels=build_single_site_axes(2, 4, "xy")[1],
        T=8.0,
        Nt=1,
        S=40000,
        amp_bound=10000.0,
        hard_amp=10000.0,
        drift_strength_hw=1.0,
        residual_tol=1e-7,
        max_frames=16,
        product_bch_reps=4,
        diagnostics=d4_diag,
        term_diagnostics=d4_terms,
        reporter=None,
    )
    d4_emitted_terms = [row for row in d4_terms if int(row.get("degree", 0)) == 4 and int(row.get("emitted", 0))]
    d4_unsupported_terms = [row for row in d4_terms if "unsupported_degree4" in str(row.get("status", ""))]
    ok_d4_compile = d4_idx >= 0 and len(d4_emitted_terms) > 0 and len(d4_unsupported_terms) == 0
    record(
        "constructive_batch2_degree4_bch",
        ok_d4_compile,
        "key={} emitted={} unsupported={} used={}".format(
            pauli_key_to_string(d4_key),
            len(d4_emitted_terms),
            len(d4_unsupported_terms),
            int(d4_emitted_terms[0].get("used_steps", 0)) if d4_emitted_terms else 0,
        ),
    )

    opt_shared = ImprovedPolynomialPontryagin(2, 4, 1, 8.0, 4, drift_strength=1.0, seed=4, verbose=False)
    opt_shared.u[:] = 0.0
    shared_idx = -1
    shared_key: tuple[tuple[int, str], ...] = tuple()
    shared_coeff = 0.0
    for j, op in enumerate(opt_shared.controls):
        for key, coeff in exact_pauli_expansion(op, 4):
            if pauli_key_degree(key) == 4 and abs(coeff) > 1e-12:
                shared_idx = j
                shared_key = key
                shared_coeff = float(np.real(coeff))
                break
        if shared_idx >= 0:
            break
    if shared_idx >= 0:
        opt_shared.u[0, shared_idx] = 2e-5 / (float(opt_shared.dt) * shared_coeff)
    shared_summary = configure_ea_shared_product_scheduler(
        opt_shared,
        T=8.0,
        Nt=1,
        S=40000,
        hard_amp=10000.0,
        drift_strength_hw=1.0,
        product_bch_reps=1,
        total_budget_frac=0.95,
        degree4_budget_frac=0.30,
        degree4_min_terms_per_slice=1,
        trim_degree2=True,
        degree3_threshold=1e-8,
        degree4_threshold=1e-10,
        pauli_tol=1e-10,
    )
    ok_shared_scheduler = (
        shared_idx >= 0
        and int(shared_summary.get("last_max_estimated_used_steps", 0)) <= int(shared_summary.get("total_budget", 0))
        and int(shared_summary.get("last_min_d4_kept_terms", 0)) >= 1
    )
    record(
        "shared_product_scheduler_p4_budget",
        ok_shared_scheduler,
        "key={} used<={}/{} d4_terms={}..{}".format(
            pauli_key_to_string(shared_key),
            int(shared_summary.get("last_max_estimated_used_steps", 0)),
            int(shared_summary.get("total_budget", 0)),
            int(shared_summary.get("last_min_d4_kept_terms", 0)),
            int(shared_summary.get("last_max_d4_kept_terms", 0)),
        ),
    )

                                                   
    rng = np.random.RandomState(0)
    d, N, Nt_small, T_small, p = 2, 2, 4, 1.5, 2
    psi0, target = build_targets(d, N, "ghz")
    opt = ImprovedPolynomialPontryagin(d, N, Nt_small, T_small, p, drift_strength=0.5, seed=1, verbose=False)
    opt.u = 0.05 * rng.randn(Nt_small, len(opt.controls))
    F0, grad = opt.compute_fidelity_and_gradient(psi0, target)
    preferred_labels = {
        "Re(H0_s0*H1_s1)",
        "Re(H1_s0*H0_s1)",
    }
    preferred = [j for j, label in enumerate(opt.control_labels) if label in preferred_labels]
    if len(preferred) < 2:
        preferred = [j for j, deg in enumerate(opt.control_degrees) if int(deg) == 2][:2]
    tests = [(k, j) for k in (0, 1, 2) for j in preferred[:2]]
    h = 1e-6
    abs_errs: list[float] = []
    rel_errs_sig: list[float] = []
    for k, j in tests:
        old = opt.u[k, j]
        opt.u[k, j] = old + h
        Fp, _ = opt.compute_fidelity_and_gradient(psi0, target)
        opt.u[k, j] = old - h
        Fm, _ = opt.compute_fidelity_and_gradient(psi0, target)
        opt.u[k, j] = old
        fd = (Fp - Fm) / (2 * h)
        an = float(grad[k, j])
        abs_err = abs(an - fd)
        abs_errs.append(abs_err)
        if abs(fd) > 2e-2:
            rel_errs_sig.append(abs_err / abs(fd))
    max_abs_err = max(abs_errs) if abs_errs else 0.0
    max_rel_sig = max(rel_errs_sig) if rel_errs_sig else 0.0
    ok_grad = max_abs_err < 1e-4 and max_rel_sig < 5e-3 and np.isfinite(F0) and len(tests) > 0
    record("gradient_spotcheck", ok_grad, f"F={F0:.6f} tested={tests} max_abs_err={max_abs_err:.3e} max_rel_sig={max_rel_sig:.3e}")

    opt_seed_a = ImprovedPolynomialPontryagin(d, N, Nt_small, T_small, p, drift_strength=0.5, seed=11, verbose=False)
    opt_seed_b = ImprovedPolynomialPontryagin(d, N, Nt_small, T_small, p, drift_strength=0.5, seed=12, verbose=False)
    opt_seed_a2 = ImprovedPolynomialPontryagin(d, N, Nt_small, T_small, p, drift_strength=0.5, seed=11, verbose=False)
    seed_diff = float(np.linalg.norm(opt_seed_a.u - opt_seed_b.u))
    seed_repeat = float(np.linalg.norm(opt_seed_a.u - opt_seed_a2.u))
    ok_seed_sensitive = seed_diff > 1e-10 and seed_repeat < 1e-12
    record("ea_seed_sensitivity", ok_seed_sensitive, f"diff={seed_diff:.3e} repeat={seed_repeat:.3e}")

                                        
    d, N, Nt_smoke, T_smoke, p_smoke = 2, 2, 6, 2.0, 2
    psi0, target = build_targets(d, N, "ghz")
    ctrl_axes, ctrl_labels = build_single_site_axes(d, N, "xy")
    H0_base = build_drift_base(d, N)
    H0_base_dense = qobj_to_dense_matrix(H0_base)
    ctrl_stack_dense = build_dense_operator_stack(ctrl_axes)
    psi0_vec = qobj_to_dense_vector(psi0)
    target_vec = qobj_to_dense_vector(target)
    opt = ImprovedPolynomialPontryagin(d, N, Nt_smoke, T_smoke, p_smoke, drift_strength=0.5, seed=2, verbose=False)
    F_eff = float(opt.optimize_improved(psi0, target, max_iter=40, reporter=None, progress_every=1000, trace_rows=[]))
    u_seed = compile_ea_to_controls(
        opt=opt,
        ctrl_labels=ctrl_labels,
        T=T_smoke,
        Nt=Nt_smoke,
        S=12,
        amp_bound=2.0,
        hard_amp=2.0,
        linear_split=True,
        verbose=False,
        compiler_eps=0.02,
        budget_frac=0.8,
        max_terms_per_slice=12,
    )
    stages = build_stages([1.0], 12, 12, 20, 0, 0.05, 0.05)
    opt_cfg = OptConfig(lr=0.05, l2=0.0, amp=2.0, clip=0.0, backtracks=0, accept_mode="hard", accept_drop=0.0, threshold=2.0, verbose=0)
    rr, u_final = run_method_stages(
        name="SELFTEST",
        psi0=psi0,
        target=target,
        H0_base=H0_base,
        drift_strength_hw=0.5,
        ctrl_axes=ctrl_axes,
        T=T_smoke,
        Nt=Nt_smoke,
        stages=stages,
        homotopy_mode="mult",
        u0=u_seed,
        opt_cfg_template=opt_cfg,
        threshold=2.0,
        seed=2,
        jitter_list=[0.0],
        verbose=0,
        reporter=None,
        progress_every=1000,
        trace_dir=None,
        trace_prefix="",
        checkpoint_manager=None,
        H0_base_dense=H0_base_dense,
        ctrl_stack_dense=ctrl_stack_dense,
        psi0_vec=psi0_vec,
        target_vec=target_vec,
    )
    ok_smoke = np.isfinite(F_eff) and np.isfinite(rr.finalF_phys) and 0.0 <= rr.finalF_phys <= 1.000001 and u_final.shape == (Nt_smoke * 12, len(ctrl_axes))
    record("method_smoke", ok_smoke, f"F_eff={F_eff:.6f} finalF={rr.finalF_phys:.6f} shape={u_final.shape}")

    if failures:
        reporter.info(f"[SELF-TEST] failed: {', '.join(failures)}")
        return 1
    reporter.info("[SELF-TEST] all checks passed")
    return 0


def main() -> None:
    main_wall_start = time.time()
    args = parse_args()
    reporter = ConsoleReporter(mode=str(args.progress))

    if args.list_presets:
        print_preset_list(reporter)
        return

    if args.self_test:
        raise SystemExit(run_self_tests(reporter))

    requested_compiler_mode = str(args.compiler_mode)
    auto_target_continuation_applied = False
    if (
        int(getattr(args, "n2_auto_target_continuation", 1))
        and int(args.N) == 2
        and requested_compiler_mode == "ea_trajectory_hardware_project"
    ):
        args.compiler_mode = "ea_target_continuation_project"
        auto_target_continuation_applied = True

    d = int(args.d)
    N = int(args.N)
    T = float(args.T)
    Nt = int(args.Nt)
    D = d**N
    scales = [float(x) for x in str(args.homotopy).split(",") if x.strip()]
    if not scales:
        scales = [1.0]
    jitter_list = [float(x) for x in str(args.robust_jitter).split(",") if x.strip()]
    if not jitter_list:
        jitter_list = [0.0]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d-%H%M%S")
    tag = ("_" + str(args.tag)) if str(args.tag) else ""
    run_slug = f"{str(args.tag)}_{ts}" if str(args.tag) else ts

    trace_dir = None
    if bool(int(args.save_traces)):
        trace_dir = Path(args.trace_dir) if str(args.trace_dir).strip() else default_output_path(outdir, "traces", run_slug, "")
        trace_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = None
    if bool(int(args.save_metadata)):
        metadata_path = Path(args.metadata_path) if str(args.metadata_path).strip() else default_output_path(outdir, "run_metadata", run_slug, ".json")

    compiler_diag_path = None
    if bool(int(args.save_compiler_diagnostics)):
        compiler_diag_path = (
            Path(args.compiler_diagnostics_path)
            if str(args.compiler_diagnostics_path).strip()
            else default_output_path(outdir, "compiler_diagnostics", run_slug, ".csv")
        )

    compiler_term_diag_path = None
    compiler_audit_summary_path = None
    constructive_project_mode = str(args.compiler_mode) == "constructive_project"
    constructive_batch2_mode = str(args.compiler_mode) == "constructive_batch2"
    constructive_compress_mode = str(args.compiler_mode) == "constructive_batch2_compress"
    constructive_batch2_family_mode = constructive_batch2_mode or constructive_compress_mode
    constructive_mode = constructive_project_mode or constructive_batch2_family_mode
    ea_projectability_S = (
        int(args.constructive_reference_S)
        if constructive_compress_mode and int(args.constructive_reference_S) > 0
        else int(args.S1)
    )
    is_product_family = str(args.compiler_mode).startswith("product") or constructive_mode
    trajectory_projection_mode = str(args.compiler_mode) == "ea_trajectory_hardware_project"
    target_continuation_mode = str(args.compiler_mode) == "ea_target_continuation_project"
    if is_product_family and bool(int(args.save_compiler_diagnostics)):
        compiler_term_diag_path = (
            Path(args.compiler_term_diagnostics_path)
            if str(args.compiler_term_diagnostics_path).strip()
            else default_output_path(outdir, "compiler_terms", run_slug, ".csv")
        )
        compiler_audit_summary_path = (
            Path(args.compiler_audit_summary_path)
            if str(args.compiler_audit_summary_path).strip()
            else default_output_path(outdir, "compiler_audit", run_slug, ".json")
        )

    constructive_audit_path = None
    constructive_audit_rows_path = None
    pauli_audit_path = None
    pauli_audit_rows_path = None
    if constructive_mode:
        constructive_audit_path = (
            Path(args.constructive_audit_path)
            if str(args.constructive_audit_path).strip()
            else default_output_path(outdir, "constructive_audit", run_slug, ".json")
        )
        constructive_audit_rows_path = (
            Path(args.constructive_audit_rows_path)
            if str(args.constructive_audit_rows_path).strip()
            else default_output_path(outdir, "constructive_audit_rows", run_slug, ".csv")
        )
    if constructive_batch2_family_mode:
        pauli_audit_path = (
            Path(args.pauli_audit_path)
            if str(args.pauli_audit_path).strip()
            else default_output_path(outdir, "pauli_expansion_audit", run_slug, ".json")
        )
        pauli_audit_rows_path = (
            Path(args.pauli_audit_rows_path)
            if str(args.pauli_audit_rows_path).strip()
            else default_output_path(outdir, "pauli_expansion_audit_rows", run_slug, ".csv")
        )

    checkpoint_root = Path(args.checkpoint_dir) / run_slug if str(args.checkpoint_dir).strip() else None
    main_checkpoint = CheckpointManager(str(checkpoint_root), "run") if checkpoint_root is not None else None

    ctrl_axes, ctrl_labels = build_single_site_axes(d, N, str(args.drives))
    base_physical_control_count = len(ctrl_axes)
    effective_control_mode = str(args.compiler_mode) == "product_effective_d2"
    hardware_projection_mode = str(args.compiler_mode) == "product_d2_hardware_project"
    trajectory_projection_mode = str(args.compiler_mode) == "ea_trajectory_hardware_project"
    target_continuation_mode = str(args.compiler_mode) == "ea_target_continuation_project"
    constructive_project_mode = str(args.compiler_mode) == "constructive_project"
    constructive_batch2_mode = str(args.compiler_mode) == "constructive_batch2"
    constructive_compress_mode = str(args.compiler_mode) == "constructive_batch2_compress"
    constructive_batch2_family_mode = constructive_batch2_mode or constructive_compress_mode
    constructive_mode = constructive_project_mode or constructive_batch2_family_mode
    effective_d2_map: dict[str, int] = {}
    H0_base = build_drift_base(d, N)
    psi0, target = build_targets(d, N, str(args.target))
    H0_base_dense = qobj_to_dense_matrix(H0_base) if should_use_dense_backend(1, H0_base.shape[0]) else None
    ctrl_stack_dense = build_dense_operator_stack(ctrl_axes)
    psi0_vec = qobj_to_dense_vector(psi0)
    target_vec = qobj_to_dense_vector(target)

    method_stages = build_stages(scales, int(args.S1), int(args.S2), int(args.iters1), int(args.iters2), float(args.lr1), float(args.lr2))
    if len(method_stages) == 0:
        raise SystemExit("No METHOD stages. Increase iters1/iters2.")

    physical_scale = 1.0 if str(args.homotopy_mode) == "mult" else float(args.drift_strength_hw)
    if str(args.baseline_mode) == "matched":
        baseline_stages = method_stages
    else:
        total_budget = int(sum(st.iters for st in method_stages))
        baseline_stages = [Stage(scale=physical_scale, S=int(args.S2), iters=total_budget, lr=float(args.lr2))]

    feasibility = compiler_feasibility_info(
        T=T,
        Nt=Nt,
        S=int(args.S1),
        compiler_eps=float(args.compiler_eps),
        hard_amp=float(args.hard_amp),
        budget_frac=float(args.compiler_budget_frac),
        max_degree=max(1, min(int(args.p), int(args.N))),
    )
    infeasible = [
        (deg, info["word_steps"])
        for deg, info in feasibility["degree_info"].items()
        if deg >= 2 and deg <= min(int(args.p), int(args.N)) and not info["fits_in_budget"]
    ]

    metadata: dict[str, Any] = {
        "version": "srgrape",
        "timestamp": ts,
        "cwd": os.getcwd(),
        "command": command_as_shell([sys.argv[0], *sys.argv[1:]]),
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "qutip_version": getattr(qt, "__version__", "unknown"),
        "preset": str(args.preset) if str(args.preset) else None,
        "args": metadata_friendly(vars(args)),
        "compiler_mode_requested": str(requested_compiler_mode),
        "compiler_mode_effective": str(args.compiler_mode),
        "n2_auto_target_continuation_applied": bool(auto_target_continuation_applied),
        "system": {
            "d": d,
            "N": N,
            "D": D,
            "Nt": Nt,
            "T": T,
            "num_physical_controls": base_physical_control_count,
            "num_base_physical_controls": base_physical_control_count,
            "num_effective_d2_controls": 0,
            "num_total_controls_for_optimization": len(ctrl_axes),
            "effective_control_mode": bool(effective_control_mode),
            "hardware_projection_mode": bool(hardware_projection_mode),
            "trajectory_projection_mode": bool(trajectory_projection_mode),
            "target_continuation_mode": bool(target_continuation_mode),
            "constructive_projection_mode": bool(constructive_mode),
            "constructive_batch2_mode": bool(constructive_batch2_mode),
            "constructive_batch2_compress_mode": bool(constructive_compress_mode),
            "compressed_grape_S1": int(args.S1),
            "ea_projectability_S": int(ea_projectability_S),
            "constructive_reference_S": int(args.constructive_reference_S),
            "effective_control_mode_note": (
                "product_effective_d2 uses direct degree-2 product axes as an effective Hamiltonian model; "
                "these controls are not hardware pulse channels."
                if effective_control_mode
                else ""
            ),
            "hardware_projection_mode_note": (
                "product_d2_hardware_project uses an effective degree-2 reference only to project a seed "
                "back into physical local-control channels"
                if hardware_projection_mode
                else ""
            ),
            "trajectory_projection_mode_note": (
                "ea_trajectory_hardware_project uses EA state checkpoints only as a reference and emits "
                "physical local-control channels"
                if trajectory_projection_mode
                else ""
            ),
            "target_continuation_mode_note": (
                "ea_target_continuation_project prioritizes physical target fidelity first, then uses "
                "late EA checkpoints only as continuation regularization; emitted controls are physical local channels"
                if target_continuation_mode
                else ""
            ),
            "constructive_projection_mode_note": (
                f"{args.compiler_mode} emits constructive physical control strings and audits EA-vs-physical "
                "slice/trajectory agreement before any GRAPE refinement"
                if constructive_mode
                else ""
            ),
            "constructive_batch2_compress_mode_note": (
                "constructive_batch2_compress audits a high-resolution constructive physical reference, "
                "then compresses that reference onto the requested physical GRAPE grid"
                if constructive_compress_mode
                else ""
            ),
            "dense_backend_enabled": bool(ctrl_stack_dense is not None and H0_base_dense is not None),
        },
        "homotopy_scales": scales,
        "method_stages": stage_dicts(method_stages),
        "baseline_stages": stage_dicts(baseline_stages),
        "compiler_feasibility": metadata_friendly(feasibility),
        "output_paths": metadata_friendly(
            {
                "outdir": outdir,
                "trace_dir": trace_dir,
                "metadata_path": metadata_path,
                "compiler_diagnostics_path": compiler_diag_path,
                "compiler_term_diagnostics_path": compiler_term_diag_path,
                "compiler_audit_summary_path": compiler_audit_summary_path,
                "constructive_audit_path": constructive_audit_path,
                "constructive_audit_rows_path": constructive_audit_rows_path,
                "pauli_audit_path": pauli_audit_path,
                "pauli_audit_rows_path": pauli_audit_rows_path,
                "checkpoint_root": checkpoint_root,
            }
        ),
    }
    maybe_write_metadata(metadata_path, metadata)

    reporter.info(f"[SYS] d={d}, N={N}, D=d^N={D}, Nt={Nt}, T={T}")
    reporter.info(f"[DRIFT] drift_strength_ea={args.drift_strength_ea} | drift_strength_hw={args.drift_strength_hw}")
    reporter.info(f"[HOM] scales={scales} mode={args.homotopy_mode}")
    reporter.info(f"[CTRL] drives={args.drives} | amp={args.amp} | l2={args.l2}")
    if auto_target_continuation_applied:
        reporter.info(
            "[AUTO] N=2 high-fidelity projection: rerouted compiler-mode "
            f"{requested_compiler_mode} -> {args.compiler_mode}. "
            "Use --n2-auto-target-continuation 0 for strict trajectory diagnostics."
        )
    if float(args.ea_control_l2) > 0 or float(args.ea_smooth_l2) > 0:
        reporter.info(f"[EA-SHAPE] control_l2={float(args.ea_control_l2):.6g} smooth_l2={float(args.ea_smooth_l2):.6g}")
    if str(args.preset):
        reporter.info(f"[PRESET] {args.preset}")
    reporter.info(
        f"[COMPILER] slice_budget={feasibility['slice_budget']} steps_per_primitive={feasibility['steps_per_primitive']} "
        f"dt_micro={feasibility['dt_micro']:.6g}"
    )
    for degree, info in feasibility["degree_info"].items():
        if degree > int(args.p):
            continue
        reporter.info(
            f"[COMPILER] degree={degree} pulses={info['word_pulses']} word_steps={info['word_steps']} "
            f"fits_budget={int(info['fits_in_budget'])}"
        )
    for degree, word_steps in infeasible:
        reporter.info(
            f"[WARN] degree-{degree} compiler words require {word_steps} microsteps but the per-slice budget is "
            f"{feasibility['slice_budget']}. Those words will be skipped."
        )
    if effective_control_mode:
        reporter.info(
            "[WARN] product_effective_d2 is diagnostic: it augments the model with direct degree-2 "
            "Pauli-product controls. Results from this mode are effective-Hamiltonian results, not "
            "hardware-pulse results."
        )
    if hardware_projection_mode:
        reporter.info(
            "[D2PROJECT] product_d2_hardware_project is hardware-executable: the effective degree-2 "
            "model is used only as a reference, and the emitted seed uses physical local controls only."
        )
    if trajectory_projection_mode:
        reporter.info(
            "[EATRAJ] ea_trajectory_hardware_project is hardware-executable: EA states are reference "
            "checkpoints only, and the emitted seed uses physical local controls only."
        )
    if target_continuation_mode:
        reporter.info(
            "[TARGETCONT] ea_target_continuation_project is hardware-executable: target fidelity is "
            "optimized first, and EA checkpoints are used only as continuation regularization."
        )
    if constructive_mode:
        if constructive_compress_mode:
            reporter.info(
                "[BATCH2COMPRESS] certified-reference compression mode: a high-resolution "
                "constructive_batch2 reference is audited, then compressed onto the requested "
                "physical GRAPE grid."
            )
        elif constructive_batch2_mode:
            reporter.info(
                "[BATCH2] whole-slice average-Hamiltonian projection mode: degree-2 EA couplings "
                "are synthesized in batch, then audited before GRAPE."
            )
        else:
            reporter.info(
                "[CONSTRUCTIVE] strict product-formula projection mode: EA generators are compiled to "
                "physical strings, then audited before GRAPE."
            )

    if int(args.state_grow):
        run_state_grow_mode(
            args=args,
            reporter=reporter,
            metadata_path=metadata_path,
            ts=ts,
            tag=tag,
            run_slug=run_slug,
            main_wall_start=main_wall_start,
        )
        return

    ea_iters = resolved_ea_iters(args)
    if args.dry_run:
        reporter.info("[DRY-RUN] No optimization will be executed.")
        reporter.info(f"[DRY-RUN] resolved_ea_iters={ea_iters}")
        if int(args.p) >= 3 and int(args.ea_nested_p2):
            nested_iters = int(args.ea_nested_p2_iters) if int(args.ea_nested_p2_iters) > 0 else ea_iters
            nested_p3_iters = int(args.ea_nested_p3_iters) if int(args.ea_nested_p3_iters) > 0 else ea_iters
            chain = "p=2 -> p=3" if int(args.p) == 3 else "p=2 -> p=3 -> p=4"
            reporter.info(
                f"[DRY-RUN] nested EA continuation enabled: chain={chain} p2_iters={nested_iters} "
                f"p3_iters={nested_p3_iters if int(args.p) >= 4 else 0} "
                f"guard={int(args.ea_nested_guard)}"
            )
        if int(args.p) >= 3 and int(args.ea_degree3_projectable):
            reporter.info(
                f"[DRY-RUN] p>=3 projector={args.ea_degree3_projector} "
                f"theta_radius={args.ea_degree3_theta_radius} "
                f"step_budget_frac={args.ea_degree3_step_budget_frac} "
                f"max_terms_per_slice={args.ea_degree3_max_terms_per_slice} "
                f"warmup={args.ea_degree3_projector_warmup} "
                f"ramp={args.ea_degree3_projector_ramp}"
            )
        if int(args.p) == 4 and int(args.ea_degree4_projectable):
            reporter.info(
                f"[DRY-RUN] p=4 cost projector enabled: "
                f"step_budget_frac={args.ea_degree4_step_budget_frac} "
                f"threshold={args.ea_degree4_cost_term_threshold} "
                f"max_terms_per_slice={args.ea_degree4_max_terms_per_slice} "
                f"warmup={args.ea_degree4_projector_warmup} "
                f"ramp={args.ea_degree4_projector_ramp}"
            )
        if int(args.ea_shared_scheduler):
            reporter.info(
                f"[DRY-RUN] shared scheduler enabled: "
                f"total_budget_frac={args.ea_shared_total_budget_frac} "
                f"d4_budget_frac={args.ea_shared_d4_budget_frac} "
                f"d4_min_terms={args.ea_shared_d4_min_terms} "
                f"trim_degree2={args.ea_shared_trim_degree2}"
            )
        if int(args.constructive_repair_enable):
            reporter.info(
                f"[DRY-RUN] constructive repair enabled: attempts={args.constructive_repair_attempts} "
                f"total_scale={args.constructive_repair_total_scale} "
                f"d4_scale={args.constructive_repair_d4_scale} "
                f"high_degree_only={args.constructive_repair_high_degree_only}"
            )
        reporter.info(f"[DRY-RUN] method stages={stage_dicts(method_stages)}")
        reporter.info(f"[DRY-RUN] baseline stages={stage_dicts(baseline_stages)}")
        reporter.info(
            f"[DRY-RUN] estimated work units: EA~{ea_iters * Nt} "
            f"METHOD~{sum(st.iters * Nt * st.S for st in method_stages)} "
            f"BASELINE~{sum(st.iters * Nt * st.S for st in baseline_stages) if int(args.compare) else 0}"
        )
        return

    if abs(scales[-1] - physical_scale) > 1e-12:
        reporter.info(
            f"[WARN] last homotopy scale={scales[-1]} does not match the physical scale={physical_scale}. "
            "Final evaluation is always performed on the physical drift."
        )

    reporter.info("")
    reporter.info(f"[EA] Running EA synthesis (p={args.p})")

    ea_trace: list[dict[str, Any]] = []
    ea_nested_trace: list[dict[str, Any]] = []
    ea_nested_summary: dict[str, Any] | None = None
    ea_nested_embedding_rows: list[dict[str, Any]] = []
    ea_nested_p3_trace: list[dict[str, Any]] = []
    ea_nested_p2_to_p3_rows: list[dict[str, Any]] = []
    ea_nested_p3_to_p4_rows: list[dict[str, Any]] = []
    opt_p2: ImprovedPolynomialPontryagin | None = None
    opt_p3_nested: ImprovedPolynomialPontryagin | None = None
    nested_source_opt: ImprovedPolynomialPontryagin | None = None
    nested_source_p = 0
    nested_source_fidelity: float | None = None
    F_p2: float | None = None
    F_p3_nested: float | None = None
    embedded_p2_in_p3_fidelity: float | None = None
    p3_guard_reverted = False
    p2_projectability_summary: dict[str, Any] | None = None
    p2_native_xy_projectability_summary: dict[str, Any] | None = None
    p3_projectability_summary: dict[str, Any] | None = None
    p3_native_xy_projectability_summary: dict[str, Any] | None = None
    p3_degree3_projectability_summary: dict[str, Any] | None = None

    if int(args.p) >= 3 and int(args.ea_nested_p2):
        nested_iters = int(args.ea_nested_p2_iters) if int(args.ea_nested_p2_iters) > 0 else ea_iters
        nested_p3_iters = int(args.ea_nested_p3_iters) if int(args.ea_nested_p3_iters) > 0 else ea_iters
        reporter.info(
            f"[EA-NEST] Solving projectable p=2 fallback first "
            f"(iters={nested_iters}, guard={int(args.ea_nested_guard)})"
        )
        opt_p2 = ImprovedPolynomialPontryagin(
            d,
            N,
            Nt,
            T,
            2,
            drift_strength=float(args.drift_strength_ea),
            seed=int(args.seed),
            verbose=bool(N >= 5),
        )
        apply_ea_path_shape_args(opt_p2, args)
        if int(args.ea_batch2_projectable):
            p2_projectability_summary = configure_ea_batch2_projectability(
                opt_p2,
                drift_strength_hw=float(args.drift_strength_hw),
                budget_frac=float(args.ea_batch2_budget_frac),
                pauli_tol=float(args.pauli_audit_tol),
            )
            reporter.info(
                "[EA-NEST] p=2 projectability enabled: projector={} theta_budget={:.6g}".format(
                    str(p2_projectability_summary.get("projector", "unknown")),
                    float(p2_projectability_summary.get("theta_l1_radius", p2_projectability_summary.get("nuclear_radius", 0.0))),
                )
            )
        if int(args.ea_native_xy_projectable) or int(args.ea_native_z_projectable):
            p2_native_xy_projectability_summary = configure_ea_native_xy_projectability(
                opt_p2,
                amp_bound=float(args.amp),
                amp_frac=float(args.ea_native_xy_amp_frac),
                include_z=bool(int(args.ea_native_z_projectable)),
                z_amp_frac=float(args.ea_native_z_amp_frac),
                pauli_tol=float(args.pauli_audit_tol),
            )
            reporter.info(
                "[EA-NEST] p=2 local projectability enabled: xy_bound={:.6g} z_bound={:.6g} include_z={} controls={}".format(
                    float(p2_native_xy_projectability_summary.get("xy_coefficient_bound", 0.0)),
                    float(p2_native_xy_projectability_summary.get("z_coefficient_bound", 0.0)),
                    int(bool(p2_native_xy_projectability_summary.get("include_z", False))),
                    int(p2_native_xy_projectability_summary.get("control_count", 0)),
                )
            )
        F_p2 = float(
            opt_p2.optimize_improved(
                psi0,
                target,
                max_iter=nested_iters,
                reporter=reporter,
                progress_every=int(args.progress_every_ea),
                trace_rows=ea_nested_trace,
                progress_label="EA:p2",
            )
        )
        reporter.info(f"[EA-NEST] p=2 fallback fidelity: F_p2={F_p2:.12f}")
        nested_source_opt = opt_p2
        nested_source_p = 2
        nested_source_fidelity = float(F_p2)

        if int(args.p) >= 4:
            reporter.info(
                f"[EA-NEST] Embedding p=2 into p=3 and optimizing p=3 "
                f"(iters={nested_p3_iters}, guard={int(args.ea_nested_guard)})"
            )
            opt_p3_nested = ImprovedPolynomialPontryagin(
                d,
                N,
                Nt,
                T,
                3,
                drift_strength=float(args.drift_strength_ea),
                seed=int(args.seed),
                verbose=bool(N >= 5),
            )
            apply_ea_path_shape_args(opt_p3_nested, args)
            embedding_p2_p3 = embed_lower_degree_solution(
                opt_p2,
                opt_p3_nested,
                tol=float(args.ea_nested_tol),
            )
            ea_nested_p2_to_p3_rows = list(embedding_p2_p3.pop("rows", []))
            if not bool(embedding_p2_p3.get("success", False)):
                raise SystemExit(
                    "[EA-NEST] p=2 basis is not nested in p=3; missing={} max_embedding_error={:.3e}".format(
                        int(embedding_p2_p3.get("missing_count", 0)),
                        float(embedding_p2_p3.get("max_embedding_error", 0.0)),
                    )
                )
            reporter.info(
                "[EA-NEST] embedded p=2 into p=3: matched={}/{} max_error={:.3e}".format(
                    int(embedding_p2_p3.get("matched_count", 0)),
                    int(embedding_p2_p3.get("lower_control_count", 0)),
                    float(embedding_p2_p3.get("max_embedding_error", 0.0)),
                )
            )
            if int(args.ea_batch2_projectable):
                p3_projectability_summary = configure_ea_batch2_projectability(
                    opt_p3_nested,
                    drift_strength_hw=float(args.drift_strength_hw),
                    budget_frac=float(args.ea_batch2_budget_frac),
                    pauli_tol=float(args.pauli_audit_tol),
                )
            if int(args.ea_native_xy_projectable) or int(args.ea_native_z_projectable):
                p3_native_xy_projectability_summary = configure_ea_native_xy_projectability(
                    opt_p3_nested,
                    amp_bound=float(args.amp),
                    amp_frac=float(args.ea_native_xy_amp_frac),
                    include_z=bool(int(args.ea_native_z_projectable)),
                    z_amp_frac=float(args.ea_native_z_amp_frac),
                    pauli_tol=float(args.pauli_audit_tol),
                )
                reporter.info(
                    "[EA-NEST] p=3 local projectability enabled: xy_bound={:.6g} z_bound={:.6g} include_z={} controls={}".format(
                        float(p3_native_xy_projectability_summary.get("xy_coefficient_bound", 0.0)),
                        float(p3_native_xy_projectability_summary.get("z_coefficient_bound", 0.0)),
                        int(bool(p3_native_xy_projectability_summary.get("include_z", False))),
                        int(p3_native_xy_projectability_summary.get("control_count", 0)),
                    )
                )
            if int(args.ea_degree3_projectable):
                if str(args.ea_degree3_projector) == "cost":
                    p3_degree3_projectability_summary = configure_ea_degree3_cost_projectability(
                        opt_p3_nested,
                        T=T,
                        Nt=Nt,
                        S=int(ea_projectability_S),
                        hard_amp=float(args.hard_amp),
                        drift_strength_hw=float(args.drift_strength_hw),
                        step_budget_frac=float(args.ea_degree3_step_budget_frac),
                        term_theta_threshold=float(args.ea_degree3_cost_term_threshold),
                        product_bch_reps=int(args.product_bch_reps),
                        max_terms_per_slice=int(args.ea_degree3_max_terms_per_slice),
                        warmup_calls=int(args.ea_degree3_projector_warmup),
                        ramp_calls=int(args.ea_degree3_projector_ramp),
                        pauli_tol=float(args.pauli_audit_tol),
                    )
                else:
                    p3_degree3_projectability_summary = configure_ea_degree3_projectability(
                        opt_p3_nested,
                        theta_radius=float(args.ea_degree3_theta_radius),
                        pauli_tol=float(args.pauli_audit_tol),
                    )
            embedded_p2_in_p3_fidelity = float(opt_p3_nested.compute_fidelity_and_gradient(psi0, target)[0])
            F_p3_nested = float(
                opt_p3_nested.optimize_improved(
                    psi0,
                    target,
                    max_iter=nested_p3_iters,
                    reporter=reporter,
                    progress_every=int(args.progress_every_ea),
                    trace_rows=ea_nested_p3_trace,
                    progress_label="EA:p3",
                )
            )
            if (
                int(args.ea_nested_guard)
                and F_p3_nested + float(args.ea_nested_tol) < float(embedded_p2_in_p3_fidelity)
            ):
                embed_lower_degree_solution(opt_p2, opt_p3_nested, tol=float(args.ea_nested_tol))
                F_p3_nested = float(embedded_p2_in_p3_fidelity)
                opt_p3_nested.last_stop_reason = f"{getattr(opt_p3_nested, 'last_stop_reason', 'unknown')}_guard_reverted_to_p2"
                p3_guard_reverted = True
                reporter.info(
                    "[EA-NEST] p=3 guard reverted to embedded p=2 fallback: F={:.12f}".format(F_p3_nested)
                )
            else:
                reporter.info(
                    "[EA-NEST] p=3 nested fidelity: F_p3={:.12f} improvement_over_p2={:+.6e}".format(
                        F_p3_nested,
                        F_p3_nested - float(embedded_p2_in_p3_fidelity),
                    )
                )
            nested_source_opt = opt_p3_nested
            nested_source_p = 3
            nested_source_fidelity = float(F_p3_nested)

    opt = ImprovedPolynomialPontryagin(
        d,
        N,
        Nt,
        T,
        int(args.p),
        drift_strength=float(args.drift_strength_ea),
        seed=int(args.seed),
        verbose=bool(N >= 5),
    )
    apply_ea_path_shape_args(opt, args)
    if nested_source_opt is not None:
        embedding_summary = embed_lower_degree_solution(
            nested_source_opt,
            opt,
            tol=float(args.ea_nested_tol),
        )
        ea_nested_embedding_rows = list(embedding_summary.pop("rows", []))
        if int(args.p) == 4:
            ea_nested_p3_to_p4_rows = list(ea_nested_embedding_rows)
        elif int(args.p) == 3:
            ea_nested_p2_to_p3_rows = list(ea_nested_embedding_rows)
        if not bool(embedding_summary.get("success", False)):
            raise SystemExit(
                "[EA-NEST] p={} basis is not nested in p={}; missing={} max_embedding_error={:.3e}".format(
                    int(nested_source_p),
                    int(args.p),
                    int(embedding_summary.get("missing_count", 0)),
                    float(embedding_summary.get("max_embedding_error", 0.0)),
                )
            )
        ea_nested_summary = {
            "enabled": True,
            "chain": "p2_to_p3" if int(args.p) == 3 else "p2_to_p3_to_p4",
            "p2_fidelity": float(F_p2 if F_p2 is not None else 0.0),
            "p2_iters": int(nested_iters if int(args.ea_nested_p2_iters) > 0 else ea_iters),
            "p2_stop_reason": getattr(opt_p2, "last_stop_reason", "unknown") if opt_p2 is not None else "not_run",
            "p2_projectability": p2_projectability_summary,
            "p2_native_xy_projectability": p2_native_xy_projectability_summary,
            "p3_fidelity": float(F_p3_nested if F_p3_nested is not None else 0.0),
            "embedded_p2_fidelity_in_p3_basis": float(embedded_p2_in_p3_fidelity if embedded_p2_in_p3_fidelity is not None else 0.0),
            "p3_improvement_over_embedded_p2": float(
                (F_p3_nested if F_p3_nested is not None else 0.0)
                - (embedded_p2_in_p3_fidelity if embedded_p2_in_p3_fidelity is not None else 0.0)
            ) if int(args.p) >= 4 else 0.0,
            "p3_guard_reverted": bool(p3_guard_reverted),
            "p3_iters": int((int(args.ea_nested_p3_iters) if int(args.ea_nested_p3_iters) > 0 else ea_iters) if int(args.p) >= 4 else 0),
            "p3_stop_reason": getattr(opt_p3_nested, "last_stop_reason", "not_run") if opt_p3_nested is not None else "not_run",
            "p3_batch2_projectability": p3_projectability_summary,
            "p3_native_xy_projectability": p3_native_xy_projectability_summary,
            "p3_degree3_projectability": p3_degree3_projectability_summary,
            "source_p": int(nested_source_p),
            "source_fidelity": float(nested_source_fidelity if nested_source_fidelity is not None else 0.0),
            "final_embedding": embedding_summary,
            "guard_enabled": bool(int(args.ea_nested_guard)),
            "guard_reverted": False,
        }
        reporter.info(
            "[EA-NEST] embedded p={} into p={}: matched={}/{} max_error={:.3e}".format(
                int(nested_source_p),
                int(args.p),
                int(embedding_summary.get("matched_count", 0)),
                int(embedding_summary.get("lower_control_count", 0)),
                float(embedding_summary.get("max_embedding_error", 0.0)),
            )
        )

    ea_projectability_summary: dict[str, Any] | None = None
    ea_native_xy_projectability_summary: dict[str, Any] | None = None
    ea_degree3_projectability_summary: dict[str, Any] | None = None
    ea_degree4_projectability_summary: dict[str, Any] | None = None
    ea_shared_scheduler_summary: dict[str, Any] | None = None
    if int(args.ea_batch2_projectable):
        ea_projectability_summary = configure_ea_batch2_projectability(
            opt,
            drift_strength_hw=float(args.drift_strength_hw),
            budget_frac=float(args.ea_batch2_budget_frac),
            pauli_tol=float(args.pauli_audit_tol),
        )
        reporter.info(
            "[EA-BATCH2] projectability constraint enabled: projector={} theta_budget={:.6g} "
            "d2_controls={} budget_frac={:.3g}".format(
                str(ea_projectability_summary.get("projector", "unknown")),
                float(ea_projectability_summary.get("theta_l1_radius", ea_projectability_summary.get("nuclear_radius", 0.0))),
                int(ea_projectability_summary.get("d2_control_count", 0)),
                float(ea_projectability_summary.get("budget_frac", 0.0)),
            )
        )
    if int(args.ea_native_xy_projectable) or int(args.ea_native_z_projectable):
        ea_native_xy_projectability_summary = configure_ea_native_xy_projectability(
            opt,
            amp_bound=float(args.amp),
            amp_frac=float(args.ea_native_xy_amp_frac),
            include_z=bool(int(args.ea_native_z_projectable)),
            z_amp_frac=float(args.ea_native_z_amp_frac),
            pauli_tol=float(args.pauli_audit_tol),
        )
        reporter.info(
            "[EA-LOCAL1] projectability constraint enabled: xy_bound={:.6g} z_bound={:.6g} include_z={} controls={} projected_slices={}".format(
                float(ea_native_xy_projectability_summary.get("xy_coefficient_bound", 0.0)),
                float(ea_native_xy_projectability_summary.get("z_coefficient_bound", 0.0)),
                int(bool(ea_native_xy_projectability_summary.get("include_z", False))),
                int(ea_native_xy_projectability_summary.get("control_count", 0)),
                int(ea_native_xy_projectability_summary.get("last_projected_slices", 0)),
            )
        )
    if int(args.ea_degree3_projectable):
        if str(args.ea_degree3_projector) == "cost":
            ea_degree3_projectability_summary = configure_ea_degree3_cost_projectability(
                opt,
                T=T,
                Nt=Nt,
                S=int(ea_projectability_S),
                hard_amp=float(args.hard_amp),
                drift_strength_hw=float(args.drift_strength_hw),
                step_budget_frac=float(args.ea_degree3_step_budget_frac),
                term_theta_threshold=float(args.ea_degree3_cost_term_threshold),
                product_bch_reps=int(args.product_bch_reps),
                max_terms_per_slice=int(args.ea_degree3_max_terms_per_slice),
                warmup_calls=int(args.ea_degree3_projector_warmup),
                ramp_calls=int(args.ea_degree3_projector_ramp),
                pauli_tol=float(args.pauli_audit_tol),
            )
            reporter.info(
                "[EA-D3] cost projectability enabled: projector={} step_budget={}/{} "
                "d3_controls={} threshold={:.3g} max_terms={}".format(
                    str(ea_degree3_projectability_summary.get("projector", "unknown")),
                    int(ea_degree3_projectability_summary.get("step_budget", 0)),
                    int(ea_projectability_S),
                    int(ea_degree3_projectability_summary.get("degree3_control_count", 0)),
                    float(ea_degree3_projectability_summary.get("term_theta_threshold", 0.0)),
                    int(ea_degree3_projectability_summary.get("max_terms_per_slice", 0)),
                )
            )
        else:
            ea_degree3_projectability_summary = configure_ea_degree3_projectability(
                opt,
                theta_radius=float(args.ea_degree3_theta_radius),
                pauli_tol=float(args.pauli_audit_tol),
            )
            reporter.info(
                "[EA-D3] projectability constraint enabled: projector={} theta_radius={:.6g} "
                "d3_controls={}".format(
                    str(ea_degree3_projectability_summary.get("projector", "unknown")),
                    float(ea_degree3_projectability_summary.get("theta_l1_radius", 0.0)),
                    int(ea_degree3_projectability_summary.get("degree3_control_count", 0)),
                )
            )
    if int(args.ea_degree4_projectable):
        ea_degree4_projectability_summary = configure_ea_degree4_cost_projectability(
            opt,
            T=T,
            Nt=Nt,
            S=int(ea_projectability_S),
            hard_amp=float(args.hard_amp),
            drift_strength_hw=float(args.drift_strength_hw),
            step_budget_frac=float(args.ea_degree4_step_budget_frac),
            term_theta_threshold=float(args.ea_degree4_cost_term_threshold),
            product_bch_reps=int(args.product_bch_reps),
            max_terms_per_slice=int(args.ea_degree4_max_terms_per_slice),
            warmup_calls=int(args.ea_degree4_projector_warmup),
            ramp_calls=int(args.ea_degree4_projector_ramp),
            pauli_tol=float(args.pauli_audit_tol),
        )
        reporter.info(
            "[EA-D4] cost projectability enabled: projector={} step_budget={}/{} "
            "d4_controls={} threshold={:.3g} max_terms={}".format(
                str(ea_degree4_projectability_summary.get("projector", "unknown")),
                int(ea_degree4_projectability_summary.get("step_budget", 0)),
                int(ea_projectability_S),
                int(ea_degree4_projectability_summary.get("degree4_control_count", 0)),
                float(ea_degree4_projectability_summary.get("term_theta_threshold", 0.0)),
                int(ea_degree4_projectability_summary.get("max_terms_per_slice", 0)),
            )
        )
    if int(args.ea_shared_scheduler):
        ea_shared_scheduler_summary = configure_ea_shared_product_scheduler(
            opt,
            T=T,
            Nt=Nt,
            S=int(ea_projectability_S),
            hard_amp=float(args.hard_amp),
            drift_strength_hw=float(args.drift_strength_hw),
            product_time_scale=float(args.product_time_scale),
            product_bch_reps=int(args.product_bch_reps),
            total_budget_frac=float(args.ea_shared_total_budget_frac),
            degree4_budget_frac=float(args.ea_shared_d4_budget_frac),
            degree4_min_terms_per_slice=int(args.ea_shared_d4_min_terms),
            trim_degree2=bool(int(args.ea_shared_trim_degree2)),
            degree3_threshold=float(args.ea_degree3_cost_term_threshold),
            degree4_threshold=float(args.ea_degree4_cost_term_threshold),
            pauli_tol=float(args.pauli_audit_tol),
        )
        reporter.info(
            "[EA-SCHED] shared scheduler enabled: budget={}/{} d4_cap={} d4_min_terms={} "
            "trim_d2={}".format(
                int(ea_shared_scheduler_summary.get("total_budget", 0)),
                int(ea_projectability_S),
                int(ea_shared_scheduler_summary.get("degree4_budget_cap", 0)),
                int(ea_shared_scheduler_summary.get("degree4_min_terms_per_slice", 0)),
                bool(ea_shared_scheduler_summary.get("trim_degree2", False)),
            )
        )
    fallback_u = opt.u.copy() if ea_nested_summary is not None else None
    fallback_fidelity = None
    if ea_nested_summary is not None:
        fallback_fidelity = float(opt.compute_fidelity_and_gradient(psi0, target)[0])
        ea_nested_summary["embedded_source_fidelity_in_final_basis"] = float(fallback_fidelity)
        reporter.info(
            "[EA-NEST] embedded p={} fidelity in p={} basis: F={:.12f}".format(
                int(nested_source_p),
                int(args.p),
                fallback_fidelity,
            )
        )

    F_eff = float(
        opt.optimize_improved(
            psi0,
            target,
            max_iter=ea_iters,
            reporter=reporter,
            progress_every=int(args.progress_every_ea),
            trace_rows=ea_trace,
            progress_label="EA",
        )
    )
    if (
        ea_nested_summary is not None
        and int(args.ea_nested_guard)
        and fallback_u is not None
        and fallback_fidelity is not None
        and F_eff + float(args.ea_nested_tol) < float(fallback_fidelity)
    ):
        opt.u = fallback_u.copy()
        F_eff = float(fallback_fidelity)
        ea_nested_summary["guard_reverted"] = True
        ea_nested_summary["final_guard_reverted_to_p"] = int(nested_source_p)
        ea_nested_summary["final_fidelity"] = float(F_eff)
        ea_nested_summary["final_improvement_over_source"] = 0.0
        opt.last_stop_reason = f"{getattr(opt, 'last_stop_reason', 'unknown')}_guard_reverted_to_p{nested_source_p}"
        reporter.info(
            "[EA-NEST] p={} guard reverted to embedded p={} fallback: F={:.12f}".format(
                int(args.p),
                int(nested_source_p),
                F_eff,
            )
        )
    elif ea_nested_summary is not None:
        final_improvement = float(F_eff - float(fallback_fidelity if fallback_fidelity is not None else 0.0))
        ea_nested_summary["final_fidelity"] = float(F_eff)
        ea_nested_summary["final_improvement_over_source"] = float(final_improvement)
        if int(args.p) == 3:
            ea_nested_summary["p3_improvement_over_embedded_p2"] = float(final_improvement)
        elif int(args.p) == 4:
            ea_nested_summary["p4_improvement_over_embedded_p3"] = float(final_improvement)
    reporter.info(f"[EA] effective fidelity (EA space): F_eff={F_eff:.12f}")
    if ea_projectability_summary is not None:
        reporter.info(
            "[EA-BATCH2] final projector max_nuclear_before={:.6g} max_nuclear_after={:.6g} projected_slices={}".format(
                float(ea_projectability_summary.get("last_max_nuclear_before", 0.0)),
                float(ea_projectability_summary.get("last_max_nuclear_after", 0.0)),
                int(ea_projectability_summary.get("last_projected_slices", 0)),
            )
        )
    if ea_degree3_projectability_summary is not None:
        if str(ea_degree3_projectability_summary.get("projector", "")) == "degree3_constructive_cost_multiterm":
            reporter.info(
                "[EA-D3] final cost projector requested_steps<={} emitted_steps<={} "
                "kept_terms={}..{} target_terms<={} packable_terms<={} projected_slices={}".format(
                    int(ea_degree3_projectability_summary.get("last_max_requested_steps", 0)),
                    int(ea_degree3_projectability_summary.get("last_max_emitted_steps", 0)),
                    int(ea_degree3_projectability_summary.get("last_min_kept_terms", 0)),
                    int(ea_degree3_projectability_summary.get("last_max_kept_terms", 0)),
                    int(ea_degree3_projectability_summary.get("last_max_target_terms", 0)),
                    int(ea_degree3_projectability_summary.get("last_max_packable_terms", 0)),
                    int(ea_degree3_projectability_summary.get("last_projected_slices", 0)),
                )
            )
        else:
            reporter.info(
                "[EA-D3] final projector max_l1_before={:.6g} max_l1_after={:.6g} projected_slices={}".format(
                    float(ea_degree3_projectability_summary.get("last_max_l1_before", 0.0)),
                    float(ea_degree3_projectability_summary.get("last_max_l1_after", 0.0)),
                    int(ea_degree3_projectability_summary.get("last_projected_slices", 0)),
                )
            )
    if ea_degree4_projectability_summary is not None:
        reporter.info(
            "[EA-D4] final cost projector requested_steps<={} emitted_steps<={} "
            "kept_terms={}..{} target_terms<={} packable_terms<={} projected_slices={}".format(
                int(ea_degree4_projectability_summary.get("last_max_requested_steps", 0)),
                int(ea_degree4_projectability_summary.get("last_max_emitted_steps", 0)),
                int(ea_degree4_projectability_summary.get("last_min_kept_terms", 0)),
                int(ea_degree4_projectability_summary.get("last_max_kept_terms", 0)),
                int(ea_degree4_projectability_summary.get("last_max_target_terms", 0)),
                int(ea_degree4_projectability_summary.get("last_max_packable_terms", 0)),
                int(ea_degree4_projectability_summary.get("last_projected_slices", 0)),
            )
        )
    if ea_shared_scheduler_summary is not None:
        reporter.info(
            "[EA-SCHED] final shared schedule used<={}/{} lower_steps<={} "
            "d3_steps<={} d4_steps<={} d4_terms={}..{} final_remaining_min={}".format(
                int(ea_shared_scheduler_summary.get("last_max_estimated_used_steps", 0)),
                int(ea_shared_scheduler_summary.get("total_budget", 0)),
                int(ea_shared_scheduler_summary.get("last_max_fixed_lower_steps", 0)),
                int(ea_shared_scheduler_summary.get("last_max_d3_steps", 0)),
                int(ea_shared_scheduler_summary.get("last_max_d4_steps", 0)),
                int(ea_shared_scheduler_summary.get("last_min_d4_kept_terms", 0)),
                int(ea_shared_scheduler_summary.get("last_max_d4_kept_terms", 0)),
                int(ea_shared_scheduler_summary.get("last_min_remaining_final", 0)),
            )
        )
    if trace_dir is not None:
        write_rows_csv(trace_dir / "ea_trace.csv", ea_trace)
        if ea_nested_trace:
            write_rows_csv(trace_dir / "ea_p2_nested_trace.csv", ea_nested_trace)
        if ea_nested_p3_trace:
            write_rows_csv(trace_dir / "ea_p3_nested_trace.csv", ea_nested_p3_trace)
        if ea_nested_p2_to_p3_rows:
            write_rows_csv(trace_dir / "ea_p2_to_p3_embedding.csv", ea_nested_p2_to_p3_rows)
        if ea_nested_p3_to_p4_rows:
            write_rows_csv(trace_dir / "ea_p3_to_p4_embedding.csv", ea_nested_p3_to_p4_rows)
        if ea_nested_embedding_rows and not (ea_nested_p2_to_p3_rows or ea_nested_p3_to_p4_rows):
            write_rows_csv(trace_dir / "ea_nested_embedding.csv", ea_nested_embedding_rows)
    if main_checkpoint is not None:
        main_checkpoint.save_npz("ea_solution", controls=opt.u, best_fidelity=np.array([F_eff], dtype=float))
        main_checkpoint.save_json(
            "ea_summary",
            {
                "best_fidelity": float(F_eff),
                "stop_reason": getattr(opt, "last_stop_reason", "unknown"),
                "trace_rows": len(ea_trace),
                "nested_p2": ea_nested_summary,
                "batch2_projectability": ea_projectability_summary,
                "native_xy_projectability": ea_native_xy_projectability_summary,
                "degree3_projectability": ea_degree3_projectability_summary,
                "degree4_projectability": ea_degree4_projectability_summary,
                "shared_scheduler": ea_shared_scheduler_summary,
            },
        )

    metadata["ea"] = {
        "resolved_iters": int(ea_iters),
        "effective_fidelity": float(F_eff),
        "trace_rows": len(ea_trace),
        "nested_p2": ea_nested_summary,
        "stop_reason": getattr(opt, "last_stop_reason", "unknown"),
        "batch2_projectability": ea_projectability_summary,
        "native_xy_projectability": ea_native_xy_projectability_summary,
        "degree3_projectability": ea_degree3_projectability_summary,
        "degree4_projectability": ea_degree4_projectability_summary,
        "shared_scheduler": ea_shared_scheduler_summary,
        "ea_only": bool(int(args.ea_only)),
    }
    maybe_write_metadata(metadata_path, metadata)

    if int(args.ea_only):
        reporter.info("[EA] ea-only requested; exiting before compilation.")
        reporter.info(f"[TIME] total wall={format_seconds(time.time() - main_wall_start)}")
        return

    reporter.info("")
    reporter.info("[SEED] Compiling EA path to physical controls")
    compiler_diagnostics: list[dict[str, Any]] = []
    compiler_term_diagnostics: list[dict[str, Any]] = []
    hardware_projection_summary: dict[str, Any] | None = None
    trajectory_projection_summary: dict[str, Any] | None = None
    trajectory_projection_failed_reasons: list[str] = []
    target_continuation_summary: dict[str, Any] | None = None
    constructive_compress_summary: dict[str, Any] | None = None
    batch2_summary: dict[str, Any] | None = None
    constructive_reference_seed_for_audit: np.ndarray | None = None
    constructive_reference_S_for_audit: int | None = None
    pauli_audit_rows: list[dict[str, Any]] = []
    batch2_pauli_cache: Optional[tuple[Any, list[dict[str, Any]], dict[str, Any]]] = None
    if target_continuation_mode:
        u_seed, target_continuation_summary = compile_ea_to_target_continuation_project(
            opt=opt,
            ctrl_axes=ctrl_axes,
            ctrl_labels=ctrl_labels,
            H0_base=H0_base,
            psi0=psi0,
            target=target,
            T=T,
            Nt=Nt,
            S=int(args.S1),
            drift_strength_hw=float(args.drift_strength_hw),
            amp_bound=float(args.amp),
            hard_amp=float(args.hard_amp),
            trajectory_reference=str(args.trajectory_reference),
            continuation_iters=str(args.target_continuation_iters),
            continuation_lrs=str(args.target_continuation_lrs),
            continuation_weights=str(args.target_continuation_weights),
            continuation_checkpoints=str(args.target_continuation_checkpoints),
            continuation_min_improve=float(args.target_continuation_min_improve),
            continuation_init=str(args.target_continuation_init),
            continuation_policy=str(args.target_continuation_policy),
            continuation_escape_weights=str(args.target_continuation_escape_weights),
            continuation_escape_checkpoints=str(args.target_continuation_escape_checkpoints),
            continuation_escape_iters=str(args.target_continuation_escape_iters),
            continuation_target_stall_tol=args.target_continuation_target_stall_tol,
            continuation_max_stages=int(args.target_continuation_max_stages),
            compiler_eps=float(args.compiler_eps),
            budget_frac=float(args.compiler_budget_frac),
            max_terms_per_slice=int(args.compiler_max_terms),
            compiler_sort_mode=str(args.compiler_sort_mode),
            product_time_scale=float(args.product_time_scale),
            projection_clip=float(args.clip),
            projection_backtracks=int(args.backtracks),
            projection_accept_mode=str(args.accept_mode),
            projection_accept_drop=float(args.accept_drop),
            projection_threshold=float(args.projection_threshold),
            diagnostics=compiler_diagnostics,
            term_diagnostics=compiler_term_diagnostics,
            reporter=reporter,
            progress_every=int(args.progress_every_grape),
            H0_base_dense=H0_base_dense,
            ctrl_stack_dense=ctrl_stack_dense,
            psi0_vec=psi0_vec,
            target_vec=target_vec,
        )
    elif trajectory_projection_mode:
        u_seed, trajectory_projection_summary = compile_ea_to_trajectory_hardware_project(
            opt=opt,
            ctrl_axes=ctrl_axes,
            ctrl_labels=ctrl_labels,
            H0_base=H0_base,
            psi0=psi0,
            target=target,
            T=T,
            Nt=Nt,
            S=int(args.S1),
            drift_strength_hw=float(args.drift_strength_hw),
            amp_bound=float(args.amp),
            hard_amp=float(args.hard_amp),
            trajectory_reference=str(args.trajectory_reference),
            trajectory_weight=float(args.trajectory_weight),
            trajectory_checkpoints=str(args.trajectory_checkpoints),
            trajectory_project_iters=int(args.trajectory_project_iters),
            trajectory_project_lr=float(args.trajectory_project_lr),
            trajectory_stage_weights=str(args.trajectory_stage_weights),
            trajectory_stage_checkpoints=str(args.trajectory_stage_checkpoints),
            trajectory_stage_iters=str(args.trajectory_stage_iters),
            trajectory_stage_lrs=str(args.trajectory_stage_lrs),
            trajectory_stage_select=str(args.trajectory_stage_select),
            trajectory_select_min_target_fidelity=float(args.eatraj_min_target_fidelity),
            trajectory_select_min_mean_fidelity=float(args.eatraj_min_trajectory_fidelity),
            trajectory_select_min_worst_fidelity=float(args.eatraj_min_trajectory_min_fidelity),
            trajectory_auto_polish=int(args.trajectory_auto_polish),
            trajectory_polish_objective=str(args.trajectory_polish_objective),
            trajectory_polish_softmin_tau=float(args.trajectory_polish_softmin_tau),
            trajectory_polish_stage_weights=str(args.trajectory_polish_stage_weights),
            trajectory_polish_stage_checkpoints=str(args.trajectory_polish_stage_checkpoints),
            trajectory_polish_stage_iters=str(args.trajectory_polish_stage_iters),
            trajectory_polish_stage_lrs=str(args.trajectory_polish_stage_lrs),
            trajectory_init=str(args.trajectory_init),
            compiler_eps=float(args.compiler_eps),
            budget_frac=float(args.compiler_budget_frac),
            max_terms_per_slice=int(args.compiler_max_terms),
            compiler_sort_mode=str(args.compiler_sort_mode),
            product_time_scale=float(args.product_time_scale),
            projection_clip=float(args.clip),
            projection_backtracks=int(args.backtracks),
            projection_accept_mode=str(args.accept_mode),
            projection_accept_drop=float(args.accept_drop),
            projection_threshold=float(args.projection_threshold),
            trajectory_optimizer=str(args.trajectory_optimizer),
            trajectory_lbfgs_maxls=int(args.trajectory_lbfgs_maxls),
            trajectory_lbfgs_ftol=float(args.trajectory_lbfgs_ftol),
            trajectory_lbfgs_gtol=float(args.trajectory_lbfgs_gtol),
            trajectory_objective=str(args.trajectory_objective),
            trajectory_softmin_tau=float(args.trajectory_softmin_tau),
            batch2_residual_tol=float(args.batch2_residual_tol),
            batch2_max_frames=int(args.batch2_max_frames),
            batch2_frame_tol=float(args.batch2_frame_tol),
            constructive_trotter_reps=int(args.constructive_trotter_reps),
            pauli_audit_tol=float(args.pauli_audit_tol),
            diagnostics=compiler_diagnostics,
            term_diagnostics=compiler_term_diagnostics,
            reporter=reporter,
            progress_every=int(args.progress_every_grape),
            H0_base_dense=H0_base_dense,
            ctrl_stack_dense=ctrl_stack_dense,
            psi0_vec=psi0_vec,
            target_vec=target_vec,
        )
        trajectory_ok, trajectory_projection_failed_reasons = ea_trajectory_projection_passed(
            trajectory_projection_summary,
            args,
        )
        trajectory_projection_summary["passed"] = bool(trajectory_ok)
        trajectory_projection_summary["fail_reasons"] = list(trajectory_projection_failed_reasons)
    elif hardware_projection_mode:
        u_seed, hardware_projection_summary = compile_ea_to_d2_hardware_project(
            opt=opt,
            ctrl_axes=ctrl_axes,
            ctrl_labels=ctrl_labels,
            H0_base=H0_base,
            psi0=psi0,
            target=target,
            T=T,
            Nt=Nt,
            S=int(args.S1),
            drift_strength_hw=float(args.drift_strength_hw),
            amp_bound=float(args.amp),
            hard_amp=float(args.hard_amp),
            budget_frac=float(args.compiler_budget_frac),
            max_terms_per_slice=int(args.compiler_max_terms),
            compiler_sort_mode=str(args.compiler_sort_mode),
            product_time_scale=float(args.product_time_scale),
            projection_iters=int(args.projection_iters),
            projection_lr=float(args.projection_lr),
            projection_threshold=float(args.projection_threshold),
            projection_init=str(args.projection_init),
            projection_clip=float(args.clip),
            projection_backtracks=int(args.backtracks),
            projection_accept_mode=str(args.accept_mode),
            projection_accept_drop=float(args.accept_drop),
            diagnostics=compiler_diagnostics,
            term_diagnostics=compiler_term_diagnostics,
            reporter=reporter,
            progress_every=int(args.progress_every_grape),
            seed=int(args.seed),
            H0_base_dense=H0_base_dense,
            ctrl_stack_dense=ctrl_stack_dense,
            psi0_vec=psi0_vec,
            target_vec=target_vec,
        )
    elif effective_control_mode:
        old_count = len(ctrl_axes)
        ctrl_axes, ctrl_labels, effective_d2_map = build_effective_d2_axes_from_opt(opt, ctrl_axes, ctrl_labels)
        ctrl_stack_dense = build_dense_operator_stack(ctrl_axes)
        metadata["system"].update(
            {
                "num_physical_controls": base_physical_control_count,
                "num_base_physical_controls": base_physical_control_count,
                "num_effective_d2_controls": len(ctrl_axes) - base_physical_control_count,
                "num_total_controls_for_optimization": len(ctrl_axes),
                "effective_d2_labels": ctrl_labels[base_physical_control_count:],
                "dense_backend_enabled": bool(ctrl_stack_dense is not None and H0_base_dense is not None),
            }
        )
        maybe_write_metadata(metadata_path, metadata)
        reporter.info(
            f"[EFFECTIVE] added {len(ctrl_axes) - old_count} direct degree-2 controls; "
            f"optimization control count is now {len(ctrl_axes)}."
        )
        u_seed = compile_ea_to_effective_d2_controls(
            opt=opt,
            effective_labels=ctrl_labels,
            T=T,
            Nt=Nt,
            S=int(args.S1),
            amp_bound=float(args.amp),
            budget_frac=float(args.compiler_budget_frac),
            max_terms_per_slice=int(args.compiler_max_terms),
            compiler_sort_mode=str(args.compiler_sort_mode),
            product_time_scale=float(args.product_time_scale),
            diagnostics=compiler_diagnostics,
            term_diagnostics=compiler_term_diagnostics,
            reporter=reporter,
            progress_label="EffectiveD2Compiler",
        )
    elif constructive_compress_mode:
        batch2_pauli_cache = build_exact_pauli_expansion_audit(opt, tol=float(args.pauli_audit_tol))
        u_seed, constructive_compress_summary, constructive_reference_seed_for_audit, constructive_reference_S_for_audit = (
            compile_ea_to_constructive_batch2_compress_project(
                opt=opt,
                ctrl_axes=ctrl_axes,
                ctrl_labels=ctrl_labels,
                H0_base=H0_base,
                psi0=psi0,
                target=target,
                T=T,
                Nt=Nt,
                S=int(args.S1),
                reference_S=int(args.constructive_reference_S),
                drift_strength_hw=float(args.drift_strength_hw),
                amp_bound=float(args.amp),
                hard_amp=float(args.hard_amp),
                product_time_scale=float(args.product_time_scale),
                term_theta_threshold=float(args.constructive_term_threshold),
                pauli_tol=float(args.pauli_audit_tol),
                residual_tol=float(args.batch2_residual_tol),
                max_frames=int(args.batch2_max_frames),
                frame_tol=float(args.batch2_frame_tol),
                product_bch_reps=int(args.product_bch_reps),
                trotter_reps=int(args.constructive_trotter_reps),
                trajectory_weight=float(args.trajectory_weight),
                trajectory_checkpoints=str(args.trajectory_checkpoints),
                trajectory_project_iters=int(args.trajectory_project_iters),
                trajectory_project_lr=float(args.trajectory_project_lr),
                projection_clip=float(args.clip),
                projection_backtracks=int(args.backtracks),
                projection_accept_mode=str(args.accept_mode),
                projection_accept_drop=float(args.accept_drop),
                projection_threshold=float(args.projection_threshold),
                compression_target_guard=bool(int(args.compression_target_guard)),
                compression_target_guard_tol=float(args.compression_target_guard_tol),
                diagnostics=compiler_diagnostics,
                term_diagnostics=compiler_term_diagnostics,
                pauli_audit_rows=pauli_audit_rows,
                pauli_expansion_cache=batch2_pauli_cache,
                reporter=reporter,
                progress_every=int(args.progress_every_grape),
                H0_base_dense=H0_base_dense,
                ctrl_stack_dense=ctrl_stack_dense,
                psi0_vec=psi0_vec,
                target_vec=target_vec,
            )
        )
        batch2_summary = dict(constructive_compress_summary.get("batch2_reference", {}))
    elif constructive_batch2_mode:
        batch2_pauli_cache = build_exact_pauli_expansion_audit(opt, tol=float(args.pauli_audit_tol))
        u_seed, batch2_summary = compile_ea_to_controls_constructive_batch2(
            opt=opt,
            ctrl_labels=ctrl_labels,
            T=T,
            Nt=Nt,
            S=int(args.S1),
            amp_bound=float(args.amp),
            hard_amp=float(args.hard_amp),
            drift_strength_hw=float(args.drift_strength_hw),
            product_time_scale=float(args.product_time_scale),
            term_theta_threshold=float(args.constructive_term_threshold),
            pauli_tol=float(args.pauli_audit_tol),
            residual_tol=float(args.batch2_residual_tol),
            max_frames=int(args.batch2_max_frames),
            frame_tol=float(args.batch2_frame_tol),
            product_bch_reps=int(args.product_bch_reps),
            trotter_reps=int(args.constructive_trotter_reps),
            diagnostics=compiler_diagnostics,
            term_diagnostics=compiler_term_diagnostics,
            pauli_audit_rows=pauli_audit_rows,
            pauli_expansion_cache=batch2_pauli_cache,
            reporter=reporter,
            progress_label="Batch2Compiler",
        )
    elif is_product_family:
        compile_max_terms = (
            len(opt.controls)
            if constructive_mode and int(args.constructive_include_all)
            else int(args.compiler_max_terms)
        )
        u_seed = compile_ea_to_controls_product_aware(
            opt=opt,
            ctrl_labels=ctrl_labels,
            T=T,
            Nt=Nt,
            S=int(args.S1),
            amp_bound=float(args.amp),
            hard_amp=float(args.hard_amp),
            drift_strength_hw=float(args.drift_strength_hw),
            compiler_eps=float(args.compiler_eps),
            budget_frac=float(args.compiler_budget_frac),
            max_terms_per_slice=int(compile_max_terms),
            product_bch_reps=int(args.product_bch_reps),
            product_max_degree=int(args.product_max_degree),
            compiler_sort_mode=str(args.compiler_sort_mode),
            compiler_operation_mode=("product" if constructive_mode else str(args.compiler_mode)),
            product_time_scale=float(args.product_time_scale),
            term_theta_threshold=(float(args.constructive_term_threshold) if constructive_mode else 0.0),
            aggregate_terms=bool(constructive_mode),
            diagnostics=compiler_diagnostics,
            term_diagnostics=compiler_term_diagnostics,
            reporter=reporter,
            progress_label="ProductCompiler",
        )
    else:
        u_seed = compile_ea_to_controls(
            opt=opt,
            ctrl_labels=ctrl_labels,
            T=T,
            Nt=Nt,
            S=int(args.S1),
            amp_bound=float(args.amp),
            hard_amp=float(args.hard_amp),
            linear_split=True,
            verbose=True,
            compiler_eps=float(args.compiler_eps),
            budget_frac=float(args.compiler_budget_frac),
            max_terms_per_slice=int(args.compiler_max_terms),
            diagnostics=compiler_diagnostics,
            reporter=reporter,
            progress_label="Compiler",
        )

    u_seed = apply_seed_gain_dither_clip(
        u_seed,
        seed_gain=float(args.seed_gain),
        seed_dither=float(args.seed_dither),
        amp=float(args.amp),
        seed=int(args.seed),
    )

    if compiler_diag_path is not None:
        write_rows_csv(compiler_diag_path, compiler_diagnostics)
    if pauli_audit_rows_path is not None:
        write_rows_csv(pauli_audit_rows_path, pauli_audit_rows)
    if pauli_audit_path is not None:
        write_json(
            pauli_audit_path,
            {
                "mode": str(args.compiler_mode),
                "summary": (batch2_summary or {}).get("pauli_expansion", {}),
                "rows": len(pauli_audit_rows),
            },
        )
    compiler_audit_summary: dict[str, Any] | None = None
    if compiler_term_diag_path is not None:
        write_rows_csv(compiler_term_diag_path, compiler_term_diagnostics)
    constructive_audit_summary: dict[str, Any] | None = None
    constructive_audit_failed_reasons: list[str] = []
    constructive_repair_summary: dict[str, Any] = {
        "enabled": bool(int(args.constructive_repair_enable)),
        "attempts": [],
        "final_repaired": False,
    }

    def refresh_product_audits() -> bool:
        nonlocal compiler_audit_summary, constructive_audit_summary, constructive_audit_failed_reasons
        compiler_audit_summary = summarize_compiler_term_diagnostics(compiler_term_diagnostics)
        compiler_audit_summary.update(
            {
                "compiler_mode": str(args.compiler_mode),
                "constructive_projection_mode": bool(constructive_mode),
                "effective_control_mode": bool(effective_control_mode),
                "effective_control_mode_note": (
                    "degree-2 Pauli products are emitted as direct effective controls; "
                    "not a hardware pulse schedule"
                    if effective_control_mode
                    else ""
                ),
                "hardware_projection_mode": bool(hardware_projection_mode),
                "hardware_projection_summary": hardware_projection_summary,
                "batch2_summary": batch2_summary,
                "product_max_degree": int(args.product_max_degree),
                "product_bch_reps": int(args.product_bch_reps),
                "product_time_scale": float(args.product_time_scale),
                "compiler_max_terms": int(len(opt.controls) if constructive_mode and int(args.constructive_include_all) else int(args.compiler_max_terms)),
                "compiler_sort_mode": str(args.compiler_sort_mode),
                "compiler_budget_frac": float(args.compiler_budget_frac),
                "S1": int(args.S1),
                "slice_budget": int(max(1, np.floor(float(args.compiler_budget_frac) * int(args.S1)))),
                "term_diagnostics_rows": len(compiler_term_diagnostics),
                "constructive_repair": constructive_repair_summary,
            }
        )
        if batch2_summary is not None:
            compiler_audit_summary["totals"].update(
                {
                    "requested_weight": float(batch2_summary.get("requested_weight", 0.0)),
                    "emitted_weight": float(batch2_summary.get("emitted_weight", 0.0)),
                    "emitted_weight_fraction": float(batch2_summary.get("emitted_weight_fraction", 0.0)),
                }
            )
        if not constructive_mode:
            constructive_audit_summary = None
            constructive_audit_failed_reasons = []
            return True
        audit_seed = constructive_reference_seed_for_audit if constructive_reference_seed_for_audit is not None else u_seed
        audit_S = int(constructive_reference_S_for_audit) if constructive_reference_S_for_audit is not None else int(args.S1)
        constructive_audit_summary = constructive_projection_audit(
            opt=opt,
            u_seed=audit_seed,
            H0_base=H0_base,
            ctrl_axes=ctrl_axes,
            psi0=psi0,
            T=T,
            Nt=Nt,
            S=audit_S,
            drift_strength_hw=float(args.drift_strength_hw),
            compiler_audit_summary=compiler_audit_summary,
            mode=str(args.compiler_mode),
        )
        if constructive_reference_seed_for_audit is not None:
            constructive_audit_summary["audited_seed_role"] = "constructive_reference"
            constructive_audit_summary["compressed_seed_shape"] = list(np.asarray(u_seed).shape)
            constructive_audit_summary["reference_seed_shape"] = list(np.asarray(audit_seed).shape)
            constructive_audit_summary["reference_S"] = int(audit_S)
            constructive_audit_summary["compressed_S"] = int(args.S1)
            constructive_audit_summary["compression_summary"] = constructive_compress_summary
        constructive_ok, constructive_audit_failed_reasons = constructive_projection_passed(
            constructive_audit_summary,
            args,
        )
        constructive_audit_summary["passed"] = bool(constructive_ok)
        constructive_audit_summary["fail_reasons"] = list(constructive_audit_failed_reasons)
        constructive_audit_summary["repair_summary"] = constructive_repair_summary
        return bool(constructive_ok)

    if is_product_family:
        constructive_ok = refresh_product_audits()
        initial_high_degree_failures = high_degree_compiler_failures(compiler_audit_summary)
        if (
            constructive_batch2_mode
            and not constructive_ok
            and int(args.constructive_repair_enable)
            and int(args.constructive_repair_attempts) > 0
            and (
                initial_high_degree_failures
                or not int(args.constructive_repair_high_degree_only)
            )
        ):
            reporter.info(
                "[REPAIR] constructive audit failed; attempting repair. high_degree_failures={}".format(
                    "; ".join(initial_high_degree_failures) if initial_high_degree_failures else "none"
                )
            )
            for attempt in range(1, int(args.constructive_repair_attempts) + 1):
                repair_total_frac = max(
                    0.05,
                    float(args.ea_shared_total_budget_frac) * (float(args.constructive_repair_total_scale) ** attempt),
                )
                repair_d4_frac = max(
                    0.0,
                    float(args.ea_shared_d4_budget_frac) * (float(args.constructive_repair_d4_scale) ** attempt),
                )
                repair_summary = configure_ea_shared_product_scheduler(
                    opt,
                    T=T,
                    Nt=Nt,
                    S=int(ea_projectability_S),
                    hard_amp=float(args.hard_amp),
                    drift_strength_hw=float(args.drift_strength_hw),
                    product_time_scale=float(args.product_time_scale),
                    product_bch_reps=int(args.product_bch_reps),
                    total_budget_frac=repair_total_frac,
                    degree4_budget_frac=repair_d4_frac,
                    degree4_min_terms_per_slice=int(args.ea_shared_d4_min_terms),
                    trim_degree2=bool(int(args.ea_shared_trim_degree2)),
                    degree3_threshold=float(args.ea_degree3_cost_term_threshold),
                    degree4_threshold=float(args.ea_degree4_cost_term_threshold),
                    pauli_tol=float(args.pauli_audit_tol),
                )
                compiler_diagnostics = []
                compiler_term_diagnostics = []
                pauli_audit_rows = []
                u_seed, batch2_summary = compile_ea_to_controls_constructive_batch2(
                    opt=opt,
                    ctrl_labels=ctrl_labels,
                    T=T,
                    Nt=Nt,
                    S=int(args.S1),
                    amp_bound=float(args.amp),
                    hard_amp=float(args.hard_amp),
                    drift_strength_hw=float(args.drift_strength_hw),
                    product_time_scale=float(args.product_time_scale),
                    term_theta_threshold=float(args.constructive_term_threshold),
                    pauli_tol=float(args.pauli_audit_tol),
                    residual_tol=float(args.batch2_residual_tol),
                    max_frames=int(args.batch2_max_frames),
                    frame_tol=float(args.batch2_frame_tol),
                    product_bch_reps=int(args.product_bch_reps),
                    trotter_reps=int(args.constructive_trotter_reps),
                    diagnostics=compiler_diagnostics,
                    term_diagnostics=compiler_term_diagnostics,
                    pauli_audit_rows=pauli_audit_rows,
                    pauli_expansion_cache=batch2_pauli_cache,
                    reporter=reporter,
                    progress_label=f"Batch2Repair{attempt}",
                )
                u_seed = apply_seed_gain_dither_clip(
                    u_seed,
                    seed_gain=float(args.seed_gain),
                    seed_dither=float(args.seed_dither),
                    amp=float(args.amp),
                    seed=int(args.seed),
                )
                constructive_ok = refresh_product_audits()
                high_degree_failures = high_degree_compiler_failures(compiler_audit_summary)
                attempt_row = {
                    "attempt": int(attempt),
                    "total_budget_frac": float(repair_total_frac),
                    "degree4_budget_frac": float(repair_d4_frac),
                    "scheduler": repair_summary,
                    "constructive_passed": bool(constructive_ok),
                    "fail_reasons": list(constructive_audit_failed_reasons),
                    "high_degree_failures": list(high_degree_failures),
                }
                constructive_repair_summary["attempts"].append(attempt_row)
                reporter.info(
                    "[REPAIR] attempt {} pass={} high_degree_failures={}".format(
                        attempt,
                        int(bool(constructive_ok)),
                        "; ".join(high_degree_failures) if high_degree_failures else "none",
                    )
                )
                if constructive_ok:
                    constructive_repair_summary["final_repaired"] = True
                    break
        if compiler_audit_summary is not None:
            compiler_audit_summary["constructive_repair"] = constructive_repair_summary
        if constructive_audit_summary is not None:
            constructive_audit_summary["repair_summary"] = constructive_repair_summary
        if compiler_diag_path is not None:
            write_rows_csv(compiler_diag_path, compiler_diagnostics)
        if compiler_term_diag_path is not None:
            write_rows_csv(compiler_term_diag_path, compiler_term_diagnostics)
        if pauli_audit_rows_path is not None:
            write_rows_csv(pauli_audit_rows_path, pauli_audit_rows)
        if pauli_audit_path is not None:
            write_json(
                pauli_audit_path,
                {
                    "mode": str(args.compiler_mode),
                    "summary": (batch2_summary or {}).get("pauli_expansion", {}),
                    "rows": len(pauli_audit_rows),
                },
            )
        if compiler_audit_summary_path is not None:
            write_json(compiler_audit_summary_path, compiler_audit_summary)
        log_compiler_audit_summary(reporter, compiler_audit_summary)
        if constructive_mode and constructive_audit_summary is not None:
            if constructive_audit_path is not None:
                write_json(constructive_audit_path, constructive_audit_summary)
            if constructive_audit_rows_path is not None:
                write_rows_csv(constructive_audit_rows_path, constructive_audit_summary.get("rows", []))
            reporter.info(
                "[CONSTRUCTIVE] emitted_weight_fraction={:.6f} min_slice_overlap={:.6f} "
                "min_checkpoint_fidelity={:.6f} final_checkpoint_fidelity={:.6f}".format(
                    float(constructive_audit_summary.get("emitted_weight_fraction", 0.0)),
                    float(constructive_audit_summary.get("min_slice_unitary_overlap", 0.0)),
                    float(constructive_audit_summary.get("min_checkpoint_state_fidelity", 0.0)),
                    float(constructive_audit_summary.get("final_checkpoint_state_fidelity", 0.0)),
                )
            )
            if constructive_ok:
                reporter.info("[CONSTRUCTIVE] audit PASS")
            else:
                reporter.info("[CONSTRUCTIVE] audit FAIL: " + "; ".join(constructive_audit_failed_reasons))
    if hardware_projection_summary is not None:
        reporter.info(
            "[D2PROJECT] effective reference target F={:.6f}; hardware target F before/after projection={:.6f}/{:.6f}".format(
                float(hardware_projection_summary.get("effective_reference_target_fidelity", 0.0)),
                float(hardware_projection_summary.get("hardware_target_fidelity_before_projection", 0.0)),
                float(hardware_projection_summary.get("hardware_target_fidelity_after_projection", 0.0)),
            )
        )
        reporter.info(
            "[D2PROJECT] hardware reference F before/after projection={:.6f}/{:.6f}; physical_controls={}".format(
                float(hardware_projection_summary.get("hardware_reference_fidelity_before_projection", 0.0)),
                float(hardware_projection_summary.get("hardware_reference_fidelity_after_projection", 0.0)),
                int(hardware_projection_summary.get("physical_control_count", len(ctrl_labels))),
            )
        )
    if trajectory_projection_summary is not None:
        ref_summary = trajectory_projection_summary.get("reference", {})
        proj_summary = trajectory_projection_summary.get("projection", {})
        reporter.info(
            "[EATRAJ] EA reference target F={:.6f}; hardware target F before/after projection={:.6f}/{:.6f}".format(
                float(ref_summary.get("target_fidelity", 0.0)),
                float(proj_summary.get("target_fidelity_before", 0.0)),
                float(proj_summary.get("target_fidelity_after", 0.0)),
            )
        )
        reporter.info(
            "[EATRAJ] trajectory match F before/after projection={:.6f}/{:.6f}; checkpoints={}".format(
                float(proj_summary.get("trajectory_fidelity_before", 0.0)),
                float(proj_summary.get("trajectory_fidelity_after", 0.0)),
                int(proj_summary.get("checkpoint_count", 0)),
            )
        )
        if bool(trajectory_projection_summary.get("passed", True)):
            reporter.info("[EATRAJ] projection gate PASS")
        else:
            reporter.info("[EATRAJ] projection gate FAIL: " + "; ".join(trajectory_projection_failed_reasons))
    if target_continuation_summary is not None:
        ref_summary = target_continuation_summary.get("reference", {})
        reporter.info(
            "[TARGETCONT] EA reference target F={:.6f}; hardware target F initial/best={:.6f}/{:.6f}".format(
                float(ref_summary.get("target_fidelity", 0.0)),
                float(target_continuation_summary.get("target_fidelity_initial", 0.0)),
                float(target_continuation_summary.get("target_fidelity_best", 0.0)),
            )
        )
        for row in target_continuation_summary.get("stages", []):
            reporter.info(
                "[TARGETCONT] stage {stage}: target {target_before:.6f}->{target_after:.6f}; "
                "traj {trajectory_before:.6f}->{trajectory_after:.6f}; weight={used_weight:.3g}; "
                "checkpoints={checkpoints}; accepted={accepted_for_next_stage}".format(**row)
            )
    if constructive_compress_summary is not None:
        ref_summary = constructive_compress_summary.get("reference", {})
        comp_summary = constructive_compress_summary.get("compression", {})
        reporter.info(
            "[BATCH2COMPRESS] constructive reference target F={:.6f}; compressed target F before/after={:.6f}/{:.6f}".format(
                float(ref_summary.get("target_fidelity", 0.0)),
                float(comp_summary.get("target_fidelity_before", 0.0)),
                float(comp_summary.get("target_fidelity_after", 0.0)),
            )
        )
        reporter.info(
            "[BATCH2COMPRESS] trajectory match F before/after={:.6f}/{:.6f}; checkpoints={}".format(
                float(comp_summary.get("trajectory_fidelity_before", 0.0)),
                float(comp_summary.get("trajectory_fidelity_after", 0.0)),
                int(comp_summary.get("checkpoint_count", 0)),
            )
        )
    if main_checkpoint is not None:
        main_checkpoint.save_npz("compiled_seed", controls=u_seed)
        main_checkpoint.save_json(
            "compiler_summary",
            {
                "rows": len(compiler_diagnostics),
                "term_rows": len(compiler_term_diagnostics),
                "audit_summary": compiler_audit_summary,
                "batch2_summary": batch2_summary,
                "hardware_projection_summary": hardware_projection_summary,
                "trajectory_projection_summary": trajectory_projection_summary,
                "target_continuation_summary": target_continuation_summary,
                "constructive_compress_summary": constructive_compress_summary,
                "constructive_audit_summary": constructive_audit_summary,
                "seed_rms": float(rms(u_seed)),
                "seed_max_abs": float(np.max(np.abs(u_seed))) if u_seed.size else 0.0,
            },
        )

    dt_seed = T / (Nt * int(args.S1))
    seed_path = outdir / f"seed_controls{tag}.csv"
    write_awg_csv(seed_path, u_seed, dt_seed, ctrl_labels)
    reporter.info(f"[SEED] wrote {seed_path}")

    opt_cfg_template = OptConfig(
        lr=float(args.lr2),
        l2=float(args.l2),
        amp=float(args.amp),
        clip=float(args.clip),
        backtracks=int(args.backtracks),
        accept_mode=str(args.accept_mode),
        accept_drop=float(args.accept_drop),
        threshold=float(args.threshold),
        verbose=int(args.verbose),
        stall_enable=bool(int(args.stall_enable)),
        stall_gnorm=float(args.stall_gnorm),
        stall_max_kicks=int(args.stall_max_kicks),
        stall_kick_sigma=float(args.stall_kick_sigma),
    )

    metadata["ea"] = {
        "resolved_iters": int(ea_iters),
        "effective_fidelity": float(F_eff),
        "trace_rows": len(ea_trace),
        "nested_p2": ea_nested_summary,
        "stop_reason": getattr(opt, "last_stop_reason", "unknown"),
        "batch2_projectability": ea_projectability_summary,
        "native_xy_projectability": ea_native_xy_projectability_summary,
        "degree3_projectability": ea_degree3_projectability_summary,
        "degree4_projectability": ea_degree4_projectability_summary,
        "shared_scheduler": ea_shared_scheduler_summary,
        "ea_only": bool(int(args.ea_only)),
    }
    metadata["seed"] = {
        "shape": list(u_seed.shape),
        "rms": float(rms(u_seed)),
        "max_abs": float(np.max(np.abs(u_seed))) if u_seed.size else 0.0,
        "seed_path": str(seed_path),
        "effective_control_mode": bool(effective_control_mode),
        "hardware_projection_mode": bool(hardware_projection_mode),
        "trajectory_projection_mode": bool(trajectory_projection_mode),
        "target_continuation_mode": bool(target_continuation_mode),
        "constructive_projection_mode": bool(constructive_mode),
        "constructive_batch2_mode": bool(constructive_batch2_mode),
        "constructive_batch2_compress_mode": bool(constructive_compress_mode),
        "base_physical_control_count": int(base_physical_control_count),
        "effective_d2_control_count": int(len(effective_d2_map)),
        "effective_d2_labels": ctrl_labels[base_physical_control_count:] if effective_control_mode else [],
        "hardware_projection_summary": hardware_projection_summary,
        "trajectory_projection_summary": trajectory_projection_summary,
        "target_continuation_summary": target_continuation_summary,
        "constructive_compress_summary": constructive_compress_summary,
        "constructive_audit_summary": constructive_audit_summary,
        "batch2_summary": batch2_summary,
        "constructive_audit_path": str(constructive_audit_path) if constructive_audit_path is not None else None,
        "constructive_audit_rows_path": str(constructive_audit_rows_path) if constructive_audit_rows_path is not None else None,
        "pauli_audit_path": str(pauli_audit_path) if pauli_audit_path is not None else None,
        "pauli_audit_rows_path": str(pauli_audit_rows_path) if pauli_audit_rows_path is not None else None,
        "compiler_diagnostics_rows": len(compiler_diagnostics),
        "compiler_diagnostics_path": str(compiler_diag_path) if compiler_diag_path is not None else None,
        "compiler_term_diagnostics_rows": len(compiler_term_diagnostics),
        "compiler_term_diagnostics_path": str(compiler_term_diag_path) if compiler_term_diag_path is not None else None,
        "compiler_audit_summary_path": str(compiler_audit_summary_path) if compiler_audit_summary_path is not None else None,
        "compiler_audit_summary": compiler_audit_summary,
    }
    maybe_write_metadata(metadata_path, metadata)

    if trajectory_projection_summary is not None and trajectory_projection_failed_reasons and int(args.eatraj_stop_on_fail):
        reporter.info("[EATRAJ] stop-on-fail requested; exiting before physical GRAPE.")
        reporter.info(f"[TIME] total wall={format_seconds(time.time() - main_wall_start)}")
        return

    if int(args.compiler_audit_only) or str(args.compiler_mode) in ("product_cost", "product_instant"):
        if str(args.compiler_mode) in ("product_cost", "product_instant") and not int(args.compiler_audit_only):
            reporter.info(f"[AUDIT] {args.compiler_mode} is diagnostic-only; exiting before physical GRAPE.")
        else:
            reporter.info("[AUDIT] compiler-audit-only requested; exiting before physical GRAPE.")
        reporter.info(f"[TIME] total wall={format_seconds(time.time() - main_wall_start)}")
        return

    if constructive_mode and constructive_audit_failed_reasons and int(args.constructive_stop_on_fail):
        reporter.info("[CONSTRUCTIVE] stopping before GRAPE because the constructive projection audit failed.")
        reporter.info(f"[TIME] total wall={format_seconds(time.time() - main_wall_start)}")
        return

    run_start = time.time()

    if not int(args.compare):
        reporter.info("")
        reporter.info("[RUN] METHOD only")
        method_trace_dir = trace_dir / "method" if trace_dir is not None else None
        if method_trace_dir is not None:
            method_trace_dir.mkdir(parents=True, exist_ok=True)
        method_checkpoint = CheckpointManager(str(checkpoint_root / "method"), "method") if checkpoint_root is not None else None
        rr, u_final = run_method_stages(
            name="METHOD",
            psi0=psi0,
            target=target,
            H0_base=H0_base,
            drift_strength_hw=float(args.drift_strength_hw),
            ctrl_axes=ctrl_axes,
            T=T,
            Nt=Nt,
            stages=method_stages,
            homotopy_mode=str(args.homotopy_mode),
            u0=u_seed,
            opt_cfg_template=opt_cfg_template,
            threshold=float(args.threshold),
            seed=int(args.seed),
            jitter_list=jitter_list,
            verbose=int(args.verbose),
            reporter=reporter,
            progress_every=int(args.progress_every_grape),
            trace_dir=method_trace_dir,
            trace_prefix="",
            checkpoint_manager=method_checkpoint,
            H0_base_dense=H0_base_dense,
            ctrl_stack_dense=ctrl_stack_dense,
            psi0_vec=psi0_vec,
            target_vec=target_vec,
        )

        dt_final = T / u_final.shape[0]
        method_path = outdir / f"method_controls{tag}.csv"
        write_awg_csv(method_path, u_final, dt_final, ctrl_labels)
        reporter.info(f"[AWG] wrote {method_path}")
        if effective_control_mode:
            reporter.info(
                f"[METHOD:EFFECTIVE] initF_effmodel={rr.initF_phys:.6f}  "
                f"finalF_effmodel={rr.finalF_phys:.6f}"
            )
        else:
            reporter.info(f"[METHOD] initF_phys={rr.initF_phys:.6f}  finalF_phys={rr.finalF_phys:.6f}")
        reporter.info(f"[TIME] total wall={format_seconds(time.time() - run_start)}")
        metadata["method_only"] = {"result": asdict(rr), "method_controls_path": str(method_path)}
        maybe_write_metadata(metadata_path, metadata)
        return

    trials = int(args.trials)
    if trials <= 0:
        raise SystemExit("--trials must be positive")

    reporter.info("")
    reporter.info(f"[COMPARE] METHOD vs BASELINE over {trials} trial(s)")
    reporter.info(f"[BASELINE] mode={args.baseline_mode} init={args.baseline_init}")

    rows: list[dict[str, Any]] = []
    best_method: tuple[float, np.ndarray] | None = None
    best_baseline: tuple[float, np.ndarray] | None = None
    trials_start = time.time()

    for t_idx in range(trials):
        trial_seed = int(args.seed) + int(t_idx)
        rng = np.random.RandomState(trial_seed)

        u0_method = u_seed.copy()
        if float(args.seed_dither) > 0:
            u0_method = u0_method + float(args.seed_dither) * rng.randn(*u0_method.shape)
        if float(args.amp) > 0:
            np.clip(u0_method, -float(args.amp), float(args.amp), out=u0_method)

        M_base0 = Nt * baseline_stages[0].S
        target_rms = None
        if str(args.baseline_init) == "rmsmatch":
            target_rms = rms(resample_controls(u0_method, M_base0))

        u0_baseline = random_controls(
            M=M_base0,
            m=len(ctrl_axes),
            rng=rng,
            amp_bound=float(args.amp),
            init_mode=str(args.baseline_init),
            sigma=float(args.baseline_sigma),
            target_rms=target_rms,
        )

        trial_trace_dir = trace_dir / f"trial{t_idx + 1:03d}" if trace_dir is not None else None
        if trial_trace_dir is not None:
            trial_trace_dir.mkdir(parents=True, exist_ok=True)
        method_checkpoint = (
            CheckpointManager(str(checkpoint_root / f"trial{t_idx + 1:03d}"), "method")
            if checkpoint_root is not None
            else None
        )
        baseline_checkpoint = (
            CheckpointManager(str(checkpoint_root / f"trial{t_idx + 1:03d}"), "baseline")
            if checkpoint_root is not None
            else None
        )

        rr_m, u_m = run_method_stages(
            name="METHOD",
            psi0=psi0,
            target=target,
            H0_base=H0_base,
            drift_strength_hw=float(args.drift_strength_hw),
            ctrl_axes=ctrl_axes,
            T=T,
            Nt=Nt,
            stages=method_stages,
            homotopy_mode=str(args.homotopy_mode),
            u0=u0_method,
            opt_cfg_template=opt_cfg_template,
            threshold=float(args.threshold),
            seed=trial_seed,
            jitter_list=jitter_list,
            verbose=0,
            reporter=reporter,
            progress_every=int(args.progress_every_grape),
            trace_dir=trial_trace_dir,
            trace_prefix="",
            checkpoint_manager=method_checkpoint,
            H0_base_dense=H0_base_dense,
            ctrl_stack_dense=ctrl_stack_dense,
            psi0_vec=psi0_vec,
            target_vec=target_vec,
        )

        rr_b, u_b = run_method_stages(
            name="BASELINE",
            psi0=psi0,
            target=target,
            H0_base=H0_base,
            drift_strength_hw=float(args.drift_strength_hw),
            ctrl_axes=ctrl_axes,
            T=T,
            Nt=Nt,
            stages=baseline_stages,
            homotopy_mode=str(args.homotopy_mode),
            u0=u0_baseline,
            opt_cfg_template=opt_cfg_template,
            threshold=float(args.threshold),
            seed=trial_seed + 10_000,
            jitter_list=jitter_list,
            verbose=0,
            reporter=reporter,
            progress_every=int(args.progress_every_grape),
            trace_dir=trial_trace_dir,
            trace_prefix="",
            checkpoint_manager=baseline_checkpoint,
            H0_base_dense=H0_base_dense,
            ctrl_stack_dense=ctrl_stack_dense,
            psi0_vec=psi0_vec,
            target_vec=target_vec,
        )

        row = {
            "trial": t_idx + 1,
            "seed": trial_seed,
            "method_initF_phys": rr_m.initF_phys,
            "method_finalF_phys": rr_m.finalF_phys,
            "method_bestF_last_stage": rr_m.bestF_last_stage,
            "method_iters_to_threshold": rr_m.iters_to_threshold if rr_m.iters_to_threshold is not None else "",
            "method_total_stage_iters": rr_m.total_stage_iters,
            "method_wall_s": rr_m.wall_s,
            "baseline_initF_phys": rr_b.initF_phys,
            "baseline_finalF_phys": rr_b.finalF_phys,
            "baseline_bestF_last_stage": rr_b.bestF_last_stage,
            "baseline_iters_to_threshold": rr_b.iters_to_threshold if rr_b.iters_to_threshold is not None else "",
            "baseline_total_stage_iters": rr_b.total_stage_iters,
            "baseline_wall_s": rr_b.wall_s,
        }
        rows.append(row)

        if best_method is None or rr_m.finalF_phys > best_method[0]:
            best_method = (rr_m.finalF_phys, u_m.copy())
        if best_baseline is None or rr_b.finalF_phys > best_baseline[0]:
            best_baseline = (rr_b.finalF_phys, u_b.copy())

        trial_label = "EFFMODEL" if effective_control_mode else "F"
        reporter.info(
            f"  trial {t_idx + 1:3d}/{trials}: METHOD {trial_label}={rr_m.finalF_phys:.6f} | "
            f"BASELINE {trial_label}={rr_b.finalF_phys:.6f}"
        )
        reporter.progress(
            "Trials",
            t_idx + 1,
            trials,
            trials_start,
            extra=f"METHOD={rr_m.finalF_phys:.6f} BASELINE={rr_b.finalF_phys:.6f}",
            force=(t_idx + 1 == trials),
        )

    if str(getattr(args, "results_csv", "")).strip():
        out_csv = Path(str(args.results_csv))
        out_csv.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_csv = outdir / f"compare{tag}_{ts}.csv"
    write_results_csv(out_csv, rows)
    reporter.info("")
    reporter.info(f"[CSV] wrote {out_csv}")

    best_method_path = None
    best_baseline_path = None
    if best_method is not None:
        u_best = best_method[1]
        dt = T / u_best.shape[0]
        best_method_path = outdir / f"best_method_controls{tag}_{ts}.csv"
        write_awg_csv(best_method_path, u_best, dt, ctrl_labels)
        reporter.info(f"[AWG] wrote {best_method_path}")

    if best_baseline is not None:
        u_best = best_baseline[1]
        dt = T / u_best.shape[0]
        best_baseline_path = outdir / f"best_baseline_controls{tag}_{ts}.csv"
        write_awg_csv(best_baseline_path, u_best, dt, ctrl_labels)
        reporter.info(f"[AWG] wrote {best_baseline_path}")

    mF = [float(r["method_finalF_phys"]) for r in rows]
    bF = [float(r["baseline_finalF_phys"]) for r in rows]

    reporter.info("")
    reporter.info("=== SUMMARY ===")
    if effective_control_mode:
        reporter.info("diagnostic model: direct degree-2 effective controls, not hardware pulses")
    reporter.info(f"threshold: {args.threshold}")
    reporter.info(
        f"METHOD   success rate: {success_rate(mF, float(args.threshold)):.3f} | mean F={np.mean(mF):.6f} | std={np.std(mF):.6f}"
    )
    reporter.info(
        f"BASELINE success rate: {success_rate(bF, float(args.threshold)):.3f} | mean F={np.mean(bF):.6f} | std={np.std(bF):.6f}"
    )
    reporter.info(f"[TIME] total wall={format_seconds(time.time() - run_start)}")

    metadata["comparison"] = {
        "effective_control_mode": bool(effective_control_mode),
        "results_csv": str(out_csv),
        "best_method_controls": str(best_method_path) if best_method_path is not None else None,
        "best_baseline_controls": str(best_baseline_path) if best_baseline_path is not None else None,
        "method_success_rate": success_rate(mF, float(args.threshold)),
        "baseline_success_rate": success_rate(bF, float(args.threshold)),
        "method_mean_fidelity": float(np.mean(mF)) if mF else None,
        "baseline_mean_fidelity": float(np.mean(bF)) if bF else None,
        "method_std_fidelity": float(np.std(mF)) if mF else None,
        "baseline_std_fidelity": float(np.std(bF)) if bF else None,
        "rows": rows,
    }
    maybe_write_metadata(metadata_path, metadata)


if __name__ == "__main__":
    main()
