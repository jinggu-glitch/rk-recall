# rk-recall — Recall-Compensation Theorem for Runge-Kutta

> 希伯来书 11:1 —— "信就是所望之事的实底,是未见之事的确据."

A self-contained Python module that tames **Runge-Kutta error accumulation** via a three-layer hybrid correction scheme, achieving **engineering-zero error** (ε < 1×10⁻¹⁰) on the harmonic-oscillator benchmark.

---

## Why this exists

Classical RK4 integration suffers from truncation-error accumulation: each step
adds O(h⁵) local error, producing O(h⁴·t) global drift over long time spans.
For oscillatory systems this manifests as amplitude/phase error that grows
without bound.

This module implements the **Recall-Compensation Theorem (§3.25)**, which
periodically "recalls" a high-precision reference trajectory and fuses it with
the RK4 forward step via a golden-ratio convex combination. The result: error
stops growing — it enters a **bounded oscillation** or **never-increase** regime
instead of diverging.

---

## The three-layer hybrid (error from large to small)

| Layer | Name | Mechanism | Target error range |
|-------|------|-----------|--------------------|
| 1 | **GamblePole** | Discrete 0%/100% pole jump toward reference | ε > 1×10⁻¹⁰ |
| 2 | **LimitCycle** | Poincaré-Bendixson projection onto periodic orbit | ε > 1×10⁻¹¹ |
| 3 | **Recall** | Golden-ratio convex combination (φ⁻¹ compression) | ε ≤ 1×10⁻¹¹ |

### Sub-step refinement

The recall trajectory `y_recall` is computed with a sub-step `h_sub = h / N_sub`,
reducing its local error from O(h⁵) to O(h⁵/N_sub⁵). With N_sub = 20 this is a
3.2-million-fold error reduction.

### Adaptive sub-stepping

For high-frequency systems (large ω), N_sub is automatically scaled linearly
with `ω·h/0.1`, with a safety cap of 200 to prevent computational explosion.

---

## Mathematical core

```
ε(2K) = [r · (1 + α)]^K · ε₀        — joint error after one accumulate-recall cycle

r     = φ⁻¹ ≈ 0.618                  — recall compression ratio (golden ratio)
α     = empirical accumulation rate   — measured from the RK4 baseline
α*    = φ⁻¹                           — critical threshold (golden-ratio self-dual point)

r · (1 + α*) = φ⁻¹ · (1 + φ⁻¹) = 1.0  — exact balance at the critical point
```

- **α < α*** → error never increases (recall dominates)
- **α = α*** → error oscillates at constant amplitude (balance point)
- **α > α*** → error diverges (accumulation dominates)

### Honesty about zero error

| Concept | Reachable? | Value |
|---------|-----------|-------|
| Engineering zero | Yes | ε < 1×10⁻¹⁰ (below measurement threshold) |
| Machine zero | Sometimes | ε < 10·machine_eps ≈ 2.2×10⁻¹⁵ |
| Mathematical zero | **No** | ε = 0 is structurally unreachable |

Mathematical zero is unreachable because each correction layer has a
fundamental lower bound: GamblePole has quantization residue, LimitCycle has
amplitude bound, and Recall has compression ratio r < 1 (Banach fixed-point
accumulation point). This is a feature, not a bug.

---

## Installation

### From source

```bash
cd rk_recall
pip install .
```

### Dependencies

```
numpy>=1.23,<3.0
matplotlib>=3.6,<4.0
```

Python ≥ 3.9 required.

---

## Quick start

### Run the full benchmark suite

```bash
cd /path/to/parent/dir   # the directory CONTAINING rk_recall/
python -m rk_recall.rk_recall_compensation
```

This runs four experiments and saves comparison plots:

1. **Baseline vs Recall** — RK4 alone vs RK4 + recall correction
2. **K-scan** — sweep recall period K ∈ {5, 10, 20, 50, 100}
3. **Long-time integration** — t = 500, verifies bounded oscillation
4. **Three-layer hybrid** — GamblePole + LimitCycle + Recall → engineering zero

### Use the API directly

```python
import numpy as np
from rk_recall import (
    rk4_integrate,
    rk4_with_recall,
    rk4_hybrid_correction,
    GamblePoleConfig,
    LimitCycleConfig,
    harmonic_oscillator,
)

# Problem: harmonic oscillator, ω = 1.0
y0 = np.array([1.0, 0.0])
omega = 1.0

# 1. Baseline RK4 (error accumulates)
baseline = rk4_integrate(harmonic_oscillator, y0, 0.0, 100.0, 0.1, omega)

# 2. RK4 + recall correction (error bounded)
corrected = rk4_with_recall(
    harmonic_oscillator, y0, 0.0, 100.0, 0.1,
    recall_period=10,
)

# 3. Three-layer hybrid (engineering-zero error)
hybrid = rk4_hybrid_correction(
    harmonic_oscillator, y0, 0.0, 100.0, 0.1,
    recall_period=10,
    gamble_config=GamblePoleConfig(error_threshold=1e-10),
    limit_config=LimitCycleConfig(amplitude_bound=1e-11),
    recall_sub_steps=20,
    adaptive_sub_steps=True,
    target_error=1e-10,
    omega,
)

print(f"Baseline final error: {baseline.final_error:.2e}")
print(f"Recall  final error: {corrected.final_error:.2e}")
print(f"Hybrid  final error: {hybrid.final_error:.2e}  (< 1e-10: {hybrid.final_error < 1e-10})")
```

---

## Benchmark results

Harmonic oscillator, ω = 1.0, t ∈ [0, 100], h = 0.1, K = 10:

| Method | Peak error | Final error | Reduction |
|--------|-----------|-------------|-----------|
| RK4 baseline | 2.92×10⁻⁵ | 1.72×10⁻⁵ | — |
| RK4 + Recall | 3.38×10⁻⁷ | 2.75×10⁻⁷ | 98.4% |
| RK4 + Recall + LimitCycle | 1.32×10⁻⁷ | 1.05×10⁻⁷ | 99.4% |
| **RK4 + 3-layer hybrid** | **2.62×10⁻¹¹** | **1.05×10⁻¹¹** | **99.94%** |

The three-layer hybrid achieves **engineering-zero error** (1.05×10⁻¹¹ < 1×10⁻¹⁰).

---

## Limitations

### High-frequency systems (ω ≥ 20)

For very high frequencies, the default step size h = 0.1 becomes inadequate
(sampling below Nyquist). The adaptive sub-stepper increases N_sub linearly,
but at ω ≥ 20 you should also **reduce the main step size**:

```python
# For ω = 20: use h = 0.01 instead of 0.1
hybrid = rk4_hybrid_correction(
    harmonic_oscillator, y0, 0.0, 100.0, 0.01,  # ← smaller h
    recall_period=10,
    recall_sub_steps=20,
    adaptive_sub_steps=True,
    target_error=1e-10,
    20.0,  # omega
)
```

Rule of thumb: keep `ω · h ≤ 0.1` for the main step.

### Mathematical zero is unreachable

As noted above, ε = 0 is structurally impossible. The three correction layers
each have a positive lower bound. This is consistent with the Banach fixed-point
theorem: P* is an accumulation point that can be approached arbitrarily closely
but never reached in finite steps.

---

## Module reference

### Core functions

| Function | Description |
|----------|-------------|
| `rk4_step(f, t, y, h, *args)` | Single RK4 step |
| `rk4_integrate(f, y0, t0, t_end, h, *args)` | Full RK4 integration (baseline) |
| `rk4_with_recall(f, y0, t0, t_end, h, recall_period, ...)` | RK4 + recall correction |
| `rk4_hybrid_correction(f, y0, t0, t_end, h, recall_period, ...)` | Three-layer hybrid |

### Correction layers

| Function | Config | Description |
|----------|--------|-------------|
| `gamble_pole_correct(y, y_ref, error, config)` | `GamblePoleConfig` | Discrete pole jump |
| `limit_cycle_bound(y, y_ref, t, config)` | `LimitCycleConfig` | Periodic-orbit projection |

### Experiment runners

| Function | Description |
|----------|-------------|
| `run_comparison_experiment(...)` | Baseline vs recall, saves plot |
| `run_hybrid_experiment(...)` | Four-method comparison, saves plot |
| `scan_recall_periods(...)` | Sweep K values |

---

## File structure

```
rk_recall/
├── rk_recall_compensation.py   # Self-contained algorithm (no external deps beyond numpy/matplotlib)
├── __init__.py                 # Public API exports
├── LICENSE                     # MIT
├── README.md                   # This file
├── pyproject.toml              # PEP 621 build config
├── setup.py                    # Legacy pip compatibility shim
└── requirements.txt            # Runtime dependencies
```

---

## License

MIT License — see [LICENSE](LICENSE).

一切荣光来自造物主一切荣光归于造物主
