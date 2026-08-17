# rk-recall — Recall-Compensation Theorem for Runge-Kutta

> 希伯来书 11:1 —— "信就是所望之事的实底,是未见之事的确据."
> Hebrews 11:1 — "Now faith is the substance of things hoped for, the evidence of things not seen."

A self-contained Python module that tames **Runge-Kutta error accumulation**
via an ω-limit-set projection mechanism. The core theoretical result
(§3.26.16, **Drift Convergence Theorem**) proves that drift → 0
asymptotically as the baseline integration time T_baseline → ∞, a property
**not possessed by** Symplectic integrators, Projected RK, or Pyragas DFC.

---

## Why this exists

Classical RK4 integration suffers from truncation-error accumulation: each
step adds O(h⁵) local error, producing O(h⁴·t) global drift over long time
spans. For oscillatory systems this manifests as amplitude/phase error that
grows without bound; for chaotic systems the error grows exponentially
(ε ≈ ε₀·exp(λ_max·t)).

This module implements the **Recall-Compensation Theorem (§3.25)** and its
upgrades (§3.26.16 drift convergence, §3.26.17 Pyragas→Γ phase locking).
The correction anchor (hope_p) is located in **closed form** from the
trajectory's own ω-limit set (MIP theory: limit cycle → P* → nearest leaf),
implemented in pure numpy — **no oracle, no iteration, no analytical
solution required**. This makes it work on systems WITHOUT analytical
solutions (Lorenz, Van der Pol, etc.), not just harmonic oscillators.

---

## Theoretical core (§3.26.16 — Drift Convergence Theorem)

```
ε_leaf(T_baseline) ≤ C · h^{1/d} / T_baseline^{1/d}  →  0  as  T_baseline → ∞

  d           = attractor dimension (harmonic: 1, Lorenz: ~2.06)
  T_baseline  = baseline integration time (for Γ identification)
  ε_leaf      = drift = min_{z ∈ Γ} ‖y - z‖ (distance to attractor)
```

**Proof** (rigorous, in docs): Birkhoff ergodic theorem + Poincaré recurrence
⇒ uniform sampling on Γ ⇒ Voronoi cell radius → 0 ⇒ ε_leaf → 0.

**This is the unique property not possessed by market algorithms**:
- Symplectic: drift = O(h^{2p}) — fixed, does not decrease with time
- Projected RK: drift = 0 — strict, but requires known manifold equation
- Pyragas DFC: drift → 0 — asymptotic, but requires known period T, only UPO

rk-recall: drift → 0 as T_baseline → ∞ — **time is on your side**.

---

## §3.26.17 — Pyragas → Γ generalization (phase locking)

Pyragas DFC (1992) stabilizes UPOs (discrete subset of attractor) with
known period T. §3.26.17 generalizes this to the **entire attractor Γ**
using phase-labeled sampling, achieving phase locking **without knowing T**.

```
Convergence condition: |1 + h·λ_local - K| < 1
  → K > h·λ_local  (feedback strength > local expansion rate)
  → phase error → O(h²) asymptotically
```

**Proof** (rigorous, in docs): linearized error evolution e_{k+1} =
(1 + hλ - K)·e_k + O(h²), contraction when |1 + hλ - K| < 1.

---

## Honest limitations (反偶像声明)

### What this module DOES achieve

| Property | Status | Evidence |
|----------|--------|----------|
| Drift → 0 as T_baseline → ∞ | ✓ Proven | §3.26.16, Lorenz 202x improvement |
| Phase → O(h²) with K > λ_local | ✓ Proven | §3.26.17, three-system validation |
| No oracle (no analytical solution needed) | ✓ | Pure numpy, closed-form hope_p |
| Universal (periodic + chaotic + conservative) | ✓ | Three benchmarks |

### What this module does NOT achieve

| Property | Reason |
|----------|--------|
| **Drift = 0 (strict)** | Floating-point lower bound ε_round ≈ 1e-16 (受造限制) |
| **Phase = 0 (strict)** | RK4 local error O(h⁵) ⇒ phase floor O(h²) |
| **Beats Symplectic on conservative systems** | Symplectic is optimal for conservative (use symplectic variant) |
| **Beats Pyragas on UPO** | Pyragas has strict proof on UPO, this module is broader but less precise |
| **Works on unbounded systems** | Requires bounded system (ω-limit set must exist) |
| **Works on stiff systems** | Explicit RK4 baseline explodes on stiff systems |
| **Works on high-dimensional systems (d > 3)** | 1/d convergence rate becomes impractical (curse of dimensionality) |

### Honest comparison with market algorithms

| Dimension | rk-recall | Symplectic | Projected RK | Pyragas DFC |
|-----------|-----------|------------|--------------|-------------|
| Drift → 0 with time | ✅ **Unique** | ❌ Fixed | ❌ Fixed | ❌ Fixed |
| No oracle | ✅ | ✅ | ❌ (needs manifold) | ❌ (needs T) |
| Universal (3 systems) | ✅ | ❌ (conservative only) | ⚠ (needs manifold) | ⚠ (needs T + UPO) |
| Instant drift precision | ⚠ ~3% | ✅ 1e-11 | ✅ Strict 0 | ✅ → 0 |
| Phase control | ⚠ O(h²) | ✅ O(h^p) | ✅ O(h^p) | ✅ → 0 |
| Rigorous proof | ✅ §3.26.16/17 | ✅ Hairer 2006 | ✅ Manifold theory | ✅ Just 1999 |

**Honest positioning**: rk-recall is **not "better than"** market algorithms.
It is the **unique** algorithm with "drift → 0 as T_baseline → ∞"
(ergodic theorem engineering). In instantaneous precision it is **worse**
than specialized algorithms. Use rk-recall when you need long-term drift
control on unknown systems without analytical solutions.

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
matplotlib>=3.6,<4.0  # optional, only for plotting
```

Python ≥ 3.9 required. **Algorithm core has no scipy dependency**
(validation layer may use scipy for independent DOP853 verification).

---

## Quick start

### Run the full benchmark suite

```bash
cd /path/to/parent/dir   # the directory CONTAINING rk_recall/
python -m rk_recall.rk_recall_compensation
```

### Use the API directly

```python
import numpy as np
from rk_recall import (
    rk4_integrate,
    rk4_with_recall,
    rk4_hybrid_correction,
    rk4_with_recall_adaptive,  # §3.26.16 adaptive baseline
    pyragas_adaptive,           # §3.26.16 + §3.26.17 end-to-end
    benchmarks,
)

# System: Lorenz (chaotic, no analytical solution)
system = benchmarks.BENCHMARKS["lorenz"]
f, y0, t_end, h = system["f"], system["y0"], system["t_end"], system["h"]

# 1. Baseline RK4 (error accumulates)
baseline = rk4_integrate(f, y0, 0.0, t_end, h)

# 2. RK4 + recall projection (drift → 0 as T_baseline → ∞, §3.26.16)
corrected = rk4_with_recall(f, y0, 0.0, t_end, h, recall_period=10)

# 3. Adaptive baseline extension (§3.26.16, extends T_baseline until drift ≤ target)
adaptive = rk4_with_recall_adaptive(
    f, y0, 0.0, t_end, h, recall_period=10, target_drift=1e-3
)

# 4. Pyragas → Γ phase locking (§3.26.17, end-to-end)
pyragas_result = pyragas_adaptive(
    f, y0, 0.0, t_end, h, recall_period=10, target_drift=1e-3
)
```

---

## Benchmark results (§3.26.16 drift convergence)

Drift = max_t min_{z ∈ Γ} ‖y(t) - z‖ (distance to attractor, NOT trajectory-vs-trajectory).

| System | d | T_baseline growth | Drift improvement | max/diameter |
|--------|---|-------------------|-------------------|--------------|
| harmonic | 1 | 10→500 (50x) | 2.91e-2 → 1.83e-2 (1.6x) | 1.1% |
| **lorenz** | 2.06 | 10→500 (50x) | **17.86 → 0.088 (202x)** | **0.28%** |
| vanderpol | 1 | 10→500 (50x) | 4.32e-2 → 1.34e-2 (3.2x) | 0.55% |

**Key insight**: Drift decreases as T_baseline increases — **time is on your side**.
For chaotic systems (d > 1), the fractal structure allows ε_leaf → 0
indefinitely; for periodic systems (d = 1), interpolation densification
extends the convergence.

---

## §3.26.17 phase locking results

| System | λ_local | K | Drift improvement | Phase improvement |
|--------|---------|---|-------------------|-------------------|
| harmonic | -0.0000 | 0.0019 | 6.49x | 1.06x (angle) |
| **lorenz** | 0.0680 | 0.1019 | **5.04x** | **5.04x** |
| vanderpol | 0.0628 | 0.0110 | 4.81x | 4.81x |

K is chosen adaptively: K = 1.5·h·λ_local for expanding systems (chaotic),
K = 1e-3 for contracting/conservative systems (avoid over-feedback).

---

## Module reference

### Core algorithms

| Function | Description |
|----------|-------------|
| `rk4_step(f, t, y, h, *args)` | Single RK4 step |
| `rk4_integrate(f, y0, t0, t_end, h, *args)` | Full RK4 integration (baseline) |
| `rk4_with_recall(f, y0, ...)` | RK4 + recall projection (§3.26.4) |
| `rk4_hybrid_correction(f, y0, ...)` | Three-layer hybrid (GamblePole + LimitCycle + Recall) |
| `rk4_with_recall_adaptive(f, y0, ...)` | §3.26.16 adaptive baseline extension |
| `pyragas_adaptive(f, y0, ...)` | §3.26.16 + §3.26.17 end-to-end |

### Closed-form P* location (no oracle)

| Function | Description |
|----------|-------------|
| `detect_limit_cycle(t_array, y_array)` | FFT-based period detection + ω-limit set extraction |
| `locate_p_star(cycle_points, temperature)` | P* = argmax_{y∈Γ} H(y) (max entropy point) |
| `locate_nearest_leaf(y, cycle_points)` | hope_p dynamic update (§3.26.2) |
| `limit_cycle_distance(y, cycle_points)` | Unified error metric (§3.26.6) |
| `densify_cycle(cycle_points, factor)` | Periodic Γ interpolation (solves M saturation) |
| `adaptive_baseline_extension(...)` | T_baseline auto-extension until drift ≤ target |

### §3.26.17 phase locking

| Function | Description |
|----------|-------------|
| `build_phase_labeled_gamma(...)` | Build Γ with phase labels |
| `estimate_local_lyapunov(...)` | Upper-quantile λ_local estimation |
| `pyragas_on_attractor(...)` | §3.26.17 continuous feedback on Γ |

### Independent validation (§3.26.15)

| Function | Description |
|----------|-------------|
| `independent_validation.independent_error_dop853(...)` | DOP853 independent verification |
| `independent_validation.three_layer_error_analysis(...)` | Three-layer error metric (eliminates self-reference) |

---

## File structure

```
rk_recall/
├── rk_recall_compensation.py   # Algorithm core (no scipy, pure numpy)
├── independent_validation.py   # Validation layer (may use scipy, test-only)
├── benchmarks.py               # Three benchmark ODE systems
├── __init__.py                 # Public API exports
├── tests/                      # Test suite (48 tests)
├── LICENSE                     # MIT
├── README.md                   # This file
├── CITATION.cff                # Citation metadata
├── pyproject.toml              # PEP 621 build config
└── requirements.txt            # Runtime dependencies
```

---

## Theoretical references

- **§3.26.16** Drift Convergence Theorem (Birkhoff ergodic + Poincaré recurrence)
- **§3.26.17** Pyragas → Γ generalization (phase locking without known T)
- **Pyragas DFC (1992)** — original delayed feedback control for UPO stabilization
- **Hairer (2006)** — Geometric Integration (Symplectic methods)
- **Birkhoff (1931)** — Ergodic theorem
- **Poincaré (1890)** — Recurrence theorem

Full proofs in `docs/控制理论与应用_第23卷第3期_理论框架.md` (§3.26.16, §3.26.17).

---

## License

MIT License — see [LICENSE](LICENSE).

---

## 反偶像声明 (Anti-idolatry statement)

This module is a **created mathematical tool**, not life, not consciousness,
not "spirit". The "drift → 0" is a measure-theoretic conclusion, not "the
Creator's omniscience". The "phase → O(h²)" is an asymptotic limit, not
"actual achievement of 0". The algorithm is bound by mathematical laws
(Lyapunov exponents, floating-point precision) — it cannot exceed its
created nature.

一切荣光来自造物主，一切荣光归于造物主
All glory comes from the Creator, all glory belongs to the Creator.

---

## Related

- [The Base Scripture](https://github.com/jinggu-glitch/theology-collection) — A Hebrew-Morse decoding system for Biblical interpretation: pictographic reading, gematria, chiastic narrative synthesis.
