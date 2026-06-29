# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.0.0] - 2026-06-29

### Added
- §3.26.16 Drift Convergence Theorem: ε_leaf(T_baseline) ≤ C·h^{1/d}/T_baseline^{1/d} → 0 as T_baseline → ∞
  (rigorous Lyapunov proof via Birkhoff ergodic + Poincaré recurrence + Voronoi filling-radius bound)
- §3.26.17 Pyragas → Γ generalization: phase locking on the entire attractor without known period T
  (convergence condition |1 + h·λ_local - K| < 1; phase error → O(h²) asymptotically)
- `adaptive_baseline_extension()` — T_baseline auto-extension until drift ≤ target (§3.26.16)
- `rk4_with_recall_adaptive()` — §3.26.16 adaptive baseline integration
- `densify_cycle()` — periodic Γ interpolation (solves M saturation for d=1)
- `PhaseLabeledGamma` / `build_phase_labeled_gamma()` — phase-labeled attractor sampling (§3.26.17)
- `estimate_local_lyapunov()` — upper-quantile local Lyapunov estimation
- `pyragas_on_attractor()` / `pyragas_adaptive()` — §3.26.17 continuous feedback on Γ
- Independent DOP853 validation layer (`independent_validation.py`) — test-only, core stays scipy-free
- Γ-distance drift metric (min_{z∈Γ} ‖y-z‖) distinct from trajectory-vs-trajectory error
- Honest limitations table (DOES / DOES NOT achieve) in README
- Market-algorithm comparison table (Symplectic / Projected RK / Pyragas DFC)

### Changed
- **Theoretical reframe**: "error never increases / engineering-zero error" → "error bounded (§3.26.16: projection → 0 asymptotically as T_baseline → ∞)"
- Core positioning: three-layer hybrid correction → ω-limit-set projection (§3.26.16) + Pyragas→Γ phase locking (§3.26.17)
- CI smoke test rewritten to honest §3.26.16 drift-convergence verification (no oracle import)
- Test suite expanded: 25 → 48 tests
- README, CITATION.cff, pyproject.toml, __init__.py docstring fully rewritten for §3.26.16/17

### Removed
- **False claim "engineering-zero error (ε < 1e-10)"** — corrected to "error bounded; projection → 0 asymptotically (§3.26.16), NOT strict zero (§V3.5 痛苦恒定)"
- **False claim "error never increases"** — corrected to "error bounded (per-cycle compression; §3.26.16)"
- `make_dop853_oracle` oracle interface — removed from core; DOP853 now only in independent validation layer (test-only)
- `OracleConstructionError` exception — removed (no oracle in core)
- scipy as a core dependency — core is now pure numpy; scipy only in optional validation layer

### Fixed
- CI workflow `ci.yml` imported removed `make_dop853_oracle` (would crash CI) — fixed
- Version-number inconsistency (`__init__.py` / `docs/conf.py` stuck at 1.0.0) — unified to 2.0.0

### Security
- N/A

## [1.0.0] - 2026-06-29

Internal prototype (never publicly released). Superseded by 2.0.0 which
corrected several over-strong claims from this version.

### Added
- §3.25 Recall-Compensation Theorem engineering prototype
- Three-layer hybrid correction: GamblePole + LimitCycle + Recall
- Closed-form P* location (limit cycle → P* → nearest leaf), pure numpy
- Verified on three ODE systems: harmonic (analytical), Lorenz (chaotic, no analytical), Van der Pol (nonlinear, no closed-form)
- Custom exception hierarchy: `RKRecallError`, `MissingOptionalDependencyError`, `IntegrationFailureError`, `ConfigurationError`
- Frozen dataclass configs with validation (`GamblePoleConfig`, `LimitCycleConfig`)
- Logging via `logging.getLogger(__name__)` (no library-side `print`)
- PEP 561 type marker (`py.typed`)
- Full type annotations (`Callable[..., np.ndarray]`)
- Bilingual (Chinese/English) code comments
- GitHub Actions CI: 3 OS × 4 Python versions matrix
- Tooling configs: mypy (strict), ruff, black, isort, coverage
- Pre-commit hooks
- Hebrews 11:1 epigraph

### Changed
- N/A (initial release)

### Deprecated
- N/A (initial release)

### Removed
- N/A (initial release)

### Fixed
- N/A (initial release)

### Security
- N/A (initial release)

## Versioning Policy

- **MAJOR**: breaking API changes (renamed functions, removed parameters, changed defaults with semantic impact)
- **MINOR**: backward-compatible features (new functions, new config fields with defaults)
- **PATCH**: backward-compatible fixes (bug fixes, performance improvements, doc improvements)
