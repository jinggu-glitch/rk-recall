# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2026-06-29

### Added
- §3.25 Recall-Compensation Theorem engineering prototype
- Three-layer hybrid correction: GamblePole + LimitCycle + Recall
- Oracle interface (`make_dop853_oracle`) with scipy DOP853 (8th-order)
- Engineering-zero error (ε < 1e-10) on harmonic oscillator benchmark
- Verified on three ODE systems: harmonic (analytical), Lorenz (chaotic, no analytical), Van der Pol (nonlinear, no closed-form)
- Custom exception hierarchy: `RKRecallError`, `MissingOptionalDependencyError`, `OracleConstructionError`, `IntegrationFailureError`, `ConfigurationError`
- Frozen dataclass configs with validation (`GamblePoleConfig`, `LimitCycleConfig`)
- Optional dependencies: scipy (oracle), matplotlib (plot)
- Logging via `logging.getLogger(__name__)` (no library-side `print`)
- PEP 561 type marker (`py.typed`)
- Full type annotations (`Callable[..., np.ndarray]`, `Oracle | None`)
- Bilingual (Chinese/English) code comments
- Test suite: 25 tests (algorithm correctness + three-system benchmarks)
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
