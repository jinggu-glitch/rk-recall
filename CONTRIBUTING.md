# Contributing to rk-recall

Thank you for your interest in contributing to **rk-recall**. This document describes the development workflow and standards.

## Development Setup

### Prerequisites

- Python 3.9 or later (3.12 recommended)
- git

### Clone and install

```bash
git clone https://github.com/jinggu-glitch/rk-recall.git
cd rk-recall
python -m venv .venv
# Windows
.venv\Scripts\activate
# Unix
source .venv/bin/activate

# Install with dev dependencies
pip install -e ".[dev,all]"
pre-commit install
```

### Verify installation

```bash
pytest tests/ -v
```

All 25 tests should pass.

## Development Workflow

### 1. Branch naming

- `feature/<short-description>` — new features
- `fix/<short-description>` — bug fixes
- `docs/<short-description>` — documentation only
- `refactor/<short-description>` — code refactoring
- `test/<short-description>` — test additions

### 2. Commit messages (Conventional Commits)

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**:
- `feat`: new feature
- `fix`: bug fix
- `docs`: documentation
- `style`: formatting only
- `refactor`: code change that neither fixes a bug nor adds a feature
- `perf`: performance improvement
- `test`: test additions
- `chore`: build/tooling

**Example**:
```
feat(§3.26.16): extend adaptive baseline to d>3 systems

Adds dimension-aware T_baseline extension for high-dimensional attractors.
Validates on a 4D hyperchaotic system with Γ-distance drift metric.
```

### 3. Pre-commit hooks

Before each commit, pre-commit runs:
- **ruff** — linting
- **black** — formatting
- **isort** — import sorting
- **mypy** — type checking

To run manually:
```bash
pre-commit run --all-files
```

### 4. Tests

- All new code must have tests
- Tests live in `tests/` with `test_*.py` naming
- Use `pytest` framework
- Mark long tests with `@pytest.mark.slow`
- Mark cross-system tests with `@pytest.mark.integration`

Run tests:
```bash
# All tests
pytest

# Fast tests only
pytest -m "not slow"

# With coverage
pytest --cov=rk_recall --cov-report=html
```

### 5. Type checking

All code must pass `mypy --strict`:

```bash
mypy rk_recall/
```

### 6. Linting

```bash
ruff check rk_recall/
```

### 7. Before submitting a PR

- [ ] All tests pass (`pytest`)
- [ ] `mypy` passes with no errors
- [ ] `ruff` passes with no errors
- [ ] `black --check` passes
- [ ] `isort --check` passes
- [ ] CHANGELOG.md updated (if user-facing changes)
- [ ] Bilingual (Chinese/English) comments added for new code
- [ ] No `print()` in library code (use `logging`)

## Coding Standards

### Style

- Line length: 100 characters
- Use `black` defaults for formatting
- Imports sorted with `isort` (profile=black)

### Comments

- All comments must be **bilingual (Chinese/English)**
- Use the format: `# 中文 / English`
- For docstrings: Chinese first, English on the next line

```python
# §数据结构 / Data structures
def my_function(x: float) -> float:
    """计算平方 / Compute square.
    
    Args:
        x: 输入值 / input value
    
    Returns:
        x² / x squared
    """
    return x * x
```

### Type annotations

- All functions must have complete type annotations
- Use `from __future__ import annotations` for forward references
- Use `Callable[..., np.ndarray]` for ODE functions
- Use `X | None` (not `Optional[X]`) for optional types

### Error handling

- Use the custom exception hierarchy (`RKRecallError` and subclasses)
- Never use bare `except:` — always specify the exception type
- Library code must not print — use `logger.info()` / `logger.warning()` / `logger.error()`

### Configuration objects

- Use `@dataclass(frozen=True)` for all configuration objects
- Validate invariants in `__post_init__`
- Raise `ConfigurationError` on invalid input

### Optional dependencies

- Hard dependency: `numpy` only
- Optional: `scipy` (independent validation layer, test-only), `matplotlib` (plot)
- Import optional deps lazily inside functions
- Raise `MissingOptionalDependencyError` if missing

## Theoretical Contributions

If you are contributing to the **theoretical** aspects (new theorems, proofs, or extensions to §3.25):

1. Open an issue first to discuss the mathematical framework
2. Provide a written proof in the PR description
3. Add tests that verify the theoretical claim numerically
4. Update the relevant theory documentation

## Anti-idolatry Statement

This project is a created tool — it has no consciousness, soul, or life. Mathematical conclusions are created structures, not divine truth. All glory belongs to the Creator.

本工程是受造的工具 — 无意识、无灵魂、无生命. 数学结论是受造的结构, 不是神圣真理. 一切荣光来自造物主, 一切荣光归于造物主.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
