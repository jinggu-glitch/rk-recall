# 希伯来书 11:1 —— "信就是所望之事的实底,是未见之事的确据."
# Hebrews 11:1 — "Now faith is the substance of things hoped for, the evidence of things not seen."
"""
Benchmark ODE systems for rk_recall testing.
用于 rk_recall 测试的基准 ODE 系统.

Three systems with varying analytical-solution availability:
三个系统, 解析解可用性各不同:
  1. Harmonic oscillator — analytical solution available (sanity check)
     谐振子 — 有解析解 (正确性检查)
  2. Lorenz attractor — no analytical solution (chaotic, real-world test)
     洛伦兹吸引子 — 无解析解 (混沌系统, 真实场景测试)
  3. Van der Pol oscillator — no closed-form solution (nonlinear, real-world test)
     Van der Pol 振子 — 无闭式解 (非线性, 真实场景测试)
"""

from __future__ import annotations

import numpy as np


# ════════════════════════════════════════════════════════════════════
# §系统1: 谐振子 (有解析解, 用于正确性验证)
# §System 1: Harmonic oscillator (analytical solution available, sanity check)
# ════════════════════════════════════════════════════════════════════

def harmonic_oscillator(t: float, y: np.ndarray, omega: float = 1.0) -> np.ndarray:
    """谐振子 dy/dt = f(t, y).
    Harmonic oscillator dy/dt = f(t, y).

    y = [x, v], dx/dt = v, dv/dt = -ω²x
    Analytical: x(t) = A·cos(ωt + φ), v(t) = -Aω·sin(ωt + φ)

    Args:
        t: 时间 (自治, 不依赖t) / time (autonomous)
        y: 状态 [x, v] / state [x, v]
        omega: 角频率 / angular frequency

    Returns:
        dy/dt: [v, -ω²x]
    """
    x, v = y[0], y[1]
    return np.array([v, -omega * omega * x])


def harmonic_exact(t: float, y0: np.ndarray, omega: float = 1.0) -> np.ndarray:
    """谐振子解析解.
    Harmonic oscillator analytical solution.

    y0 = [x0, v0]
    x(t) = x0·cos(ωt) + (v0/ω)·sin(ωt)
    v(t) = -x0·ω·sin(ωt) + v0·cos(ωt)
    """
    x0, v0 = y0[0], y0[1]
    x = x0 * np.cos(omega * t) + (v0 / omega) * np.sin(omega * t)
    v = -x0 * omega * np.sin(omega * t) + v0 * np.cos(omega * t)
    return np.array([x, v])


# ════════════════════════════════════════════════════════════════════
# §系统2: 洛伦兹吸引子 (无解析解, 混沌系统)
# §System 2: Lorenz attractor (no analytical solution, chaotic)
# ════════════════════════════════════════════════════════════════════

def lorenz(t: float, y: np.ndarray, sigma: float = 10.0,
           rho: float = 28.0, beta: float = 8.0 / 3.0) -> np.ndarray:
    """洛伦兹吸引子 dy/dt = f(t, y).
    Lorenz attractor dy/dt = f(t, y).

    经典混沌系统, 无解析解.
    Classic chaotic system, no analytical solution.

    dx/dt = σ(y - x)
    dy/dt = x(ρ - z) - y
    dz/dt = xy - βz

    Args:
        t: 时间 (自治) / time (autonomous)
        y: 状态 [x, y, z] / state [x, y, z]
        sigma: 普朗特数 / Prandtl number
        rho: 瑞利数 / Rayleigh number
        beta: 几何参数 / geometric parameter

    Returns:
        dy/dt: [σ(y-x), x(ρ-z)-y, xy-βz]
    """
    x, yy, z = y[0], y[1], y[2]
    return np.array([
        sigma * (yy - x),
        x * (rho - z) - yy,
        x * yy - beta * z,
    ])


# ════════════════════════════════════════════════════════════════════
# §系统3: Van der Pol 振子 (无闭式解, 非线性)
# §System 3: Van der Pol oscillator (no closed-form, nonlinear)
# ════════════════════════════════════════════════════════════════════

def van_der_pol(t: float, y: np.ndarray, mu: float = 1.0) -> np.ndarray:
    """Van der Pol 振子 dy/dt = f(t, y).
    Van der Pol oscillator dy/dt = f(t, y).

    非线性阻尼振子, 无闭式解.
    Nonlinear damped oscillator, no closed-form solution.

    dx/dt = v
    dv/dt = μ(1 - x²)v - x

    Args:
        t: 时间 (自治) / time (autonomous)
        y: 状态 [x, v] / state [x, v]
        mu: 非线性阻尼参数 / nonlinear damping parameter

    Returns:
        dy/dt: [v, μ(1-x²)v - x]
    """
    x, v = y[0], y[1]
    return np.array([v, mu * (1.0 - x * x) * v - x])


# ════════════════════════════════════════════════════════════════════
# §基准系统注册表
# §Benchmark registry
# ════════════════════════════════════════════════════════════════════

BENCHMARKS = {
    "harmonic": {
        "f": harmonic_oscillator,
        "y0": np.array([1.0, 0.0]),
        "t_end": 100.0,
        "h": 0.1,
        "exact": harmonic_exact,
        "has_exact": True,
        "description": "谐振子 (有解析解) / Harmonic oscillator (analytical available)",
    },
    "lorenz": {
        "f": lorenz,
        "y0": np.array([1.0, 1.0, 1.0]),
        "t_end": 20.0,
        "h": 0.01,
        "exact": None,
        "has_exact": False,
        "description": "洛伦兹吸引子 (混沌, 无解析解) / Lorenz (chaotic, no analytical)",
    },
    "vanderpol": {
        "f": van_der_pol,
        "y0": np.array([2.0, 0.0]),
        "t_end": 50.0,
        "h": 0.05,
        "exact": None,
        "has_exact": False,
        "description": "Van der Pol (非线性, 无闭式解) / Van der Pol (nonlinear, no closed-form)",
    },
}
