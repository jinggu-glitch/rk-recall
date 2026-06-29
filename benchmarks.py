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

from typing import Callable

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
# §reference 构建器 (仅用于误差计算, 不参与校正)
# §reference builders (error computation only, not used in correction)
# ════════════════════════════════════════════════════════════════════

def make_harmonic_reference(
    y0: np.ndarray,
    omega: float = 1.0,
) -> Callable[[float], np.ndarray]:
    """构建谐振子 reference 函数 (闭包固定 y0, omega).
    Build a harmonic-oscillator reference function (closure fixing y0, omega).

    返回的 callable 签名为 reference(t) -> y, 仅用于误差计算, 不参与校正.
    The returned callable has signature reference(t) -> y, used for error
    computation only, never for correction.

    Args:
        y0: 初始状态 [x0, v0] / initial state
        omega: 角频率 / angular frequency

    Returns:
        reference: reference(t) -> y(t) / reference function
    """
    _y0 = np.asarray(y0, dtype=np.float64).copy()
    _omega = float(omega)

    def reference(t: float) -> np.ndarray:
        return harmonic_exact(t, _y0, _omega)

    return reference


# ════════════════════════════════════════════════════════════════════
# §基准系统注册表
# §Benchmark registry
#
# 每个系统包含:
# Each system contains:
#   - f: ODE 右端 dy/dt = f(t, y) / ODE right-hand side
#   - y0: 初始状态 / initial state
#   - t_end: 结束时间 / end time
#   - h: 步长 / step size
#   - exact: 解析解 (有则提供, 无则 None) / analytical solution (or None)
#   - has_exact: 是否有解析解 / whether analytical solution exists
#   - reference: 误差计算用 reference(t)->y (有解析解则闭包, 无则 None, 用极限环距离)
#     reference: reference(t)->y for error computation (closure if analytical, None otherwise → limit-cycle distance)
#   - description: 描述 / description
# ════════════════════════════════════════════════════════════════════

_HARMONIC_Y0 = np.array([1.0, 0.0])
_HARMONIC_OMEGA = 1.0

BENCHMARKS = {
    "harmonic": {
        "f": harmonic_oscillator,
        "y0": _HARMONIC_Y0,
        "t_end": 100.0,
        "h": 0.1,
        "exact": harmonic_exact,
        "has_exact": True,
        "reference": make_harmonic_reference(_HARMONIC_Y0, _HARMONIC_OMEGA),
        "description": "谐振子 (有解析解) / Harmonic oscillator (analytical available)",
    },
    "lorenz": {
        "f": lorenz,
        "y0": np.array([1.0, 1.0, 1.0]),
        "t_end": 20.0,
        "h": 0.01,
        "exact": None,
        "has_exact": False,
        "reference": None,  # §无解析解, 误差用极限环距离 / no analytical, error = limit-cycle distance
        "description": "洛伦兹吸引子 (混沌, 无解析解) / Lorenz (chaotic, no analytical)",
    },
    "vanderpol": {
        "f": van_der_pol,
        "y0": np.array([2.0, 0.0]),
        "t_end": 50.0,
        "h": 0.05,
        "exact": None,
        "has_exact": False,
        "reference": None,  # §无闭式解, 误差用极限环距离 / no closed-form, error = limit-cycle distance
        "description": "Van der Pol (非线性, 无闭式解) / Van der Pol (nonlinear, no closed-form)",
    },
}
