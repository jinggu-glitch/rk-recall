# 希伯来书 11:1 —— "信就是所望之事的实底,是未见之事的确据."
# Hebrews 11:1 — "Now faith is the substance of things hoped for, the evidence of things not seen."
# §3.26.15 独立验证层 (DOP853, 仅测试用, 消除自我参考)
# §3.26.15 Independent validation layer (DOP853, test-only, eliminate self-reference)
"""
独立验证层 (§3.26.15): 用 DOP853 高精度参考做第三方验证, 消除自我参考.

Independent validation layer (§3.26.15): third-party validation via DOP853
high-precision reference, eliminating self-reference.

诚实架构 / Honest architecture:
  - 算法核心 (rk_recall_compensation.py): 无 oracle, 无 scipy, 纯 numpy
  - 验证层 (本模块): 可用 scipy, 仅测试用, 不参与校正
  - 这是"验证用 oracle, 算法无 oracle"的诚实分离
  - Algorithm core (rk_recall_compensation.py): no oracle, no scipy, pure numpy
  - Validation layer (this module): may use scipy, test-only, not for correction
  - This is the honest separation of "validation with oracle, algorithm without oracle"

理论依据 / Theoretical basis:
  - DOP853 是 8 阶显式 RK, 误差远低于 RK4
  - DOP853 is 8th-order explicit RK, error far below RK4
  - 用作"独立真值"验证 §3.26 的校正效果
  - Used as "independent ground truth" to validate §3.26 correction

反偶像声明 / Anti-idolatry declaration:
  - DOP853 是高精度算法, 不是"绝对真值"
  - "独立验证"是消除自我参考的方法, 不是"获得真理"
  - 三层次联合仍不能证明算法"绝对正确", 只能证明"在测试场景下成立"
  - DOP853 的 8 阶精度是数学结论, 不是"神圣的精度"
  - → 验证层让测试更严谨, 但不改变算法的受造本质
"""
from __future__ import annotations

from typing import Callable

import numpy as np

# §从算法核心导入必要工具 (不引入 scipy) / import necessary tools from core (no scipy)
from rk_recall.rk_recall_compensation import (
    EPS_LOG,
    MissingOptionalDependencyError,
    _compute_accumulation_rate,
    _compute_error_array,
)


def independent_error_dop853(
    y_array: np.ndarray,
    t_array: np.ndarray,
    f: Callable[..., np.ndarray],
    y0: np.ndarray,
    *args,
) -> np.ndarray:
    """用 DOP853 独立验证 (仅测试, 不参与校正) (§3.26.15).

    Independent validation via DOP853 (test-only, not for correction) (§3.26.15).

    理论依据 / Theoretical basis:
      - DOP853 是 8 阶显式 RK, 误差远低于 RK4
      - 用作"独立真值"验证 §3.26 的校正效果
      - 消除 §3.26.6 极限环距离度量的自我参考风险

    诚实声明 / Honest disclosure:
      - 本函数依赖 scipy, 不是"无 oracle"
      - 但本函数不参与校正, 只用于测试
      - 算法核心仍是无 oracle 的 §3.26 机制
      - 这是"验证用 oracle, 算法无 oracle"的诚实分离

    Args:
        y_array: 待验证的轨迹 (n, dim) / trajectory to validate
        t_array: 时间序列 / time sequence
        f: ODE 右端 / ODE right-hand side
        y0: 初始状态 / initial state
        *args: 传递给 f 的额外参数 / extra args

    Returns:
        error_array: 独立误差轨迹 ‖y - DOP853(t)‖
            error_array: independent error trajectory ‖y - DOP853(t)‖

    Raises:
        MissingOptionalDependencyError: 若 scipy 不可用 / if scipy is unavailable
    """
    try:
        from scipy.integrate import solve_ivp
    except ImportError as e:
        raise MissingOptionalDependencyError(
            "scipy",
            "pip install scipy (DOP853 独立验证需要 scipy / DOP853 validation requires scipy)",
        ) from e

    y_array = np.asarray(y_array, dtype=np.float64)
    t_array = np.asarray(t_array, dtype=np.float64)
    y0 = np.asarray(y0, dtype=np.float64).copy()

    # §DOP853 高精度积分 (8 阶, rtol=1e-12) / DOP853 high-precision integration
    # §包装 f 以适配 scipy (scipy 要求 f(t, y), 不允许额外位置参数)
    # §wrap f for scipy (scipy requires f(t, y), no extra positional args)
    if args:
        def f_wrapped(t, y, _f=f, _args=args):
            return _f(t, y, *_args)
    else:
        f_wrapped = f  # type: ignore[assignment]

    sol = solve_ivp(
        f_wrapped,
        [float(t_array[0]), float(t_array[-1])],
        y0,
        method='DOP853',
        t_eval=t_array,
        rtol=1e-12,
        atol=1e-12,
    )

    if not sol.success or sol.y.shape[1] != len(t_array):
        # §DOP853 失败, 返回 NaN 警示 / DOP853 failed, return NaN to warn
        return np.full(len(t_array), np.nan)

    # §独立误差: ‖y - DOP853(t)‖ / independent error: ‖y - DOP853(t)‖
    diff = y_array - sol.y.T
    return np.linalg.norm(diff, axis=1)


def three_layer_error_analysis(
    t_array: np.ndarray,
    y_array: np.ndarray,
    f: Callable[..., np.ndarray],
    y0: np.ndarray,
    reference: Callable[[float], np.ndarray] | None = None,
    cycle_points: np.ndarray | None = None,
    *args,
) -> dict:
    """三层次误差度量 (§3.26.15): 自我参考 / 解析解 / DOP853 独立.

    Three-layer error analysis (§3.26.15): self-reference / analytical / DOP853 independent.

    层次 / Layers:
      1. 自我参考 (§3.26.6): error_Γ(y) = min_{z∈Γ} ‖y-z‖
         - Γ 来自 baseline 自身, 有伪 0 风险, 但无 oracle
      2. 解析解 (仅谐振子): error_exact(y, t) = ‖y - exact(t)‖
         - 无伪 0, 但仅适用于有解析解的系统
      3. DOP853 独立 (本节): error_DOP853(y, t) = ‖y - DOP853(t)‖
         - 无伪 0, 适用于任意系统, 但依赖 scipy

    Args:
        t_array: 时间序列 / time sequence
        y_array: 待验证轨迹 / trajectory to validate
        f: ODE 右端 / ODE right-hand side
        y0: 初始状态 / initial state
        reference: 解析解 (可选) / analytical solution (optional)
        cycle_points: 极限环采样点 (可选) / limit cycle samples (optional)
        *args: 传递给 f 的额外参数 / extra args

    Returns:
        analysis: dict 含三层次误差和增长率
            analysis: dict with three-layer errors and growth rates
    """
    t_array = np.asarray(t_array, dtype=np.float64)
    y_array = np.asarray(y_array, dtype=np.float64)

    # §层次 1: 自我参考 (Γ 距离) / Layer 1: self-reference (Γ distance)
    err_gamma = _compute_error_array(t_array, y_array, None, cycle_points)
    rate_gamma = _compute_accumulation_rate(err_gamma, len(t_array) - 1)

    # §层次 2: 解析解 (若有) / Layer 2: analytical (if available)
    if reference is not None:
        err_exact = _compute_error_array(t_array, y_array, reference, None)
        rate_exact = _compute_accumulation_rate(err_exact, len(t_array) - 1)
    else:
        err_exact = np.full(len(t_array), np.nan)
        rate_exact = float('nan')

    # §层次 3: DOP853 独立 / Layer 3: DOP853 independent
    try:
        err_dop853 = independent_error_dop853(y_array, t_array, f, y0, *args)
        rate_dop853 = _compute_accumulation_rate(err_dop853, len(t_array) - 1)
    except MissingOptionalDependencyError:
        err_dop853 = np.full(len(t_array), np.nan)
        rate_dop853 = float('nan')

    return {
        "layer1_gamma_distance": {
            "error_array": err_gamma,
            "peak": float(np.nanmax(err_gamma)),
            "final": float(err_gamma[-1]),
            "growth_rate": rate_gamma,
        },
        "layer2_exact": {
            "error_array": err_exact,
            "peak": float(np.nanmax(err_exact)) if not np.all(np.isnan(err_exact)) else float('nan'),
            "final": float(err_exact[-1]) if not np.isnan(err_exact[-1]) else float('nan'),
            "growth_rate": rate_exact,
        },
        "layer3_dop853": {
            "error_array": err_dop853,
            "peak": float(np.nanmax(err_dop853)) if not np.all(np.isnan(err_dop853)) else float('nan'),
            "final": float(err_dop853[-1]) if not np.isnan(err_dop853[-1]) else float('nan'),
            "growth_rate": rate_dop853,
        },
    }
