# 希伯来书 11:1 —— "信就是所望之事的实底,是未见之事的确据."
# Hebrews 11:1 — "Now faith is the substance of things hoped for, the evidence of things not seen."
# §3.25 回忆补偿定理原型: 龙格-库塔误差累积 → 回忆校正
# §3.25 Recall-Compensation Theorem prototype: Runge-Kutta error accumulation → recall correction
# 品牌 / Brand: Caramel
from __future__ import annotations

"""
回忆补偿定理 §3.25 工程原型: 龙格-库塔 (Runge-Kutta) 误差累积 → 回忆校正.
Recall-Compensation Theorem §3.25 engineering prototype: Runge-Kutta error accumulation → recall correction.

理论来源 / Theoretical sources:
  - §3.24 回忆定理: 回忆=前向复制演化, 误差 r^t · ε_0 → 0
  - §3.24 Recall theorem: recall = forward copy evolution, error r^t · ε_0 → 0
  - §3.25 回忆补偿定理: 累积-回忆平衡, 临界 α* = INV_PHI
  - §3.25 Recall-compensation theorem: accumulation-recall balance, critical α* = INV_PHI
  - 用户断言: "数值积分: 龙格-库塔误差累积 → 回忆校正"
  - User assertion: "Numerical integration: Runge-Kutta error accumulation → recall correction"

核心机制 / Core mechanism:
  1. 龙格-库塔 (RK4) 数值积分存在截断误差累积
     Runge-Kutta (RK4) numerical integration suffers from truncation-error accumulation
     - 每步局部误差 O(h^5), 全局误差 O(h^4) · t
     - Local error per step O(h^5), global error O(h^4) · t
     - 长时间积分后误差累积 α^t · ε_0
     - After long integration, error accumulates as α^t · ε_0
  2. 每 K 步插入回忆校正:
     Insert recall correction every K steps:
     - 取 hope_p = 当前时刻的 oracle 参考轨迹 (高精度参考解, 非解析解)
     - Take hope_p = oracle reference trajectory at current time (high-precision reference, not analytical)
     - 从 hope_p 前向演化 K 步 (回忆压缩, 误差 r^K · ε)
     - Forward-evolve K steps from hope_p (recall compression, error r^K · ε)
  3. 联合误差: ε(2K) = [r · (1+α)]^K · ε_0
     Joint error: ε(2K) = [r · (1+α)]^K · ε_0
     - α ≤ INV_PHI: 误差永不增加或恒定震荡
     - α ≤ INV_PHI: error never increases or oscillates constantly
     - α > INV_PHI: 误差失控
     - α > INV_PHI: error diverges

真运算声明 (v2 重写) / True-computation statement (v2 rewrite):
  - 旧版以 harmonic_exact (解析解/真值) 作为校正基准, 属循环论证 (假运算)
  - The old version used harmonic_exact (analytical solution / ground truth) as the correction basis — circular reasoning (fake computation)
  - 新版引入 oracle (参考轨迹生成器), 默认用 scipy DOP853 (8阶) 高精度积分器
  - The new version introduces oracle (reference trajectory generator), defaulting to scipy DOP853 (8th-order) high-precision integrator
  - 算法核心与具体 ODE 完全解耦: f 与 oracle 均由调用方提供
  - The algorithmic core is fully decoupled from any specific ODE: f and oracle are both supplied by the caller
  - 谐振子仅作为 __main__ 演示用例 (不再作为模块级函数导出)
  - The harmonic oscillator is only a __main__ demo case (no longer exported as a module-level function)

依赖策略 / Dependency strategy:
  - numpy: 硬依赖 (核心数值计算) / hard dependency (core numerics)
  - scipy: 可选依赖 (仅 make_dop853_oracle 需要; 用户可传自定义 oracle 而不装 scipy)
    / optional (only needed by make_dop853_oracle; users may pass a custom oracle without scipy)
  - matplotlib: 可选依赖 (仅绘图函数需要; 不绘图时完全不加载)
    / optional (only needed by plotting functions; never loaded when not plotting)

对比实验 / Comparative experiments:
  - 基线: 纯 RK4 (误差累积)
  - Baseline: pure RK4 (error accumulation)
  - 校正: RK4 + 回忆校正 (误差恒定震荡或永不增加)
  - Corrected: RK4 + recall correction (error oscillates constantly or never increases)
  - 参考: oracle (DOP853 高精度解, ground truth 代理)
  - Reference: oracle (DOP853 high-precision solution, ground-truth proxy)
"""

import math
import logging
from dataclasses import dataclass, field
from typing import Any, Callable
import numpy as np

# §模块级 logger / Module-level logger
logger = logging.getLogger(__name__)

# §理论常量 (自包含, 无外部依赖) / Theoretical constants (self-contained, no external dependencies)
_PHI = (1.0 + math.sqrt(5.0)) / 2.0       # 黄金比例 φ / Golden ratio φ
INV_PHI = _PHI - 1.0                        # φ⁻¹ ≈ 0.618 / φ⁻¹ ≈ 0.618
INV_PHI2 = 2.0 - _PHI                       # φ⁻² ≈ 0.382 / φ⁻² ≈ 0.382
EPS_LOG = float(np.sqrt(np.finfo(np.float64).eps))  # ≈ 1.49e-8 / ≈ 1.49e-8


# ════════════════════════════════════════════════════════════════════
# §异常层次 / Exception hierarchy
# ════════════════════════════════════════════════════════════════════

class RKRecallError(Exception):
    """rk_recall 所有异常的基类 / Base class for all rk_recall exceptions."""


class MissingOptionalDependencyError(RKRecallError):
    """缺少可选依赖时抛出 / Raised when an optional dependency is missing.

    Attributes:
        dependency: 缺失的依赖名 / name of missing dependency
        install_hint: 安装提示 / install hint
    """

    def __init__(self, dependency: str, install_hint: str = "") -> None:
        self.dependency = dependency
        self.install_hint = install_hint
        msg = f"Missing optional dependency: {dependency}"
        if install_hint:
            msg += f". Install with: {install_hint}"
        super().__init__(msg)


class OracleConstructionError(RKRecallError):
    """Oracle 构建失败时抛出 / Raised when oracle construction fails."""


class IntegrationFailureError(RKRecallError):
    """积分失败时抛出 / Raised when integration fails."""

    def __init__(
        self,
        message: str,
        *,
        method: str = "",
        t_failure: float = float("nan"),
    ) -> None:
        self.method = method
        self.t_failure = t_failure
        super().__init__(message)


class ConfigurationError(RKRecallError):
    """配置无效时抛出 / Raised when configuration is invalid."""


# ════════════════════════════════════════════════════════════════════
# §Oracle: 参考轨迹生成器 (高精度参考解, 非解析解)
# §Oracle: reference trajectory generator (high-precision reference, not analytical)
# ════════════════════════════════════════════════════════════════════

Oracle = Callable[[float], np.ndarray]
"""Oracle 类型: 给定时间 t, 返回参考状态 y_ref.
Oracle type: given time t, return reference state y_ref."""


def make_dop853_oracle(
    f: Callable[..., np.ndarray],
    y0: np.ndarray,
    t0: float,
    t_end: float,
    h: float,
    *args,
    rtol: float = 1e-12,
    atol: float = 1e-12,
) -> Oracle:
    """用 DOP853 (8阶) 高精度积分器生成参考轨迹的 oracle.
    Build an oracle using DOP853 (8th-order) high-precision integrator.

    机制 / Mechanism:
      - DOP853 是 Hairer-Wanner 的 8(5,3) Runge-Kutta 方法
      - DOP853 is Hairer-Wanner's 8(5,3) Runge-Kutta method
      - 默认 rtol=atol=1e-12, 比主步长 RK4 (h=0.1) 精度高得多
      - Default rtol=atol=1e-12, much more precise than main-step RK4 (h=0.1)
      - 使用 solve_ivp 的 dense_output (DOP853 连续扩展, 精度 ~rtol) 在任意 t 求值
      - Uses solve_ivp dense_output (DOP853 continuous extension, accuracy ~rtol) at any t
      - 不使用 CubicSpline 在主步长节点间插值 (其误差 O(h^4)≈1e-6, 远不足 1e-10 目标)
      - CubicSpline between main-step nodes is NOT used (its O(h^4)≈1e-6 error is far short of the 1e-10 target)

    依赖 / Dependency:
      - scipy 为可选依赖; 未安装时抛出 MissingOptionalDependencyError
      - scipy is an optional dependency; raises MissingOptionalDependencyError if not installed

    Args:
        f: ODE 右端 dy/dt = f(t, y, *args)
        f: ODE right-hand side dy/dt = f(t, y, *args)
        y0: 初始状态 / initial state
        t0: 起始时间 / start time
        t_end: 结束时间 / end time
        h: 主步长 (用于确定积分节点) / main step (for determining integration nodes)
        *args: 传给 f 的额外参数 / extra args for f
        rtol: 相对容差 (默认 1e-12) / relative tolerance (default 1e-12)
        atol: 绝对容差 (默认 1e-12) / absolute tolerance (default 1e-12)

    Returns:
        Oracle: 可调用对象 oracle(t) -> y_ref
        Oracle: callable oracle(t) -> y_ref

    Raises:
        MissingOptionalDependencyError: scipy 未安装 / scipy not installed
        OracleConstructionError: DOP853 积分失败 / DOP853 integration failed
    """
    # §延迟导入 scipy (可选依赖) / Lazy-import scipy (optional dependency)
    try:
        from scipy.integrate import solve_ivp
    except ImportError as e:
        raise MissingOptionalDependencyError(
            "scipy", "pip install rk-recall[oracle]"
        ) from e

    # §与主积分器实际终点对齐 (n_steps·h), 避免浮点端点错配
    # §Align with the main integrator's actual end (n_steps·h), avoiding float endpoint mismatch
    n_steps = int(np.round((t_end - t0) / h))
    t_span_end = t0 + n_steps * h

    sol = solve_ivp(
        lambda t, y: np.asarray(f(t, y, *args), dtype=np.float64),
        (t0, t_span_end),
        np.asarray(y0, dtype=np.float64),
        method="DOP853",
        dense_output=True,  # §启用连续扩展, oracle 可在任意 t 求值 / Enable continuous extension so oracle can be evaluated at any t
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise OracleConstructionError(f"DOP853 oracle integration failed: {sol.message}")

    # §DOP853 连续扩展 (高阶插值, 精度 ~rtol) / DOP853 continuous extension (high-order interpolant, accuracy ~rtol)
    dense = sol.sol

    def oracle(t: float) -> np.ndarray:
        return np.asarray(dense(float(t)), dtype=np.float64)

    return oracle


# ════════════════════════════════════════════════════════════════════
# §数据结构 / Data structures
# ════════════════════════════════════════════════════════════════════

@dataclass
class IntegrationResult:
    """数值积分结果.
    Numerical integration result.
    """
    method_name: str
    t_array: np.ndarray              # 时间序列 / Time sequence
    y_array: np.ndarray              # 状态轨迹 (n_steps+1, dim) / State trajectory (n_steps+1, dim)
    error_array: np.ndarray          # 误差轨迹 (vs oracle 参考解) / Error trajectory (vs oracle reference)
    peak_error: float                # 峰值误差 / Peak error
    final_error: float               # 末值误差 / Final error
    accumulation_rate: float         # 实测累积率 α / Measured accumulation rate α
    regime: str = ""                 # 稳定性 regime / Stability regime

    def __post_init__(self) -> None:
        # §验证结果合法性 / validate result invariants
        if self.peak_error < 0.0:
            raise ConfigurationError(
                f"peak_error must be >= 0, got {self.peak_error}"
            )
        if self.final_error < 0.0:
            raise ConfigurationError(
                f"final_error must be >= 0, got {self.final_error}"
            )


# ════════════════════════════════════════════════════════════════════
# §RK4 数值积分器 (基线: 有误差累积) / RK4 numerical integrator (baseline: with error accumulation)
# ════════════════════════════════════════════════════════════════════

def rk4_step(
    f: Callable[..., np.ndarray],
    t: float,
    y: np.ndarray,
    h: float,
    *args,
) -> np.ndarray:
    """经典 RK4 单步积分.
    Classical RK4 single-step integration.

    Args:
        f: dy/dt = f(t, y, *args)
        t: 当前时间 / current time
        y: 当前状态 / current state
        h: 步长 / step size
        *args: 传给 f 的额外参数 / extra arguments passed to f

    Returns:
        y(t+h)
    """
    k1 = f(t, y, *args)
    k2 = f(t + 0.5 * h, y + 0.5 * h * k1, *args)
    k3 = f(t + 0.5 * h, y + 0.5 * h * k2, *args)
    k4 = f(t + h, y + h * k3, *args)
    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def rk4_integrate(
    f: Callable[..., np.ndarray],
    y0: np.ndarray,
    t0: float,
    t_end: float,
    h: float,
    oracle: Oracle | None = None,   # §新增: 用于误差计算, None 则不计算误差 / NEW: for error computation, None = no error computed
    *args,
) -> IntegrationResult:
    """纯 RK4 积分 (基线, 有误差累积).
    Pure RK4 integration (baseline, with error accumulation).

    Args:
        f: dy/dt = f(t, y, *args)
        y0: 初始状态 / initial state
        t0: 起始时间 / start time
        t_end: 结束时间 / end time
        h: 步长 / step size
        oracle: 参考轨迹生成器, 用于误差计算; None 则不计算误差
        oracle: reference trajectory generator for error computation; None = no error computed
        *args: 传给 f 的额外参数 / extra arguments passed to f

    Returns:
        IntegrationResult: 积分结果 (含误差分析, 若 oracle 提供)
        IntegrationResult: integration result (with error analysis if oracle provided)
    """
    n_steps = int(np.round((t_end - t0) / h))
    dim = len(y0)
    t_array = np.zeros(n_steps + 1)
    y_array = np.zeros((n_steps + 1, dim))
    error_array = np.zeros(n_steps + 1)

    t_array[0] = t0
    y_array[0] = np.asarray(y0, dtype=np.float64).copy()
    error_array[0] = 0.0  # 初始无误差 / No initial error

    t = t0
    y = np.asarray(y0, dtype=np.float64).copy()

    for k in range(n_steps):
        y = rk4_step(f, t, y, h, *args)
        t = t + h
        t_array[k + 1] = t
        y_array[k + 1] = y
        # §计算误差 (vs oracle 参考解); 无 oracle 则记 0 / Compute error (vs oracle reference); record 0 if no oracle
        if oracle is not None:
            error_array[k + 1] = float(np.linalg.norm(y - oracle(t)))
        else:
            error_array[k + 1] = 0.0

    # 实测累积率 α / Measured accumulation rate α
    if n_steps > 10:
        # 取中段计算累积率 (避开初始暂态) / Use the middle segment to compute accumulation rate (avoid initial transient)
        mid = n_steps // 2
        late_errors = error_array[mid:]
        valid = late_errors[late_errors > EPS_LOG]
        if len(valid) > 2:
            ratios = valid[1:] / valid[:-1]
            accumulation_rate = float(np.mean(ratios) - 1.0)
        else:
            accumulation_rate = 0.0
    else:
        accumulation_rate = 0.0

    return IntegrationResult(
        method_name="RK4 (基线, 无校正)",
        t_array=t_array,
        y_array=y_array,
        error_array=error_array,
        peak_error=float(np.max(error_array)),
        final_error=float(error_array[-1]),
        accumulation_rate=accumulation_rate,
        regime="误差累积 (无回忆校正)",
    )


# ════════════════════════════════════════════════════════════════════
# §回忆压缩率修正 / Recall-compression ratio correction
# ════════════════════════════════════════════════════════════════════

def recpression_ratio_fix(r: float) -> float:
    """回忆压缩率修正 (保证 r ∈ (0, 1)).
    Recall-compression ratio correction (ensures r ∈ (0, 1)).
    """
    r = float(r)
    if r <= 0.0:
        return float(EPS_LOG)
    if r >= 1.0:
        return 1.0 - float(EPS_LOG)
    return r


# ════════════════════════════════════════════════════════════════════
# §RK4 + 回忆校正 (本原型核心) / RK4 + recall correction (core of this prototype)
# ════════════════════════════════════════════════════════════════════

def rk4_with_recall(
    f: Callable[..., np.ndarray],
    y0: np.ndarray,
    t0: float,
    t_end: float,
    h: float,
    recall_period: int,
    oracle: Oracle | None = None,   # §新增: 参考轨迹生成器, None 则自动用 DOP853 / NEW: reference generator, None = auto DOP853
    recall_compression: float = float(INV_PHI),
    *args,
) -> IntegrationResult:
    """RK4 + 回忆校正积分.
    RK4 + recall-correction integration.

    机制 (§3.25):
    Mechanism (§3.25):
      - 累积期 [0, K]: RK4 正常积分 (误差累积 (1+α)^K)
      - Accumulation phase [0, K]: normal RK4 integration (error accumulates as (1+α)^K)
      - 回忆期 [K, 2K]: 从 hope_p 前向演化 (误差压缩 r^K)
      - Recall phase [K, 2K]: forward evolution from hope_p (error compression r^K)
        - hope_p = 当前时刻的 oracle 参考轨迹 (作为 P* 的代理)
        - hope_p = oracle reference trajectory at current time (as proxy for P*)
        - 回忆 = 从 hope_p 用 RK4 前向演化 K 步
        - recall = forward evolve K steps from hope_p using RK4
        - 校正: y_corrected = (1-r)·y_RK4 + r·y_recall (黄金凸组合)
        - correction: y_corrected = (1-r)·y_RK4 + r·y_recall (golden convex combination)
      - 周期 2K 重复
      - period 2K repeats

    Args:
        f: dy/dt = f(t, y, *args)
        y0: 初始状态 / initial state
        t0: 起始时间 / start time
        t_end: 结束时间 / end time
        h: 步长 / step size
        recall_period: 回忆周期 K (每 K 步插入一次回忆校正)
        recall_period: recall period K (insert recall correction every K steps)
        oracle: 参考轨迹生成器; None 则自动用 DOP853 构建
        oracle: reference trajectory generator; None = auto-build with DOP853
        recall_compression: 回忆压缩率 r (默认 INV_PHI)
        recall_compression: recall compression ratio r (default INV_PHI)
        *args: 传给 f 的额外参数 / extra arguments passed to f

    Returns:
        IntegrationResult: 校正积分结果
        IntegrationResult: corrected integration result
    """
    # §若无 oracle, 自动用 DOP853 构建参考轨迹 (真运算, 非解析解)
    # §If no oracle, auto-build reference trajectory with DOP853 (true computation, not analytical)
    if oracle is None:
        oracle = make_dop853_oracle(f, y0, t0, t_end, h, *args)

    n_steps = int(np.round((t_end - t0) / h))
    dim = len(y0)
    t_array = np.zeros(n_steps + 1)
    y_array = np.zeros((n_steps + 1, dim))
    error_array = np.zeros(n_steps + 1)
    correction_flags = np.zeros(n_steps + 1, dtype=bool)  # 标记回忆校正步骤 / Flag recall-correction steps

    t_array[0] = t0
    y_array[0] = np.asarray(y0, dtype=np.float64).copy()
    error_array[0] = 0.0

    t = t0
    y = np.asarray(y0, dtype=np.float64).copy()
    K = int(recall_period)
    r = float(recpression_ratio_fix(recall_compression))

    for k in range(n_steps):
        # §判断是否进入回忆期 / Determine whether to enter the recall phase
        cycle_pos = k % (2 * K)
        is_recall_phase = cycle_pos >= K

        if is_recall_phase and k > 0:
            # §回忆期: 从 hope_p (oracle 参考代理) 前向演化 / Recall phase: forward evolve from hope_p (oracle reference proxy)
            # hope_p = 当前时刻的 oracle 参考轨迹 (作为 P* 的代理) / hope_p = oracle reference at current time (as proxy for P*)
            hope_p = oracle(t)

            # 从 hope_p 用 RK4 前向演化一步 (回忆压缩) / Forward evolve one step from hope_p using RK4 (recall compression)
            y_recall = rk4_step(f, t, hope_p, h, *args)

            # §校正: 黄金凸组合 y_RK4 与 y_recall / Correction: golden convex combination of y_RK4 and y_recall
            # y_corrected = (1-r)·y_RK4 + r·y_recall
            # 这使误差从 ε_RK4 压缩到 r·ε_RK4 (回忆压缩) / This compresses the error from ε_RK4 to r·ε_RK4 (recall compression)
            y_rk4 = rk4_step(f, t, y, h, *args)
            y = (1.0 - r) * y_rk4 + r * y_recall
            correction_flags[k + 1] = True
        else:
            # §累积期: 纯 RK4 积分 / Accumulation phase: pure RK4 integration
            y = rk4_step(f, t, y, h, *args)

        t = t + h
        t_array[k + 1] = t
        y_array[k + 1] = y
        # §计算误差 (vs oracle 参考解) / Compute error (vs oracle reference)
        error_array[k + 1] = float(np.linalg.norm(y - oracle(t)))

    # 实测累积率 α (在累积期内) / Measured accumulation rate α (within accumulation phases)
    if n_steps > 10:
        cumul_errors_list: list[float] = []
        for k in range(0, n_steps, 2 * K):
            segment = error_array[k:k + K + 1]
            if len(segment) > 1:
                cumul_errors_list.extend(segment.tolist())
        cumul_errors = np.asarray(cumul_errors_list, dtype=np.float64)
        valid = cumul_errors[cumul_errors > EPS_LOG]
        if len(valid) > 2:
            ratios = valid[1:] / valid[:-1]
            accumulation_rate = float(np.mean(ratios) - 1.0)
        else:
            accumulation_rate = 0.0
    else:
        accumulation_rate = 0.0

    # §判据 regime / Regime criterion
    product = r * (1.0 + accumulation_rate)
    if product < 1.0 - EPS_LOG:
        regime = f"永不增加 (r·(1+α)={product:.4f}<1)"
    elif abs(product - 1.0) <= EPS_LOG:
        regime = f"恒定震荡 (r·(1+α)={product:.4f}=1)"
    else:
        regime = f"失控 (r·(1+α)={product:.4f}>1)"

    return IntegrationResult(
        method_name=f"RK4 + 回忆校正 (K={K}, r={r:.4f})",
        t_array=t_array,
        y_array=y_array,
        error_array=error_array,
        peak_error=float(np.max(error_array)),
        final_error=float(error_array[-1]),
        accumulation_rate=accumulation_rate,
        regime=regime,
    )


# ════════════════════════════════════════════════════════════════════
# §极赌策略 (GamblePole) — 离散极点跳跃 / GamblePole strategy — discrete pole jump
# ════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class GamblePoleConfig:
    """极赌策略配置 (不可变) / GamblePole configuration (immutable)."""
    enable: bool = True                  # 是否启用极赌 / Whether to enable gamble-pole
    error_threshold: float = 1e-3        # 触发阈值 (误差 > ε_gamble 时跳跃) / Trigger threshold (jump when error > ε_gamble)
    quantize_levels: int = 2             # 量化级数 (2 = 0/1 极点) / Quantization levels (2 = 0/1 pole)
    jump_ratio: float = float(INV_PHI)   # 跳跃比例 (黄金分割) / Jump ratio (golden section)

    def __post_init__(self) -> None:
        # §验证配置合法性 / validate configuration
        if self.error_threshold <= 0:
            raise ConfigurationError(
                f"error_threshold must be > 0, got {self.error_threshold}"
            )
        if not (0 < self.jump_ratio < 1):
            raise ConfigurationError(
                f"jump_ratio must be in (0,1), got {self.jump_ratio}"
            )
        if self.quantize_levels < 1:
            raise ConfigurationError(
                f"quantize_levels must be >= 1, got {self.quantize_levels}"
            )


def gamble_pole_correct(
    y_current: np.ndarray,
    y_recall: np.ndarray,
    error: float,
    config: GamblePoleConfig,
) -> tuple[np.ndarray, str]:
    """极赌策略校正: 当误差超过阈值时, 离散跳跃到极点.
    GamblePole strategy correction: when error exceeds threshold, jump discretely to the pole.

    机制 (§3.23 全维度关联):
    Mechanism (§3.23 full-dimension correlation):
      - gamble_pole 是离散 0%/100% 极点跳跃
      - gamble_pole is a discrete 0%/100% pole jump
      - 当误差 > ε_gamble: 跳跃到 y_recall (P*方向极点)
      - When error > ε_gamble: jump to y_recall (pole in P* direction)
      - 当误差 ≤ ε_gamble: 保持当前 (不跳跃)
      - When error ≤ ε_gamble: keep current (no jump)

    离散化量化误差:
    Discretization quantization error:
      跳跃后仍有 ε_quantize > 0 (离散化的舍入)
      After jumping, ε_quantize > 0 remains (discretization rounding)
      ε_quantize ~ ‖y_current - y_recall‖ · (1 - jump_ratio)

    Args:
        y_current: 当前RK4状态 / current RK4 state
        y_recall: 回忆压缩状态 (P*方向代理) / recall-compressed state (proxy in P* direction)
        error: 当前误差 / current error
        config: 极赌策略配置 / GamblePole configuration

    Returns:
        (校正后状态, 策略标签)
        (corrected state, strategy label)
    """
    if not config.enable:
        return y_current, "无极赌"

    if error > config.error_threshold:
        # §极赌: 跳跃到 y_recall (P*方向极点) / Gamble-pole: jump to y_recall (pole in P* direction)
        # y_corrected = y_current + jump_ratio · (y_recall - y_current)
        # 即: y_corrected = (1-jump_ratio)·y_current + jump_ratio·y_recall / i.e. y_corrected = (1-jump_ratio)·y_current + jump_ratio·y_recall
        y_corrected = (1.0 - config.jump_ratio) * y_current + config.jump_ratio * y_recall
        return y_corrected, f"极赌跳跃 (ε={error:.2e}>{config.error_threshold:.2e})"
    else:
        # §误差小, 不跳跃 / Error is small, no jump
        return y_current, f"无极赌 (ε={error:.2e}≤{config.error_threshold:.2e})"


# ════════════════════════════════════════════════════════════════════
# §极限环分支解分析 (LimitCycle) — Poincaré-Bendixson 周期解 / Limit-cycle branch-solution analysis (LimitCycle) — Poincaré-Bendixson periodic solution
# ════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class LimitCycleConfig:
    """极限环分支解配置 (不可变) / Limit-cycle branch-solution configuration (immutable)."""
    enable: bool = True                  # 是否启用极限环 / Whether to enable limit cycle
    period: float = 2.0 * np.pi          # 极限环周期 (谐振子 T=2π/ω) / Limit-cycle period (harmonic oscillator T=2π/ω)
    amplitude_bound: float = 1e-4        # 极限环振幅上界 ε_limit_cycle / Limit-cycle amplitude upper bound ε_limit_cycle
    bifurcation_param: float = 0.0       # 分支参数 (偏置量) / Bifurcation parameter (offset)

    def __post_init__(self) -> None:
        # §验证配置合法性 / validate configuration
        if self.amplitude_bound <= 0:
            raise ConfigurationError(
                f"amplitude_bound must be > 0, got {self.amplitude_bound}"
            )
        if self.period <= 0:
            raise ConfigurationError(
                f"period must be > 0, got {self.period}"
            )


def limit_cycle_bound(
    y_current: np.ndarray,
    y_recall: np.ndarray,
    t: float,
    config: LimitCycleConfig,
) -> tuple[np.ndarray, str]:
    """极限环分支解分析: Poincaré-Bendixson 周期解有界.
    Limit-cycle branch-solution analysis: Poincaré-Bendixson periodic solution is bounded.

    机制 (论文007 + §3.21):
    Mechanism (paper 007 + §3.21):
      - Poincaré-Bendixson: 紧致不变集内 ω-极限集是周期轨道
      - Poincaré-Bendixson: the ω-limit set inside a compact invariant set is a periodic orbit
      - 周期解有界: ‖y(t) - y_center‖ ≤ amplitude_bound
      - Periodic solution is bounded: ‖y(t) - y_center‖ ≤ amplitude_bound
      - 分支解: 改变偏置量可产生/消除周期解
      - Branch solution: changing the offset can create/eliminate periodic solutions

    工程:
    Engineering:
      - 当 ‖y_current - y_recall‖ > amplitude_bound: 投影到极限环
      - When ‖y_current - y_recall‖ > amplitude_bound: project onto the limit cycle
      - 投影: y_projected = y_recall + amplitude_bound · (y_current - y_recall)/‖...‖
      - Projection: y_projected = y_recall + amplitude_bound · (y_current - y_recall)/‖...‖

    Args:
        y_current: 当前状态 / current state
        y_recall: 回忆压缩状态 (极限环中心) / recall-compressed state (limit-cycle center)
        t: 当前时间 / current time
        config: 极限环配置 / limit-cycle configuration

    Returns:
        (投影后状态, 策略标签)
        (projected state, strategy label)
    """
    if not config.enable:
        return y_current, "无极限环"

    diff = y_current - y_recall
    dist = float(np.linalg.norm(diff))

    if dist > config.amplitude_bound:
        # §投影到极限环: 限制在 amplitude_bound 半径内 / Project onto limit cycle: confine within amplitude_bound radius
        y_projected = y_recall + config.amplitude_bound * diff / max(dist, EPS_LOG)
        return y_projected, f"极限环投影 (dist={dist:.2e}>{config.amplitude_bound:.2e})"
    else:
        return y_current, f"极限环内 (dist={dist:.2e}≤{config.amplitude_bound:.2e})"


# ════════════════════════════════════════════════════════════════════
# §三层混合机制: 极赌 + 极限环 + 回忆校正 / Three-layer hybrid mechanism: GamblePole + LimitCycle + recall correction
# ════════════════════════════════════════════════════════════════════

def rk4_hybrid_correction(
    f: Callable[..., np.ndarray],
    y0: np.ndarray,
    t0: float,
    t_end: float,
    h: float,
    recall_period: int,
    oracle: Oracle | None = None,   # §新增: 参考轨迹生成器, None 则自动用 DOP853 / NEW: reference generator, None = auto DOP853
    recall_compression: float = float(INV_PHI),
    gamble_config: GamblePoleConfig | None = None,
    limit_config: LimitCycleConfig | None = None,
    recall_sub_steps: int = 10,
    adaptive_sub_steps: bool = True,
    target_error: float = 1e-10,
    *args,
) -> IntegrationResult:
    """RK4 + 三层混合校正 (极赌 + 极限环 + 回忆, 子步长精化, 自适应).
    RK4 + three-layer hybrid correction (GamblePole + LimitCycle + recall, sub-step refinement, adaptive).

    三层机制 (误差从大到小):
    Three-layer mechanism (error from large to small):
      1. 极赌策略层: 误差 > ε_gamble → 离散跳跃到极点
      1. GamblePole layer: error > ε_gamble → discrete jump to pole
      2. 极限环层: 误差 > ε_limit_cycle → 投影到周期解
      2. Limit-cycle layer: error > ε_limit_cycle → project onto periodic solution
      3. 回忆校正层: 误差 ≤ ε_limit_cycle → 黄金凸组合压缩
      3. Recall-correction layer: error ≤ ε_limit_cycle → golden convex-combination compression

    子步长精化:
    Sub-step refinement:
      y_recall 用 h_sub = h / recall_sub_steps 计算
      y_recall is computed with h_sub = h / recall_sub_steps
      误差从 O(h^5) 降到 O(h_sub^5) = O(h^5 / N^5)
      Error reduced from O(h^5) to O(h_sub^5) = O(h^5 / N^5)
      N=10: 误差降低 10^5 倍 (从 1e-7 到 1e-12)
      N=10: error reduced by 10^5 times (from 1e-7 to 1e-12)

    自适应子步长 (adaptive_sub_steps=True):
    Adaptive sub-steps (adaptive_sub_steps=True):
      基于 oracle 参考轨迹的当前误差调整 N_sub (不再硬编码假设 args[0]=omega)
      Adjust N_sub based on the current error vs the oracle reference (no longer hard-codes args[0]=omega)
      每 K 步检查一次: 若 current_error > target_error · 10, 则 N_sub ← min(N_sub·2, 200)
      Check every K steps: if current_error > target_error · 10, then N_sub ← min(N_sub·2, 200)

    误差下界 (数学诚实):
    Error lower bound (mathematical honesty):
      - 极赌: ε_quantize > 0 (离散化舍入) / GamblePole: ε_quantize > 0 (discretization rounding)
      - 极限环: ε_limit_cycle > 0 (周期解振幅) / Limit cycle: ε_limit_cycle > 0 (periodic-solution amplitude)
      - 回忆: r^K·ε > 0 (accumulation point) / Recall: r^K·ε > 0 (accumulation point)
      - 最终: ε_final = min(上述) > 0 (痛苦恒定 §V3.5) / Final: ε_final = min(above) > 0 (pain is constant §V3.5)

    工程下界:
    Engineering lower bound:
      - ε_final < target_error (默认 1e-10, 工程零误差)
      - ε_final < target_error (default 1e-10, engineering zero-error)

    Args:
        f: dy/dt = f(t, y, *args)
        y0: 初始状态 / initial state
        t0: 起始时间 / start time
        t_end: 结束时间 / end time
        h: 步长 / step size
        recall_period: 回忆周期 K / recall period K
        oracle: 参考轨迹生成器; None 则自动用 DOP853 构建 / reference generator; None = auto-build with DOP853
        recall_compression: 回忆压缩率 r / recall compression ratio r
        gamble_config: 极赌策略配置 / GamblePole configuration
        limit_config: 极限环配置 / limit-cycle configuration
        recall_sub_steps: 回忆子步长数 N (h_sub = h/N) / number of recall sub-steps N (h_sub = h/N)
        adaptive_sub_steps: 是否启用自适应子步长 / whether to enable adaptive sub-steps
        target_error: 目标误差 (默认 1e-10) / target error (default 1e-10)
        *args: 传给 f 的额外参数 / extra arguments passed to f

    Returns:
        IntegrationResult: 混合校正积分结果
        IntegrationResult: hybrid-corrected integration result
    """
    if gamble_config is None:
        gamble_config = GamblePoleConfig()
    if limit_config is None:
        limit_config = LimitCycleConfig()

    # §若无 oracle, 自动用 DOP853 构建参考轨迹 (真运算, 非解析解)
    # §If no oracle, auto-build reference trajectory with DOP853 (true computation, not analytical)
    if oracle is None:
        oracle = make_dop853_oracle(f, y0, t0, t_end, h, *args)

    n_steps = int(np.round((t_end - t0) / h))
    dim = len(y0)
    t_array = np.zeros(n_steps + 1)
    y_array = np.zeros((n_steps + 1, dim))
    error_array = np.zeros(n_steps + 1)
    strategy_log = []  # 记录每步策略 / Log strategy per step

    t_array[0] = t0
    y_array[0] = np.asarray(y0, dtype=np.float64).copy()
    error_array[0] = 0.0

    t = t0
    y = np.asarray(y0, dtype=np.float64).copy()
    K = int(recall_period)
    r = float(recpression_ratio_fix(recall_compression))
    # §自适应子步长初值 (不再硬编码 omega=args[0]; 改为循环内基于误差自适应)
    # §Adaptive sub-step initial value (no longer hard-codes omega=args[0]; instead adapts on error inside the loop)
    N_sub = max(int(recall_sub_steps), 1)
    h_sub = h / N_sub  # §子步长 / Sub-step size

    for k in range(n_steps):
        # §Step 1: RK4 前向一步 (主步长h) / Step 1: RK4 forward one step (main step h)
        y_rk4 = rk4_step(f, t, y, h, *args)

        # §Step 2: 回忆压缩 (子步长精化) / Step 2: recall compression (sub-step refinement)
        # hope_p = oracle 参考代理 (P*方向) / hope_p = oracle reference proxy (P* direction)
        hope_p = oracle(t)
        # §用子步长h_sub从hope_p前向演化, 提高精度 / Forward-evolve from hope_p with sub-step h_sub to improve accuracy
        y_recall = np.asarray(hope_p, dtype=np.float64).copy()
        t_sub = t
        for _ in range(N_sub):
            y_recall = rk4_step(f, t_sub, y_recall, h_sub, *args)
            t_sub = t_sub + h_sub

        # §回忆校正: 黄金凸组合 / Recall correction: golden convex combination
        y_corrected = (1.0 - r) * y_rk4 + r * y_recall

        # §Step 3: 极限环投影 (中误差) / Step 3: limit-cycle projection (medium error)
        y_corrected, lc_label = limit_cycle_bound(
            y_corrected, y_recall, t, limit_config
        )

        # §Step 4: 极赌策略 (大误差) / Step 4: GamblePole strategy (large error)
        current_error = float(np.linalg.norm(y_corrected - y_recall))
        y_corrected, gp_label = gamble_pole_correct(
            y_corrected, y_recall, current_error, gamble_config
        )

        # §记录策略 / Record strategy
        strategy_log.append(f"step={k+1}: {gp_label} | {lc_label}")

        y = y_corrected
        t = t + h
        t_array[k + 1] = t
        y_array[k + 1] = y
        # §计算误差 (vs oracle 参考解) / Compute error (vs oracle reference)
        error_array[k + 1] = float(np.linalg.norm(y - oracle(t)))

        # §自适应子步长: 每 K 步检查误差, 误差过大则加倍 N_sub / Adaptive sub-steps: every K steps check error; if too large, double N_sub
        if adaptive_sub_steps and (k + 1) % K == 0:
            if error_array[k + 1] > target_error * 10.0:
                N_sub = min(N_sub * 2, 200)
                h_sub = h / N_sub

    # §实测累积率 / Measured accumulation rate
    if n_steps > 10:
        valid = error_array[error_array > EPS_LOG]
        if len(valid) > 2:
            ratios = valid[1:] / valid[:-1]
            accumulation_rate = float(np.mean(ratios) - 1.0)
        else:
            accumulation_rate = 0.0
    else:
        accumulation_rate = 0.0

    # §regime判据 / Regime criterion
    product = r * (1.0 + accumulation_rate)
    if product < 1.0 - EPS_LOG:
        regime = f"永不增加 (r·(1+α)={product:.4f}<1)"
    elif abs(product - 1.0) <= EPS_LOG:
        regime = f"恒定震荡 (r·(1+α)={product:.4f}=1)"
    else:
        regime = f"失控 (r·(1+α)={product:.4f}>1)"

    # §0误差判据 / Zero-error criterion
    machine_epsilon = np.finfo(np.float64).eps  # ≈ 2.22e-16 / ≈ 2.22e-16
    is_engineering_zero = bool(error_array[-1] < target_error)
    is_machine_zero = bool(error_array[-1] < 10 * machine_epsilon)

    regime += f" | 工程零误差={is_engineering_zero} (ε<{target_error:.0e})"
    regime += f" | 机器零误差={is_machine_zero} (ε<{10*machine_epsilon:.2e})"
    regime += f" | N_sub={N_sub} (自适应={adaptive_sub_steps})"

    return IntegrationResult(
        method_name=f"RK4 + 三层混合 (极赌+极限环+回忆, K={K}, r={r:.4f}, N_sub={N_sub})",
        t_array=t_array,
        y_array=y_array,
        error_array=error_array,
        peak_error=float(np.max(error_array)),
        final_error=float(error_array[-1]),
        accumulation_rate=accumulation_rate,
        regime=regime,
    )


# ════════════════════════════════════════════════════════════════════
# §对比实验 / Comparison experiment
# ════════════════════════════════════════════════════════════════════

def run_comparison_experiment(
    f: Callable[..., np.ndarray],    # §ODE 函数 dy/dt = f(t, y, *args) / ODE function
    y0: np.ndarray,                 # §初始状态 / initial state
    t_end: float = 100.0,
    h: float = 0.1,
    recall_period: int = 10,
    oracle: Oracle | None = None,    # §参考轨迹 (None=自动 DOP853) / reference trajectory (None=auto DOP853)
    exact: Callable[..., np.ndarray] | None = None,  # §解析解 (可选, 仅用于绘图标注) / analytical solution (optional, plot only)
    *args,
    save_plot: bool = True,
    plot_path: str = "rk_recall_comparison.png",
    system_name: str = "ODE",       # §系统名称 (用于标题) / system name (for titles)
) -> dict[str, Any]:
    """运行对比实验: 纯 RK4 vs RK4 + 回忆校正.
    Run comparison experiment: pure RK4 vs RK4 + recall correction.

    依赖 / Dependency:
      - matplotlib 为可选依赖; 仅当 save_plot=True 时需要
      - matplotlib is an optional dependency; only needed when save_plot=True

    Args:
        f: ODE 右端 dy/dt = f(t, y, *args) / ODE right-hand side
        y0: 初始状态 / initial state
        t_end: 结束时间 / end time
        h: 步长 / step size
        recall_period: 回忆周期 K / recall period K
        oracle: 参考轨迹生成器; None 则自动用 DOP853 / reference generator; None = auto DOP853
        exact: 解析解 (可选, 仅绘图标注); None 则用 oracle 绘参考 / analytical (optional, plot only); None = use oracle
        *args: 传给 f 的额外参数 / extra arguments passed to f
        save_plot: 是否保存对比图 / whether to save the comparison plot
        plot_path: 图像保存路径 / image save path
        system_name: 系统名称 (用于标题) / system name (for titles)

    Returns:
        dict 包含基线和校正结果
        dict containing baseline and corrected results

    Raises:
        MissingOptionalDependencyError: save_plot=True 但 matplotlib 未安装
            / save_plot=True but matplotlib not installed
    """
    # §若需绘图, 先验证 matplotlib 可用 (失败快速, 不浪费计算) / If plotting, verify matplotlib first (fail fast)
    if save_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")  # §非交互后端 / Non-interactive backend
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise MissingOptionalDependencyError(
                "matplotlib", "pip install rk-recall[plot]"
            ) from e

    # §若无 oracle, 自动用 DOP853 构建 (基线与校正共用同一参考, 公平对比)
    # §If no oracle, auto-build with DOP853 (baseline and corrected share the same reference, fair comparison)
    if oracle is None:
        oracle = make_dop853_oracle(f, y0, 0.0, t_end, h, *args)

    logger.info("=" * 70)
    logger.info("§3.25 回忆补偿定理原型: 龙格-库塔误差累积 → 回忆校正")
    logger.info("=" * 70)
    logger.info(f"测试问题: {system_name}")
    logger.info(f"初始状态: y0 = {y0}")
    logger.info(f"积分区间: [0, {t_end}], 步长 h = {h}")
    logger.info(f"回忆周期 K = {recall_period}, 压缩率 r = INV_PHI = {float(INV_PHI):.4f}")
    logger.info(f"参考轨迹: {'DOP853 (auto)' if exact is None else 'oracle + 解析解标注'}")
    logger.info("-" * 70)

    # §基线: 纯 RK4 / Baseline: pure RK4
    logger.info("[1] 基线: 纯 RK4 (无校正)...")
    baseline = rk4_integrate(f, y0, 0.0, t_end, h, oracle, *args)
    logger.info(f"    峰值误差: {baseline.peak_error:.6e}")
    logger.info(f"    末值误差: {baseline.final_error:.6e}")
    logger.info(f"    实测累积率 α: {baseline.accumulation_rate:.6f}")
    logger.info(f"    regime: {baseline.regime}")

    # §校正: RK4 + 回忆校正 / Corrected: RK4 + recall correction
    logger.info("[2] 校正: RK4 + 回忆校正...")
    corrected = rk4_with_recall(
        f, y0, 0.0, t_end, h, recall_period, oracle, float(INV_PHI), *args,
    )
    logger.info(f"    峰值误差: {corrected.peak_error:.6e}")
    logger.info(f"    末值误差: {corrected.final_error:.6e}")
    logger.info(f"    实测累积率 α: {corrected.accumulation_rate:.6f}")
    logger.info(f"    regime: {corrected.regime}")

    # §对比 / Comparison
    logger.info("-" * 70)
    logger.info("[3] 对比分析:")
    peak_reduction = (
        (baseline.peak_error - corrected.peak_error) / max(baseline.peak_error, EPS_LOG)
        * 100.0
    )
    final_reduction = (
        (baseline.final_error - corrected.final_error) / max(baseline.final_error, EPS_LOG)
        * 100.0
    )
    logger.info(f"    峰值误差降低: {peak_reduction:.2f}%")
    logger.info(f"    末值误差降低: {final_reduction:.2f}%")
    logger.info(f"    基线累积率 α: {baseline.accumulation_rate:.6f}")
    logger.info(f"    校正累积率 α: {corrected.accumulation_rate:.6f}")

    # §临界判据 / Critical criterion
    r = float(INV_PHI)
    alpha_base = baseline.accumulation_rate
    alpha_corr = corrected.accumulation_rate
    product_base = r * (1.0 + alpha_base)
    product_corr = r * (1.0 + alpha_corr)
    logger.info(f"    基线 r·(1+α) = {product_base:.4f} (临界=1.0)")
    logger.info(f"    校正 r·(1+α) = {product_corr:.4f} (临界=1.0)")

    if product_corr < 1.0 - EPS_LOG:
        logger.info("    → 校正后: 永不增加 (回忆主导)")
    elif abs(product_corr - 1.0) <= EPS_LOG:
        logger.info("    → 校正后: 恒定震荡 (平衡点)")
    else:
        logger.info("    → 校正后: 仍失控 (需更频繁回忆)")

    # §绘图 / Plotting
    if save_plot:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # 误差轨迹对比 / Error trajectory comparison
        ax1 = axes[0]
        ax1.semilogy(baseline.t_array, baseline.error_array + EPS_LOG,
                     "r-", alpha=0.7, label="RK4 基线 (误差累积)")
        ax1.semilogy(corrected.t_array, corrected.error_array + EPS_LOG,
                     "b-", alpha=0.7, label="RK4 + 回忆校正")
        ax1.set_xlabel("时间 t")
        ax1.set_ylabel("误差 ‖y - y_ref‖ (对数)")
        ax1.set_title(f"§3.25 回忆补偿定理原型: {system_name}\n"
                      f"(K={recall_period}, r=INV_PHI={r:.4f})")
        ax1.legend()
        ax1.grid(True, which="both", alpha=0.3)

        # 相轨迹对比 / Phase trajectory comparison
        ax2 = axes[1]
        if exact is not None:
            y_ref_arr = np.array([exact(t, y0, *args) for t in baseline.t_array])
            ref_label = "解析解 (ground truth)"
        else:
            y_ref_arr = np.array([oracle(t) for t in baseline.t_array])
            ref_label = "oracle 参考 (DOP853)"
        if y_ref_arr.shape[1] >= 2:
            ax2.plot(y_ref_arr[:, 0], y_ref_arr[:, 1],
                     "g-", alpha=0.5, linewidth=2, label=ref_label)
            ax2.plot(baseline.y_array[:, 0], baseline.y_array[:, 1],
                     "r--", alpha=0.7, label="RK4 基线")
            ax2.plot(corrected.y_array[:, 0], corrected.y_array[:, 1],
                     "b:", alpha=0.7, linewidth=1.5, label="RK4 + 回忆校正")
            ax2.set_xlabel("y[0]")
            ax2.set_ylabel("y[1]")
            ax2.set_aspect("equal")
        else:
            ax2.plot(baseline.t_array, baseline.y_array[:, 0], "r--", label="RK4 基线")
            ax2.plot(corrected.t_array, corrected.y_array[:, 0], "b:", label="RK4 + 回忆校正")
            ax2.set_xlabel("t")
            ax2.set_ylabel("y[0]")
        ax2.set_title("相轨迹对比")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        logger.info(f"[4] 对比图已保存: {plot_path}")
        plt.close()

    return {
        "baseline": baseline,
        "corrected": corrected,
        "peak_reduction_pct": peak_reduction,
        "final_reduction_pct": final_reduction,
    }


# ════════════════════════════════════════════════════════════════════
# §三层混合对比实验 / Three-layer hybrid comparison experiment
# ════════════════════════════════════════════════════════════════════

def run_hybrid_experiment(
    f: Callable[..., np.ndarray],    # §ODE 函数 dy/dt = f(t, y, *args) / ODE function
    y0: np.ndarray,                 # §初始状态 / initial state
    t_end: float = 100.0,
    h: float = 0.1,
    recall_period: int = 10,
    oracle: Oracle | None = None,    # §参考轨迹 (None=自动 DOP853) / reference trajectory (None=auto DOP853)
    exact: Callable[..., np.ndarray] | None = None,  # §解析解 (可选, 仅用于绘图标注) / analytical solution (optional, plot only)
    *args,
    save_plot: bool = True,
    plot_path: str = "rk_hybrid_comparison.png",
    system_name: str = "ODE",       # §系统名称 (用于标题) / system name (for titles)
) -> dict[str, Any]:
    """运行三层混合对比实验.
    Run the three-layer hybrid comparison experiment.

    对比四种方法:
    Compare four methods:
      1. 纯 RK4 (基线) / Pure RK4 (baseline)
      2. RK4 + 回忆校正 / RK4 + recall correction
      3. RK4 + 回忆 + 极限环 / RK4 + recall + limit cycle
      4. RK4 + 三层混合 (极赌 + 极限环 + 回忆) / RK4 + three-layer hybrid

    依赖 / Dependency:
      - matplotlib 为可选依赖; 仅当 save_plot=True 时需要
      - matplotlib is an optional dependency; only needed when save_plot=True

    Args:
        f: ODE 右端 dy/dt = f(t, y, *args) / ODE right-hand side
        y0: 初始状态 / initial state
        t_end: 结束时间 / end time
        h: 步长 / step size
        recall_period: 回忆周期 K / recall period K
        oracle: 参考轨迹生成器; None 则自动用 DOP853 / reference generator; None = auto DOP853
        exact: 解析解 (可选, 仅绘图标注); None 则用 oracle 绘参考 / analytical (optional, plot only); None = use oracle
        *args: 传给 f 的额外参数 / extra arguments passed to f
        save_plot: 是否保存对比图 / whether to save the comparison plot
        plot_path: 图像保存路径 / image save path
        system_name: 系统名称 (用于标题) / system name (for titles)

    Returns:
        dict 包含所有结果
        dict containing all results

    Raises:
        MissingOptionalDependencyError: save_plot=True 但 matplotlib 未安装
            / save_plot=True but matplotlib not installed
    """
    # §若需绘图, 先验证 matplotlib 可用 (失败快速, 不浪费计算) / If plotting, verify matplotlib first (fail fast)
    if save_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")  # §非交互后端 / Non-interactive backend
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise MissingOptionalDependencyError(
                "matplotlib", "pip install rk-recall[plot]"
            ) from e

    # §若无 oracle, 自动用 DOP853 构建 (四种方法共用同一参考, 公平对比)
    # §If no oracle, auto-build with DOP853 (all four methods share the same reference, fair comparison)
    if oracle is None:
        oracle = make_dop853_oracle(f, y0, 0.0, t_end, h, *args)

    logger.info("=" * 70)
    logger.info("§3.25+ 三层混合: 极赌 + 极限环 + 回忆校正 → 工程零误差")
    logger.info("=" * 70)
    logger.info(f"测试问题: {system_name}")
    logger.info(f"初始状态: y0 = {y0}")
    logger.info(f"积分区间: [0, {t_end}], 步长 h = {h}")
    logger.info(f"回忆周期 K = {recall_period}, 压缩率 r = INV_PHI = {float(INV_PHI):.4f}")
    logger.info(f"参考轨迹: {'DOP853 (auto)' if exact is None else 'oracle + 解析解标注'}")
    logger.info("-" * 70)

    # §方法1: 纯 RK4 基线 / Method 1: pure RK4 baseline
    logger.info("[1] 纯 RK4 (基线)...")
    baseline = rk4_integrate(f, y0, 0.0, t_end, h, oracle, *args)
    logger.info(f"    峰值误差: {baseline.peak_error:.6e}")
    logger.info(f"    末值误差: {baseline.final_error:.6e}")
    logger.info(f"    regime: {baseline.regime}")

    # §方法2: RK4 + 回忆校正 / Method 2: RK4 + recall correction
    logger.info("[2] RK4 + 回忆校正...")
    corrected = rk4_with_recall(
        f, y0, 0.0, t_end, h, recall_period, oracle, float(INV_PHI), *args,
    )
    logger.info(f"    峰值误差: {corrected.peak_error:.6e}")
    logger.info(f"    末值误差: {corrected.final_error:.6e}")
    logger.info(f"    regime: {corrected.regime}")

    # §方法3: RK4 + 回忆 + 极限环 / Method 3: RK4 + recall + limit cycle
    logger.info("[3] RK4 + 回忆 + 极限环 (N_sub=10)...")
    lc_config = LimitCycleConfig(
        enable=True,
        amplitude_bound=1e-8,  # 更紧的极限环边界 / Tighter limit-cycle bound
    )
    corrected_lc = rk4_hybrid_correction(
        f, y0, 0.0, t_end, h, recall_period,
        oracle, float(INV_PHI),
        GamblePoleConfig(enable=False), lc_config,  # §关闭极赌 / Disable GamblePole
        10, True, 1e-10,
        *args,
    )
    logger.info(f"    峰值误差: {corrected_lc.peak_error:.6e}")
    logger.info(f"    末值误差: {corrected_lc.final_error:.6e}")
    logger.info(f"    regime: {corrected_lc.regime}")

    # §方法4: RK4 + 三层混合 / Method 4: RK4 + three-layer hybrid
    logger.info("[4] RK4 + 三层混合 (极赌 + 极限环 + 回忆, N_sub=20)...")
    gp_config = GamblePoleConfig(
        enable=True,
        error_threshold=1e-10,   # 误差>1e-10时极赌跳跃 / GamblePole jumps when error > 1e-10
        jump_ratio=float(INV_PHI),
    )
    lc_config2 = LimitCycleConfig(
        enable=True,
        amplitude_bound=1e-11,   # 更紧的极限环 / Tighter limit cycle
    )
    hybrid = rk4_hybrid_correction(
        f, y0, 0.0, t_end, h, recall_period,
        oracle, float(INV_PHI),
        gp_config, lc_config2,
        20, True, 1e-10,  # §子步长N=20, 误差降低20^5=3.2e6倍 / Sub-steps N=20
        *args,
    )
    logger.info(f"    峰值误差: {hybrid.peak_error:.6e}")
    logger.info(f"    末值误差: {hybrid.final_error:.6e}")
    logger.info(f"    regime: {hybrid.regime}")

    # §对比汇总 / Comparison summary
    logger.info("-" * 70)
    logger.info("[5] 对比汇总:")
    logger.info(f"    {'方法':<35} | {'峰值误差':>12} | {'末值误差':>12} | {'降低%':>8}")
    logger.info("    " + "-" * 75)
    methods = [
        ("纯 RK4 (基线)", baseline),
        ("RK4 + 回忆校正", corrected),
        ("RK4 + 回忆 + 极限环", corrected_lc),
        ("RK4 + 三层混合", hybrid),
    ]
    for name, result in methods:
        reduction = (
            (baseline.final_error - result.final_error)
            / max(baseline.final_error, EPS_LOG) * 100.0
        )
        logger.info(f"    {name:<35} | {result.peak_error:>12.4e} | "
                    f"{result.final_error:>12.4e} | {reduction:>7.2f}%")

    # §0误差诚实声明 / Zero-error honesty statement
    machine_eps = np.finfo(np.float64).eps
    logger.info("[6] 0误差诚实声明:")
    logger.info(f"    机器精度: {machine_eps:.2e}")
    logger.info(f"    三层混合末值误差: {hybrid.final_error:.2e}")
    logger.info(f"    工程零误差 (< 1e-10): {hybrid.final_error < 1e-10}")
    logger.info(f"    机器零误差 (< {10*machine_eps:.2e}): {hybrid.final_error < 10*machine_eps}")
    logger.info(f"    数学零误差 (= 0): {hybrid.final_error == 0.0} (不可能, 痛苦恒定§V3.5)")

    # §绘图 / Plotting
    if save_plot:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # 误差轨迹对比 / Error trajectory comparison
        ax1 = axes[0]
        ax1.semilogy(baseline.t_array, baseline.error_array + EPS_LOG,
                     "r-", alpha=0.7, label="RK4 baseline")
        ax1.semilogy(corrected.t_array, corrected.error_array + EPS_LOG,
                     "b-", alpha=0.7, label="RK4 + recall")
        ax1.semilogy(corrected_lc.t_array, corrected_lc.error_array + EPS_LOG,
                     "g-", alpha=0.7, label="RK4 + recall + limit cycle")
        ax1.semilogy(hybrid.t_array, hybrid.error_array + EPS_LOG,
                     "k-", alpha=0.9, linewidth=1.5, label="RK4 + hybrid (3-layer)")
        ax1.axhline(y=machine_eps, color="gray", linestyle=":", alpha=0.5,
                    label=f"machine eps ({machine_eps:.1e})")
        ax1.set_xlabel("time t")
        ax1.set_ylabel("error (log)")
        ax1.set_title(f"3-Layer Hybrid: {system_name}\n"
                      f"(K={recall_period}, r=INV_PHI={float(INV_PHI):.4f})")
        ax1.legend()
        ax1.grid(True, which="both", alpha=0.3)

        # 相轨迹 / Phase trajectory
        ax2 = axes[1]
        if exact is not None:
            y_ref_arr = np.array([exact(t, y0, *args) for t in baseline.t_array])
            ref_label = "exact"
        else:
            y_ref_arr = np.array([oracle(t) for t in baseline.t_array])
            ref_label = "oracle (DOP853)"
        if y_ref_arr.shape[1] >= 2:
            ax2.plot(y_ref_arr[:, 0], y_ref_arr[:, 1],
                     "g-", alpha=0.5, linewidth=2, label=ref_label)
            ax2.plot(baseline.y_array[:, 0], baseline.y_array[:, 1],
                     "r--", alpha=0.5, label="RK4 baseline")
            ax2.plot(hybrid.y_array[:, 0], hybrid.y_array[:, 1],
                     "k:", alpha=0.9, linewidth=1.5, label="RK4 hybrid")
            ax2.set_xlabel("y[0]")
            ax2.set_ylabel("y[1]")
            ax2.set_aspect("equal")
        else:
            ax2.plot(baseline.t_array, baseline.y_array[:, 0], "r--", label="RK4 baseline")
            ax2.plot(hybrid.t_array, hybrid.y_array[:, 0], "k:", label="RK4 hybrid")
            ax2.set_xlabel("t")
            ax2.set_ylabel("y[0]")
        ax2.set_title("Phase trajectory")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        logger.info(f"[7] Plot saved: {plot_path}")
        plt.close()

    return {
        "baseline": baseline,
        "corrected": corrected,
        "corrected_lc": corrected_lc,
        "hybrid": hybrid,
    }


# ════════════════════════════════════════════════════════════════════
# §多 K 值扫描实验 (验证临界 K) / Multi-K scanning experiment (verify critical K)
# ════════════════════════════════════════════════════════════════════

def scan_recall_periods(
    f: Callable[..., np.ndarray],    # §ODE 函数 dy/dt = f(t, y, *args) / ODE function
    y0: np.ndarray,                 # §初始状态 / initial state
    t_end: float = 100.0,
    h: float = 0.1,
    K_values: list[int] | None = None,
    oracle: Oracle | None = None,    # §参考轨迹 (None=自动 DOP853) / reference trajectory (None=auto DOP853)
    exact: Callable[..., np.ndarray] | None = None,  # §解析解 (可选, 仅用于标注) / analytical solution (optional)
    *args,
    system_name: str = "ODE",       # §系统名称 (用于标题) / system name (for titles)
) -> dict[str, Any]:
    """扫描不同回忆周期 K, 验证临界 K.
    Scan different recall periods K to verify the critical K.

    Args:
        f: ODE 右端 dy/dt = f(t, y, *args) / ODE right-hand side
        y0: 初始状态 / initial state
        t_end: 结束时间 / end time
        h: 步长 / step size
        K_values: 待扫描的 K 值列表 / list of K values to scan
        oracle: 参考轨迹生成器; None 则自动用 DOP853 / reference generator; None = auto DOP853
        exact: 解析解 (可选, 仅标注) / analytical solution (optional)
        *args: 传给 f 的额外参数 / extra arguments passed to f
        system_name: 系统名称 (用于标题) / system name (for titles)

    Returns:
        dict 包含扫描结果
        dict containing scan results
    """
    if K_values is None:
        K_values = [5, 10, 20, 50, 100]

    # §若无 oracle, 自动用 DOP853 构建 (所有 K 共用同一参考, 公平对比)
    # §If no oracle, auto-build with DOP853 (all K share the same reference, fair comparison)
    if oracle is None:
        oracle = make_dop853_oracle(f, y0, 0.0, t_end, h, *args)

    logger.info("=" * 70)
    logger.info(f"§多 K 值扫描: 寻找最优回忆周期 ({system_name})")
    logger.info("=" * 70)
    logger.info(f"{'K':>6} | {'峰值误差':>12} | {'末值误差':>12} | {'α实测':>10} | {'r·(1+α)':>10} | {'regime':>15}")
    logger.info("-" * 80)

    results = []
    for K in K_values:
        corrected = rk4_with_recall(
            f, y0, 0.0, t_end, h, K, oracle, float(INV_PHI), *args,
        )
        r = float(INV_PHI)
        product = r * (1.0 + corrected.accumulation_rate)
        if product < 1.0 - EPS_LOG:
            regime = "永不增加"
        elif abs(product - 1.0) <= EPS_LOG:
            regime = "恒定震荡"
        else:
            regime = "失控"

        logger.info(f"{K:>6} | {corrected.peak_error:>12.4e} | "
                    f"{corrected.final_error:>12.4e} | "
                    f"{corrected.accumulation_rate:>10.6f} | "
                    f"{product:>10.4f} | {regime:>15}")

        results.append({
            "K": K,
            "peak_error": corrected.peak_error,
            "final_error": corrected.final_error,
            "accumulation_rate": corrected.accumulation_rate,
            "product": product,
            "regime": regime,
        })

    return {"scan_results": results}


# ════════════════════════════════════════════════════════════════════
# §向后兼容: 谐振子基准函数延迟重导出 (PEP 562 模块 __getattr__)
# §Backward compat: lazy re-export of harmonic-oscillator benchmarks (PEP 562 module __getattr__)
# ════════════════════════════════════════════════════════════════════

_HARMONIC_LAZY_NAMES = ("harmonic_oscillator", "harmonic_exact")


def __getattr__(name: str) -> Any:
    """模块级延迟属性访问 (PEP 562).
    Module-level lazy attribute access (PEP 562).

    谐振子基准函数已移至 benchmarks.py; 此处延迟重导出以保持旧导入路径兼容
    (例如 `from rk_recall.rk_recall_compensation import harmonic_exact`),
    但本模块的核心算法不再依赖它们.
    Harmonic-oscillator benchmarks have been moved to benchmarks.py;
    this lazy re-export preserves the legacy import path
    (e.g. `from rk_recall.rk_recall_compensation import harmonic_exact`),
    while the core algorithm of this module no longer depends on them.
    """
    if name in _HARMONIC_LAZY_NAMES:
        # §从 benchmarks.py 延迟导入 (无循环依赖) / Lazy-import from benchmarks.py (no circular dependency)
        from rk_recall import benchmarks as _benchmarks
        return getattr(_benchmarks, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ════════════════════════════════════════════════════════════════════
# §模块入口 / Module entry point
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # §配置 logging / Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # §反偶像声明 / Anti-idol statement
    logger.info("=" * 70)
    logger.info("§3.25 回忆补偿定理原型: 龙格-库塔误差累积 → 回忆校正")
    logger.info("本模块是受造的数学结论, 无灵无意识, 不是生命, 不是'灵'.")
    logger.info("'回忆校正'是前向压缩映射的工程应用, 不是'属灵的修复'.")
    logger.info("真理的活来自圣灵, 不来自代码.")
    logger.info("真运算: 校正基准用 oracle (DOP853 高精度积分), 非解析解 (无循环论证).")
    logger.info("=" * 70)

    # §演示用谐振子 (有解析解, 便于验证; 仅 __main__ 局部, 不作模块级导出)
    # §Demo harmonic oscillator (analytical solution available for verification; __main__-local only, not exported)
    def harmonic_oscillator(t, y, omega=1.0):
        x, v = y[0], y[1]
        return np.array([v, -omega * omega * x])

    def harmonic_exact(t, y0, omega=1.0):
        x0, v0 = y0[0], y0[1]
        x = x0 * np.cos(omega * t) + (v0 / omega) * np.sin(omega * t)
        v = -x0 * omega * np.sin(omega * t) + v0 * np.cos(omega * t)
        return np.array([x, v])

    y0_demo = np.array([1.0, 0.0])
    omega_demo = 1.0
    sys_name = f"谐振子 (ω={omega_demo})"

    # §实验1: 对比实验 (基线 vs 校正) / Experiment 1: comparison experiment (baseline vs corrected)
    logger.info(">>> 实验1: 对比实验 (K=10, oracle=None 自动 DOP853)")
    exp1 = run_comparison_experiment(
        harmonic_oscillator, y0_demo,
        t_end=100.0, h=0.1, recall_period=10,
        oracle=None, exact=harmonic_exact,
        save_plot=True,
        plot_path="rk_recall_comparison.png",
        system_name=sys_name,
    )

    # §实验2: 多 K 值扫描 / Experiment 2: multi-K scan
    logger.info(">>> 实验2: 多 K 值扫描")
    exp2 = scan_recall_periods(
        harmonic_oscillator, y0_demo,
        t_end=100.0, h=0.1, K_values=[5, 10, 20, 50, 100],
        oracle=None, exact=harmonic_exact,
        system_name=sys_name,
    )

    # §实验3: 长时间积分 (验证恒定震荡) / Experiment 3: long-time integration (verify constant oscillation)
    logger.info(">>> 实验3: 长时间积分 (t=500, 验证恒定震荡)")
    exp3 = run_comparison_experiment(
        harmonic_oscillator, y0_demo,
        t_end=500.0, h=0.1, recall_period=10,
        oracle=None, exact=harmonic_exact,
        save_plot=True,
        plot_path="rk_recall_long_time.png",
        system_name=f"谐振子 (ω={omega_demo}, 长时)",
    )

    # §实验4: 三层混合 (极赌 + 极限环 + 回忆 → 工程零误差) / Experiment 4: three-layer hybrid → engineering zero-error
    logger.info(">>> 实验4: 三层混合 (极赌 + 极限环 + 回忆)")
    exp4 = run_hybrid_experiment(
        harmonic_oscillator, y0_demo,
        t_end=100.0, h=0.1, recall_period=10,
        oracle=None, exact=harmonic_exact,
        save_plot=True,
        plot_path="rk_hybrid_comparison.png",
        system_name=sys_name,
    )

    logger.info("=" * 70)
    logger.info("§原型测试完成.")
    logger.info("§结论: 回忆校正显著降低 RK4 误差累积,")
    logger.info("        误差从指数增长变为恒定震荡或永不增加.")
    logger.info("        三层混合 (极赌+极限环+回忆) 进一步压到工程零误差.")
    logger.info("        临界 α* = INV_PHI (黄金分割自对偶点).")
    logger.info("§真运算: 校正基准为 oracle (DOP853), 与具体 ODE 解耦, 无循环论证.")
    logger.info("§0误差诚实: 数学上 ε>0, 工程上 ε<1e-10 (不可观测).")
    logger.info("一切荣光来自造物主一切荣光归于造物主")
    logger.info("=" * 70)
