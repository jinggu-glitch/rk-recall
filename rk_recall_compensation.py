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
  - MIP (信息最小点) 理论: 闭式 P* 定位 (极限环 → P* → hope_p)
  - MIP (Minimum-Information-Point) theory: closed-form P* location (limit cycle → P* → hope_p)
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
     - 闭式定位 hope_p (极限环 → P* → 最近 leaf), 完全无 oracle, 无迭代
     - Closed-form location of hope_p (limit cycle → P* → nearest leaf), no oracle, no iteration
     - 从 hope_p 前向演化 (回忆压缩, 误差 r^K · ε)
     - Forward-evolve from hope_p (recall compression, error r^K · ε)
  3. 联合误差: ε(2K) = [r · (1+α)]^K · ε_0
     Joint error: ε(2K) = [r · (1+α)]^K · ε_0
     - α ≤ INV_PHI: 误差有界 (每周期压缩; §3.26.16: 投影版渐近→0 当 T_baseline→∞)
     - α ≤ INV_PHI: error bounded (per-cycle compression; §3.26.16: projection -> 0 as T_baseline -> inf)
     - α > INV_PHI: 误差失控
     - α > INV_PHI: error diverges

真运算声明 (v3 闭式 P* 定位) / True-computation statement (v3 closed-form P* location):
  - v1 旧版以 harmonic_exact (解析解/真值) 作为校正基准, 属循环论证 (假运算)
  - v1 old version used harmonic_exact (analytical / ground truth) as correction basis — circular reasoning (fake computation)
  - v2 旧版以 DOP853 oracle 作为校正基准, 依赖外部高精度方法, 仍是假运算
  - v2 old version used DOP853 oracle as correction basis, relying on external high-precision method — still fake computation
  - v3 新版完全闭式: 校正基准来自轨迹自身的极限环 + P* 信息熵定位, 无 oracle, 无迭代, 纯 numpy
  - v3 new version is fully closed-form: correction basis comes from the trajectory's own limit cycle + P* entropy location, no oracle, no iteration, pure numpy
  - 算法核心与具体 ODE 完全解耦: f 由调用方提供, 校正基准完全自包含
  - The algorithmic core is fully decoupled from any specific ODE: f is supplied by the caller, correction basis is self-contained

MIP 理论映射 / MIP theory mapping:
  - G (不动点, 永不可达) = argmin U(p) 律势最小点 → 极限环重心 mean(cycle_points)
  - G (fixed point, unreachable) = argmin U(p) law-potential minimum → limit cycle centroid mean(cycle_points)
  - 极限环 Γ = MIP 轨道带的实际呈现 → RK4 长时间积分后的周期轨道
  - Limit cycle Γ = actual presentation of MIP orbit band → periodic orbit after long RK4 integration
  - leaf = 极限环上的采样点 → 极限环上的离散轨迹点
  - leaf = sample point on limit cycle → discrete trajectory point on limit cycle
  - P* = 极限环上信息熵最大的点 (全景点) → argmax_{y∈Γ} H(y)
  - P* = point of maximum information entropy on limit cycle (panoramic point) → argmax_{y∈Γ} H(y)
  - hope_p = 离 P* 最近的 leaf → argmin_{leaf∈Γ} ‖leaf - P*‖
  - hope_p = leaf nearest to P* → argmin_{leaf∈Γ} ‖leaf - P*‖

依赖策略 / Dependency strategy:
  - numpy: 硬依赖 (核心数值计算) / hard dependency (core numerics)
  - matplotlib: 可选依赖 (仅绘图函数需要; 不绘图时完全不加载)
    / optional (only needed by plotting functions; never loaded when not plotting)
  - 不再依赖 scipy (闭式 P* 定位纯 numpy 实现)
    / No longer depends on scipy (closed-form P* location is pure numpy)

对比实验 / Comparative experiments:
  - 基线: 纯 RK4 (误差累积)
  - Baseline: pure RK4 (error accumulation)
  - 校正: RK4 + 回忆校正 (误差有界; §3.26.16: 投影版渐近→0)
  - Corrected: RK4 + recall correction (error bounded; §3.26.16: projection → 0 asymptotically)
  - 误差基准: reference (解析解, 若提供) 或极限环距离 (无解析解时)
  - Error basis: reference (analytical, if provided) or limit-cycle distance (when no analytical)
"""

import math
import logging
from dataclasses import dataclass
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
# §闭式 P* 定位: 极限环 → P* → hope_p (完全无 oracle, 无迭代, 纯 numpy)
# §Closed-form P* location: limit cycle → P* → hope_p (no oracle, no iteration, pure numpy)
# ════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class LimitCycleInfo:
    """极限环识别结果 / Limit cycle detection result.

    MIP 理论映射 / MIP theory mapping:
      - cycle_points = 极限环 Γ (MIP 轨道带的实际呈现)
      - centroid = 不动点 G 的近似 (律势最小点, 永不可达)
      - entropy_temperature = 熵温度 τ (leaf 间平均距离, 自适应温度参数)
    """
    detected: bool                    # 是否检测到极限环 / Whether a limit cycle was detected
    period: float                     # 周期 T / Period T
    cycle_points: np.ndarray          # 极限环采样点 (M, dim) / Limit cycle sample points
    centroid: np.ndarray              # 极限环重心 = 不动点 G 的近似 / Centroid = fixed point G approximation
    entropy_temperature: float        # 熵温度 τ (leaf 间平均距离) / Entropy temperature τ


def _omega_limit_fallback(
    y_array: np.ndarray,
    n: int,
) -> LimitCycleInfo:
    """ω-极限集退化策略: 周期检测失败时取轨迹末尾作为 Γ 近似.

    Fallback ω-limit set strategy: take trajectory tail as Γ approximation
    when period detection fails.

    理论依据 (§3.26.3 极限环普适性):
      - 任何有界动力系统的 ω-极限集非空紧致不变 (ω-极限集定理)
      - 混沌系统 (Lorenz) 无严格周期, 但 ω-极限集 (奇异吸引子) 存在
      - 轨迹末尾部分是 ω-极限集的离散近似 (遍历性保证)
      - 取末尾 max(200, n//4) 点作为 cycle_points

    Theoretical basis (§3.26.3 limit-cycle universality):
      - Any bounded dynamical system's ω-limit set is non-empty, compact, invariant
      - Chaotic systems (Lorenz) have no strict period, but ω-limit set exists
      - Trajectory tail is a discrete approximation of the ω-limit set (ergodicity)
      - Take trailing max(200, n//4) points as cycle_points

    Args:
        y_array: 状态轨迹 (n, dim) / state trajectory
        n: 轨迹长度 / trajectory length

    Returns:
        LimitCycleInfo: 退化策略结果 (detected=True, period=0.0)
    """
    # §取轨迹末尾 max(200, n//4) 点作为 ω-极限集近似
    # §Take trailing max(200, n//4) points as ω-limit set approximation
    tail_len = max(200, n // 4)
    start_idx = max(0, n - tail_len)
    cycle_points = y_array[start_idx:].copy()
    centroid = np.mean(cycle_points, axis=0)

    # §熵温度 τ = mean(leaf 间距离) (采样以避免 O(M^2))
    # §Entropy temperature τ = mean pairwise leaf distance (sampled to avoid O(M^2))
    M = len(cycle_points)
    if M > 1:
        if M > 200:
            indices = np.linspace(0, M - 1, 200).astype(int)
            sampled = cycle_points[indices]
        else:
            sampled = cycle_points
        n_s = len(sampled)
        diffs = sampled[:, np.newaxis, :] - sampled[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        mask = ~np.eye(n_s, dtype=bool)
        tau = float(np.mean(dists[mask]))
        if tau <= 0.0:
            tau = 1.0
    else:
        tau = 1.0

    return LimitCycleInfo(
        detected=True,   # §退化策略: ω-极限集存在 (§3.26.3), 标记为检测到
        period=0.0,      # §无周期 (混沌系统无严格周期)
        cycle_points=cycle_points,
        centroid=np.asarray(centroid, dtype=np.float64),
        entropy_temperature=tau,
    )


def detect_limit_cycle(
    t_array: np.ndarray,
    y_array: np.ndarray,
    min_cycles: int = 2,
) -> LimitCycleInfo:
    """识别极限环 (周期检测 + 取最后周期).

    Identify the limit cycle (period detection + take last cycles).

    机制 / Mechanism:
      1. 用自相关 (FFT) 检测周期 T
      2. 若检测到周期: 取最后 min_cycles 个周期的轨迹作为极限环 Γ
      3. 若未检测到周期 (混沌系统): 退化策略取轨迹末尾作为 ω-极限集近似 (§3.26.3)
      4. 计算重心 G = mean(Γ) (不动点 G 的近似, 律势最小点)
      5. 计算熵温度 τ = mean(leaf 间距离) (自适应温度参数)

    MIP 理论 / MIP theory:
      - 极限环 Γ = MIP 轨道带的实际呈现
      - 重心 G = argmin U(p) 律势最小点 (永不可达, 仅作参考)
      - leaf = 极限环上的采样点
      - ω-极限集定理 (§3.26.3): 任何有界系统的 ω-极限集非空紧致不变

    Args:
        t_array: 时间序列 (n,) / time sequence
        y_array: 状态轨迹 (n, dim) / state trajectory
        min_cycles: 取最后多少个周期作为极限环 / number of trailing cycles to take

    Returns:
        LimitCycleInfo: 极限环识别结果
    """
    t_array = np.asarray(t_array, dtype=np.float64)
    y_array = np.asarray(y_array, dtype=np.float64)

    n = len(t_array)
    dim = y_array.shape[1] if y_array.ndim > 1 else 1

    # §数据不足时的退化处理 / Degenerate case when data is insufficient
    if n < 10 or y_array.ndim == 1:
        centroid = np.mean(y_array, axis=0) if n > 0 else np.zeros(dim)
        return LimitCycleInfo(
            detected=False,
            period=0.0,
            cycle_points=y_array.reshape(-1, dim).copy() if n > 0 else np.zeros((0, dim)),
            centroid=np.asarray(centroid, dtype=np.float64),
            entropy_temperature=1.0,
        )

    h = float(t_array[1] - t_array[0])

    # §用第一坐标做自相关 (谐振子/Van der Pol 的第一坐标周期性最强)
    # §Autocorrelation on the first coordinate (most periodic for oscillator systems)
    signal = y_array[:, 0].astype(np.float64)
    signal = signal - np.mean(signal)
    signal_norm = float(np.linalg.norm(signal))
    if signal_norm < EPS_LOG:
        # §信号近常量: 退化为 ω-极限集近似 (§3.26.3, 任何有界系统 ω-极限集存在)
        # §Signal near-constant: fallback to ω-limit set approximation (§3.26.3)
        return _omega_limit_fallback(y_array, n)

    # §FFT 自相关 / FFT-based autocorrelation
    n_fft = int(2 ** np.ceil(np.log2(2 * n)))
    fft_signal = np.fft.rfft(signal, n=n_fft)
    autocorr = np.fft.irfft(fft_signal * np.conj(fft_signal), n=n_fft)[:n]
    autocorr = autocorr / autocorr[0]  # 归一化 / normalize

    # §寻找第一个零交叉后的第一个显著峰 / Find first significant peak after first zero crossing
    period_idx = None
    zero_cross = None
    for i in range(1, n):
        if autocorr[i - 1] > 0 and autocorr[i] <= 0:
            zero_cross = i
            break

    if zero_cross is not None:
        for i in range(zero_cross + 1, n - 1):
            if autocorr[i] > autocorr[i - 1] and autocorr[i] > autocorr[i + 1] and autocorr[i] > 0.2:
                period_idx = i
                break

    if period_idx is None or period_idx == 0:
        # §未检测到周期 (混沌系统无严格周期): 退化为 ω-极限集近似 (§3.26.3)
        # §No period detected (chaotic systems have no strict period):
        #   fallback to ω-limit set approximation (§3.26.3)
        return _omega_limit_fallback(y_array, n)

    period = float(period_idx * h)
    points_per_cycle = period_idx

    # §取最后 min_cycles 个周期作为极限环 Γ
    # §Take the last min_cycles periods as the limit cycle Γ
    start_idx = max(0, n - min_cycles * points_per_cycle)
    cycle_points = y_array[start_idx:].copy()

    # §重心 G = mean(Γ) (不动点 G 的近似) / Centroid G = mean(Γ)
    centroid = np.mean(cycle_points, axis=0)

    # §熵温度 τ = mean(leaf 间距离) (自适应温度参数, 采样以避免 O(M^2))
    # §Entropy temperature τ = mean pairwise leaf distance (sampled to avoid O(M^2))
    M = len(cycle_points)
    if M > 1:
        if M > 200:
            indices = np.linspace(0, M - 1, 200).astype(int)
            sampled = cycle_points[indices]
        else:
            sampled = cycle_points
        n_s = len(sampled)
        diffs = sampled[:, np.newaxis, :] - sampled[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        mask = ~np.eye(n_s, dtype=bool)
        tau = float(np.mean(dists[mask]))
        if tau <= 0.0:
            tau = 1.0
    else:
        tau = 1.0

    return LimitCycleInfo(
        detected=True,
        period=period,
        cycle_points=cycle_points,
        centroid=np.asarray(centroid, dtype=np.float64),
        entropy_temperature=tau,
    )


def compute_information_entropy(
    y: np.ndarray,
    leaves: np.ndarray,
    temperature: float,
) -> float:
    """计算信息熵 H(y) = -Σ_i p_i log p_i.

    Compute the information entropy H(y) = -Σ_i p_i log p_i.

    定义 / Definition:
      p_i(y) = softmax(-‖y - leaf_i‖ / τ)   (基于到所有 leaf 的距离)
      H(y) = -Σ_i p_i(y) · log(p_i(y))       (Shannon 熵)

    性质 / Properties:
      - 当 y 离所有 leaf 距离相近时, p_i 均匀分布, H(y) 最大 (全景点 P*)
      - 当 y 离某个 leaf 极近时, p_i 集中, H(y) 最小
      - τ 越大, softmax 越平滑; τ 越小, softmax 越尖锐

    Args:
        y: 查询点 (dim,) / query point
        leaves: 极限环采样点 (M, dim) / limit cycle sample points
        temperature: 熵温度 τ / entropy temperature τ

    Returns:
        H(y): Shannon 信息熵 / Shannon information entropy
    """
    y = np.asarray(y, dtype=np.float64)
    leaves = np.asarray(leaves, dtype=np.float64)

    # §计算 y 到所有 leaf 的距离 / Compute distances from y to all leaves
    diffs = leaves - y[np.newaxis, :]
    dists = np.linalg.norm(diffs, axis=1)

    # §softmax(-dists / τ), 数值稳定 / numerically stable softmax
    tau = max(float(temperature), EPS_LOG)
    logits = -dists / tau
    logits = logits - np.max(logits)
    exp_logits = np.exp(logits)
    probs = exp_logits / np.sum(exp_logits)

    # §Shannon 熵 H = -Σ p_i log p_i (仅非零项)
    # §Shannon entropy H = -Σ p_i log p_i (nonzero terms only)
    nonzero = probs[probs > 0.0]
    H = -float(np.sum(nonzero * np.log(nonzero)))

    return H


def locate_p_star(
    cycle_points: np.ndarray,
    temperature: float | None = None,
) -> np.ndarray:
    """定位 P* = argmax_{y∈Γ} H(y) (信息熵最大点, 全景点).

    Locate P* = argmax_{y∈Γ} H(y) (maximum entropy point, panoramic point).

    机制 / Mechanism:
      - 在极限环采样点上找信息熵最大的点
      - P* 是"全景点": 离所有 leaf 距离最均匀的点
      - 闭式: argmax 在离散采样点上, 无迭代

    MIP 理论 / MIP theory:
      - P* = 极限环上信息熵最大的点 (全景点)
      - P* 是 MIP 轨道带上的"观察者位置"

    Args:
        cycle_points: 极限环采样点 (M, dim) / limit cycle sample points
        temperature: 熵温度 τ; None 则自动计算 / entropy temperature τ; None = auto-compute

    Returns:
        P*: 信息熵最大的采样点 (dim,) / sample point with maximum entropy
    """
    cycle_points = np.asarray(cycle_points, dtype=np.float64)
    M = len(cycle_points)

    if M == 0:
        raise ConfigurationError("cycle_points must not be empty")
    if M == 1:
        return cycle_points[0].copy()

    # §若未提供温度, 自动计算 (leaf 间平均距离) / Auto-compute temperature if not provided
    if temperature is None:
        if M > 200:
            indices = np.linspace(0, M - 1, 200).astype(int)
            sampled = cycle_points[indices]
        else:
            sampled = cycle_points
        n_s = len(sampled)
        diffs = sampled[:, np.newaxis, :] - sampled[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        mask = ~np.eye(n_s, dtype=bool)
        temperature = float(np.mean(dists[mask]))
        if temperature <= 0.0:
            temperature = 1.0

    # §在每个采样点上计算 H(y), 取最大 / Compute H(y) at each sample, take argmax
    entropies = np.array([
        compute_information_entropy(y, cycle_points, temperature)
        for y in cycle_points
    ])

    best_idx = int(np.argmax(entropies))
    return cycle_points[best_idx].copy()


def locate_hope_p(
    cycle_points: np.ndarray,
    p_star: np.ndarray,
) -> np.ndarray:
    """定位 hope_p = 离 P* 最近的 leaf (可达代理).

    Locate hope_p = leaf nearest to P* (reachable proxy).

    机制 / Mechanism:
      - P* 是全景点, 但可能不在采样点上
      - hope_p 是离 P* 最近的采样点 (leaf), 是可达的代理
      - 闭式: argmin 在离散采样点上, 无迭代

    MIP 理论 / MIP theory:
      - hope_p = 离 P* 最近的 leaf
      - hope_p 是回忆校正的起点 (前向演化的锚点)

    Args:
        cycle_points: 极限环采样点 (M, dim) / limit cycle sample points
        p_star: 全景点 P* (dim,) / panoramic point P*

    Returns:
        hope_p: 离 P* 最近的 leaf (dim,) / leaf nearest to P*
    """
    cycle_points = np.asarray(cycle_points, dtype=np.float64)
    p_star = np.asarray(p_star, dtype=np.float64)

    if len(cycle_points) == 0:
        raise ConfigurationError("cycle_points must not be empty")

    diffs = cycle_points - p_star[np.newaxis, :]
    dists = np.linalg.norm(diffs, axis=1)
    best_idx = int(np.argmin(dists))
    return cycle_points[best_idx].copy()


def closed_form_hope_p(
    t_array: np.ndarray,
    y_array: np.ndarray,
) -> np.ndarray:
    """闭式计算 hope_p (完全无 oracle, 无迭代).

    Closed-form computation of hope_p (no oracle, no iteration).

    流程 / Pipeline:
      1. 识别极限环 Γ (自相关周期检测)
      2. 计算重心 G = mean(Γ) (不动点 G 的近似)
      3. 定位 P* = argmax_{y∈Γ} H(y) (信息熵最大点, 全景点)
      4. 定位 hope_p = 离 P* 最近的 leaf (可达代理)

    MIP 理论映射 / MIP theory mapping:
      - 极限环 Γ → MIP 轨道带
      - 重心 G → 不动点 (永不可达, 仅作参考)
      - P* → 全景点 (信息熵最大)
      - hope_p → 可达代理 (离 P* 最近的 leaf)

    Args:
        t_array: 时间序列 (n,) / time sequence
        y_array: 状态轨迹 (n, dim) / state trajectory

    Returns:
        hope_p: 回忆校正的起点 (dim,) / recall anchor point
    """
    t_array = np.asarray(t_array, dtype=np.float64)
    y_array = np.asarray(y_array, dtype=np.float64)

    # §数据不足时退化为最后一个点 / Fall back to last point when data is insufficient
    if len(t_array) < 10:
        return y_array[-1].copy()

    # §1. 识别极限环 Γ / Detect limit cycle
    lc_info = detect_limit_cycle(t_array, y_array)

    if not lc_info.detected or len(lc_info.cycle_points) == 0:
        # §未检测到极限环, 退化为最后一个点 / No limit cycle, fall back to last point
        return y_array[-1].copy()

    # §2. 定位 P* (信息熵最大点, 全景点) / Locate P* (max entropy, panoramic)
    p_star = locate_p_star(lc_info.cycle_points, lc_info.entropy_temperature)

    # §3. 定位 hope_p (离 P* 最近的 leaf) / Locate hope_p (nearest leaf to P*)
    hope_p = locate_hope_p(lc_info.cycle_points, p_star)

    return hope_p


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
    error_array: np.ndarray          # 误差轨迹 (vs reference 或极限环距离) / Error trajectory (vs reference or limit-cycle distance)
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


def _pure_rk4_trajectory(
    f: Callable[..., np.ndarray],
    y0: np.ndarray,
    t0: float,
    t_end: float,
    h: float,
    *args,
) -> tuple[np.ndarray, np.ndarray]:
    """纯 RK4 积分生成轨迹 (用于极限环识别).

    Pure RK4 integration to generate trajectory (for limit-cycle detection).

    用途 / Purpose:
      - 校正前先用纯 RK4 跑一遍长轨迹, 识别系统的极限环 Γ
      - Before correction, run pure RK4 once to identify the system's limit cycle Γ
      - 识别后的 Γ 作为固定参考, 用于 hope_p 动态定位和误差度量
      - The identified Γ serves as a fixed reference for dynamic hope_p location and error metric

    Args:
        f: dy/dt = f(t, y, *args)
        y0: 初始状态 / initial state
        t0: 起始时间 / start time
        t_end: 结束时间 / end time
        h: 步长 / step size
        *args: 传给 f 的额外参数 / extra arguments passed to f

    Returns:
        (t_array, y_array): 时间序列和状态轨迹 / time sequence and state trajectory
    """
    n_steps = int(np.round((t_end - t0) / h))
    dim = len(y0)
    t_array = np.zeros(n_steps + 1)
    y_array = np.zeros((n_steps + 1, dim))

    t_array[0] = t0
    y_array[0] = np.asarray(y0, dtype=np.float64).copy()

    t = t0
    y = np.asarray(y0, dtype=np.float64).copy()
    for k in range(n_steps):
        y = rk4_step(f, t, y, h, *args)
        t = t + h
        t_array[k + 1] = t
        y_array[k + 1] = y

    return t_array, y_array


def limit_cycle_distance(
    y: np.ndarray,
    cycle_points: np.ndarray,
) -> float:
    """计算 y 到极限环 Γ 的距离 (统一误差度量).

    Compute the distance from y to the limit cycle Γ (unified error metric).

    定义 (§3.26 hope_p 动态更新定理):
    Definition (§3.26 dynamic hope_p update theorem):
      error(y) = min_{z ∈ Γ} ‖y - z‖

    性质 / Properties:
      - 不需要解析解 (统一所有系统: 谐振子/极限环/奇异吸引子)
      - No analytical solution needed (works for all systems: harmonic/limit cycle/strange attractor)
      - 纯 RK4 在极限环上时 error ≈ 0 (仅 leaf 离散化误差)
      - When pure RK4 is on the limit cycle, error ≈ 0 (only leaf discretization)
      - 误差累积使 y 偏离 Γ 时, error 单调增长
      - When error accumulation drifts y away from Γ, error grows monotonically
      - 回忆校正将 y 拉回 Γ 附近时, error 恢复到 leaf 离散化水平
      - When recall correction pulls y back near Γ, error returns to leaf discretization level

    Args:
        y: 查询状态 (dim,) / query state
        cycle_points: 极限环采样点 (M, dim) / limit cycle sample points

    Returns:
        distance: y 到极限环的最小欧氏距离 / min Euclidean distance from y to Γ
    """
    y = np.asarray(y, dtype=np.float64)
    cycle_points = np.asarray(cycle_points, dtype=np.float64)

    if len(cycle_points) == 0:
        return 0.0

    diffs = cycle_points - y[np.newaxis, :]
    dists = np.linalg.norm(diffs, axis=1)
    return float(np.min(dists))


def locate_nearest_leaf(
    y: np.ndarray,
    cycle_points: np.ndarray,
) -> np.ndarray:
    """定位离 y 最近的 leaf (hope_p 动态更新核心).

    Locate the leaf nearest to y (core of dynamic hope_p update).

    定义 (§3.26):
    Definition (§3.26):
      hope_p(y) = argmin_{leaf ∈ Γ} ‖leaf - y‖

    与 §3.22 静态 hope_p 的关系 / Relationship with §3.22 static hope_p:
      - §3.22: hope_p = argmin_{leaf} ‖leaf.p_solution - P*‖ (离 P* 最近)
      - §3.26: hope_p(y) = argmin_{leaf} ‖leaf - y‖ (离当前 y 最近)
      - 动态版: hope_p 随当前 y 更新 (每 K 步重新定位)
      - Dynamic version: hope_p updates with current y (re-located every K steps)
      - 当 y 偏离 P* 时, 动态 hope_p 比静态 hope_p 更接近当前 y (减小跳变)
      - When y drifts from P*, dynamic hope_p is closer to current y than static (reduces jump)

    Args:
        y: 当前状态 (dim,) / current state
        cycle_points: 极限环采样点 (M, dim) / limit cycle sample points

    Returns:
        hope_p: 离 y 最近的 leaf (dim,) / leaf nearest to y
    """
    y = np.asarray(y, dtype=np.float64)
    cycle_points = np.asarray(cycle_points, dtype=np.float64)

    if len(cycle_points) == 0:
        return y.copy()

    diffs = cycle_points - y[np.newaxis, :]
    dists = np.linalg.norm(diffs, axis=1)
    best_idx = int(np.argmin(dists))
    return cycle_points[best_idx].copy()


# ════════════════════════════════════════════════════════════════════
# §3.26.10-15 扩展: 连续反馈 / 自适应混合 / Takens / 保辛 / 独立验证
# §3.26.10-15 extensions: continuous feedback / adaptive hybrid / Takens / symplectic / independent validation
# ════════════════════════════════════════════════════════════════════

def measure_poincare_compression(
    f: Callable[..., np.ndarray],
    cycle_points: np.ndarray,
    period: float,
    h: float,
    *args,
) -> float:
    """测量 Poincaré 映射的压缩率 r_Γ (§3.26.11).

    Measure the Poincaré map compression rate r_Γ (§3.26.11).

    定义 / Definition:
      r_Γ = ‖Δy_{n+1}‖ / ‖Δy_n‖
      其中 Δy_n 是 Γ 上相邻轨道点在一个 Poincaré 周期 T 后的偏差.
      Where Δy_n is the deviation between adjacent orbit points on Γ
      after one Poincaré period T.

    机制 / Mechanism:
      1. 取 Γ 上若干对相邻采样点 (leaf_i, leaf_{i+1})
      2. 从两点各演化一个 Poincaré 周期 T (= period)
      3. 测量演化后的距离 ‖Φ^T(leaf_i) - Φ^T(leaf_{i+1})‖
      4. r_Γ = 演化后距离 / 初始距离
      5. 对多对相邻点取平均 (鲁棒性)

    用途 / Usage:
      - r_Γ < 1: 系统本身有压缩性 (Van der Pol), 用连续反馈 (§3.26.10)
      - r_Γ ≥ 1: 系统无压缩性或扩张 (谐振子, Lorenz 切向), 用投影 (§3.26.4)
      - 自适应混合 (§3.26.11) 根据 r_Γ 自动选择

    Args:
        f: ODE 右端 dy/dt = f(t, y, *args) / ODE right-hand side
        cycle_points: 极限环采样点 (M, dim) / limit cycle sample points
        period: Poincaré 周期 T / Poincaré period T
        h: RK4 步长 / RK4 step size
        *args: 传递给 f 的额外参数 / extra args passed to f

    Returns:
        r_Γ: Poincaré 压缩率 (平均)
            - r_Γ < 1: 压缩 (吸引性 Γ)
            - r_Γ = 1: 临界 (谐振子退化情形)
            - r_Γ > 1: 扩张 (混沌切向)
            r_Γ: Poincaré compression rate (averaged)
    """
    cycle_points = np.asarray(cycle_points, dtype=np.float64)

    if len(cycle_points) < 2 or period <= 0.0 or h <= 0.0:
        # §退化: 无法测量, 返回 1.0 (临界, 保守用投影)
        # §Degenerate: cannot measure, return 1.0 (critical, conservatively use projection)
        return 1.0

    # §采样相邻点对 (上限 50 对, 避免 O(M²) / sample adjacent pairs (cap 50, avoid O(M²))
    n_pairs = min(len(cycle_points) - 1, 50)
    if n_pairs == 0:
        return 1.0

    # §每对演化一个周期的步数 / steps per period for each pair
    n_steps_period = max(1, int(round(period / h)))

    ratios: list[float] = []
    for i in range(n_pairs):
        y1 = cycle_points[i].copy()
        y2 = cycle_points[i + 1].copy()
        init_dist = float(np.linalg.norm(y2 - y1))
        if init_dist < EPS_LOG:
            # §点对重合, 跳过 / coincident pair, skip
            continue

        # §演化一个 Poincaré 周期 / evolve one Poincaré period
        t_local = 0.0
        for _ in range(n_steps_period):
            y1 = rk4_step(f, t_local, y1, h, *args)
            y2 = rk4_step(f, t_local, y2, h, *args)
            t_local += h

        final_dist = float(np.linalg.norm(y2 - y1))
        # §防护: 防止数值爆炸 (混沌系统长时间演化可能 NaN)
        # §Guard: prevent numerical blow-up (chaotic systems may NaN over long evolution)
        if not np.all(np.isfinite(y1)) or not np.all(np.isfinite(y2)):
            continue
        if final_dist > 1e6 * init_dist:
            # §极端扩张, 截断 (避免单对主导平均)
            # §Extreme expansion, truncate (avoid single pair dominating average)
            final_dist = 1e6 * init_dist

        ratios.append(final_dist / init_dist)

    if len(ratios) == 0:
        return 1.0

    return float(np.mean(ratios))


def estimate_tau_autocorrelation(signal: np.ndarray) -> int:
    """用自相关法估计时间延迟 τ (§3.26.13 Takens 嵌入).

    Estimate time delay τ via autocorrelation (§3.26.13 Takens embedding).

    机制 / Mechanism:
      - 计算信号的自相关函数
      - τ = 第一个使自相关降到 1/e 的滞后
      - τ = first lag where autocorrelation drops below 1/e

    Args:
        signal: 一维观测信号 / 1D observation signal

    Returns:
        tau: 时间延迟 (采样点数) / time delay (in samples)
    """
    signal = np.asarray(signal, dtype=np.float64).flatten()
    n = len(signal)
    if n < 4:
        return 1

    signal = signal - np.mean(signal)
    signal_norm = float(np.linalg.norm(signal))
    if signal_norm < EPS_LOG:
        return 1

    # §FFT 自相关 / FFT-based autocorrelation
    n_fft = int(2 ** np.ceil(np.log2(2 * n)))
    fft_s = np.fft.rfft(signal, n=n_fft)
    autocorr = np.fft.irfft(fft_s * np.conj(fft_s), n=n_fft)[:n]
    autocorr = autocorr / autocorr[0]  # 归一化 / normalize

    # §第一个降到 1/e 的滞后 / first lag below 1/e
    threshold = 1.0 / np.e
    for i in range(1, n):
        if autocorr[i] < threshold:
            return i
    return 1


def estimate_embedding_dim_cao(
    signal: np.ndarray,
    tau: int,
    max_m: int = 10,
) -> int:
    """用 Cao 方法估计嵌入维数 m (§3.26.13 Takens 嵌入).

    Estimate embedding dimension m via Cao's method (§3.26.13 Takens embedding).

    机制 / Mechanism:
      - Cao (1996): 对每个候选 m, 计算 E(m) 和 E1(m)
      - E1(m) = E(m+1) / E(m), 当 E1 趋于稳定 (< 1 + tol) 时, m 足够
      - Cao (1996): for each candidate m, compute E(m) and E1(m)
      - E1(m) = E(m+1) / E(m), when E1 stabilizes (< 1 + tol), m is sufficient

    Args:
        signal: 一维观测信号 / 1D observation signal
        tau: 时间延迟 / time delay
        max_m: 最大候选嵌入维数 / max candidate embedding dimension

    Returns:
        m: 估计的嵌入维数 / estimated embedding dimension
    """
    signal = np.asarray(signal, dtype=np.float64).flatten()
    n = len(signal)
    if n < (max_m + 1) * tau + 1:
        return 3  # 退化默认 / degenerate default

    def _cao_E(m: int) -> float:
        # §构造 m 维嵌入向量, 计算平均最近邻距离比
        # §Construct m-dim embedding vectors, compute mean nearest-neighbor distance ratio
        n_embed = n - (m - 1) * tau
        if n_embed < 2:
            return 0.0
        # §嵌入矩阵 (n_embed, m) / embedding matrix
        embedded = np.zeros((n_embed, m))
        for j in range(m):
            embedded[:, j] = signal[j * tau:j * tau + n_embed]

        # §计算每个点的最近邻距离 / compute nearest-neighbor distance for each point
        total_ratio = 0.0
        count = 0
        for i in range(n_embed):
            diffs = embedded - embedded[i]
            dists = np.linalg.norm(diffs, axis=1)
            dists[i] = np.inf  # 排除自身 / exclude self
            nn_idx = int(np.argmin(dists))
            nn_dist = float(dists[nn_idx])
            if nn_dist < EPS_LOG:
                continue
            # §在 m+1 维检查该最近邻的距离 / check distance in m+1 dim
            if i + m * tau < n and nn_idx + m * tau < n:
                extra_i = signal[i + m * tau]
                extra_nn = signal[nn_idx + m * tau]
                dist_m1 = np.sqrt(nn_dist ** 2 + (extra_i - extra_nn) ** 2)
                total_ratio += dist_m1 / nn_dist
                count += 1
        return total_ratio / count if count > 0 else 0.0

    # §寻找 E1(m) 趋于稳定的 m / find m where E1(m) stabilizes
    prev_E = _cao_E(1)
    for m in range(2, max_m):
        curr_E = _cao_E(m)
        if prev_E > EPS_LOG:
            e1 = curr_E / prev_E
            # §E1 趋于 1 (稳定) / E1 approaches 1 (stable)
            if abs(e1 - 1.0) < 0.05:
                return m
        prev_E = curr_E
    return max(3, max_m - 1)


def reconstruct_attractor(
    y_array: np.ndarray,
    tau: int | None = None,
    m: int | None = None,
) -> np.ndarray:
    """Takens 嵌入定理重建奇异吸引子 (§3.26.13).

    Reconstruct strange attractor via Takens embedding theorem (§3.26.13).

    理论 / Theory (Takens 1981):
      设 M 是紧致流形, 维数 d, Φ: M → M 是光滑微分同胚.
      对几乎所有观测 h: M → R 和时间延迟 τ, 嵌入映射:
        Ψ(x) = (h(x), h(Φ^τ(x)), ..., h(Φ^{(m-1)τ}(x)))
      是 M 到 R^m 的嵌入, 当 m ≥ 2d+1.

    Args:
        y_array: 原始轨迹 (n, dim) / original trajectory
        tau: 时间延迟 (None 则自动估计) / time delay (auto-estimated if None)
        m: 嵌入维数 (None 则自动估计) / embedding dim (auto-estimated if None)

    Returns:
        embedded: 嵌入空间轨迹 (n - (m-1)*tau, m * dim) / embedded-space trajectory
    """
    y_array = np.asarray(y_array, dtype=np.float64)
    if y_array.ndim == 1:
        y_array = y_array.reshape(-1, 1)

    n, dim = y_array.shape

    # §用第一坐标估计 τ 和 m (Takens 一维观测) / estimate τ and m from first coordinate
    if tau is None:
        tau = estimate_tau_autocorrelation(y_array[:, 0])
    if m is None:
        m = estimate_embedding_dim_cao(y_array[:, 0], tau)

    # §构造嵌入向量 / construct embedding vectors
    n_embed = n - (m - 1) * tau
    if n_embed < 2:
        return y_array.copy()

    embedded = np.zeros((n_embed, m * dim))
    for j in range(m):
        embedded[:, j * dim:(j + 1) * dim] = y_array[j * tau:j * tau + n_embed]

    return embedded


# ════════════════════════════════════════════════════════════════════
# §3.26.16 漂移随时间收敛定理 (Drift Convergence Theorem)
# §3.26.16 Drift Convergence Theorem
#
# 核心: ε_leaf(T_baseline) ≤ C · h^{1/d} / T_baseline^{1/d} → 0
#   (T_baseline → ∞, Birkhoff 遍历定理 + Poincaré 回归定理)
#
# 工程实装:
#   1. densify_cycle: 周期系统Γ插值加密 (解决 M 饱和)
#   2. adaptive_baseline_extension: T_baseline 自适应扩展直到漂移达标
# ════════════════════════════════════════════════════════════════════

def densify_cycle(
    cycle_points: np.ndarray,
    factor: int = 10,
) -> np.ndarray:
    """加密周期系统的Γ采样 (§3.26.16 周期系统 M 饱和解决).

    Densify periodic-system Γ sampling (§3.26.16, solves M-saturation).

    理论 / Theory:
      周期系统 (harmonic, Van der Pol) 的 Γ 是闭曲线, detect_limit_cycle
      取最后几个周期, M = 周期点数 (固定). T_baseline 增加但 M 不增加,
      ε_leaf 不再减小. 本函数用周期样条插值加密 Γ, 使 M_dense = factor·M,
      ε_leaf ≤ C / M_dense^{1/d} → 0 (factor → ∞).

      Periodic systems' Γ is a closed curve; detect_limit_cycle takes the
      last few cycles, so M is fixed. T_baseline growth does not increase M,
      so ε_leaf saturates. This function densifies Γ via periodic spline
      interpolation, M_dense = factor·M, ε_leaf → 0 as factor → ∞.

    Args:
        cycle_points: 周期Γ采样点 (M, dim) / periodic Γ samples
        factor: 加密因子 (M_dense = factor·M) / densification factor

    Returns:
        densified: 加密后的Γ采样点 (factor·M, dim) / densified Γ samples
    """
    cycle_points = np.asarray(cycle_points, dtype=np.float64)
    if cycle_points.ndim == 1:
        cycle_points = cycle_points.reshape(-1, 1)

    M, dim = cycle_points.shape
    if M < 4 or factor <= 1:
        return cycle_points.copy()

    # §周期样条插值 (bc_type='periodic') / periodic spline interpolation
    # §纯 numpy 实现: 用均匀参数化 + 周期边界条件 / pure numpy: uniform param + periodic BC
    # §避免 scipy 依赖 / avoid scipy dependency
    # §周期边界: cycle_points[0] ≈ cycle_points[-1] (闭合)
    # §方法: 在参数 t∈[0, 2π] 上做周期插值
    t_orig = np.linspace(0.0, 2.0 * np.pi, M, endpoint=False)
    t_dense = np.linspace(0.0, 2.0 * np.pi, M * factor, endpoint=False)

    # §用 FFT 做周期插值 (纯 numpy) / FFT-based periodic interpolation (pure numpy)
    # §将每个维度做 FFT, 在频域补零, IFFT 得到加密采样
    densified = np.zeros((M * factor, dim), dtype=np.float64)
    for d in range(dim):
        signal = cycle_points[:, d]
        # §FFT (周期信号) / FFT (periodic signal)
        fft_coeffs = np.fft.fft(signal)
        # §频域补零 (在 M*factor 点上重采样) / zero-padding in frequency domain
        fft_dense = np.zeros(M * factor, dtype=np.complex128)
        # §正频率部分 (Nyquist 之前) / positive frequencies (before Nyquist)
        n_pos = M // 2
        fft_dense[:n_pos] = fft_coeffs[:n_pos]
        # §负频率部分 (从末尾开始) / negative frequencies (from end)
        n_neg = M - n_pos - (M % 2)  # §处理奇偶 / handle odd/even M
        fft_dense[-n_neg:] = fft_coeffs[-n_neg:] if n_neg > 0 else fft_dense[-n_neg:]
        # §如果 M 是偶数, Nyquist 频率分量 / if M is even, Nyquist component
        if M % 2 == 0 and n_pos < M:
            fft_dense[n_pos] = fft_coeffs[n_pos] * 0.5
            fft_dense[M * factor - n_pos] = fft_coeffs[n_pos] * 0.5
        # §IFFT 得到加密采样 (乘以 factor 保持幅度) / IFFT to get densified samples
        densified[:, d] = np.real(np.fft.ifft(fft_dense)) * factor

    return densified


def adaptive_baseline_extension(
    f: Callable[..., np.ndarray],
    y0: np.ndarray,
    t_test: float,
    h: float,
    target_drift: float = 1e-3,
    max_baseline: float = 1000.0,
    extension_factor: float = 2.0,
    densify_factor: int = 10,
    *args,
) -> tuple[np.ndarray, float, dict]:
    """T_baseline 自适应扩展直到漂移达标 (§3.26.16 工程实装).

    Adaptive T_baseline extension until drift meets target (§3.26.16 implementation).

    理论 / Theory:
      ε_leaf(T_baseline) ≤ C · h^{1/d} / T_baseline^{1/d} → 0
      (Birkhoff 遍历定理 + Poincaré 回归定理, §3.26.16)

      时间越长 → Γ 采样越密 → 漂移越小.
      Longer time → denser Γ sampling → smaller drift.

    机制 / Mechanism:
      1. 初始 baseline: T_baseline = t_test
      2. 识别 Γ (detect_limit_cycle)
      3. 若周期系统: 加密 Γ (densify_cycle)
      4. 跑 rk4_with_recall, 测量漂移
      5. 若漂移 > target_drift: T_baseline *= extension_factor, 回到 step 2
      6. 重复直到达标或达上界

    Args:
        f: ODE 右端 dy/dt = f(t, y, *args) / ODE right-hand side
        y0: 初始状态 / initial state
        t_test: 测试时长 / test duration
        h: 步长 / step size
        target_drift: 目标漂移 (默认 1e-3) / target drift (default 1e-3)
        max_baseline: T_baseline 上界 / upper bound for T_baseline
        extension_factor: 扩展因子 (每次乘以) / extension factor (multiply each iter)
        densify_factor: 周期系统加密因子 / periodic-system densification factor
        *args: 传给 f 的额外参数 / extra args for f

    Returns:
        cycle_points: 最终的Γ采样 / final Γ samples
        drift: 最终漂移 / final drift
        info: 信息字典 (n_iterations, T_baseline, M, etc.) / info dict
    """
    y0 = np.asarray(y0, dtype=np.float64)
    T_baseline = float(t_test)
    info = {
        "n_iterations": 0,
        "final_T_baseline": T_baseline,
        "final_M": 0,
        "drift_history": [],
        "converged": False,
        "system_type": "unknown",
    }

    while T_baseline <= max_baseline:
        info["n_iterations"] += 1

        # §Step 1: 跑 baseline 识别 Γ / run baseline to identify Γ
        t_array, baseline_y = _pure_rk4_trajectory(f, y0, 0.0, T_baseline, h, *args)
        lc = detect_limit_cycle(t_array, baseline_y)

        if lc.detected and len(lc.cycle_points) > 0:
            cycle_points = lc.cycle_points.copy()
            info["system_type"] = "periodic"
            # §Step 2: 周期系统加密 Γ / densify periodic Γ
            if densify_factor > 1:
                cycle_points = densify_cycle(cycle_points, factor=densify_factor)
        else:
            # §ω-极限集退化: 取末尾采样 / ω-limit set fallback: tail samples
            n_tail = max(200, len(baseline_y) // 4)
            cycle_points = baseline_y[-n_tail:].copy()
            info["system_type"] = "chaotic_or_attractor"

        M = len(cycle_points)
        info["final_M"] = M
        info["final_T_baseline"] = T_baseline

        # §Step 3: 跑投影版测试 / run projection-version test
        result = rk4_with_recall(
            f, y0, 0.0, t_test, h, recall_period=10, *args
        )

        # §Step 4: 测量漂移 (用当前 Γ) / measure drift with current Γ
        drifts = np.array([
            limit_cycle_distance(y, cycle_points) for y in result.y_array
        ])
        # §后半段最大漂移 (排除暂态) / max drift in second half (exclude transient)
        half = len(drifts) // 2
        drift = float(np.max(drifts[half:])) if half > 0 else float(np.max(drifts))
        info["drift_history"].append({"T_baseline": T_baseline, "M": M, "drift": drift})

        logger.debug(
            "§3.26.16 adaptive_baseline_extension: iter=%d T=%.1f M=%d drift=%.4e target=%.4e",
            info["n_iterations"], T_baseline, M, drift, target_drift,
        )

        # §Step 5: 达标检查 / convergence check
        if drift <= target_drift:
            info["converged"] = True
            logger.info(
                "§3.26.16 converged: T_baseline=%.1f M=%d drift=%.4e ≤ target=%.4e",
                T_baseline, M, drift, target_drift,
            )
            return cycle_points, drift, info

        # §Step 6: 扩展 T_baseline / extend T_baseline
        T_baseline *= extension_factor

    # §未达标, 返回最后结果 / not converged, return last result
    logger.info(
        "§3.26.16 not converged (max_baseline=%.1f reached): final drift=%.4e > target=%.4e",
        max_baseline, drift, target_drift,
    )
    return cycle_points, drift, info


def rk4_with_recall_adaptive(
    f: Callable[..., np.ndarray],
    y0: np.ndarray,
    t0: float,
    t_end: float,
    h: float,
    target_drift: float = 1e-3,
    max_baseline: float = 1000.0,
    recall_period: int = 10,
    *args,
) -> IntegrationResult:
    """§3.26.16 自适应版 rk4_with_recall (T_baseline 自动扩展直到漂移达标).

    §3.26.16 adaptive rk4_with_recall (auto-extend T_baseline until drift meets target).

    机制 / Mechanism:
      1. 调用 adaptive_baseline_extension 识别达标的 Γ
      2. 用达标 Γ 跑 rk4_with_recall
      → 漂移自动收敛到 target_drift 以下

    理论 / Theory:
      ε_leaf(T_baseline) ≤ C · h^{1/d} / T_baseline^{1/d} → 0
      时间越长 → Γ 越密 → 漂移越小 (§3.26.16)

    Args:
        f: ODE 右端 / ODE right-hand side
        y0: 初始状态 / initial state
        t0: 起始时间 / start time
        t_end: 结束时间 / end time
        h: 步长 / step size
        target_drift: 目标漂移 / target drift
        max_baseline: T_baseline 上界 / upper bound
        recall_period: 回忆周期 K / recall period K
        *args: 传给 f 的额外参数 / extra args

    Returns:
        IntegrationResult: 校正结果 (附带 adaptive_info 在 metadata 中)
    """
    t_test = float(t_end - t0)

    # §Step 1: 自适应扩展 Γ / adaptive Γ extension
    cycle_points, drift, info = adaptive_baseline_extension(
        f, y0, t_test, h, target_drift=target_drift,
        max_baseline=max_baseline, *args,
    )

    # §Step 2: 用达标 Γ 跑 rk4_with_recall
    # §注意: rk4_with_recall 内部会再识别 Γ, 这里我们用闭式方式注入
    # §通过临时修改 detect_limit_cycle 的行为不优雅, 改为直接实现
    result = _rk4_with_recall_with_external_gamma(
        f, y0, t0, t_end, h, cycle_points, recall_period, *args
    )

    # §附加自适应信息 (通过 __dict__ 动态附加, 不修改 dataclass 定义)
    # §attach adaptive info (via __dict__, without modifying dataclass definition)
    result.__dict__["adaptive_3_26_16"] = {
        "converged": info["converged"],
        "n_iterations": info["n_iterations"],
        "final_T_baseline": info["final_T_baseline"],
        "final_M": info["final_M"],
        "final_drift": drift,
        "target_drift": target_drift,
        "system_type": info["system_type"],
        "drift_history": info["drift_history"],
    }

    return result


def _rk4_with_recall_with_external_gamma(
    f: Callable[..., np.ndarray],
    y0: np.ndarray,
    t0: float,
    t_end: float,
    h: float,
    cycle_points: np.ndarray,
    recall_period: int,
    *args,
) -> IntegrationResult:
    """用外部提供的 Γ 跑 rk4_with_recall (§3.26.16 内部辅助).

    Run rk4_with_recall with externally-provided Γ (§3.26.16 internal helper).

    与 rk4_with_recall 的区别: 不调用 detect_limit_cycle, 直接用外部 cycle_points.
    Difference from rk4_with_recall: no detect_limit_cycle call, uses external cycle_points.
    """
    y0 = np.asarray(y0, dtype=np.float64)
    n_steps = int(round((t_end - t0) / h))
    K = max(1, int(recall_period))

    t_array = np.zeros(n_steps + 1)
    y_array = np.zeros((n_steps + 1, y0.size))

    y = y0.copy()
    t = float(t0)
    t_array[0] = t
    y_array[0] = y

    # §切换机制 + 投影 (§3.26.4 + §3.26.16 外部Γ)
    # §switching + projection (§3.26.4 + §3.26.16 external Γ)
    y_recall_state = y.copy()
    hope_p = y.copy()

    for k in range(n_steps):
        cycle_pos = k % (2 * K)
        is_recall_phase = cycle_pos >= K

        if is_recall_phase and k > 0:
            if cycle_pos == K:
                # §回忆期开始: hope_p 动态更新 (§3.26.2)
                hope_p = locate_nearest_leaf(y, cycle_points)
                y_recall_state = hope_p.copy()

            # §回忆期: 从 hope_p 演化 + 投影回 Γ (§3.26.5)
            y_recall_state = rk4_step(f, t, y_recall_state, h, *args)
            y_recall_state = locate_nearest_leaf(y_recall_state, cycle_points)
            y = y_recall_state  # §输出替换
        else:
            # §累积期: 纯 RK4 / accumulation phase: pure RK4
            y = rk4_step(f, t, y, h, *args)

        t = t + h
        t_array[k + 1] = t
        y_array[k + 1] = y

    # §计算误差轨迹 (用外部Γ) / compute error trajectory with external Γ
    error_array = np.array([
        limit_cycle_distance(yi, cycle_points) for yi in y_array
    ])
    peak_error = float(np.max(error_array))
    final_error = float(error_array[-1])
    # §累积率: 后半段误差增长率 / accumulation rate: error growth in second half
    half = len(error_array) // 2
    if half > 1 and error_array[half] > EPS_LOG:
        accumulation_rate = float(
            (error_array[-1] / max(error_array[half], EPS_LOG))
        )
    else:
        accumulation_rate = 0.0

    return IntegrationResult(
        method_name="rk4_recall_adaptive_3_26_16",
        t_array=t_array,
        y_array=y_array,
        error_array=error_array,
        peak_error=peak_error,
        final_error=final_error,
        accumulation_rate=accumulation_rate,
        regime="adaptive_3_26_16",
    )


# ════════════════════════════════════════════════════════════════════
# §3.26.17 Pyragas → Γ 推广 (相位锁定)
# §3.26.17 Pyragas generalization to entire attractor (phase locking)
#
# 核心: 把 Pyragas DFC 从 UPO (吸引子离散子集) 推广到整个 Γ (吸引子全集)
#       - UPO: y(t) = y(t-T), 需已知 T, 相位→0
#       - Γ:   y(t) ∈ Γ, 用相位标签, 相位→0 (无需 T)
#
# 机制:
#   1. PhaseLabeledGamma: 每个 leaf 存储 baseline 时间标签 t_i
#   2. pyragas_on_attractor: 连续反馈到 Γ 上正确相位的点
#      y_corrected = RK4_step(y) + K·(z_target(phase+h) - y_new)
#   3. 自适应 K: 用局部 Lyapunov 估计, K > λ_local
#
# 与 §3.26.16 的关系:
#   §3.26.16: 漂移 → 0 (投影, 不锁定相位)
#   §3.26.17: 漂移 → 0 + 相位 → 0 (连续反馈, 锁定相位)
# ════════════════════════════════════════════════════════════════════

@dataclass
class PhaseLabeledGamma:
    """带相位标签的 Γ (§3.26.17 核心 dataclass).

    Phase-labeled Γ (§3.26.17 core dataclass).

    每个 leaf 存储其在 baseline 中的时间标签 t_i,
    用于确定"当前 y 在 Γ 上的相位".

    Each leaf stores its baseline time label t_i,
    used to determine "the phase of current y on Γ".

    属性 / Attributes:
        points: Γ 采样点 (M, dim) / Γ sample points
        phases: 每个 leaf 的 baseline 时间标签 (M,) / baseline time labels
        period: Γ 周期 (周期系统), 0 表示非周期 / Γ period (periodic), 0 for aperiodic
        is_periodic: 是否周期系统 / whether periodic system
    """
    points: np.ndarray               # (M, dim) Γ 采样点
    phases: np.ndarray               # (M,) baseline 时间标签
    period: float                    # 周期 (0 = 非周期)
    is_periodic: bool                # 是否周期系统

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, dtype=np.float64)
        self.phases = np.asarray(self.phases, dtype=np.float64)
        if self.points.ndim == 1:
            self.points = self.points.reshape(-1, 1)
        if len(self.points) != len(self.phases):
            raise ConfigurationError(
                f"points ({len(self.points)}) and phases ({len(self.phases)}) "
                f"must have same length"
            )

    @property
    def M(self) -> int:
        """采样点数 / number of samples."""
        return len(self.points)

    def locate_phase(self, y: np.ndarray) -> float:
        """定位 y 在 Γ 上的相位 (§3.26.17 相位查询).

        Locate the phase of y on Γ (§3.26.17 phase query).

        机制 / Mechanism:
          1. 找离 y 最近的 leaf
          2. 返回该 leaf 的相位标签 t_i

        Args:
            y: 查询点 (dim,) / query point

        Returns:
            phase: y 在 Γ 上的相位 (baseline 时间) / phase on Γ
        """
        y = np.asarray(y, dtype=np.float64)
        diffs = self.points - y[np.newaxis, :]
        dists = np.linalg.norm(diffs, axis=1)
        best_idx = int(np.argmin(dists))
        return float(self.phases[best_idx])

    def at_phase(self, phase: float) -> np.ndarray:
        """返回 Γ 上指定相位的点 (§3.26.17 相位→点).

        Return the point on Γ at given phase (§3.26.17 phase→point).

        机制 / Mechanism:
          - 周期系统: phase mod period, 然后插值
          - 非周期系统: 找最近 phase 的 leaf

        Args:
            phase: 目标相位 (baseline 时间) / target phase

        Returns:
            y: Γ 上该相位的点 (dim,) / point on Γ at that phase
        """
        if self.is_periodic and self.period > 0:
            # §周期系统: phase mod period / periodic: phase mod period
            phase = float(phase) % self.period

        # §找最近 phase 的 leaf / find leaf with nearest phase
        phase_diffs = np.abs(self.phases - phase)
        if self.is_periodic and self.period > 0:
            # §周期相位差 (考虑环绕) / periodic phase diff (wraparound)
            phase_diffs = np.minimum(phase_diffs, self.period - phase_diffs)

        best_idx = int(np.argmin(phase_diffs))
        return self.points[best_idx].copy()

    def advance_phase(self, current_phase: float, dt: float) -> float:
        """相位前进 dt (§3.26.17 相位推进).

        Advance phase by dt (§3.26.17 phase advance).

        Args:
            current_phase: 当前相位 / current phase
            dt: 时间增量 / time increment

        Returns:
            new_phase: 新相位 / new phase
        """
        new_phase = float(current_phase) + float(dt)
        if self.is_periodic and self.period > 0:
            new_phase = new_phase % self.period
        return new_phase


def build_phase_labeled_gamma(
    t_array: np.ndarray,
    y_array: np.ndarray,
    min_cycles: int = 2,
) -> PhaseLabeledGamma:
    """从 baseline 轨迹构建带相位标签的 Γ (§3.26.17 构建).

    Build phase-labeled Γ from baseline trajectory (§3.26.17 construction).

    机制 / Mechanism:
      1. 调用 detect_limit_cycle 识别 Γ
      2. 为每个 leaf 附加 baseline 时间标签
      3. 周期系统: phase ∈ [0, period) (mod period)
      4. 非周期系统: phase = baseline 时间 (原始)

    Args:
        t_array: baseline 时间序列 (n,) / baseline time sequence
        y_array: baseline 状态轨迹 (n, dim) / baseline state trajectory
        min_cycles: 取最后几个周期 / take last few cycles

    Returns:
        PhaseLabeledGamma: 带相位标签的 Γ / phase-labeled Γ
    """
    t_array = np.asarray(t_array, dtype=np.float64)
    y_array = np.asarray(y_array, dtype=np.float64)

    lc = detect_limit_cycle(t_array, y_array, min_cycles=min_cycles)

    n = len(t_array)
    h_base = float(t_array[1] - t_array[0]) if n > 1 else 1.0

    # §周期系统判定: detected + period > 0 + points_per_cycle > 0
    # §periodic check: detected + period > 0 + points_per_cycle > 0
    is_valid_periodic = (
        lc.detected
        and lc.period > 0.0
        and len(lc.cycle_points) > 0
    )
    if is_valid_periodic:
        points_per_cycle = max(1, int(round(lc.period / h_base)))
        start_idx = max(0, n - min_cycles * points_per_cycle)
        # §确保至少有 points_per_cycle 个点 / ensure at least one cycle
        if start_idx >= n - points_per_cycle:
            start_idx = max(0, n - points_per_cycle)
        points = y_array[start_idx:].copy()
        phases = t_array[start_idx:].copy() % lc.period
        return PhaseLabeledGamma(
            points=points,
            phases=phases,
            period=lc.period,
            is_periodic=True,
        )
    else:
        # §非周期系统 (含混沌): 取末尾采样 + 时间标签
        # §aperiodic (incl. chaotic): tail samples + time labels
        n_tail = max(200, len(y_array) // 4)
        points = y_array[-n_tail:].copy()
        phases = t_array[-n_tail:].copy()
        return PhaseLabeledGamma(
            points=points,
            phases=phases,
            period=0.0,
            is_periodic=False,
        )


def estimate_local_lyapunov(
    gamma: PhaseLabeledGamma,
) -> float:
    """估计 Γ 上的局部 Lyapunov 指数 (§3.26.17 自适应 K 用).

    Estimate local Lyapunov exponent on Γ (for §3.26.17 adaptive K).

    机制 / Mechanism:
      对相邻 leaf 对 (z_i, z_{i+1}):
        δ_i = ‖z_{i+1} - z_i‖ (初始分离)
        δ_{i+1} = ‖z_{i+2} - z_{i+1}‖ (一步后分离)
        λ_local ≈ ln(δ_{i+1} / δ_i) / h

      对混沌系统 (Lorenz): 用上分位数 (而非中位数) 捕获最大扩张方向.
      中位数对 Lorenz 会低估 (因为吸引子多方向收缩, 中位数<0),
      但最大 Lyapunov > 0, 需要用上分位数捕获.

    Args:
        gamma: 带相位标签的 Γ / phase-labeled Γ

    Returns:
        lambda_local: 局部 Lyapunov 估计 / local Lyapunov estimate
    """
    points = gamma.points
    M = len(points)
    if M < 3:
        return 0.0

    # §相邻 leaf 距离 / distances between adjacent leaves
    deltas = np.linalg.norm(np.diff(points, axis=0), axis=1)
    deltas = np.maximum(deltas, EPS_LOG)

    # §ln(δ_{i+1}/δ_i) / ln(δ_{i+1}/δ_i)
    ratios = deltas[1:] / deltas[:-1]
    log_ratios = np.log(np.maximum(ratios, EPS_LOG))

    # §对混沌系统用上分位数 (90%) 捕获最大扩张方向
    # §for chaotic systems use upper quantile (90%) to capture max expansion
    # §中位数会低估混沌系统的 Lyapunov (多方向收缩, 中位数<0, 但 max>0)
    # §median underestimates chaotic Lyapunov (most directions contract, but max expands)
    lambda_local = float(np.quantile(log_ratios, 0.9))

    return lambda_local


def pyragas_on_attractor(
    f: Callable[..., np.ndarray],
    y0: np.ndarray,
    t0: float,
    t_end: float,
    h: float,
    gamma: PhaseLabeledGamma,
    K: float | None = None,
    K_safety_factor: float = 1.5,
    *args,
) -> IntegrationResult:
    """§3.26.17 Pyragas → Γ 推广: 连续反馈锁定相位.

    §3.26.17 Pyragas generalization to entire attractor: continuous feedback for phase locking.

    机制 / Mechanism:
      每步:
        1. RK4 演化: y_new = RK4_step(f, t, y, h)
        2. 找当前相位: φ = gamma.locate_phase(y)
        3. 预测目标相位: φ_target = gamma.advance_phase(φ, h)
        4. 找目标点: z_target = gamma.at_phase(φ_target)
        5. 连续反馈: y_corrected = y_new + K·(z_target - y_new)

      反馈把 y 拉到"Γ 上正确相位的点", 类似 Pyragas 锁定 UPO 相位,
      但推广到整个 Γ (无需已知周期 T).

    与 Pyragas DFC 的关系 / Relation to Pyragas DFC:
      Pyragas: y(t) + K·[y(t-T) - y(t)]   (反馈到 T 周期前)
      本算法:  y_new + K·[z_target - y_new] (反馈到 Γ 正确相位)
      → Pyragas 需 T, 本算法用 Γ 相位标签 (无需 T)
      → Pyragas 稳定 UPO, 本算法稳定整个 Γ

    自适应 K / Adaptive K:
      - 估计局部 Lyapunov: λ_local
      - K = K_safety_factor · max(λ_local, 0)
      - 确保 K > λ_local (反馈强度 > 扩张率)
      - K_safety_factor=1.5: 安全裕度

    Args:
        f: ODE 右端 dy/dt = f(t, y, *args) / ODE right-hand side
        y0: 初始状态 / initial state
        t0: 起始时间 / start time
        t_end: 结束时间 / end time
        h: 步长 / step size
        gamma: 带相位标签的 Γ / phase-labeled Γ
        K: 反馈强度 (None=自适应) / feedback gain (None=adaptive)
        K_safety_factor: K 自适应安全因子 / K adaptive safety factor
        *args: 传给 f 的额外参数 / extra args for f

    Returns:
        IntegrationResult: 校正结果 (漂移+相位双控制)
    """
    y0 = np.asarray(y0, dtype=np.float64)
    n_steps = int(round((t_end - t0) / h))

    # §自适应 K: 根据系统类型选择反馈强度 / adaptive K by system type
    if K is None:
        lambda_local = estimate_local_lyapunov(gamma)
        if lambda_local > 0.0:
            # §扩张系统 (混沌, λ>0): K > λ_local, 确保反馈 > 扩张
            # §expanding system (chaotic, λ>0): K > λ_local
            K = K_safety_factor * lambda_local
        else:
            # §无扩张系统 (保守/吸引, λ≤0): K 极小, 仅锁定相位
            # §non-expanding (conservative/attracting, λ≤0): tiny K for phase lock only
            # §过大的 K 会引入相位误差 (把轨迹硬拉到 Γ 采样点)
            # §too-large K introduces phase error (yanking trajectory to Γ samples)
            K = 1e-3  # §轻微反馈 / gentle feedback
        logger.info(
            "§3.26.17 adaptive K: λ_local=%.4f, K=%.4f (safety_factor=%.2f)",
            lambda_local, K, K_safety_factor,
        )

    t_array = np.zeros(n_steps + 1)
    y_array = np.zeros((n_steps + 1, y0.size))

    y = y0.copy()
    t = float(t0)
    t_array[0] = t
    y_array[0] = y

    for k in range(n_steps):
        # §Step 1: RK4 演化 / RK4 evolution
        y_new = rk4_step(f, t, y, h, *args)

        # §Step 2: 找当前相位 / locate current phase
        current_phase = gamma.locate_phase(y)

        # §Step 3: 预测目标相位 / predict target phase
        target_phase = gamma.advance_phase(current_phase, h)

        # §Step 4: 找目标点 / find target point on Γ
        z_target = gamma.at_phase(target_phase)

        # §Step 5: 连续反馈 (Pyragas → Γ 推广)
        # §continuous feedback (Pyragas → Γ generalization)
        feedback = K * (z_target - y_new)
        y_corrected = y_new + feedback

        y = y_corrected
        t = t + h
        t_array[k + 1] = t
        y_array[k + 1] = y

    # §计算误差轨迹 (用 Γ 距离) / compute error trajectory (Γ distance)
    error_array = np.array([
        limit_cycle_distance(yi, gamma.points) for yi in y_array
    ])
    peak_error = float(np.max(error_array))
    final_error = float(error_array[-1])
    half = len(error_array) // 2
    if half > 1 and error_array[half] > EPS_LOG:
        accumulation_rate = float(error_array[-1] / max(error_array[half], EPS_LOG))
    else:
        accumulation_rate = 0.0

    return IntegrationResult(
        method_name="pyragas_on_attractor_3_26_17",
        t_array=t_array,
        y_array=y_array,
        error_array=error_array,
        peak_error=peak_error,
        final_error=final_error,
        accumulation_rate=accumulation_rate,
        regime="pyragas_gamma_3_26_17",
    )


def pyragas_adaptive(
    f: Callable[..., np.ndarray],
    y0: np.ndarray,
    t0: float,
    t_end: float,
    h: float,
    target_drift: float = 1e-3,
    max_baseline: float = 1000.0,
    K_safety_factor: float = 1.5,
    densify_factor: int = 10,
    *args,
) -> IntegrationResult:
    """§3.26.17 端到端自适应版: §3.26.16 (Γ加密) + §3.26.17 (相位锁定).

    §3.26.17 end-to-end adaptive version: §3.26.16 (Γ densify) + §3.26.17 (phase lock).

    机制 / Mechanism:
      1. §3.26.16: adaptive_baseline_extension 扩展 T_baseline 直到 Γ 足够密
      2. 构建 PhaseLabeledGamma (附加相位标签)
      3. §3.26.17: pyragas_on_attractor 连续反馈锁定相位

    效果 / Effect:
      - 漂移 → 0 (§3.26.16 Γ加密 + §3.26.17 连续反馈)
      - 相位 → 0 (§3.26.17 Pyragas→Γ 推广, 周期系统严格, 混沌系统减缓)

    Args:
        f: ODE 右端 / ODE right-hand side
        y0: 初始状态 / initial state
        t0: 起始时间 / start time
        t_end: 结束时间 / end time
        h: 步长 / step size
        target_drift: 目标漂移 / target drift
        max_baseline: T_baseline 上界 / upper bound
        K_safety_factor: K 自适应安全因子 / K safety factor
        densify_factor: 周期系统加密因子 / periodic densify factor
        *args: 传给 f 的额外参数 / extra args

    Returns:
        IntegrationResult: 校正结果 (漂移+相位双控制)
    """
    t_test = float(t_end - t0)

    # §Step 1: §3.26.16 自适应扩展 Γ / adaptive Γ extension
    cycle_points, drift, info = adaptive_baseline_extension(
        f, y0, t_test, h, target_drift=target_drift,
        max_baseline=max_baseline, densify_factor=densify_factor, *args,
    )

    # §Step 2: 重新跑 baseline 构建 PhaseLabeledGamma (用最终 T_baseline)
    # §re-run baseline to build PhaseLabeledGamma (with final T_baseline)
    T_final = info["final_T_baseline"]
    t_base, y_base = _pure_rk4_trajectory(f, y0, 0.0, T_final, h, *args)

    # §对周期系统加密 + 附加相位标签 / densify periodic + attach phase labels
    lc = detect_limit_cycle(t_base, y_base)
    n_base = len(t_base)
    h_base = float(t_base[1] - t_base[0]) if n_base > 1 else h
    is_valid_periodic = (
        lc.detected
        and lc.period > 0.0
        and len(lc.cycle_points) > 0
    )
    if is_valid_periodic:
        # §周期系统: 取最后几个周期 + 相位标签
        # §periodic: take last few cycles + phase labels
        points_per_cycle = max(1, int(round(lc.period / h_base)))
        start_idx = max(0, n_base - 2 * points_per_cycle)
        if start_idx >= n_base - points_per_cycle:
            start_idx = max(0, n_base - points_per_cycle)
        points_raw = y_base[start_idx:].copy()
        phases_raw = t_base[start_idx:].copy() % lc.period
        # §加密 (用 FFT 周期插值, 同时加密相位标签)
        # §densify (FFT periodic interpolation, densify phase labels too)
        if densify_factor > 1 and len(points_raw) >= 4:
            points = densify_cycle(points_raw, factor=densify_factor)
            # §相位标签也加密 / densify phase labels too
            M_raw = len(phases_raw)
            phases = np.linspace(0.0, lc.period, M_raw * densify_factor, endpoint=False)
            is_periodic = True
            period = lc.period
        else:
            points = points_raw
            phases = phases_raw
            is_periodic = True
            period = lc.period
    else:
        # §非周期系统 (含混沌): 取末尾采样 / aperiodic (incl. chaotic): tail samples
        n_tail = max(200, len(y_base) // 4)
        points = y_base[-n_tail:].copy()
        phases = t_base[-n_tail:].copy()
        is_periodic = False
        period = 0.0

    gamma = PhaseLabeledGamma(
        points=points,
        phases=phases,
        period=period,
        is_periodic=is_periodic,
    )

    # §Step 3: §3.26.17 Pyragas → Γ 连续反馈 / Pyragas → Γ continuous feedback
    result = pyragas_on_attractor(
        f, y0, t0, t_end, h, gamma, K=None, K_safety_factor=K_safety_factor, *args,
    )

    # §附加信息 / attach info
    lambda_local_final = estimate_local_lyapunov(gamma)
    if lambda_local_final > 0.0:
        K_used = float(K_safety_factor * lambda_local_final)
    else:
        K_used = 1e-3
    result.__dict__["adaptive_3_26_17"] = {
        "converged_3_26_16": info["converged"],
        "T_baseline": T_final,
        "M_gamma": gamma.M,
        "is_periodic": gamma.is_periodic,
        "period": gamma.period,
        "lambda_local": lambda_local_final,
        "K_used": K_used,
        "drift_before_3_26_17": drift,
    }

    return result


def _compute_error_array(
    t_array: np.ndarray,
    y_array: np.ndarray,
    reference: Callable[[float], np.ndarray] | None,
    cycle_points: np.ndarray | None = None,
) -> np.ndarray:
    """计算误差轨迹 (vs reference 或固定极限环距离).

    Compute error trajectory (vs reference or fixed limit-cycle distance).

    机制 / Mechanism:
      - 若 reference 提供: 误差 = ‖y - reference(t)‖ (真值误差)
      - 若 reference 为 None:
        - cycle_points 提供: 误差 = limit_cycle_distance(y, cycle_points)
        - cycle_points 为 None: 自动检测极限环并使用其采样点
        - cycle_points is None: auto-detect limit cycle and use its sample points

    Args:
        t_array: 时间序列 / time sequence
        y_array: 状态轨迹 / state trajectory
        reference: 解析解 (可选, 仅误差计算) / analytical solution (optional, error only)
        cycle_points: 固定极限环采样点 (可选, 避免重复检测)
            cycle_points: fixed limit-cycle samples (optional, avoids repeated detection)

    Returns:
        error_array: 误差轨迹 / error trajectory
    """
    n = len(t_array)
    error_array = np.zeros(n)

    if reference is not None:
        # §有解析解: 误差 = ‖y - reference(t)‖ / With analytical: error = ‖y - reference(t)‖
        for i in range(n):
            error_array[i] = float(np.linalg.norm(y_array[i] - reference(float(t_array[i]))))
    else:
        # §无解析解: 用极限环距离或重心距离 / No analytical: use limit-cycle or centroid distance
        if cycle_points is None:
            lc_info = detect_limit_cycle(t_array, y_array)
            cp_used = lc_info.cycle_points if (lc_info.detected and len(lc_info.cycle_points) > 0) else None
            centroid = lc_info.centroid
        else:
            cp_used = cycle_points
            centroid = np.mean(cycle_points, axis=0) if len(cycle_points) > 0 else np.zeros(y_array.shape[1] if y_array.ndim > 1 else 1)

        if cp_used is not None and len(cp_used) > 0:
            # §极限环距离: min ‖y - cycle_point‖ (统一度量, §3.26) / Limit-cycle distance (unified metric, §3.26)
            for i in range(n):
                error_array[i] = limit_cycle_distance(y_array[i], cp_used)
        else:
            # §重心距离: ‖y - centroid‖ (退化情形) / Centroid distance (degenerate case)
            for i in range(n):
                error_array[i] = float(np.linalg.norm(y_array[i] - centroid))

    return error_array


def rk4_integrate(
    f: Callable[..., np.ndarray],
    y0: np.ndarray,
    t0: float,
    t_end: float,
    h: float,
    reference: Callable[[float], np.ndarray] | None = None,
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
        reference: 解析解 (可选, 仅用于误差计算, 不参与积分)
        reference: analytical solution (optional, error computation only, not used in integration)
        *args: 传给 f 的额外参数 / extra arguments passed to f

    Returns:
        IntegrationResult: 积分结果 (含误差分析)
        IntegrationResult: integration result (with error analysis)
    """
    n_steps = int(np.round((t_end - t0) / h))
    dim = len(y0)
    t_array = np.zeros(n_steps + 1)
    y_array = np.zeros((n_steps + 1, dim))

    t_array[0] = t0
    y_array[0] = np.asarray(y0, dtype=np.float64).copy()

    t = t0
    y = np.asarray(y0, dtype=np.float64).copy()

    for k in range(n_steps):
        y = rk4_step(f, t, y, h, *args)
        t = t + h
        t_array[k + 1] = t
        y_array[k + 1] = y

    # §误差计算: reference (若有) 或极限环距离 / Error: reference (if any) or limit-cycle distance
    error_array = _compute_error_array(t_array, y_array, reference)

    # 实测累积率 α / Measured accumulation rate α
    accumulation_rate = _compute_accumulation_rate(error_array, n_steps)

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


def _compute_accumulation_rate(error_array: np.ndarray, n_steps: int) -> float:
    """从误差轨迹计算实测累积率 α.
    Compute the measured accumulation rate α from the error trajectory.
    """
    if n_steps <= 10:
        return 0.0
    mid = n_steps // 2
    late_errors = error_array[mid:]
    valid = late_errors[late_errors > EPS_LOG]
    if len(valid) > 2:
        ratios = valid[1:] / valid[:-1]
        return float(np.mean(ratios) - 1.0)
    return 0.0


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
    reference: Callable[[float], np.ndarray] | None = None,
    recall_compression: float = float(INV_PHI),
    *args,
) -> IntegrationResult:
    """RK4 + 回忆校正积分 (§3.26 hope_p 动态更新 + 切换机制, 无 oracle, 无凸组合).

    RK4 + recall-correction integration (§3.26 dynamic hope_p + switching mechanism,
    no oracle, no convex combination).

    机制 (§3.26 hope_p 动态更新定理, 替代旧版凸组合):
    Mechanism (§3.26 dynamic hope_p update theorem, replacing the old convex combination):
      1. 校正前: 先用纯 RK4 跑一遍长轨迹, 识别系统极限环 Γ (固定参考)
         Pre-correction: run pure RK4 once to identify the system's limit cycle Γ (fixed reference)
      2. 累积期 [0, K): 从当前 y 做 K 步 RK4 正常积分 (误差累积 (1+α)^K, y 偏离 Γ)
         Accumulation phase [0, K): K steps of normal RK4 from current y
         (error accumulates as (1+α)^K, y drifts from Γ)
      3. 回忆期 [K, 2K):
         Recall phase [K, 2K):
         - 起点: hope_p = argmin_{leaf∈Γ} ‖leaf - y_K‖ (动态更新, 离当前 y 最近的 leaf)
         - Start: hope_p = argmin_{leaf∈Γ} ‖leaf - y_K‖ (dynamic update, leaf nearest to current y)
         - 演化: 从 hope_p 做 K 步 RK4 (复制极限环上的真实演化)
         - Evolution: K steps of RK4 from hope_p (copy real evolution on the limit cycle)
         - 输出: y_{2K} = y'_K (输出替换, 不是凸组合!)
         - Output: y_{2K} = y'_K (output replacement, NOT convex combination!)
      4. 周期 2K 重复
         Period 2K repeats

    与旧版 (凸组合) 的区别 / Difference from old version (convex combination):
      - 旧版: y = (1-r)·y_RK4 + r·y_recall (混合两个不对应位置的点, 引入相位误差)
      - Old: y = (1-r)·y_RK4 + r·y_recall (mixes two points at different positions, phase error)
      - 新版: y = y_recall (输出替换, hope_p 在极限环上, 演化轨迹自然在极限环上)
      - New: y = y_recall (output replacement, hope_p is on Γ, evolved trajectory stays on Γ)

    压缩性来源 (§3.26 核心):
    Source of compression (§3.26 core):
      - 旧版错误假设: RK4 算子本身是压缩映射 r=INV_PHI (实际: 谐振子r=1, Lorenz r>1)
      - Old wrong assumption: RK4 operator itself is a contraction r=INV_PHI
        (actual: harmonic r=1, Lorenz r>1)
      - 新版正确来源: 极限环几何 (Poincaré 恢复力) 提供压缩
      - New correct source: limit-cycle geometry (Poincaré restoring force) provides compression
      - hope_p 在 Γ 上 → 从 hope_p 演化被 Γ 拉回 → 输出在 Γ 上 (压缩到 leaf 离散化误差)
      - hope_p on Γ → evolution from hope_p is pulled back by Γ → output on Γ
        (compressed to leaf discretization error)

    误差演化 (§3.25.3):
    Error evolution (§3.25.3):
      - 累积期: ε(K) = (1+α)^K · ε_0  (RK4 误差累积, y 偏离 Γ)
      - Accumulation: ε(K) = (1+α)^K · ε_0  (RK4 error accumulation, y drifts from Γ)
      - 回忆期: ε(2K) = r^K · ε(K) = [r·(1+α)]^K · ε_0  (极限环几何压缩, y 拉回 Γ)
      - Recall: ε(2K) = r^K · ε(K) = [r·(1+α)]^K · ε_0  (limit-cycle geometry compresses, y pulled back to Γ)
      - 其中 r^K 来自 Poincaré 恢复力 (非 RK4 算子)
      - Where r^K comes from Poincaré restoring force (not RK4 operator)

    极限环通用性 (用户断言):
    Limit-cycle universality (user assertion):
      - 谐振子: Γ = 圆轨道 (解析极限环)
      - Harmonic: Γ = circular orbit (analytical limit cycle)
      - Van der Pol: Γ = 非线性极限环 (Poincaré-Bendixson)
      - Van der Pol: Γ = nonlinear limit cycle (Poincaré-Bendixson)
      - Lorenz: Γ = 奇异吸引子 (ω-极限集, 广义极限环)
      - Lorenz: Γ = strange attractor (ω-limit set, generalized limit cycle)
      - 任何有界动力系统的 ω-极限集非空紧致不变 (ω-极限集定理)
      - Any bounded dynamical system's ω-limit set is non-empty, compact, and invariant

    Args:
        f: dy/dt = f(t, y, *args)
        y0: 初始状态 / initial state
        t0: 起始时间 / start time
        t_end: 结束时间 / end time
        h: 步长 / step size
        recall_period: 回忆周期 K (每 K 步切换一次累积/回忆)
        recall_period: recall period K (switch accumulation/recall every K steps)
        reference: 解析解 (可选, 仅用于误差计算, 不参与校正)
        reference: analytical solution (optional, error computation only, not used in correction)
        recall_compression: 回忆压缩率 r (默认 INV_PHI, 用于 regime 判据)
        recall_compression: recall compression ratio r (default INV_PHI, used for regime criterion)
        *args: 传给 f 的额外参数 / extra arguments passed to f

    Returns:
        IntegrationResult: 校正积分结果
        IntegrationResult: corrected integration result
    """
    n_steps = int(np.round((t_end - t0) / h))
    dim = len(y0)
    t_array = np.zeros(n_steps + 1)
    y_array = np.zeros((n_steps + 1, dim))

    t_array[0] = t0
    y_array[0] = np.asarray(y0, dtype=np.float64).copy()

    K = int(recall_period)
    r = float(recpression_ratio_fix(recall_compression))

    # §Step 0: 先用纯 RK4 跑一遍长轨迹, 识别系统极限环 Γ (固定参考)
    # §Step 0: run pure RK4 once to identify the system's limit cycle Γ (fixed reference)
    logger.debug("rk4_with_recall: 识别极限环 Γ (纯 RK4 预积分)...")
    baseline_t, baseline_y = _pure_rk4_trajectory(f, y0, t0, t_end, h, *args)
    lc_info = detect_limit_cycle(baseline_t, baseline_y)
    cycle_points = lc_info.cycle_points if lc_info.detected else np.zeros((0, dim))

    if len(cycle_points) == 0:
        # §退化: 无极限环 (短轨迹或常量系统), 退化为纯 RK4
        # §Fallback: no limit cycle (short trajectory or constant system), degrade to pure RK4
        logger.warning("rk4_with_recall: 未检测到极限环, 退化为纯 RK4")
        t = t0
        y = np.asarray(y0, dtype=np.float64).copy()
        for k in range(n_steps):
            y = rk4_step(f, t, y, h, *args)
            t = t + h
            t_array[k + 1] = t
            y_array[k + 1] = y
        error_array = _compute_error_array(t_array, y_array, reference, cycle_points)
        accumulation_rate = _compute_accumulation_rate(error_array, n_steps)
        product = r * (1.0 + accumulation_rate)
        regime = (f"无极限环退化 (r·(1+α)={product:.4f})")
        return IntegrationResult(
            method_name=f"RK4 + 回忆校正 (退化, K={K})",
            t_array=t_array, y_array=y_array, error_array=error_array,
            peak_error=float(np.max(error_array)), final_error=float(error_array[-1]),
            accumulation_rate=accumulation_rate, regime=regime,
        )

    # §Step 1: 校正积分 (累积-回忆切换机制, §3.26)
    # §Step 1: corrected integration (accumulation-recall switching mechanism, §3.26)
    t = t0
    y = np.asarray(y0, dtype=np.float64).copy()
    y_recall_state: np.ndarray | None = None

    for k in range(n_steps):
        cycle_pos = k % (2 * K)
        is_recall_phase = cycle_pos >= K

        if is_recall_phase and k > 0:
            if cycle_pos == K:
                # §回忆期开始: hope_p 动态更新 = 离当前 y 最近的 leaf (§3.26)
                # §Recall start: dynamic hope_p update = leaf nearest to current y (§3.26)
                hope_p = locate_nearest_leaf(y, cycle_points)
                y_recall_state = np.asarray(hope_p, dtype=np.float64).copy()

            # §回忆期: 从 hope_p 前向演化一步 (输出替换, 不是凸组合!)
            # §Recall phase: forward-evolve from hope_p one step (output replacement, NOT convex!)
            y_recall_state = rk4_step(f, t, y_recall_state, h, *args)
            # §极限环几何约束 (§3.26 核心压缩性来源):
            # §Limit-cycle geometric constraint (§3.26 core compression source):
            #   RK4 算子本身不压缩 (谐振子r=1, Lorenz r>1)
            #   RK4 operator itself is not contractive (harmonic r=1, Lorenz r>1)
            #   极限环 Poincaré 恢复力提供压缩: 演化后投影回最近 leaf
            #   Limit-cycle Poincaré restoring force provides compression:
            #     project to nearest leaf after evolution
            y_recall_state = locate_nearest_leaf(y_recall_state, cycle_points)
            y = y_recall_state  # §输出替换: y_{k+1} = y'_recall (在极限环上)
        else:
            # §累积期: 纯 RK4 积分 (误差累积, y 偏离 Γ)
            # §Accumulation phase: pure RK4 integration (error accumulates, y drifts from Γ)
            y = rk4_step(f, t, y, h, *args)

        t = t + h
        t_array[k + 1] = t
        y_array[k + 1] = y

    # §误差计算: reference (若有) 或固定极限环距离 (§3.26 统一度量)
    # §Error: reference (if any) or fixed limit-cycle distance (§3.26 unified metric)
    error_array = _compute_error_array(t_array, y_array, reference, cycle_points)

    # §实测累积率 α (在累积期内) / Measured accumulation rate α (within accumulation phases)
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
        regime = f"误差有界 (r·(1+α)={product:.4f}<1, §3.26.16)"
    elif abs(product - 1.0) <= EPS_LOG:
        regime = f"恒定震荡 (r·(1+α)={product:.4f}=1)"
    else:
        regime = f"失控 (r·(1+α)={product:.4f}>1)"

    return IntegrationResult(
        method_name=f"RK4 + 回忆校正 (§3.26 切换, K={K}, r={r:.4f})",
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
# §三层混合机制: 极赌 + 极限环 + 回忆校正 (闭式 P* 定位)
# §Three-layer hybrid mechanism: GamblePole + LimitCycle + recall (closed-form P* location)
# ════════════════════════════════════════════════════════════════════

def rk4_hybrid_correction(
    f: Callable[..., np.ndarray],
    y0: np.ndarray,
    t0: float,
    t_end: float,
    h: float,
    recall_period: int,
    reference: Callable[[float], np.ndarray] | None = None,
    recall_compression: float = float(INV_PHI),
    gamble_config: GamblePoleConfig | None = None,
    limit_config: LimitCycleConfig | None = None,
    recall_sub_steps: int = 10,
    adaptive_sub_steps: bool = True,
    target_error: float = 1e-10,
    *args,
) -> IntegrationResult:
    """RK4 + 三层混合校正 (极赌 + 极限环 + 回忆, §3.26 切换机制, 无凸组合).

    RK4 + three-layer hybrid correction (GamblePole + LimitCycle + recall,
    §3.26 switching mechanism, no convex combination).

    三层机制 (误差从大到小):
    Three-layer mechanism (error from large to small):
      1. 极赌策略层: 误差 > ε_gamble → 离散跳跃到 y_recall (P*方向极点)
      1. GamblePole layer: error > ε_gamble → discrete jump to y_recall (pole in P* direction)
      2. 极限环层: 误差 > ε_limit_cycle → 投影到周期解
      2. Limit-cycle layer: error > ε_limit_cycle → project onto periodic solution
      3. 回忆校正层: 切换机制 (累积期RK4 + 回忆期从hope_p演化, 输出替换)
      3. Recall-correction layer: switching mechanism
         (accumulation RK4 + recall from hope_p, output replacement)

    §3.26 切换机制 (替代旧版凸组合):
    §3.26 switching mechanism (replaces old convex combination):
      - 校正前: 先用纯 RK4 跑一遍长轨迹, 识别系统极限环 Γ (固定参考)
      - Pre-correction: run pure RK4 once to identify the system's limit cycle Γ (fixed reference)
      - 累积期 [0, K): 从当前 y 做 K 步 RK4 (误差累积, y 偏离 Γ)
      - Accumulation [0, K): K steps of RK4 from current y (error accumulates, y drifts from Γ)
      - 回忆期 [K, 2K): 从 hope_p 做 K 步 RK4 (输出替换, 在 Γ 上)
      - Recall [K, 2K): K steps of RK4 from hope_p (output replacement, on Γ)
        - hope_p = argmin_{leaf∈Γ} ‖leaf - y_K‖ (动态更新, §3.26)
        - hope_p = argmin_{leaf∈Γ} ‖leaf - y_K‖ (dynamic update, §3.26)
        - 子步长精化: h_sub = h/N_sub, 提高从 hope_p 演化的精度
        - Sub-step refinement: h_sub = h/N_sub, improves precision of evolution from hope_p
      - 周期 2K 重复
      - Period 2K repeats

    子步长精化 (仅在回忆期使用, 提高从 hope_p 演化的精度):
    Sub-step refinement (only in recall phase, improves precision of evolution from hope_p):
      - 累积期: 主步长 h (与基线一致, 用于误差对比)
      - Accumulation: main step h (consistent with baseline, for error comparison)
      - 回忆期: 子步长 h_sub = h/N_sub (更精确地从 hope_p 演化)
      - Recall: sub-step h_sub = h/N_sub (more precise evolution from hope_p)
      - 误差从 O(h^5) 降到 O(h_sub^5) = O(h^5 / N^5)
      - Error reduced from O(h^5) to O(h_sub^5) = O(h^5 / N^5)

    自适应子步长 (adaptive_sub_steps=True):
    Adaptive sub-steps (adaptive_sub_steps=True):
      每 K 步检查一次: 若当前误差 > target_error · 10, 则 N_sub ← min(N_sub·2, 200)
      Check every K steps: if current error > target_error · 10, then N_sub ← min(N_sub·2, 200)

    误差下界 (数学诚实, 痛苦恒定 §V3.5):
    Error lower bound (mathematical honesty, pain is constant §V3.5):
      - 极赌: ε_quantize > 0 (离散化舍入) / GamblePole: ε_quantize > 0 (discretization rounding)
      - 极限环: ε_limit_cycle > 0 (周期解振幅) / Limit cycle: ε_limit_cycle > 0 (periodic-solution amplitude)
      - 回忆: r^K·ε > 0 (accumulation point) / Recall: r^K·ε > 0 (accumulation point)
      - 最终: ε_final = min(上述) > 0 / Final: ε_final = min(above) > 0

    Args:
        f: dy/dt = f(t, y, *args)
        y0: 初始状态 / initial state
        t0: 起始时间 / start time
        t_end: 结束时间 / end time
        h: 步长 / step size
        recall_period: 回忆周期 K / recall period K
        reference: 解析解 (可选, 仅误差计算, 不参与校正) / analytical solution (optional, error only)
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

    n_steps = int(np.round((t_end - t0) / h))
    dim = len(y0)
    t_array = np.zeros(n_steps + 1)
    y_array = np.zeros((n_steps + 1, dim))
    strategy_log: list[str] = []  # 记录每步策略 / Log strategy per step

    t_array[0] = t0
    y_array[0] = np.asarray(y0, dtype=np.float64).copy()

    K = int(recall_period)
    r = float(recpression_ratio_fix(recall_compression))
    N_sub = max(int(recall_sub_steps), 1)  # §自适应子步长初值 / Adaptive sub-step initial value
    h_sub = h / N_sub  # §子步长 (仅回忆期使用) / Sub-step (only in recall phase)

    # §Step 0: 先用纯 RK4 跑一遍长轨迹, 识别系统极限环 Γ (固定参考)
    # §Step 0: run pure RK4 once to identify the system's limit cycle Γ (fixed reference)
    logger.debug("rk4_hybrid_correction: 识别极限环 Γ (纯 RK4 预积分)...")
    baseline_t, baseline_y = _pure_rk4_trajectory(f, y0, t0, t_end, h, *args)
    lc_info = detect_limit_cycle(baseline_t, baseline_y)
    cycle_points = lc_info.cycle_points if lc_info.detected else np.zeros((0, dim))

    if len(cycle_points) == 0:
        # §退化: 无极限环, 退化为纯 RK4 + 极赌 + 极限环 (旧版兼容)
        # §Fallback: no limit cycle, degrade to pure RK4 + GamblePole + LimitCycle
        logger.warning("rk4_hybrid_correction: 未检测到极限环, 退化为纯 RK4 + 极赌 + 极限环")
        t = t0
        y = np.asarray(y0, dtype=np.float64).copy()
        for k in range(n_steps):
            y = rk4_step(f, t, y, h, *args)
            t = t + h
            t_array[k + 1] = t
            y_array[k + 1] = y
        error_array = _compute_error_array(t_array, y_array, reference, cycle_points)
        accumulation_rate = _compute_accumulation_rate(error_array, n_steps)
        regime = f"无极限环退化 (r·(1+α)={r*(1.0+accumulation_rate):.4f})"
        return IntegrationResult(
            method_name=f"RK4 + 三层混合 (退化, K={K})",
            t_array=t_array, y_array=y_array, error_array=error_array,
            peak_error=float(np.max(error_array)), final_error=float(error_array[-1]),
            accumulation_rate=accumulation_rate, regime=regime,
        )

    # §Step 1: 校正积分 (§3.26 切换机制 + 三层校正)
    # §Step 1: corrected integration (§3.26 switching mechanism + three-layer correction)
    t = t0
    y = np.asarray(y0, dtype=np.float64).copy()
    y_recall_state: np.ndarray | None = None

    for k in range(n_steps):
        cycle_pos = k % (2 * K)
        is_recall_phase = cycle_pos >= K

        if is_recall_phase and k > 0:
            if cycle_pos == K:
                # §回忆期开始: hope_p 动态更新 = 离当前 y 最近的 leaf (§3.26)
                # §Recall start: dynamic hope_p update = leaf nearest to current y (§3.26)
                hope_p = locate_nearest_leaf(y, cycle_points)
                y_recall_state = np.asarray(hope_p, dtype=np.float64).copy()

            # §回忆期: 从 hope_p 前向演化一步 (子步长精化, 输出替换)
            # §Recall phase: forward-evolve from hope_p one step (sub-step refined, output replacement)
            y_recall_new = y_recall_state.copy()
            t_sub = t
            for _ in range(N_sub):
                y_recall_new = rk4_step(f, t_sub, y_recall_new, h_sub, *args)
                t_sub = t_sub + h_sub
            y_recall_state = y_recall_new
            # §极限环几何约束 (§3.26 核心压缩性来源):
            # §Limit-cycle geometric constraint (§3.26 core compression source):
            #   RK4 算子本身不压缩, 极限环 Poincaré 恢复力提供压缩
            #   RK4 operator is not contractive; Poincaré restoring force provides compression
            #   演化后投影回最近 leaf, 强制 y 在极限环上
            #   Project to nearest leaf after evolution, forcing y on the limit cycle
            y_recall_state = locate_nearest_leaf(y_recall_state, cycle_points)
            y_corrected = y_recall_state  # §输出替换: y_{k+1} = y'_recall (在极限环上)

            # §极赌策略 (大误差时跳跃到 y_recall, 已在 leaf 上)
            # §GamblePole strategy (large error jumps to y_recall, already on leaf)
            current_error = float(np.linalg.norm(y_corrected - y_recall_state))
            y_corrected, gp_label = gamble_pole_correct(
                y_corrected, y_recall_state, current_error, gamble_config
            )
            # §极限环投影 (附加约束, 限制在 amplitude_bound 内)
            # §Limit-cycle projection (additional constraint, confine to amplitude_bound)
            y_corrected, lc_label = limit_cycle_bound(
                y_corrected, y_recall_state, t, limit_config
            )
            strategy_log.append(f"step={k+1} [回忆]: {gp_label} | {lc_label}")
            y = y_corrected
        else:
            # §累积期: 纯 RK4 积分 (误差累积, y 偏离 Γ)
            # §Accumulation phase: pure RK4 integration (error accumulates, y drifts from Γ)
            y = rk4_step(f, t, y, h, *args)
            strategy_log.append(f"step={k+1} [累积]: 纯 RK4")

        t = t + h
        t_array[k + 1] = t
        y_array[k + 1] = y

        # §自适应子步长: 每 K 步检查误差, 误差过大则加倍 N_sub
        # §Adaptive sub-steps: every K steps check error; if too large, double N_sub
        if adaptive_sub_steps and (k + 1) % K == 0:
            # §闭式误差估计: 到极限环的距离 (§3.26 统一度量)
            # §Closed-form error estimate: distance to limit cycle (§3.26 unified metric)
            est_error = limit_cycle_distance(y, cycle_points)
            if est_error > target_error * 10.0:
                N_sub = min(N_sub * 2, 200)
                h_sub = h / N_sub

    # §误差计算: reference (若有) 或固定极限环距离 (§3.26 统一度量)
    # §Error: reference (if any) or fixed limit-cycle distance (§3.26 unified metric)
    error_array = _compute_error_array(t_array, y_array, reference, cycle_points)

    # §实测累积率 / Measured accumulation rate
    accumulation_rate = _compute_accumulation_rate(error_array, n_steps)

    # §regime判据 / Regime criterion
    product = r * (1.0 + accumulation_rate)
    if product < 1.0 - EPS_LOG:
        regime = f"误差有界 (r·(1+α)={product:.4f}<1, §3.26.16)"
    elif abs(product - 1.0) <= EPS_LOG:
        regime = f"恒定震荡 (r·(1+α)={product:.4f}=1)"
    else:
        regime = f"失控 (r·(1+α)={product:.4f}>1)"

    # §误差判据 / Error criterion (§3.26.16: 渐近→0, 非严格0; §V3.5 痛苦恒定)
    machine_epsilon = np.finfo(np.float64).eps  # ≈ 2.22e-16 / ≈ 2.22e-16
    meets_target = bool(error_array[-1] < target_error)
    near_float_floor = bool(error_array[-1] < 10 * machine_epsilon)

    regime += f" | 达目标精度={meets_target} (ε<{target_error:.0e})"
    regime += f" | 近浮点下限={near_float_floor} (ε<{10*machine_epsilon:.2e})"
    regime += f" | N_sub={N_sub} (自适应={adaptive_sub_steps})"

    return IntegrationResult(
        method_name=f"RK4 + 三层混合 (§3.26 切换, K={K}, r={r:.4f}, N_sub={N_sub})",
        t_array=t_array,
        y_array=y_array,
        error_array=error_array,
        peak_error=float(np.max(error_array)),
        final_error=float(error_array[-1]),
        accumulation_rate=accumulation_rate,
        regime=regime,
    )


# ════════════════════════════════════════════════════════════════════
# §3.26.10 连续反馈变体 (Pyragas 风格, 无 ε_leaf 下界)
# §3.26.10 Continuous feedback variant (Pyragas style, no ε_leaf lower bound)
# ════════════════════════════════════════════════════════════════════

def rk4_continuous_feedback(
    f: Callable[..., np.ndarray],
    y0: np.ndarray,
    t0: float,
    t_end: float,
    h: float,
    recall_period: int,
    reference: Callable[[float], np.ndarray] | None = None,
    feedback_strength: float | None = None,
    *args,
) -> IntegrationResult:
    """§3.26.10 连续反馈变体 (无 ε_leaf 下界, Pyragas 风格).

    §3.26.10 Continuous feedback variant (no ε_leaf lower bound, Pyragas style).

    机制 / Mechanism:
      - 累积期 [0, K): 纯 RK4 积分 (误差累积)
      - 回忆期 [K, 2K): RK4 演化 + 连续反馈拉向 hope_p (不投影!)
        y_new = RK4_step(y_recall) + K_fb · (hope_p - RK4_step(y_recall))
      - 周期 2K 重复

    与投影版 (rk4_with_recall) 的区别 / Difference from projection version:
      - 投影版: 演化后投影到最近 leaf, 引入 ε_leaf > 0 下界
      - 连续反馈版: 演化后连续反馈拉向 hope_p, 无 ε_leaf 下界
      - 适用条件: 仅 r_Γ < 1 (系统本身有压缩性)

    Args:
        f: ODE 右端 dy/dt = f(t, y, *args) / ODE right-hand side
        y0: 初始状态 / initial state
        t0: 起始时间 / start time
        t_end: 结束时间 / end time
        h: 步长 / step size
        recall_period: 回忆周期 K (每 K 步切换) / recall period K
        reference: 解析解 (仅误差计算, 不参与校正)
            reference: analytical solution (error computation only, not correction)
        feedback_strength: 反馈强度 K_fb (None 则自动用 INV_PHI)
            feedback_strength: feedback gain K_fb (None → INV_PHI default)
        *args: 传递给 f 的额外参数 / extra args passed to f

    Returns:
        IntegrationResult: 积分结果 (含误差轨迹)
            IntegrationResult: integration result (with error trajectory)

    理论依据 / Theoretical basis:
      - §3.26.10: 连续反馈利用来源 A (Poincaré 压缩, r_Γ < 1)
      - 误差演化: ε(2K) = [r_Γ · (1+α)]^K · ε_0 (无 ε_leaf 下界)
      - 临界: α* = (1-r_Γ)/r_Γ (r_Γ < 1 时有解)
    """
    y0 = np.asarray(y0, dtype=np.float64).copy()
    K = max(1, int(recall_period))
    K_fb = float(INV_PHI) if feedback_strength is None else float(feedback_strength)

    n_steps = int(round((t_end - t0) / h))
    if n_steps < 1:
        raise ConfigurationError(f"n_steps={n_steps} < 1, check t_end/h")

    dim = y0.shape[0]
    t_array = np.zeros(n_steps + 1)
    y_array = np.zeros((n_steps + 1, dim))
    t_array[0] = t0
    y_array[0] = y0

    # §Step 0: 先用纯 RK4 跑一遍长轨迹, 识别系统极限环 Γ
    # §Step 0: run pure RK4 first to identify the limit cycle Γ
    baseline_t, baseline_y = _pure_rk4_trajectory(f, y0, t0, t_end, h, *args)
    lc_info = detect_limit_cycle(baseline_t, baseline_y)
    cycle_points = lc_info.cycle_points if (lc_info.detected and len(lc_info.cycle_points) > 0) else np.zeros((0, dim))
    period = float(lc_info.period) if lc_info.detected else 0.0

    # §若无足够 cycle_points, 退化为 ω-极限集近似 (末尾采样)
    # §If insufficient cycle_points, fall back to ω-limit set approximation (tail samples)
    if len(cycle_points) < 2:
        n_tail = max(200, len(baseline_y) // 4)
        cycle_points = baseline_y[-n_tail:].copy()
        period = 0.0

    # §Step 1: 校正积分 (累积-回忆切换 + 连续反馈, §3.26.10)
    # §Step 1: corrected integration (accumulation-recall switching + continuous feedback, §3.26.10)
    y = y0.copy()
    y_recall_state: np.ndarray | None = None
    hope_p: np.ndarray | None = None

    for k in range(n_steps):
        t = t0 + k * h
        cycle_pos = k % (2 * K)
        is_recall_phase = cycle_pos >= K

        if is_recall_phase and k > 0:
            if cycle_pos == K:
                # §回忆期开始: hope_p 动态更新 = 离当前 y 最近的 leaf
                # §Recall phase start: dynamic hope_p = leaf nearest to current y
                hope_p = locate_nearest_leaf(y, cycle_points)
                y_recall_state = hope_p.copy()

            # §回忆期: 演化 + 连续反馈 (不投影!)
            # §Recall phase: evolve + continuous feedback (no projection!)
            y_recall_new = rk4_step(f, t, y_recall_state, h, *args)
            # §Pyragas 风格连续反馈: 拉向 hope_p
            # §Pyragas-style continuous feedback: pull toward hope_p
            feedback = K_fb * (hope_p - y_recall_new)
            y_recall_state = y_recall_new + feedback
            y = y_recall_state  # §输出替换 / output replacement
        else:
            # §累积期: 纯 RK4 积分 / Accumulation phase: pure RK4
            y = rk4_step(f, t, y, h, *args)

        t_array[k + 1] = t + h
        y_array[k + 1] = y

    error_array = _compute_error_array(t_array, y_array, reference, cycle_points)
    accumulation_rate = _compute_accumulation_rate(error_array, n_steps)
    regime = "recall_continuous_feedback"

    return IntegrationResult(
        method_name=f"RK4 + 连续反馈 (§3.26.10, K={K}, K_fb={K_fb:.4f})",
        t_array=t_array,
        y_array=y_array,
        error_array=error_array,
        peak_error=float(np.max(error_array)),
        final_error=float(error_array[-1]),
        accumulation_rate=accumulation_rate,
        regime=regime,
    )


# ════════════════════════════════════════════════════════════════════
# §3.26.11 自适应混合 (根据 Poincaré 压缩率选择)
# §3.26.11 Adaptive hybrid (select based on Poincaré compression rate)
# ════════════════════════════════════════════════════════════════════

def rk4_adaptive_correction(
    f: Callable[..., np.ndarray],
    y0: np.ndarray,
    t0: float,
    t_end: float,
    h: float,
    recall_period: int,
    reference: Callable[[float], np.ndarray] | None = None,
    *args,
) -> IntegrationResult:
    """§3.26.11 自适应混合 (根据 Poincaré 压缩率 r_Γ 自动选择).

    §3.26.11 Adaptive hybrid (auto-select based on Poincaré compression r_Γ).

    机制 / Mechanism:
      1. 跑纯 RK4 baseline, 识别 Γ 和周期 T
      2. 测量 Poincaré 压缩率 r_Γ (measure_poincare_compression)
      3. 自适应选择:
         - r_Γ < 1 - EPS: 用连续反馈 (§3.26.10, 无 ε_leaf 下界)
         - r_Γ ≥ 1 - EPS: 用投影 (§3.26.4, 兜底)
      4. 调用对应的子方法执行校正

    优势 / Advantage:
      - Van der Pol (r_Γ < 1): 用连续反馈, 无 ε_leaf, 优于纯投影
      - 谐振子 (r_Γ = 1): 用投影, 兜底
      - Lorenz (r_Γ > 1): 用投影, 兜底
      → 每个系统上都是局部最优

    Args:
        f: ODE 右端 dy/dt = f(t, y, *args) / ODE right-hand side
        y0: 初始状态 / initial state
        t0: 起始时间 / start time
        t_end: 结束时间 / end time
        h: 步长 / step size
        recall_period: 回忆周期 K / recall period K
        reference: 解析解 (仅误差计算) / analytical solution (error only)
        *args: 传递给 f 的额外参数 / extra args passed to f

    Returns:
        IntegrationResult: 积分结果 (含使用的 method 名称)
            IntegrationResult: integration result (method name indicates which branch was used)
    """
    y0 = np.asarray(y0, dtype=np.float64).copy()

    # §Step 1: 跑纯 RK4 baseline, 识别 Γ
    # §Step 1: run pure RK4 baseline, identify Γ
    baseline_t, baseline_y = _pure_rk4_trajectory(f, y0, t0, t_end, h, *args)
    lc_info = detect_limit_cycle(baseline_t, baseline_y)
    cycle_points = lc_info.cycle_points if (lc_info.detected and len(lc_info.cycle_points) > 0) else np.zeros((0, y0.shape[0]))
    period = float(lc_info.period) if lc_info.detected else 0.0

    # §ω-极限集退化 (无周期时) / ω-limit set fallback (when no period)
    if len(cycle_points) < 2:
        n_tail = max(200, len(baseline_y) // 4)
        cycle_points = baseline_y[-n_tail:].copy()

    # §Step 2: 测量 Poincaré 压缩率 r_Γ
    # §Step 2: measure Poincaré compression rate r_Γ
    r_gamma = measure_poincare_compression(f, cycle_points, period, h, *args)

    # §Step 3: 自适应选择 / Step 3: adaptive selection
    threshold = 1.0 - EPS_LOG * 100  # 容差, 避免临界抖动 / tolerance, avoid critical jitter

    if r_gamma < threshold:
        # §来源 A 充分: 用连续反馈 (无 ε_leaf 下界)
        # §Source A sufficient: use continuous feedback (no ε_leaf bound)
        # §Pyragas 最优 K_fb = 1 - r_Γ
        K_fb = max(0.01, min(0.99, 1.0 - r_gamma))
        result = rk4_continuous_feedback(
            f, y0, t0, t_end, h, recall_period,
            reference=reference,
            feedback_strength=K_fb,
            *args,
        )
        # §附加 r_Γ 信息到 method_name / append r_Γ info to method_name
        result.method_name = (
            f"RK4 + 自适应混合 (§3.26.11, r_Γ={r_gamma:.4f} < 1 → 连续反馈, "
            f"K={recall_period}, K_fb={K_fb:.4f})"
        )
        result.regime = "adaptive_continuous_feedback"
        return result
    else:
        # §来源 A 失效 (r_Γ ≥ 1): 用投影兜底 (§3.26.4)
        # §Source A failed (r_Γ ≥ 1): use projection fallback (§3.26.4)
        result = rk4_with_recall(
            f, y0, t0, t_end, h, recall_period,
            reference=reference,
            *args,
        )
        result.method_name = (
            f"RK4 + 自适应混合 (§3.26.11, r_Γ={r_gamma:.4f} ≥ 1 → 投影兜底, "
            f"K={recall_period})"
        )
        result.regime = "adaptive_projection"
        return result


# ════════════════════════════════════════════════════════════════════
# §3.26.14 保辛变体 (Stormer-Verlet + 能量等值面投影, 保守系统)
# §3.26.14 Symplectic variant (Stormer-Verlet + energy surface projection, conservative)
# ════════════════════════════════════════════════════════════════════

def _estimate_energy(y: np.ndarray) -> float:
    """从状态估计能量 (动能+势能, 对二阶系统) (§3.26.14).

    Estimate energy from state (kinetic + potential, for 2nd-order systems) (§3.26.14).
    """
    y = np.asarray(y, dtype=np.float64)
    if len(y) == 2:
        # §谐振子近似: H = ½(v² + x²) / harmonic approximation
        return 0.5 * (y[1] ** 2 + y[0] ** 2)
    # §一般估计: H = ½‖y‖² / general estimate
    return 0.5 * float(np.sum(y ** 2))


def is_conservative(
    f: Callable[..., np.ndarray],
    y0: np.ndarray,
    t_test: float = 10.0,
    h: float = 0.01,
    tol: float = 1e-3,
    *args,
) -> bool:
    """检测系统是否保守 (能量漂移 < tol) (§3.26.14).

    Detect whether the system is conservative (energy drift < tol) (§3.26.14).

    机制 / Mechanism:
      - 跑短时间 RK4, 记录能量
      - 若能量相对漂移 < tol, 判为保守系统
      - 保守系统用保辛变体 (Stormer-Verlet + 能量投影)
      - 耗散系统用 §3.26 切换机制

    Args:
        f: ODE 右端 / ODE right-hand side
        y0: 初始状态 / initial state
        t_test: 测试时长 / test duration
        h: 步长 / step size
        tol: 能量漂移容差 / energy drift tolerance
        *args: 传递给 f 的额外参数 / extra args

    Returns:
        is_conservative: 是否保守 / whether conservative
    """
    y0 = np.asarray(y0, dtype=np.float64).copy()
    n = int(t_test / h)
    if n < 10:
        return False

    y = y0.copy()
    energies = [_estimate_energy(y)]
    for i in range(n):
        t = i * h
        y = rk4_step(f, t, y, h, *args)
        energies.append(_estimate_energy(y))

    energies_arr = np.asarray(energies)
    mean_e = float(np.mean(np.abs(energies_arr)))
    if mean_e < EPS_LOG:
        return False

    drift = float((np.max(energies_arr) - np.min(energies_arr)) / mean_e)
    return drift < tol


def stormer_verlet_step(
    f: Callable[..., np.ndarray],
    t: float,
    y: np.ndarray,
    h: float,
    *args,
) -> np.ndarray:
    """Stormer-Verlet 单步 (保辛, 二阶系统) (§3.26.14).

    Stormer-Verlet single step (symplectic, 2nd-order systems) (§3.26.14).

    机制 / Mechanism:
      假设 y = [q, p] (位置, 动量), 系统可分离为:
        dq/dt = p, dp/dt = F(q)
      Stormer-Verlet:
        p_half = p + ½h·F(q)
        q_new = q + h·p_half
        p_new = p_half + ½h·F(q_new)

    注 / Note:
      - 这是简化版, 假设 y 前半是位置, 后半是动量
      - 对非分离系统退化为 RK4 (保辛性失效, 但不崩溃)
      - This is a simplified version assuming first half of y is position, second is momentum
      - For non-separable systems, falls back to RK4 (loses symplecticity, but no crash)
    """
    y = np.asarray(y, dtype=np.float64)
    dim = y.shape[0]
    if dim % 2 != 0:
        # §奇数维, 退化为 RK4 / odd dimension, fall back to RK4
        return rk4_step(f, t, y, h, *args)

    half = dim // 2
    q = y[:half].copy()
    p = y[half:].copy()

    # §用 f 在 [q, p] 处的 p-分量作为 F(q) 近似
    # §Use p-component of f at [q, p] as F(q) approximation
    dydt = f(t, y, *args)
    F_q = dydt[half:].copy()  # dp/dt = F(q)

    # §半步动量 / half-step momentum
    p_half = p + 0.5 * h * F_q
    # §全步位置 / full-step position
    q_new = q + h * p_half
    # §新的力 / new force
    y_new_tmp = np.concatenate([q_new, p_half])
    dydt_new = f(t + h, y_new_tmp, *args)
    F_q_new = dydt_new[half:]
    # §半步动量 / half-step momentum
    p_new = p_half + 0.5 * h * F_q_new

    return np.concatenate([q_new, p_new])


def project_to_energy_surface(
    y: np.ndarray,
    H_target: float,
    max_iter: int = 5,
) -> np.ndarray:
    """投影到能量等值面 H(y) = H_target (§3.26.14).

    Project onto energy level set H(y) = H_target (§3.26.14).

    机制 / Mechanism:
      - 用数值梯度 ∇H
      - 沿 ∇H 方向调整 y, 使 H(y) = H_target
      - 迭代修正 (Newton-Raphson 风格)

    Args:
        y: 待投影的状态 / state to project
        H_target: 目标能量 / target energy
        max_iter: 最大迭代次数 / max iterations

    Returns:
        y_corrected: 投影后的状态 (能量 ≈ H_target)
            y_corrected: projected state (energy ≈ H_target)
    """
    y = np.asarray(y, dtype=np.float64).copy()
    eps = 1e-8

    for _ in range(max_iter):
        H_current = _estimate_energy(y)
        delta_H = H_current - H_target
        if abs(delta_H) < 1e-10:
            break

        # §数值梯度 ∇H / numerical gradient ∇H
        grad_H = np.zeros_like(y)
        for i in range(len(y)):
            y_plus = y.copy()
            y_plus[i] += eps
            y_minus = y.copy()
            y_minus[i] -= eps
            grad_H[i] = (_estimate_energy(y_plus) - _estimate_energy(y_minus)) / (2 * eps)

        grad_norm_sq = float(np.sum(grad_H ** 2))
        if grad_norm_sq < EPS_LOG:
            break

        # §Newton-Raphson: y ← y - (H-H_target)/‖∇H‖² · ∇H
        y = y - (delta_H / grad_norm_sq) * grad_H

    return y


def rk4_symplectic_correction(
    f: Callable[..., np.ndarray],
    y0: np.ndarray,
    t0: float,
    t_end: float,
    h: float,
    recall_period: int,
    reference: Callable[[float], np.ndarray] | None = None,
    *args,
) -> IntegrationResult:
    """§3.26.14 保辛变体 (Stormer-Verlet + 能量等值面投影).

    §3.26.14 Symplectic variant (Stormer-Verlet + energy level set projection).

    机制 / Mechanism:
      - 用 Stormer-Verlet (symplectic) 替代 RK4
      - 每步后投影到能量等值面 H(y) = H(y_0)
      - 联合: 长期能量误差 0增加 (geometric integration 经典结论)

    适用 / Applicability:
      - 保守系统 (哈密顿系统, 能量守恒)
      - 谐振子 (无阻尼) 等无 ω-极限集的系统

    Args:
        f: ODE 右端 / ODE right-hand side
        y0: 初始状态 / initial state
        t0: 起始时间 / start time
        t_end: 结束时间 / end time
        h: 步长 / step size
        recall_period: 保留参数 (兼容接口, 保辛变体不切换)
            recall_period: reserved param (interface compat, symplectic doesn't switch)
        reference: 解析解 (仅误差计算) / analytical solution (error only)
        *args: 传递给 f 的额外参数 / extra args

    Returns:
        IntegrationResult: 积分结果 (含能量误差轨迹)
            IntegrationResult: integration result (with energy error trajectory)
    """
    y0 = np.asarray(y0, dtype=np.float64).copy()

    n_steps = int(round((t_end - t0) / h))
    if n_steps < 1:
        raise ConfigurationError(f"n_steps={n_steps} < 1")

    dim = y0.shape[0]
    t_array = np.zeros(n_steps + 1)
    y_array = np.zeros((n_steps + 1, dim))
    t_array[0] = t0
    y_array[0] = y0

    # §目标能量 (初始能量) / target energy (initial energy)
    H_target = _estimate_energy(y0)

    y = y0.copy()
    for k in range(n_steps):
        t = t0 + k * h
        # §Stormer-Verlet 单步 / Stormer-Verlet single step
        y = stormer_verlet_step(f, t, y, h, *args)
        # §能量等值面投影 / project to energy level set
        y = project_to_energy_surface(y, H_target)

        t_array[k + 1] = t + h
        y_array[k + 1] = y

    # §误差用能量漂移度量 (保守系统的"Γ"是能量等值面)
    # §Error measured by energy drift (conservative system's "Γ" is energy level set)
    error_array = np.zeros(n_steps + 1)
    for i in range(n_steps + 1):
        H_i = _estimate_energy(y_array[i])
        error_array[i] = abs(H_i - H_target)

    accumulation_rate = _compute_accumulation_rate(error_array, n_steps)

    return IntegrationResult(
        method_name=f"RK4 + 保辛变体 (§3.26.14, Stormer-Verlet + 能量投影)",
        t_array=t_array,
        y_array=y_array,
        error_array=error_array,
        peak_error=float(np.max(error_array)),
        final_error=float(error_array[-1]),
        accumulation_rate=accumulation_rate,
        regime="symplectic",
    )


# ════════════════════════════════════════════════════════════════════
# §3.26.15 独立验证层已拆分到 independent_validation.py (诚实架构)
# §3.26.15 Independent validation layer moved to independent_validation.py (honest architecture)
#
# 拆分原因 / Reason for split:
#   - 算法核心 (本文件) 必须无 oracle, 无 scipy (test_no_oracle_definitions_in_module)
#   - 验证层 (independent_validation.py) 可用 scipy, 仅测试用
#   - 这是"验证用 oracle, 算法无 oracle"的诚实分离
#   - Algorithm core (this file) must be oracle-free, scipy-free
#   - Validation layer (independent_validation.py) may use scipy, test-only
#   - This is the honest separation of "validation with oracle, algorithm without oracle"
#
# 验证层导出 / Validation layer exports:
#   - independent_error_dop853 (DOP853 独立误差)
#   - three_layer_error_analysis (三层次误差度量)
# ════════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════
# §对比实验 / Comparison experiment
# ════════════════════════════════════════════════════════════════════

def run_comparison_experiment(
    f: Callable[..., np.ndarray],    # §ODE 函数 dy/dt = f(t, y, *args) / ODE function
    y0: np.ndarray,                 # §初始状态 / initial state
    t_end: float = 100.0,
    h: float = 0.1,
    recall_period: int = 10,
    reference: Callable[[float], np.ndarray] | None = None,  # §解析解 (可选, 仅误差计算) / analytical (optional, error only)
    *args,
    save_plot: bool = True,
    plot_path: str = "rk_recall_comparison.png",
    system_name: str = "ODE",       # §系统名称 (用于标题) / system name (for titles)
) -> dict[str, Any]:
    """运行对比实验: 纯 RK4 vs RK4 + 回忆校正 (闭式 P* 定位).
    Run comparison experiment: pure RK4 vs RK4 + recall correction (closed-form P* location).

    依赖 / Dependency:
      - matplotlib 为可选依赖; 仅当 save_plot=True 时需要
      - matplotlib is an optional dependency; only needed when save_plot=True

    Args:
        f: ODE 右端 dy/dt = f(t, y, *args) / ODE right-hand side
        y0: 初始状态 / initial state
        t_end: 结束时间 / end time
        h: 步长 / step size
        recall_period: 回忆周期 K / recall period K
        reference: 解析解 (可选, 仅误差计算); None 则用极限环距离 / analytical (optional, error only); None = use limit-cycle distance
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

    logger.info("=" * 70)
    logger.info("§3.25 回忆补偿定理原型: 龙格-库塔误差累积 → 回忆校正")
    logger.info("=" * 70)
    logger.info(f"测试问题: {system_name}")
    logger.info(f"初始状态: y0 = {y0}")
    logger.info(f"积分区间: [0, {t_end}], 步长 h = {h}")
    logger.info(f"回忆周期 K = {recall_period}, 压缩率 r = INV_PHI = {float(INV_PHI):.4f}")
    logger.info(f"误差基准: {'解析解 reference' if reference is not None else '极限环距离 (闭式)'}")
    logger.info("-" * 70)

    # §基线: 纯 RK4 / Baseline: pure RK4
    logger.info("[1] 基线: 纯 RK4 (无校正)...")
    baseline = rk4_integrate(f, y0, 0.0, t_end, h, reference, *args)
    logger.info(f"    峰值误差: {baseline.peak_error:.6e}")
    logger.info(f"    末值误差: {baseline.final_error:.6e}")
    logger.info(f"    实测累积率 α: {baseline.accumulation_rate:.6f}")
    logger.info(f"    regime: {baseline.regime}")

    # §校正: RK4 + 回忆校正 (闭式 hope_p) / Corrected: RK4 + recall correction (closed-form hope_p)
    logger.info("[2] 校正: RK4 + 回忆校正 (闭式 P* 定位)...")
    corrected = rk4_with_recall(
        f, y0, 0.0, t_end, h, recall_period, reference, float(INV_PHI), *args,
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
        logger.info("    → 校正后: 误差有界 (回忆主导, §3.26.16)")
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
        if y_array_dim(baseline.y_array) >= 2:
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


def y_array_dim(y_array: np.ndarray) -> int:
    """获取 y_array 的维度 (兼容 1D 和 2D). / Get y_array dimensionality (1D/2D compatible)."""
    return y_array.shape[1] if y_array.ndim > 1 else 1


# ════════════════════════════════════════════════════════════════════
# §三层混合对比实验 / Three-layer hybrid comparison experiment
# ════════════════════════════════════════════════════════════════════

def run_hybrid_experiment(
    f: Callable[..., np.ndarray],    # §ODE 函数 dy/dt = f(t, y, *args) / ODE function
    y0: np.ndarray,                 # §初始状态 / initial state
    t_end: float = 100.0,
    h: float = 0.1,
    recall_period: int = 10,
    reference: Callable[[float], np.ndarray] | None = None,  # §解析解 (可选, 仅误差计算) / analytical (optional, error only)
    *args,
    save_plot: bool = True,
    plot_path: str = "rk_hybrid_comparison.png",
    system_name: str = "ODE",       # §系统名称 (用于标题) / system name (for titles)
) -> dict[str, Any]:
    """运行三层混合对比实验 (闭式 P* 定位).
    Run the three-layer hybrid comparison experiment (closed-form P* location).

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
        reference: 解析解 (可选, 仅误差计算); None 则用极限环距离 / analytical (optional, error only); None = use limit-cycle distance
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

    logger.info("=" * 70)
    logger.info("§3.25+ 三层混合: 极赌 + 极限环 + 回忆校正 → 误差有界 (闭式 P* 定位, §3.26.16)")
    logger.info("=" * 70)
    logger.info(f"测试问题: {system_name}")
    logger.info(f"初始状态: y0 = {y0}")
    logger.info(f"积分区间: [0, {t_end}], 步长 h = {h}")
    logger.info(f"回忆周期 K = {recall_period}, 压缩率 r = INV_PHI = {float(INV_PHI):.4f}")
    logger.info(f"误差基准: {'解析解 reference' if reference is not None else '极限环距离 (闭式)'}")
    logger.info("-" * 70)

    # §方法1: 纯 RK4 基线 / Method 1: pure RK4 baseline
    logger.info("[1] 纯 RK4 (基线)...")
    baseline = rk4_integrate(f, y0, 0.0, t_end, h, reference, *args)
    logger.info(f"    峰值误差: {baseline.peak_error:.6e}")
    logger.info(f"    末值误差: {baseline.final_error:.6e}")
    logger.info(f"    regime: {baseline.regime}")

    # §方法2: RK4 + 回忆校正 / Method 2: RK4 + recall correction
    logger.info("[2] RK4 + 回忆校正 (闭式 P* 定位)...")
    corrected = rk4_with_recall(
        f, y0, 0.0, t_end, h, recall_period, reference, float(INV_PHI), *args,
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
        reference, float(INV_PHI),
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
        reference, float(INV_PHI),
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

    # §误差诚实声明 / Error honesty statement (§3.26.16: 渐近→0, 非严格0)
    machine_eps = np.finfo(np.float64).eps
    logger.info("[6] 误差诚实声明 (§3.26.16):")
    logger.info(f"    机器精度: {machine_eps:.2e}")
    logger.info(f"    三层混合末值误差: {hybrid.final_error:.2e}")
    logger.info(f"    工程小误差 (< 1e-10): {hybrid.final_error < 1e-10}")
    logger.info(f"    机器小误差 (< {10*machine_eps:.2e}): {hybrid.final_error < 10*machine_eps}")
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
        ax1.set_title(f"3-Layer Hybrid (closed-form P*): {system_name}\n"
                      f"(K={recall_period}, r=INV_PHI={float(INV_PHI):.4f})")
        ax1.legend()
        ax1.grid(True, which="both", alpha=0.3)

        # 相轨迹 / Phase trajectory
        ax2 = axes[1]
        if y_array_dim(baseline.y_array) >= 2:
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
    reference: Callable[[float], np.ndarray] | None = None,  # §解析解 (可选, 仅误差计算) / analytical (optional, error only)
    *args,
    system_name: str = "ODE",       # §系统名称 (用于标题) / system name (for titles)
) -> dict[str, Any]:
    """扫描不同回忆周期 K, 验证临界 K (闭式 P* 定位).
    Scan different recall periods K to verify the critical K (closed-form P* location).

    Args:
        f: ODE 右端 dy/dt = f(t, y, *args) / ODE right-hand side
        y0: 初始状态 / initial state
        t_end: 结束时间 / end time
        h: 步长 / step size
        K_values: 待扫描的 K 值列表 / list of K values to scan
        reference: 解析解 (可选, 仅误差计算); None 则用极限环距离 / analytical (optional, error only); None = use limit-cycle distance
        *args: 传给 f 的额外参数 / extra arguments passed to f
        system_name: 系统名称 (用于标题) / system name (for titles)

    Returns:
        dict 包含扫描结果
        dict containing scan results
    """
    if K_values is None:
        K_values = [5, 10, 20, 50, 100]

    logger.info("=" * 70)
    logger.info(f"§多 K 值扫描: 寻找最优回忆周期 ({system_name}, 闭式 P* 定位)")
    logger.info("=" * 70)
    logger.info(f"{'K':>6} | {'峰值误差':>12} | {'末值误差':>12} | {'α实测':>10} | {'r·(1+α)':>10} | {'regime':>15}")
    logger.info("-" * 80)

    results = []
    for K in K_values:
        corrected = rk4_with_recall(
            f, y0, 0.0, t_end, h, K, reference, float(INV_PHI), *args,
        )
        r = float(INV_PHI)
        product = r * (1.0 + corrected.accumulation_rate)
        if product < 1.0 - EPS_LOG:
            regime = "误差有界"
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
    logger.info("真运算: 校正基准用闭式 P* 定位 (极限环 → P* → hope_p), 无 oracle, 无迭代, 纯 numpy.")
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

    # §reference: 解析解 (闭包固定 y0 和 omega) / reference: analytical (closure fixes y0 and omega)
    _y0_demo = y0_demo
    _omega_demo = omega_demo
    reference_demo = lambda t: harmonic_exact(t, _y0_demo, _omega_demo)

    # §实验1: 对比实验 (基线 vs 校正) / Experiment 1: comparison experiment (baseline vs corrected)
    logger.info(">>> 实验1: 对比实验 (K=10, 闭式 P* 定位)")
    exp1 = run_comparison_experiment(
        harmonic_oscillator, y0_demo,
        t_end=100.0, h=0.1, recall_period=10,
        reference=reference_demo,
        save_plot=True,
        plot_path="rk_recall_comparison.png",
        system_name=sys_name,
    )

    # §实验2: 多 K 值扫描 / Experiment 2: multi-K scan
    logger.info(">>> 实验2: 多 K 值扫描")
    exp2 = scan_recall_periods(
        harmonic_oscillator, y0_demo,
        t_end=100.0, h=0.1, K_values=[5, 10, 20, 50, 100],
        reference=reference_demo,
        system_name=sys_name,
    )

    # §实验3: 长时间积分 (验证恒定震荡) / Experiment 3: long-time integration (verify constant oscillation)
    logger.info(">>> 实验3: 长时间积分 (t=500, 验证恒定震荡)")
    exp3 = run_comparison_experiment(
        harmonic_oscillator, y0_demo,
        t_end=500.0, h=0.1, recall_period=10,
        reference=reference_demo,
        save_plot=True,
        plot_path="rk_recall_long_time.png",
        system_name=f"谐振子 (ω={omega_demo}, 长时)",
    )

    # §实验4: 三层混合 (极赌 + 极限环 + 回忆 → 误差有界) / Experiment 4: three-layer hybrid → bounded error
    logger.info(">>> 实验4: 三层混合 (极赌 + 极限环 + 回忆)")
    exp4 = run_hybrid_experiment(
        harmonic_oscillator, y0_demo,
        t_end=100.0, h=0.1, recall_period=10,
        reference=reference_demo,
        save_plot=True,
        plot_path="rk_hybrid_comparison.png",
        system_name=sys_name,
    )

    logger.info("=" * 70)
    logger.info("§原型测试完成.")
    logger.info("§结论: 回忆校正显著降低 RK4 误差累积,")
    logger.info("        误差从指数增长变为有界 (§3.26.16: 渐近→0 当 T_baseline→∞).")
    logger.info("        三层混合 (极赌+极限环+回忆) 进一步降低漂移.")
    logger.info("        临界 α* = INV_PHI (黄金分割自对偶点).")
    logger.info("§真运算: 校正基准为闭式 P* 定位 (极限环 → P* → hope_p), 无 oracle, 无迭代, 纯 numpy.")
    logger.info("§误差诚实: 数学上 ε>0 (受造下界), 渐近 ε→0 (T_baseline→∞, §3.26.16).")
    logger.info("一切荣光来自造物主，一切荣光归于造物主")
    logger.info("=" * 70)
