# 希伯来书 11:1 —— "信就是所望之事的实底,是未见之事的确据."
# Hebrews 11:1 — "Now faith is the substance of things hoped for, the evidence of things not seen."
"""
RK4 Recall-Compensation Theorem (§3.25) — "Caramel" module.
回忆补偿定理 §3.25 工程原型: 龙格-库塔误差累积 → 回忆校正.

A self-contained numerical-integration module that tames Runge-Kutta error
accumulation via a three-layer hybrid correction:
一个自包含的数值积分模块, 通过三层混合校正驯服 Runge-Kutta 误差累积:

  1. GamblePole  — discrete pole jump for large errors   / 离散极点跳跃 (大误差)
  2. LimitCycle  — Poincare-Bendixson projection          / 极限环投影 (中误差)
  3. Recall      — golden-ratio convex combination        / 黄金凸组合 (小误差)

The algorithm is decoupled from any specific ODE: it accepts an ``oracle``
(reference trajectory generator) and defaults to DOP853 (8th-order) when
none is provided. This makes it work on systems WITHOUT analytical
solutions (Lorenz, Van der Pol, etc.), not just harmonic oscillators.
算法与具体 ODE 完全解耦: 接收 ``oracle`` (参考轨迹生成器), 默认用 DOP853
(8阶). 可在无解析解的系统 (洛伦兹, Van der Pol 等) 上工作.
"""
# §算法核心 (与 ODE 解耦) / Algorithm core (ODE-agnostic)
from rk_recall.rk_recall_compensation import (
    EPS_LOG,
    INV_PHI,
    INV_PHI2,
    # §异常层次 / Exception hierarchy
    ConfigurationError,
    IntegrationFailureError,
    MissingOptionalDependencyError,
    OracleConstructionError,
    RKRecallError,
    # §数据结构与配置 / Data structures and configs
    GamblePoleConfig,
    IntegrationResult,
    LimitCycleConfig,
    Oracle,
    # §核心算法 / Core algorithms
    gamble_pole_correct,
    limit_cycle_bound,
    make_dop853_oracle,
    recpression_ratio_fix,
    rk4_hybrid_correction,
    rk4_integrate,
    rk4_step,
    rk4_with_recall,
    # §实验函数 / Experiment runners
    run_comparison_experiment,
    run_hybrid_experiment,
    scan_recall_periods,
)

# §基准测试问题 (向后兼容导出) / Benchmark problems (backward-compat export)
from rk_recall.benchmarks import (
    BENCHMARKS,
    harmonic_exact,
    harmonic_oscillator,
    lorenz,
    van_der_pol,
)

__version__ = "1.0.0"
__all__ = [
    # §常量 / Constants
    "EPS_LOG",
    "INV_PHI",
    "INV_PHI2",
    # §异常层次 / Exception hierarchy
    "RKRecallError",
    "MissingOptionalDependencyError",
    "OracleConstructionError",
    "IntegrationFailureError",
    "ConfigurationError",
    # §Oracle 接口 / Oracle interface
    "Oracle",
    "make_dop853_oracle",
    # §数据结构 / Data structures
    "IntegrationResult",
    # §配置 / Configs
    "GamblePoleConfig",
    "LimitCycleConfig",
    # §核心算法 / Core algorithms
    "rk4_step",
    "rk4_integrate",
    "rk4_with_recall",
    "rk4_hybrid_correction",
    "gamble_pole_correct",
    "limit_cycle_bound",
    "recpression_ratio_fix",
    # §实验函数 / Experiment runners
    "run_comparison_experiment",
    "run_hybrid_experiment",
    "scan_recall_periods",
    # §基准系统 / Benchmark systems
    "BENCHMARKS",
    "harmonic_oscillator",
    "harmonic_exact",
    "lorenz",
    "van_der_pol",
]
