# 希伯来书 11:1 —— "信就是所望之事的实底,是未见之事的确据."
# Hebrews 11:1 — "Now faith is the substance of things hoped for, the evidence of things not seen."
"""
RK4 Recall-Compensation Theorem (§3.25) — "Caramel" module.
回忆补偿定理 §3.25 工程原型: 龙格-库塔误差累积 → 回忆校正.

A self-contained numerical-integration module that tames Runge-Kutta error
accumulation via ω-limit-set projection (§3.26.16 drift convergence) and
Pyragas→Γ phase locking (§3.26.17).
一个自包含的数值积分模块, 通过 ω-极限集投影 (§3.26.16 漂移收敛) 和
Pyragas→Γ 相位锁定 (§3.26.17) 驯服 Runge-Kutta 误差累积.

Core theoretical results (proven, see docs):
  - §3.26.16: ε_leaf(T_baseline) ≤ C·h^{1/d}/T_baseline^{1/d} → 0
  - §3.26.17: phase error → O(h²) when |1 + h·λ_local - K| < 1

The algorithm is fully decoupled from any specific ODE and uses NO external
oracle: the correction anchor (hope_p) is located in closed form from the
trajectory's own ω-limit set (MIP theory: limit cycle → P* → nearest leaf),
implemented in pure numpy. This makes it work on systems WITHOUT analytical
solutions (Lorenz, Van der Pol, etc.), not just harmonic oscillators.
算法与具体 ODE 完全解耦, 且不使用任何外部 oracle: 校正锚点 (hope_p) 由轨迹
自身的 ω-极限集闭式定位 (MIP 理论: 极限环 → P* → 最近 leaf), 纯 numpy 实现.
可在无解析解的系统 (洛伦兹, Van der Pol 等) 上工作.
"""
# §算法核心 (与 ODE 解耦, 闭式 P* 定位) / Algorithm core (ODE-agnostic, closed-form P* location)
from rk_recall.rk_recall_compensation import (
    EPS_LOG,
    INV_PHI,
    INV_PHI2,
    # §异常层次 / Exception hierarchy
    ConfigurationError,
    IntegrationFailureError,
    MissingOptionalDependencyError,
    RKRecallError,
    # §数据结构与配置 / Data structures and configs
    GamblePoleConfig,
    IntegrationResult,
    LimitCycleConfig,
    LimitCycleInfo,
    # §闭式 P* 定位 (无 oracle, 无迭代, 纯 numpy) / Closed-form P* location (no oracle, no iteration, pure numpy)
    closed_form_hope_p,
    compute_information_entropy,
    detect_limit_cycle,
    limit_cycle_distance,
    locate_hope_p,
    locate_nearest_leaf,
    locate_p_star,
    # §核心算法 / Core algorithms
    gamble_pole_correct,
    limit_cycle_bound,
    recpression_ratio_fix,
    rk4_hybrid_correction,
    rk4_integrate,
    rk4_step,
    rk4_with_recall,
    # §3.26.10-15 扩展算法 / Extended algorithms
    estimate_embedding_dim_cao,
    estimate_tau_autocorrelation,
    is_conservative,
    measure_poincare_compression,
    project_to_energy_surface,
    reconstruct_attractor,
    rk4_adaptive_correction,
    rk4_continuous_feedback,
    rk4_symplectic_correction,
    stormer_verlet_step,
    # §3.26.16 漂移随时间收敛定理 (时间自适应漂移控制) / Drift Convergence Theorem
    adaptive_baseline_extension,
    densify_cycle,
    rk4_with_recall_adaptive,
    # §3.26.17 Pyragas → Γ 推广 (相位锁定) / Pyragas generalization (phase locking)
    PhaseLabeledGamma,
    build_phase_labeled_gamma,
    estimate_local_lyapunov,
    pyragas_adaptive,
    pyragas_on_attractor,
    # §实验函数 / Experiment runners
    run_comparison_experiment,
    run_hybrid_experiment,
    scan_recall_periods,
)

# §3.26.15 独立验证层 (从独立模块导入, 算法核心保持无 scipy)
# §3.26.15 Independent validation layer (imported from separate module, core stays scipy-free)
from rk_recall.independent_validation import (
    independent_error_dop853,
    three_layer_error_analysis,
)

# §基准测试问题 (向后兼容导出) / Benchmark problems (backward-compat export)
from rk_recall.benchmarks import (
    BENCHMARKS,
    harmonic_exact,
    harmonic_oscillator,
    lorenz,
    van_der_pol,
)

__version__ = "2.0.0"
__all__ = [
    # §常量 / Constants
    "EPS_LOG",
    "INV_PHI",
    "INV_PHI2",
    # §异常层次 / Exception hierarchy
    "RKRecallError",
    "MissingOptionalDependencyError",
    "IntegrationFailureError",
    "ConfigurationError",
    # §数据结构 / Data structures
    "IntegrationResult",
    "LimitCycleInfo",
    # §配置 / Configs
    "GamblePoleConfig",
    "LimitCycleConfig",
    # §闭式 P* 定位 / Closed-form P* location
    "detect_limit_cycle",
    "compute_information_entropy",
    "limit_cycle_distance",
    "locate_hope_p",
    "locate_nearest_leaf",
    "locate_p_star",
    "closed_form_hope_p",
    # §核心算法 / Core algorithms
    "rk4_step",
    "rk4_integrate",
    "rk4_with_recall",
    "rk4_hybrid_correction",
    "gamble_pole_correct",
    "limit_cycle_bound",
    "recpression_ratio_fix",
    # §3.26.10-15 扩展算法 / Extended algorithms
    "estimate_embedding_dim_cao",
    "estimate_tau_autocorrelation",
    "independent_error_dop853",
    "is_conservative",
    "measure_poincare_compression",
    "project_to_energy_surface",
    "reconstruct_attractor",
    "rk4_adaptive_correction",
    "rk4_continuous_feedback",
    "rk4_symplectic_correction",
    "stormer_verlet_step",
    "three_layer_error_analysis",
    # §3.26.16 漂移随时间收敛定理 / Drift Convergence Theorem
    "adaptive_baseline_extension",
    "densify_cycle",
    "rk4_with_recall_adaptive",
    # §3.26.17 Pyragas → Γ 推广 / Pyragas generalization (phase locking)
    "PhaseLabeledGamma",
    "build_phase_labeled_gamma",
    "estimate_local_lyapunov",
    "pyragas_adaptive",
    "pyragas_on_attractor",
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
