# 希伯来书 11:1 —— "信就是所望之事的实底,是未见之事的确据."
# Hebrews 11:1 — "Now faith is the substance of things hoped for, the evidence of things not seen."
"""
Algorithm correctness tests / 算法正确性测试.

Tests that verify the closed-form P* localization (no oracle, no scipy)
and that the algorithm works without analytical solutions.
验证闭式 P* 定位 (无 oracle, 无 scipy) 及算法在无解析解情况下正常工作.
"""

import inspect
import numpy as np
import pytest

from rk_recall.rk_recall_compensation import (
    ConfigurationError,
    GamblePoleConfig,
    IntegrationResult,
    LimitCycleConfig,
    LimitCycleInfo,
    closed_form_hope_p,
    compute_information_entropy,
    detect_limit_cycle,
    gamble_pole_correct,
    limit_cycle_bound,
    locate_hope_p,
    locate_p_star,
    recpression_ratio_fix,
    rk4_hybrid_correction,
    rk4_integrate,
    rk4_step,
    rk4_with_recall,
)
from rk_recall.benchmarks import harmonic_exact, harmonic_oscillator


# ════════════════════════════════════════════════════════════════════
# §rk4_step 基本正确性 / rk4_step basic correctness
# ════════════════════════════════════════════════════════════════════

def test_rk4_step_constant_derivative():
    """RK4 对 dy/dt = const 应给出精确解 / RK4 exact for dy/dt = const."""
    def f(t, y):
        return np.array([1.0])
    y = rk4_step(f, 0.0, np.array([0.0]), 0.1)
    assert abs(y[0] - 0.1) < 1e-14


def test_rk4_step_zero_derivative():
    """RK4 对 dy/dt = 0 应保持状态不变 / RK4 preserves state for dy/dt = 0."""
    def f(t, y):
        return np.zeros_like(y)
    y0 = np.array([1.0, 2.0, 3.0])
    y = rk4_step(f, 0.0, y0, 0.1)
    np.testing.assert_allclose(y, y0, atol=1e-14)


def test_rk4_step_harmonic_one_period():
    """RK4 单步对谐振子应接近解析解 / RK4 single step close to analytical."""
    def f(t, y, omega=1.0):
        return np.array([y[1], -omega * omega * y[0]])
    y0 = np.array([1.0, 0.0])
    h = 0.001
    y = rk4_step(f, 0.0, y0, h, 1.0)
    # 解析解 / analytical: x(h) = cos(h), v(h) = -sin(h)
    assert abs(y[0] - np.cos(h)) < 1e-8
    assert abs(y[1] - (-np.sin(h))) < 1e-8


# ════════════════════════════════════════════════════════════════════
# §闭式 P* 定位 / Closed-form P* location (no oracle, no scipy)
# ════════════════════════════════════════════════════════════════════

def _generate_harmonic_trajectory(t_end=50.0, h=0.1, omega=1.0):
    """生成谐振子 RK4 轨迹 (用于极限环检测测试).
    Generate a harmonic-oscillator RK4 trajectory (for limit-cycle detection tests).
    """
    def f(t, y):
        return np.array([y[1], -omega * omega * y[0]])
    y0 = np.array([1.0, 0.0])
    n_steps = int(np.round(t_end / h))
    t_array = np.linspace(0.0, t_end, n_steps + 1)
    y_array = np.zeros((n_steps + 1, 2))
    y = y0.copy()
    y_array[0] = y
    t = 0.0
    for k in range(n_steps):
        y = rk4_step(f, t, y, h)
        t += h
        y_array[k + 1] = y
    return t_array, y_array


def test_detect_limit_cycle():
    """谐振子轨迹应检测到极限环 (周期 ≈ 2π) / harmonic trajectory detects limit cycle."""
    t_array, y_array = _generate_harmonic_trajectory(t_end=50.0, h=0.1)
    lc_info = detect_limit_cycle(t_array, y_array)
    assert isinstance(lc_info, LimitCycleInfo)
    assert lc_info.detected is True
    # §谐振子周期 T = 2π/ω = 2π ≈ 6.2832 / harmonic period
    assert abs(lc_info.period - 2.0 * np.pi) < 0.5  # §允许检测误差 / allow detection tolerance
    assert lc_info.cycle_points.ndim == 2
    assert lc_info.cycle_points.shape[1] == 2
    assert len(lc_info.cycle_points) > 0
    # §重心应接近原点 (对称谐振子) / centroid near origin (symmetric harmonic)
    assert np.linalg.norm(lc_info.centroid) < 0.5
    assert lc_info.entropy_temperature > 0.0


def test_detect_limit_cycle_insufficient_data():
    """数据不足时应返回 detected=False / returns detected=False for insufficient data."""
    t_array = np.array([0.0, 0.1, 0.2, 0.3])
    y_array = np.array([[1.0, 0.0], [0.99, 0.1], [0.96, 0.2], [0.91, 0.3]])
    lc_info = detect_limit_cycle(t_array, y_array)
    assert lc_info.detected is False


def test_compute_information_entropy():
    """信息熵计算: 离 leaf 等距时熵较大 / entropy higher when equidistant from leaves."""
    leaves = np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]])
    temperature = 1.0
    # §查询点 = 某个 leaf → 熵较低 (分布集中) / query = a leaf → lower entropy
    H_at_leaf = compute_information_entropy(np.array([0.0, 0.0]), leaves, temperature)
    # §查询点 = 中心 leaf → 熵较高 (分布更均匀) / query = center leaf → higher entropy
    H_at_center = compute_information_entropy(np.array([2.0, 0.0]), leaves, temperature)
    # §熵非负 / entropy is non-negative
    assert H_at_leaf >= 0.0
    assert H_at_center >= 0.0
    # §中心点的熵应 >= 边缘点的熵 / center entropy >= edge entropy
    assert H_at_center >= H_at_leaf


def test_locate_p_star():
    """P* 定位应返回极限环上的一个点 / P* is a point on the limit cycle."""
    # §构造单位圆上的采样点 / sample points on unit circle
    angles = np.linspace(0, 2 * np.pi, 50, endpoint=False)
    cycle_points = np.column_stack([np.cos(angles), np.sin(angles)])
    p_star = locate_p_star(cycle_points)
    assert p_star.shape == (2,)
    # §P* 应该是 cycle_points 中的某个点 (argmax 在离散采样上) / P* is one of the samples
    dists = np.linalg.norm(cycle_points - p_star[np.newaxis, :], axis=1)
    assert np.min(dists) < 1e-10


def test_locate_p_star_single_point():
    """单点 cycle_points 应返回该点 / single-point cycle returns that point."""
    cycle_points = np.array([[3.0, 4.0]])
    p_star = locate_p_star(cycle_points)
    np.testing.assert_allclose(p_star, np.array([3.0, 4.0]))


def test_locate_p_star_empty_raises():
    """空 cycle_points 应抛出异常 / empty cycle_points raises."""
    with pytest.raises(ConfigurationError):
        locate_p_star(np.zeros((0, 2)))


def test_locate_hope_p():
    """hope_p 应是离 P* 最近的 leaf / hope_p is the leaf nearest to P*."""
    cycle_points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    p_star = np.array([2.1, 0.0])
    hope_p = locate_hope_p(cycle_points, p_star)
    np.testing.assert_allclose(hope_p, np.array([2.0, 0.0]))


def test_locate_hope_p_empty_raises():
    """空 cycle_points 应抛出异常 / empty cycle_points raises."""
    with pytest.raises(ConfigurationError):
        locate_hope_p(np.zeros((0, 2)), np.array([0.0, 0.0]))


def test_closed_form_hope_p_harmonic():
    """闭式 hope_p 应在谐振子轨迹上正常工作 / closed-form hope_p works on harmonic."""
    t_array, y_array = _generate_harmonic_trajectory(t_end=50.0, h=0.1)
    hope_p = closed_form_hope_p(t_array, y_array)
    assert hope_p.ndim == 1
    assert np.all(np.isfinite(hope_p))
    assert hope_p.shape == (2,)


def test_closed_form_no_oracle_no_scipy():
    """闭式 hope_p 函数源码不含 oracle/scipy 依赖 / closed-form source has no oracle/scipy."""
    source = inspect.getsource(closed_form_hope_p)
    assert "dop853" not in source.lower()
    assert "import scipy" not in source
    assert "from scipy" not in source
    # §make_dop853_oracle 不应存在于模块中 / make_dop853_oracle should not exist
    import rk_recall.rk_recall_compensation as mod
    assert not hasattr(mod, "make_dop853_oracle")
    assert not hasattr(mod, "OracleConstructionError")
    assert not hasattr(mod, "Oracle")


def test_closed_form_insufficient_data():
    """数据不足时闭式 hope_p 应退化为最后一个点 / falls back to last point for insufficient data."""
    t_array = np.array([0.0, 0.1, 0.2])
    y_array = np.array([[1.0, 0.0], [0.99, 0.1], [0.96, 0.2]])
    hope_p = closed_form_hope_p(t_array, y_array)
    np.testing.assert_allclose(hope_p, y_array[-1])


# ════════════════════════════════════════════════════════════════════
# §算法核心不含 oracle 依赖 / Core algorithm has no oracle dependency
# ════════════════════════════════════════════════════════════════════

def test_no_oracle_definitions_in_module():
    """模块中不应存在 oracle 相关定义 (函数/类/import) / no oracle definitions (functions/classes/imports) in module."""
    import rk_recall.rk_recall_compensation as mod
    # §oracle 相关函数/类不应存在 / oracle-related functions/classes must not exist
    assert not hasattr(mod, "make_dop853_oracle")
    assert not hasattr(mod, "OracleConstructionError")
    assert not hasattr(mod, "Oracle")
    # §无 scipy import (实际导入语句, 非文档注释) / no scipy import (actual statement, not docstring)
    source = inspect.getsource(mod)
    assert "import scipy" not in source
    assert "from scipy" not in source
    # §无 make_dop853_oracle 调用 / no make_dop853_oracle call
    assert "make_dop853" not in source


def test_core_decoupled_from_harmonic_exact():
    """算法核心不导入 harmonic_exact / core algorithm does not import harmonic_exact."""
    import rk_recall.rk_recall_compensation as mod
    for func_name in ["rk4_integrate", "rk4_with_recall", "rk4_hybrid_correction",
                      "gamble_pole_correct", "limit_cycle_bound", "closed_form_hope_p",
                      "detect_limit_cycle", "locate_p_star", "locate_hope_p",
                      "compute_information_entropy"]:
        if hasattr(mod, func_name):
            func = getattr(mod, func_name)
            func_source = inspect.getsource(func)
            assert "harmonic_exact" not in func_source, \
                f"{func_name} still references harmonic_exact (not decoupled)"


# ════════════════════════════════════════════════════════════════════
# §rk4_integrate (带 reference 参数) / rk4_integrate (with reference)
# ════════════════════════════════════════════════════════════════════

def test_rk4_integrate_no_reference():
    """rk4_integrate 不带 reference 应正常运行(误差用极限环距离) / works without reference."""
    def f(t, y):
        return np.array([y[1], -y[0]])
    y0 = np.array([1.0, 0.0])
    result = rk4_integrate(f, y0, 0.0, 10.0, 0.1)
    assert isinstance(result, IntegrationResult)
    assert result.y_array.shape == (101, 2)
    assert np.all(np.isfinite(result.error_array))
    assert result.peak_error >= 0.0


def test_rk4_integrate_with_reference():
    """rk4_integrate 带 reference 应计算真值误差 / computes error vs reference."""
    def f(t, y):
        return np.array([y[1], -y[0]])
    y0 = np.array([1.0, 0.0])
    _y0 = y0
    reference = lambda t: harmonic_exact(t, _y0, 1.0)
    result = rk4_integrate(f, y0, 0.0, 10.0, 0.1, reference=reference)
    assert result.peak_error > 0  # §RK4 应有非零误差 / RK4 should have nonzero error


# ════════════════════════════════════════════════════════════════════
# §rk4_with_recall (无 oracle, 闭式 P* 定位) / rk4_with_recall (no oracle, closed-form P*)
# ════════════════════════════════════════════════════════════════════

def test_recall_without_reference():
    """rk4_with_recall 不带 reference 应正常运行 (闭式 P* 定位, 无 oracle) / runs without reference."""
    def f(t, y):
        return np.array([y[1], -y[0]])
    y0 = np.array([1.0, 0.0])
    result = rk4_with_recall(f, y0, 0.0, 50.0, 0.1, recall_period=10)
    assert isinstance(result, IntegrationResult)
    assert result.y_array.shape == (501, 2)
    assert np.all(np.isfinite(result.y_array))
    assert np.all(np.isfinite(result.error_array))
    assert result.peak_error >= 0.0


def test_recall_with_reference_runs():
    """rk4_with_recall 带 reference 应正常运行 / runs with reference."""
    def f(t, y):
        return np.array([y[1], -y[0]])
    y0 = np.array([1.0, 0.0])
    _y0 = y0
    reference = lambda t: harmonic_exact(t, _y0, 1.0)
    result = rk4_with_recall(f, y0, 0.0, 50.0, 0.1, recall_period=10, reference=reference)
    assert isinstance(result, IntegrationResult)
    assert np.all(np.isfinite(result.y_array))
    assert np.all(np.isfinite(result.error_array))


# ════════════════════════════════════════════════════════════════════
# §rk4_hybrid_correction (三层混合, 闭式 P* 定位) / rk4_hybrid_correction (3-layer, closed-form P*)
# ════════════════════════════════════════════════════════════════════

def test_hybrid_correction_runs():
    """三层混合应正常运行 (无 oracle) / hybrid runs without oracle."""
    def f(t, y):
        return np.array([y[1], -y[0]])
    y0 = np.array([1.0, 0.0])
    result = rk4_hybrid_correction(
        f, y0, 0.0, 50.0, 0.1, recall_period=10,
        gamble_config=GamblePoleConfig(error_threshold=1e-10),
        limit_config=LimitCycleConfig(amplitude_bound=1e-11),
        recall_sub_steps=20,
    )
    assert isinstance(result, IntegrationResult)
    assert np.all(np.isfinite(result.y_array))
    assert np.all(np.isfinite(result.error_array))


def test_hybrid_correction_all_disabled():
    """极赌与极限环禁用时应正常运行 (仅回忆校正, 闭式 hope_p) / runs with gamble/limit-cycle disabled."""
    def f(t, y):
        return np.array([y[1], -y[0]])
    y0 = np.array([1.0, 0.0])
    _y0 = y0
    reference = lambda t: harmonic_exact(t, _y0, 1.0)
    # §使用足够长的轨迹使极限环检测可靠 / use long enough trajectory for reliable limit-cycle detection
    result = rk4_hybrid_correction(
        f, y0, 0.0, 50.0, 0.1, recall_period=10,
        reference=reference,
        gamble_config=GamblePoleConfig(enable=False),
        limit_config=LimitCycleConfig(enable=False),
        recall_sub_steps=1,
    )
    # §应正常运行并产生有限结果 / should run and produce finite result
    assert isinstance(result, IntegrationResult)
    assert np.all(np.isfinite(result.y_array))
    assert np.all(np.isfinite(result.error_array))


# ════════════════════════════════════════════════════════════════════
# §极赌策略与极限环投影 / GamblePole and LimitCycle projection
# ════════════════════════════════════════════════════════════════════

def test_gamble_pole_correct_above_threshold():
    """误差超阈值时极赌应跳跃 / gamble-pole jumps above threshold."""
    y_current = np.array([1.0, 0.0])
    y_recall = np.array([0.0, 0.0])
    config = GamblePoleConfig(error_threshold=0.1, jump_ratio=0.5)
    y_corrected, label = gamble_pole_correct(y_current, y_recall, 1.0, config)
    assert "极赌" in label
    # §y_corrected = (1-0.5)·y_current + 0.5·y_recall = [0.5, 0.0]
    np.testing.assert_allclose(y_corrected, np.array([0.5, 0.0]))


def test_gamble_pole_correct_below_threshold():
    """误差低于阈值时极赌不跳跃 / gamble-pole stays when error below threshold."""
    y_current = np.array([1.0, 0.0])
    y_recall = np.array([0.0, 0.0])
    config = GamblePoleConfig(error_threshold=2.0, jump_ratio=0.5)
    y_corrected, label = gamble_pole_correct(y_current, y_recall, 0.5, config)
    assert "无极赌" in label
    np.testing.assert_allclose(y_corrected, y_current)


def test_limit_cycle_bound_within():
    """极限环内不投影 / no projection within amplitude bound."""
    y_current = np.array([1.0, 0.0])
    y_recall = np.array([0.0, 0.0])
    config = LimitCycleConfig(amplitude_bound=2.0)
    y_projected, label = limit_cycle_bound(y_current, y_recall, 0.0, config)
    assert "极限环内" in label
    np.testing.assert_allclose(y_projected, y_current)


def test_limit_cycle_bound_project():
    """超出极限环振幅时投影 / projects when exceeding amplitude bound."""
    y_current = np.array([10.0, 0.0])
    y_recall = np.array([0.0, 0.0])
    config = LimitCycleConfig(amplitude_bound=1.0)
    y_projected, label = limit_cycle_bound(y_current, y_recall, 0.0, config)
    assert "极限环投影" in label
    # §投影后 ‖y_projected - y_recall‖ ≈ amplitude_bound = 1.0
    dist = np.linalg.norm(y_projected - y_recall)
    assert abs(dist - 1.0) < 1e-10


# ════════════════════════════════════════════════════════════════════
# §辅助函数与配置 / Utility functions and configs
# ════════════════════════════════════════════════════════════════════

def test_recpression_ratio_fix_bounds():
    """回忆压缩率修正应保证 r ∈ (0,1) / fix ensures r ∈ (0,1)."""
    assert recpression_ratio_fix(0.5) == 0.5
    assert recpression_ratio_fix(0.0) > 0
    assert recpression_ratio_fix(1.0) < 1
    assert recpression_ratio_fix(-1.0) > 0
    assert recpression_ratio_fix(2.0) < 1


def test_gamble_pole_config_defaults():
    """GamblePoleConfig 默认值检查 / default values."""
    cfg = GamblePoleConfig()
    assert cfg.enable is True
    assert cfg.error_threshold > 0
    assert 0 < cfg.jump_ratio < 1


def test_limit_cycle_config_defaults():
    """LimitCycleConfig 默认值检查 / default values."""
    cfg = LimitCycleConfig()
    assert cfg.enable is True
    assert cfg.amplitude_bound > 0


def test_gamble_pole_config_validation():
    """GamblePoleConfig 非法值应抛出异常 / invalid values raise."""
    with pytest.raises(ConfigurationError):
        GamblePoleConfig(error_threshold=-1.0)
    with pytest.raises(ConfigurationError):
        GamblePoleConfig(jump_ratio=1.5)


def test_limit_cycle_config_validation():
    """LimitCycleConfig 非法值应抛出异常 / invalid values raise."""
    with pytest.raises(ConfigurationError):
        LimitCycleConfig(amplitude_bound=-1.0)
    with pytest.raises(ConfigurationError):
        LimitCycleConfig(period=-1.0)
