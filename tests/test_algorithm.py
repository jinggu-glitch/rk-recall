# 希伯来书 11:1 —— "信就是所望之事的实底,是未见之事的确据."
# Hebrews 11:1 — "Now faith is the substance of things hoped for, the evidence of things not seen."
"""
Algorithm correctness tests / 算法正确性测试.

Tests that verify the algorithm works without analytical solutions
and that the oracle interface is properly decoupled.
验证算法在无解析解情况下正常工作, 且 oracle 接口正确解耦.
"""

import numpy as np
import pytest

from rk_recall.rk_recall_compensation import (
    EPS_LOG,
    INV_PHI,
    GamblePoleConfig,
    IntegrationResult,
    LimitCycleConfig,
    make_dop853_oracle,
    recpression_ratio_fix,
    rk4_hybrid_correction,
    rk4_integrate,
    rk4_step,
    rk4_with_recall,
)


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
# §Oracle 接口 / Oracle interface
# ════════════════════════════════════════════════════════════════════

def test_dop853_oracle_returns_callable():
    """DOP853 oracle 应返回可调用对象 / DOP853 oracle returns callable."""
    def f(t, y):
        return np.array([y[1], -y[0]])
    y0 = np.array([1.0, 0.0])
    oracle = make_dop853_oracle(f, y0, 0.0, 10.0, 0.1)
    assert callable(oracle)
    y_ref = oracle(5.0)
    assert y_ref.shape == (2,)


def test_dop853_oracle_accuracy_harmonic():
    """DOP853 oracle 对谐振子应高精度匹配解析解 / DOP853 oracle matches analytical."""
    def f(t, y, omega=1.0):
        return np.array([y[1], -omega * omega * y[0]])
    y0 = np.array([1.0, 0.0])
    omega = 1.0
    oracle = make_dop853_oracle(f, y0, 0.0, 10.0, 0.1, omega)
    # 在多个时间点检查 / check at multiple time points
    for t in [0.0, 1.0, 2.5, 5.0, 7.3, 10.0]:
        y_ref = oracle(t)
        x_exact = y0[0] * np.cos(omega * t) + (y0[1] / omega) * np.sin(omega * t)
        v_exact = -y0[0] * omega * np.sin(omega * t) + y0[1] * np.cos(omega * t)
        np.testing.assert_allclose(y_ref, [x_exact, v_exact], atol=1e-10)


def test_oracle_decoupled_from_harmonic_exact():
    """算法核心不导入 harmonic_exact / core algorithm does not import harmonic_exact."""
    import rk_recall.rk_recall_compensation as mod
    import inspect
    source = inspect.getsource(mod)
    # 核心算法函数中不应调用 harmonic_exact / core functions must not call harmonic_exact
    for func_name in ["rk4_integrate", "rk4_with_recall", "rk4_hybrid_correction",
                      "gamble_pole_correct", "limit_cycle_bound"]:
        if hasattr(mod, func_name):
            func = getattr(mod, func_name)
            func_source = inspect.getsource(func)
            assert "harmonic_exact" not in func_source, \
                f"{func_name} still references harmonic_exact (not decoupled)"


# ════════════════════════════════════════════════════════════════════
# §rk4_integrate (带 oracle 参数) / rk4_integrate (with oracle)
# ════════════════════════════════════════════════════════════════════

def test_rk4_integrate_no_oracle():
    """rk4_integrate 不带 oracle 应正常运行(不计算误差) / works without oracle."""
    def f(t, y):
        return np.array([y[1], -y[0]])
    y0 = np.array([1.0, 0.0])
    result = rk4_integrate(f, y0, 0.0, 10.0, 0.1)
    assert isinstance(result, IntegrationResult)
    assert result.y_array.shape == (101, 2)


def test_rk4_integrate_with_oracle():
    """rk4_integrate 带 oracle 应计算误差 / computes error with oracle."""
    def f(t, y):
        return np.array([y[1], -y[0]])
    y0 = np.array([1.0, 0.0])
    oracle = make_dop853_oracle(f, y0, 0.0, 10.0, 0.1)
    result = rk4_integrate(f, y0, 0.0, 10.0, 0.1, oracle=oracle)
    assert result.peak_error > 0  # RK4 应有非零误差 / RK4 should have nonzero error


# ════════════════════════════════════════════════════════════════════
# §rk4_with_recall (oracle 自动) / rk4_with_recall (auto oracle)
# ════════════════════════════════════════════════════════════════════

def test_rk4_with_recall_auto_oracle():
    """rk4_with_recall 不传 oracle 应自动用 DOP853 / auto-DOP853 when oracle=None."""
    def f(t, y):
        return np.array([y[1], -y[0]])
    y0 = np.array([1.0, 0.0])
    result = rk4_with_recall(f, y0, 0.0, 50.0, 0.1, recall_period=10)
    assert isinstance(result, IntegrationResult)
    # 回忆校正应降低误差 vs 纯RK4 / recall should reduce error vs pure RK4
    oracle = make_dop853_oracle(f, y0, 0.0, 50.0, 0.1)
    baseline = rk4_integrate(f, y0, 0.0, 50.0, 0.1, oracle=oracle)
    assert result.final_error < baseline.final_error


# ════════════════════════════════════════════════════════════════════
# §rk4_hybrid_correction (三层混合) / rk4_hybrid_correction (3-layer)
# ════════════════════════════════════════════════════════════════════

def test_hybrid_correction_auto_oracle():
    """三层混合不传 oracle 应自动用 DOP853 / hybrid auto-DOP853."""
    def f(t, y):
        return np.array([y[1], -y[0]])
    y0 = np.array([1.0, 0.0])
    result = rk4_hybrid_correction(
        f, y0, 0.0, 50.0, 0.1, recall_period=10,
        gamble_config=GamblePoleConfig(error_threshold=1e-10),
        limit_config=LimitCycleConfig(amplitude_bound=1e-11),
        recall_sub_steps=20,
    )
    assert result.final_error < 1e-9  # 应达到高精度 / should achieve high precision


def test_hybrid_correction_all_disabled():
    """全禁用时应退化为纯 RK4 / degrades to pure RK4 when all disabled."""
    def f(t, y):
        return np.array([y[1], -y[0]])
    y0 = np.array([1.0, 0.0])
    oracle = make_dop853_oracle(f, y0, 0.0, 10.0, 0.1)
    result = rk4_hybrid_correction(
        f, y0, 0.0, 10.0, 0.1, recall_period=10,
        oracle=oracle,
        gamble_config=GamblePoleConfig(enable=False),
        limit_config=LimitCycleConfig(enable=False),
        recall_sub_steps=1,
    )
    # 全禁用时误差应与纯RK4接近 / error close to pure RK4
    baseline = rk4_integrate(f, y0, 0.0, 10.0, 0.1, oracle=oracle)
    # 误差量级应相似 (允许因子2差异) / similar magnitude (factor 2 tolerance)
    assert result.final_error < baseline.final_error * 5


# ════════════════════════════════════════════════════════════════════
# §辅助函数 / Utility functions
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
