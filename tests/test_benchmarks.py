# 希伯来书 11:1 —— "信就是所望之事的实底,是未见之事的确据."
# Hebrews 11:1 — "Now faith is the substance of things hoped for, the evidence of things not seen."
"""
Benchmark tests on three ODE systems / 三系统基准测试.

Tests the algorithm on systems with and without analytical solutions,
proving it is not circular-reasoning with ground truth.
在有无解析解的系统上测试算法, 证明非循环论证.
"""

import numpy as np
import pytest

from rk_recall.rk_recall_compensation import (
    GamblePoleConfig,
    LimitCycleConfig,
    make_dop853_oracle,
    rk4_hybrid_correction,
    rk4_integrate,
    rk4_with_recall,
)
from rk_recall.benchmarks import (
    BENCHMARKS,
    harmonic_exact,
    harmonic_oscillator,
    lorenz,
    van_der_pol,
)


# ════════════════════════════════════════════════════════════════════
# §系统1: 谐振子 (有解析解 — 正确性基准)
# §System 1: Harmonic oscillator (analytical — correctness baseline)
# ════════════════════════════════════════════════════════════════════

class TestHarmonicOscillator:
    """谐振子测试 (有解析解, 验证算法正确性).
    Harmonic oscillator tests (analytical solution, verify correctness)."""

    def setup_method(self):
        self.y0 = np.array([1.0, 0.0])
        self.omega = 1.0
        self.t_end = 50.0
        self.h = 0.1
        # §用闭包固定 omega, 使 f 签名为 f(t, y) / fix omega via closure
        omega = self.omega
        self.f = lambda t, y: harmonic_oscillator(t, y, omega)

    def test_baseline_rk4_has_error(self):
        """纯 RK4 应有累积误差 / pure RK4 has accumulating error."""
        oracle = make_dop853_oracle(self.f, self.y0, 0.0, self.t_end, self.h)
        result = rk4_integrate(self.f, self.y0, 0.0, self.t_end, self.h, oracle=oracle)
        assert result.peak_error > 1e-6  # RK4 应有明显误差 / RK4 should show visible error

    def test_recall_reduces_error(self):
        """回忆校正应显著降低误差 / recall significantly reduces error."""
        oracle = make_dop853_oracle(self.f, self.y0, 0.0, self.t_end, self.h)
        baseline = rk4_integrate(self.f, self.y0, 0.0, self.t_end, self.h, oracle=oracle)
        corrected = rk4_with_recall(
            self.f, self.y0, 0.0, self.t_end, self.h,
            recall_period=10, oracle=oracle,
        )
        assert corrected.final_error < baseline.final_error * 0.1  # 至少降低90% / at least 90% reduction

    def test_hybrid_achieves_engineering_zero(self):
        """三层混合应达到工程零误差 / hybrid achieves engineering zero."""
        result = rk4_hybrid_correction(
            self.f, self.y0, 0.0, self.t_end, self.h, recall_period=10,
            gamble_config=GamblePoleConfig(error_threshold=1e-10),
            limit_config=LimitCycleConfig(amplitude_bound=1e-11),
            recall_sub_steps=20,
        )
        assert result.final_error < 1e-9  # 工程零误差 / engineering zero

    def test_oracle_matches_analytical(self):
        """DOP853 oracle 应匹配解析解 / DOP853 oracle matches analytical."""
        oracle = make_dop853_oracle(self.f, self.y0, 0.0, self.t_end, self.h)
        for t in [0.0, 5.0, 12.3, 25.0, 49.9]:
            y_ref = oracle(t)
            y_exact = harmonic_exact(t, self.y0, self.omega)
            np.testing.assert_allclose(y_ref, y_exact, atol=1e-10)


# ════════════════════════════════════════════════════════════════════
# §系统2: 洛伦兹吸引子 (无解析解 — 真运算验证)
# §System 2: Lorenz attractor (no analytical — real computation)
# ════════════════════════════════════════════════════════════════════

class TestLorenzAttractor:
    """洛伦兹吸引子测试 (无解析解, 验证真运算).
    Lorenz attractor tests (no analytical, verify real computation)."""

    def setup_method(self):
        self.y0 = np.array([1.0, 1.0, 1.0])
        self.t_end = 10.0
        self.h = 0.01
        self.f = lambda t, y: lorenz(t, y)

    def test_baseline_rk4_runs(self):
        """纯 RK4 在洛伦兹上应正常运行 / pure RK4 runs on Lorenz."""
        result = rk4_integrate(self.f, self.y0, 0.0, self.t_end, self.h)
        assert result.y_array.shape == (int(self.t_end / self.h) + 1, 3)
        # 状态应保持有界 (洛伦兹吸引子有界) / state bounded (Lorenz is bounded)
        assert np.all(np.abs(result.y_array) < 100)

    def test_hybrid_runs_without_analytical(self):
        """三层混合在无解析解系统上应正常运行 / hybrid runs without analytical."""
        result = rk4_hybrid_correction(
            self.f, self.y0, 0.0, self.t_end, self.h, recall_period=10,
            gamble_config=GamblePoleConfig(error_threshold=1e-6),
            limit_config=LimitCycleConfig(amplitude_bound=1e-4),
            recall_sub_steps=5,
        )
        assert result.y_array.shape[0] > 0
        assert np.all(np.isfinite(result.y_array))

    def test_hybrid_stays_close_to_oracle(self):
        """三层混合应保持接近 oracle 参考解 / hybrid stays close to oracle reference."""
        result = rk4_hybrid_correction(
            self.f, self.y0, 0.0, self.t_end, self.h, recall_period=10,
            gamble_config=GamblePoleConfig(error_threshold=1e-6),
            limit_config=LimitCycleConfig(amplitude_bound=1e-4),
            recall_sub_steps=5,
        )
        # 误差 vs DOP853 oracle 应有界 / error vs DOP853 oracle should be bounded
        assert result.peak_error < 1.0  # 混沌系统误差有界即可 / bounded is sufficient for chaotic


# ════════════════════════════════════════════════════════════════════
# §系统3: Van der Pol 振子 (无闭式解 — 真运算验证)
# §System 3: Van der Pol oscillator (no closed-form — real computation)
# ════════════════════════════════════════════════════════════════════

class TestVanDerPol:
    """Van der Pol 振子测试 (无闭式解, 验证真运算).
    Van der Pol oscillator tests (no closed-form, verify real computation)."""

    def setup_method(self):
        self.y0 = np.array([2.0, 0.0])
        self.mu = 1.0
        self.t_end = 30.0
        self.h = 0.05
        mu = self.mu
        self.f = lambda t, y: van_der_pol(t, y, mu)

    def test_baseline_rk4_runs(self):
        """纯 RK4 在 Van der Pol 上应正常运行 / pure RK4 runs on Van der Pol."""
        result = rk4_integrate(self.f, self.y0, 0.0, self.t_end, self.h)
        assert result.y_array.shape == (int(self.t_end / self.h) + 1, 2)
        assert np.all(np.isfinite(result.y_array))

    def test_hybrid_runs_without_analytical(self):
        """三层混合在无闭式解系统上应正常运行 / hybrid runs without closed-form."""
        result = rk4_hybrid_correction(
            self.f, self.y0, 0.0, self.t_end, self.h, recall_period=10,
            gamble_config=GamblePoleConfig(error_threshold=1e-8),
            limit_config=LimitCycleConfig(amplitude_bound=1e-6),
            recall_sub_steps=10,
        )
        assert result.y_array.shape[0] > 0
        assert np.all(np.isfinite(result.y_array))

    def test_hybrid_reduces_error_vs_baseline(self):
        """三层混合应比纯RK4误差更小 / hybrid has smaller error than pure RK4."""
        oracle = make_dop853_oracle(self.f, self.y0, 0.0, self.t_end, self.h)
        baseline = rk4_integrate(self.f, self.y0, 0.0, self.t_end, self.h, oracle=oracle)
        hybrid = rk4_hybrid_correction(
            self.f, self.y0, 0.0, self.t_end, self.h, recall_period=10,
            oracle=oracle,
            gamble_config=GamblePoleConfig(error_threshold=1e-8),
            limit_config=LimitCycleConfig(amplitude_bound=1e-6),
            recall_sub_steps=10,
        )
        assert hybrid.final_error < baseline.final_error

    def test_hybrid_high_precision(self):
        """三层混合在Van der Pol上应达到高精度 / hybrid achieves high precision."""
        result = rk4_hybrid_correction(
            self.f, self.y0, 0.0, self.t_end, self.h, recall_period=10,
            gamble_config=GamblePoleConfig(error_threshold=1e-10),
            limit_config=LimitCycleConfig(amplitude_bound=1e-11),
            recall_sub_steps=20,
        )
        # Van der Pol 比 benchmark 应达到高精度 / should achieve good precision
        assert result.final_error < 1e-6
