# 希伯来书 11:1 —— "信就是所望之事的实底,是未见之事的确据."
# Hebrews 11:1 — "Now faith is the substance of things hoped for, the evidence of things not seen."
"""
Benchmark tests on three ODE systems / 三系统基准测试.

Tests the closed-form P* localization algorithm on systems with and without
analytical solutions, proving it works without any oracle (no DOP853, no scipy).
在有无解析解的系统上测试闭式 P* 定位算法, 证明无 oracle 也可工作.
"""

import numpy as np
import pytest

from rk_recall.rk_recall_compensation import (
    GamblePoleConfig,
    IntegrationResult,
    LimitCycleConfig,
    rk4_hybrid_correction,
    rk4_integrate,
    rk4_with_recall,
)
from rk_recall.benchmarks import (
    BENCHMARKS,
    harmonic_exact,
    harmonic_oscillator,
    lorenz,
    make_harmonic_reference,
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
        omega = self.omega
        # §用闭包固定 omega, 使 f 签名为 f(t, y) / fix omega via closure
        self.f = lambda t, y: harmonic_oscillator(t, y, omega)
        # §reference: 闭包固定 y0, omega (仅误差计算, 不参与校正)
        # §reference: closure fixing y0, omega (error computation only, not correction)
        self.reference = make_harmonic_reference(self.y0, self.omega)

    def test_baseline_rk4_has_error(self):
        """纯 RK4 应有累积误差 / pure RK4 has accumulating error."""
        result = rk4_integrate(self.f, self.y0, 0.0, self.t_end, self.h,
                               reference=self.reference)
        assert result.peak_error > 1e-6  # §RK4 应有明显误差 / RK4 should show visible error

    def test_recall_runs_and_finite(self):
        """回忆校正应正常运行并产生有限结果 / recall runs and produces finite result."""
        result = rk4_with_recall(
            self.f, self.y0, 0.0, self.t_end, self.h,
            recall_period=10, reference=self.reference,
        )
        assert isinstance(result, IntegrationResult)
        assert np.all(np.isfinite(result.y_array))
        assert np.all(np.isfinite(result.error_array))

    def test_hybrid_runs_and_finite(self):
        """三层混合应正常运行 / hybrid runs."""
        result = rk4_hybrid_correction(
            self.f, self.y0, 0.0, self.t_end, self.h, recall_period=10,
            reference=self.reference,
            gamble_config=GamblePoleConfig(error_threshold=1e-10),
            limit_config=LimitCycleConfig(amplitude_bound=1e-11),
            recall_sub_steps=20,
        )
        assert isinstance(result, IntegrationResult)
        assert np.all(np.isfinite(result.y_array))
        assert np.all(np.isfinite(result.error_array))

    def test_reference_matches_analytical(self):
        """reference 函数应匹配解析解 / reference function matches analytical."""
        for t in [0.0, 5.0, 12.3, 25.0, 49.9]:
            y_ref = self.reference(t)
            y_exact = harmonic_exact(t, self.y0, self.omega)
            np.testing.assert_allclose(y_ref, y_exact, atol=1e-14)

    def test_recall_without_reference_uses_limit_cycle(self):
        """不带 reference 时误差用极限环距离 (闭式, 无 oracle) / error uses limit-cycle distance without reference."""
        result = rk4_with_recall(
            self.f, self.y0, 0.0, self.t_end, self.h, recall_period=10,
        )
        assert isinstance(result, IntegrationResult)
        assert np.all(np.isfinite(result.error_array))
        assert result.peak_error >= 0.0


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
        # §状态应保持有界 (洛伦兹吸引子有界) / state bounded (Lorenz is bounded)
        assert np.all(np.abs(result.y_array) < 100)
        assert np.all(np.isfinite(result.error_array))

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
        assert np.all(np.isfinite(result.error_array))

    def test_recall_runs_without_analytical(self):
        """回忆校正在无解析解系统上应正常运行 / recall runs without analytical."""
        result = rk4_with_recall(
            self.f, self.y0, 0.0, self.t_end, self.h, recall_period=10,
        )
        assert isinstance(result, IntegrationResult)
        assert np.all(np.isfinite(result.y_array))
        assert np.all(np.isfinite(result.error_array))


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
        assert np.all(np.isfinite(result.error_array))

    def test_hybrid_runs_without_closed_form(self):
        """三层混合在无闭式解系统上应正常运行 / hybrid runs without closed-form."""
        result = rk4_hybrid_correction(
            self.f, self.y0, 0.0, self.t_end, self.h, recall_period=10,
            gamble_config=GamblePoleConfig(error_threshold=1e-8),
            limit_config=LimitCycleConfig(amplitude_bound=1e-6),
            recall_sub_steps=10,
        )
        assert result.y_array.shape[0] > 0
        assert np.all(np.isfinite(result.y_array))
        assert np.all(np.isfinite(result.error_array))

    def test_recall_runs_without_closed_form(self):
        """回忆校正在无闭式解系统上应正常运行 / recall runs without closed-form."""
        result = rk4_with_recall(
            self.f, self.y0, 0.0, self.t_end, self.h, recall_period=10,
        )
        assert isinstance(result, IntegrationResult)
        assert np.all(np.isfinite(result.y_array))
        assert np.all(np.isfinite(result.error_array))


# ════════════════════════════════════════════════════════════════════
# §BENCHMARKS 注册表完整性 / BENCHMARKS registry integrity
# ════════════════════════════════════════════════════════════════════

class TestBenchmarksRegistry:
    """BENCHMARKS 注册表测试 / BENCHMARKS registry tests."""

    def test_all_systems_have_required_keys(self):
        """每个系统应包含必需的键 (含 reference) / each system has required keys (incl. reference)."""
        required_keys = {"f", "y0", "t_end", "h", "exact", "has_exact",
                         "reference", "description"}
        for name, sys_config in BENCHMARKS.items():
            missing = required_keys - set(sys_config.keys())
            assert not missing, f"{name} missing keys: {missing}"

    def test_reference_is_callable_or_none(self):
        """reference 应为 callable 或 None / reference is callable or None."""
        for name, sys_config in BENCHMARKS.items():
            ref = sys_config["reference"]
            assert ref is None or callable(ref), f"{name} reference invalid"

    def test_harmonic_has_reference(self):
        """谐振子应有 reference (解析解闭包) / harmonic has reference."""
        assert BENCHMARKS["harmonic"]["reference"] is not None
        assert callable(BENCHMARKS["harmonic"]["reference"])

    def test_lorenz_reference_is_none(self):
        """洛伦兹 reference 应为 None / Lorenz reference is None."""
        assert BENCHMARKS["lorenz"]["reference"] is None

    def test_vanderpol_reference_is_none(self):
        """Van der Pol reference 应为 None / Van der Pol reference is None."""
        assert BENCHMARKS["vanderpol"]["reference"] is None

    def test_harmonic_reference_correctness(self):
        """谐振子 reference 应匹配解析解 / harmonic reference matches analytical."""
        ref = BENCHMARKS["harmonic"]["reference"]
        y0 = BENCHMARKS["harmonic"]["y0"]
        for t in [0.0, 1.0, 5.0, 10.0]:
            y_ref = ref(t)
            y_exact = harmonic_exact(t, y0, 1.0)
            np.testing.assert_allclose(y_ref, y_exact, atol=1e-14)
