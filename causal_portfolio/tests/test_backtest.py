"""Tests for backtest engine, metrics, and manifold optimizer."""

import numpy as np
import pytest

from causal_portfolio.backtest.engine import CPCMBacktester, BacktestResult
from causal_portfolio.backtest.metrics import (
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    calmar_ratio,
    average_turnover,
)
from causal_portfolio.optimizer.manifold import ManifoldOptimizer, estimate_covariance
from causal_portfolio.optimizer.diagnostics import (
    martingale_defect,
    structural_coherence_score,
)
from causal_portfolio.solvers.v1_linear import V1LinearSolver


# ── Metrics tests ───────────────────────────────────────────────────

class TestMetrics:
    def test_sharpe_positive_returns(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.01, 252)  # positive mean with variance
        s = sharpe_ratio(returns)
        assert s > 0

    def test_sharpe_zero_returns(self):
        returns = np.zeros(100)
        assert sharpe_ratio(returns) == 0.0

    def test_sharpe_constant_returns_zero(self):
        """Constant returns have zero vol → Sharpe = 0."""
        returns = np.ones(252) * 0.001
        assert sharpe_ratio(returns) == 0.0

    def test_sortino_positive_mean(self):
        """Sortino should be positive for positive-mean returns with variance."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.002, 0.01, 500)
        so = sortino_ratio(returns)
        assert so > 0

    def test_max_drawdown_negative(self):
        rng = np.random.default_rng(42)
        returns = rng.standard_normal(200) * 0.02
        mdd = max_drawdown(returns)
        assert mdd < 0

    def test_max_drawdown_no_drawdown(self):
        returns = np.ones(100) * 0.01
        mdd = max_drawdown(returns)
        assert abs(mdd) < 1e-10

    def test_calmar_ratio(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.01, 252)  # has drawdowns
        c = calmar_ratio(returns)
        assert c != 0.0  # should compute a ratio

    def test_average_turnover(self):
        weights = np.array([
            [0.5, 0.5],
            [0.6, 0.4],
            [0.5, 0.5],
        ])
        t = average_turnover(weights)
        assert t > 0

    def test_average_turnover_no_change(self):
        weights = np.ones((10, 3)) / 3
        assert average_turnover(weights) < 1e-10


# ── ManifoldOptimizer tests ─────────────────────────────────────────

class TestManifoldOptimizer:
    def test_weights_sum_to_one(self):
        rng = np.random.default_rng(42)
        solver = V1LinearSolver(alpha=0.1)
        drivers = rng.standard_normal((200, 3))
        returns = drivers @ rng.standard_normal((3, 5)) + rng.standard_normal((200, 5)) * 0.1
        solver.fit(drivers, returns)

        cov = np.cov(returns, rowvar=False)
        opt = ManifoldOptimizer(risk_aversion=1.0, max_weight=0.5)
        w = opt.optimize(solver, drivers[-1], cov)

        assert abs(np.sum(np.abs(w)) - 1.0) < 1e-10

    def test_max_weight_respected(self):
        rng = np.random.default_rng(42)
        solver = V1LinearSolver(alpha=0.1)
        drivers = rng.standard_normal((200, 3))
        returns = drivers @ rng.standard_normal((3, 5)) + rng.standard_normal((200, 5)) * 0.1
        solver.fit(drivers, returns)

        cov = np.cov(returns, rowvar=False)
        max_w = 0.3
        opt = ManifoldOptimizer(max_weight=max_w)
        w = opt.optimize(solver, drivers[-1], cov)

        assert np.all(np.abs(w) <= max_w + 1e-10)

    def test_tangent_projector_idempotent(self):
        opt = ManifoldOptimizer()
        J = np.random.randn(5, 3)
        P = opt._tangent_projector(J)
        # P @ P = P (idempotent)
        np.testing.assert_allclose(P @ P, P, atol=1e-10)

    def test_long_only(self):
        rng = np.random.default_rng(42)
        solver = V1LinearSolver(alpha=0.1)
        drivers = rng.standard_normal((200, 3))
        returns = drivers @ rng.standard_normal((3, 5)) + rng.standard_normal((200, 5)) * 0.1
        solver.fit(drivers, returns)

        cov = np.cov(returns, rowvar=False)
        opt = ManifoldOptimizer(long_only=True)
        w = opt.optimize(solver, drivers[-1], cov)

        assert np.all(w >= -1e-10)


# ── Covariance estimation ──────────────────────────────────────────

class TestCovarianceEstimation:
    def test_ledoit_wolf_positive_definite(self):
        rng = np.random.default_rng(42)
        returns = rng.standard_normal((100, 5))
        cov = estimate_covariance(returns, method="ledoit_wolf")
        eigenvalues = np.linalg.eigvalsh(cov)
        assert np.all(eigenvalues > 0)

    def test_sample_cov(self):
        rng = np.random.default_rng(42)
        returns = rng.standard_normal((100, 5))
        cov = estimate_covariance(returns, method="sample")
        assert cov.shape == (5, 5)


# ── Backtest engine ────────────────────────────────────────────────

class TestCPCMBacktester:
    @pytest.fixture
    def backtest_data(self):
        """Synthetic data for backtesting."""
        rng = np.random.default_rng(42)
        T = 500
        m = 3
        n_assets = 5

        drivers = rng.standard_normal((T, m))
        beta = rng.standard_normal((m, n_assets)) * 0.3
        returns = drivers @ beta + rng.standard_normal((T, n_assets)) * 0.02
        return drivers, returns

    def test_backtest_runs(self, backtest_data):
        drivers, returns = backtest_data
        solver = V1LinearSolver(alpha=1.0)
        optimizer = ManifoldOptimizer()
        bt = CPCMBacktester(
            solver=solver, optimizer=optimizer,
            use_ekf=True, rebalance_freq=20, train_window=200,
        )
        result = bt.run(returns, drivers)
        assert isinstance(result, BacktestResult)
        assert len(result.returns_series) == len(returns) - 200

    def test_backtest_produces_sharpe(self, backtest_data):
        drivers, returns = backtest_data
        solver = V1LinearSolver(alpha=1.0)
        optimizer = ManifoldOptimizer()
        bt = CPCMBacktester(
            solver=solver, optimizer=optimizer,
            use_ekf=False, rebalance_freq=20, train_window=200,
        )
        result = bt.run(returns, drivers)
        # With strong signal, Sharpe should be meaningfully different from 0
        assert abs(result.sharpe) > 0.01

    def test_insufficient_data_raises(self, backtest_data):
        drivers, returns = backtest_data
        solver = V1LinearSolver()
        optimizer = ManifoldOptimizer()
        bt = CPCMBacktester(
            solver=solver, optimizer=optimizer,
            train_window=600,  # > T
        )
        with pytest.raises(ValueError, match="Need T > train_window"):
            bt.run(returns, drivers)

    def test_summary_string(self, backtest_data):
        drivers, returns = backtest_data
        solver = V1LinearSolver()
        optimizer = ManifoldOptimizer()
        bt = CPCMBacktester(
            solver=solver, optimizer=optimizer,
            use_ekf=False, rebalance_freq=50, train_window=200,
        )
        result = bt.run(returns, drivers)
        s = result.summary()
        assert "Sharpe" in s
        assert "MaxDD" in s


# ── Diagnostics ─────────────────────────────────────────────────────

class TestDiagnostics:
    def test_martingale_defect_shape(self):
        rng = np.random.default_rng(42)
        solver = V1LinearSolver()
        drivers = rng.standard_normal((200, 3))
        returns = drivers @ rng.standard_normal((3, 4)) + rng.standard_normal((200, 4)) * 0.1
        solver.fit(drivers, returns)

        port_values = np.cumprod(1 + rng.standard_normal(100) * 0.01)
        filtered = rng.standard_normal((100, 3))

        defects = martingale_defect(port_values, filtered, solver)
        assert len(defects) == 99

    def test_coherence_score_non_negative(self):
        defects = np.random.randn(100)
        score = structural_coherence_score(defects)
        assert score >= 0
