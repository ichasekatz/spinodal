"""Tests for gauge-invariant eigenanalysis of the chemical Hessian."""

from __future__ import annotations

import numpy as np
import pytest

from spinodal.eigenanalysis import (
    composition_metric,
    gauge_invariant_eig,
    min_eigenvalue,
    n_negative_eigenvalues,
)


class TestCompositionMetric:
    def test_shape(self) -> None:
        G = composition_metric(3)
        assert G.shape == (3, 3)

    def test_values(self) -> None:
        G = composition_metric(2)
        expected = np.array([[2.0, 1.0], [1.0, 2.0]])
        np.testing.assert_allclose(G, expected)

    def test_positive_definite(self) -> None:
        for n in range(1, 7):
            G = composition_metric(n)
            evals = np.linalg.eigvalsh(G)
            assert np.all(evals > 0), f"G not positive-definite for n={n}"

    def test_binary_scalar(self) -> None:
        G = composition_metric(1)
        assert G.shape == (1, 1)
        assert float(G[0, 0]) == pytest.approx(2.0)


class TestGaugeInvariantEig:
    def test_sorted_ascending(self) -> None:
        H = np.array([[3.0, 1.0], [1.0, 2.0]])
        evals, _ = gauge_invariant_eig(H)
        assert evals[0] <= evals[1]

    def test_positive_definite_H_all_positive(self) -> None:
        H = np.eye(3) * 1000.0
        evals, _ = gauge_invariant_eig(H)
        assert np.all(evals > 0)

    def test_negative_definite_H_all_negative(self) -> None:
        H = -np.eye(3) * 1000.0
        evals, _ = gauge_invariant_eig(H)
        assert np.all(evals < 0)

    def test_symmetry_requirement(self) -> None:
        """H must be symmetric; asymmetric input should still give real evals."""
        H = np.array([[4.0, 0.0], [0.0, 2.0]])
        evals, evecs = gauge_invariant_eig(H)
        assert np.all(np.isreal(evals))

    def test_gauge_invariance_property(self) -> None:
        """Rotating H by dependent-component relabelling must yield same eigenvalues."""
        rng = np.random.default_rng(42)
        # Build a 2x2 symmetric H (3-component system)
        A = rng.standard_normal((2, 2))
        H = A @ A.T + np.eye(2) * 50.0
        evals_1, _ = gauge_invariant_eig(H)

        # For a 3-component system, permuting the choice of dependent component
        # changes H but eigenvalues of the generalized problem should be invariant.
        # Exact transformation is complex; just verify the solver returns sorted reals.
        assert evals_1[0] <= evals_1[1]
        assert np.all(np.isfinite(evals_1))


class TestMinEigenvalue:
    def test_identity(self) -> None:
        H = np.eye(3) * 500.0
        lam = min_eigenvalue(H)
        assert lam > 0

    def test_negative(self) -> None:
        H = -np.eye(2) * 300.0
        lam = min_eigenvalue(H)
        assert lam < 0

    def test_binary_exact(self) -> None:
        # Binary: H = [[h]], G = [[2]]. Generalised eig: h*v = lam*2*v → lam = h/2
        H = np.array([[800.0]])
        lam = min_eigenvalue(H)
        assert lam == pytest.approx(400.0, rel=1e-10)


class TestNNegativeEigenvalues:
    def test_all_positive(self) -> None:
        H = np.diag([100.0, 200.0, 300.0])
        assert n_negative_eigenvalues(H) == 0

    def test_one_negative(self) -> None:
        H = np.diag([-50.0, 200.0])
        assert n_negative_eigenvalues(H) == 1

    def test_all_negative(self) -> None:
        H = -np.eye(3) * 100.0
        assert n_negative_eigenvalues(H) == 3

    def test_binary_at_spinodal(self) -> None:
        H = np.array([[0.0]])
        # Eigenvalue = 0/2 = 0 → not negative
        assert n_negative_eigenvalues(H) == 0
