"""Gauge-invariant eigenanalysis of the chemical Gibbs energy Hessian.

Eigenvalues of H in Gibbs simplex composition space depend on the choice of
dependent component (gauge). This module implements the gauge-invariant approach
via the generalized eigenvalue problem H v = lambda G v, where G is the metric
tensor of the composition simplex.

For an N-component system, the metric tensor in composition space (with any
one component as the dependent) is G_ij = delta_ij + 1, which follows from
the inner product of simplex basis vectors in R^N.

A negative eigenvalue lambda_k implies the system is unstable to composition
fluctuations in the direction v_k (spinodal instability). The spinodal boundary
is where the smallest eigenvalue crosses zero.

References:
    Singh, S.P. et al. (2024), iterative Gram–Schmidt change-of-basis procedure.
    Kadirvel et al. (2024), explicit transformation matrices up to N=10.
    Kunselman (2026), Ch. 6, Eqs. 6.5–6.7.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh


def composition_metric(n: int) -> np.ndarray:
    r"""Return the composition simplex metric tensor G for N-1 independent components.

    For an N-component system, the N-1 simplex basis vectors
    v_i = e_i - e_N (i = 1, ..., N-1) have inner products
    G_ij = v_i · v_j = delta_ij + 1.

    Args:
        n (int): Number of independent composition variables (N-1 for N components).

    Returns:
        np.ndarray: (n, n) positive-definite metric tensor G.
    """
    return np.eye(n) + np.ones((n, n))


def gauge_invariant_eig(
    H: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Solve the gauge-invariant generalized eigenvalue problem H v = lambda G v.

    Eigenvalues are independent of the choice of dependent component and
    represent the principal curvatures of the Gibbs energy surface in
    gauge-invariant Cartesian composition space.

    Args:
        H (np.ndarray): (n, n) chemical Hessian in J/mol, from
            ``spinodal.hessian.chemical_hessian``.

    Returns:
        eigenvalues (np.ndarray): (n,) gauge-invariant eigenvalues in J/mol,
            sorted ascending. Negative values indicate spinodal instability.
        eigenvectors (np.ndarray): (n, n) matrix whose columns are the
            corresponding eigenvectors in Gibbs simplex composition coordinates,
            normalized such that v^T G v = 1.
    """
    n = H.shape[0]
    G = composition_metric(n)
    # scipy.linalg.eigh solves A v = lambda B v for symmetric A and positive-
    # definite B; eigenvalues are returned sorted ascending.
    eigenvalues, eigenvectors = eigh(H, G)
    return eigenvalues, eigenvectors


def n_negative_eigenvalues(H: np.ndarray) -> int:
    """Count the number of negative gauge-invariant eigenvalues of H.

    A result of k means the phase is unstable to composition fluctuations in
    k independent directions. The chemical spinodal corresponds to k >= 1.

    Args:
        H (np.ndarray): (n, n) chemical Hessian in J/mol.

    Returns:
        int: Number of negative gauge-invariant eigenvalues (0 means stable,
            >= 1 means spinodally unstable).
    """
    evals, _ = gauge_invariant_eig(H)
    return int(np.sum(evals < 0.0))


def min_eigenvalue(H: np.ndarray) -> float:
    """Return the smallest (most negative) gauge-invariant eigenvalue of H.

    The chemical spinodal boundary is where this quantity crosses zero from below.

    Args:
        H (np.ndarray): (n, n) chemical Hessian in J/mol.

    Returns:
        float: Smallest eigenvalue in J/mol. Negative means spinodal instability.
    """
    evals, _ = gauge_invariant_eig(H)
    return float(evals[0])
