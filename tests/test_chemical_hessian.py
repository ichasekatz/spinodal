"""Tests for the chemical Hessian via Jansson derivatives."""

from __future__ import annotations

import numpy as np
import pytest
from pycalphad import variables as v

from spinodal.hessian import chemical_hessian
from spinodal._exceptions import DependentComponentError


class TestChemicalHessianRegularSolution:
    """Validate Hessian against the analytical result for a regular solution.

    For G_mix = RT[x ln x + (1-x) ln(1-x)] + Omega x(1-x):
        H[0,0] = d²G/dx² = RT/(x(1-x)) - 2*Omega
    The binary (1×1) Hessian entry must equal this.
    """

    @pytest.mark.parametrize("x_b", [0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9])
    def test_diagonal_entry_analytical(
        self, reg_sol_dbf, reg_sol_params, x_b: float
    ) -> None:
        p = reg_sol_params
        T = 1500.0  # above critical T so system is stable
        conds = {v.T: T, v.P: p["P"], v.X("B"): x_b}
        H, indep, dep = chemical_hessian(reg_sol_dbf, p["components"], p["phase_name"], conds)

        # Analytical second derivative of ideal + regular-solution mixing Gibbs energy
        R = p["gas_constant"]
        omega = p["omega"]
        d2G_dx2 = R * T / (x_b * (1.0 - x_b)) - 2.0 * omega

        # Binary H is 1×1; gauge-invariant eigenvalue = H[0,0] / 2
        # but H[0,0] itself equals dmu_A/dx_B - dmu_dep/dx_B = d(µ_A - µ_B)/dx_B
        # For a symmetric binary: H[0,0] = d²G_m/dx_B² = analytical value
        assert H.shape == (1, 1)
        assert float(H[0, 0]) == pytest.approx(d2G_dx2, rel=1e-3)

    def test_indep_dep_components(self, reg_sol_dbf, reg_sol_params) -> None:
        p = reg_sol_params
        conds = {v.T: 1500.0, v.P: p["P"], v.X("B"): 0.3}
        H, indep, dep = chemical_hessian(reg_sol_dbf, p["components"], p["phase_name"], conds)
        assert len(indep) == 1
        assert dep.upper() in ("A", "B")
        assert indep[0].upper() != dep.upper()

    def test_explicit_dep_component(self, reg_sol_dbf, reg_sol_params) -> None:
        p = reg_sol_params
        # dep_component="A" requires X(B) fixed (B is perturbed)
        conds_xb = {v.T: 1500.0, v.P: p["P"], v.X("B"): 0.3}
        H_a, _, dep_a = chemical_hessian(
            reg_sol_dbf, p["components"], p["phase_name"], conds_xb, dep_component="A"
        )
        # dep_component="B" requires X(A) fixed (A is perturbed); x_A = 1 - x_B = 0.7
        conds_xa = {v.T: 1500.0, v.P: p["P"], v.X("A"): 0.7}
        H_b, _, dep_b = chemical_hessian(
            reg_sol_dbf, p["components"], p["phase_name"], conds_xa, dep_component="B"
        )
        assert dep_a == "A"
        assert dep_b == "B"
        # d²G/dx² is the same scalar regardless of which component is chosen as dependent
        assert float(H_a[0, 0]) == pytest.approx(float(H_b[0, 0]), rel=1e-3)

    def test_hessian_symmetry(self, reg_sol_dbf, reg_sol_params) -> None:
        """H must be symmetric (1×1 trivially; still check shape and dtype)."""
        p = reg_sol_params
        conds = {v.T: 1500.0, v.P: p["P"], v.X("B"): 0.5}
        H, _, _ = chemical_hessian(reg_sol_dbf, p["components"], p["phase_name"], conds)
        np.testing.assert_allclose(H, H.T, atol=1e-6)

    def test_hessian_negative_below_critical(self, reg_sol_dbf, reg_sol_params) -> None:
        p = reg_sol_params
        T_below = 0.5 * p["T_critical"]  # deeply inside spinodal
        conds = {v.T: T_below, v.P: p["P"], v.X("B"): 0.5}
        H, _, _ = chemical_hessian(reg_sol_dbf, p["components"], p["phase_name"], conds)
        assert float(H[0, 0]) < 0.0

    def test_hessian_positive_above_critical(self, reg_sol_dbf, reg_sol_params) -> None:
        p = reg_sol_params
        T_above = 2.0 * p["T_critical"]
        conds = {v.T: T_above, v.P: p["P"], v.X("B"): 0.5}
        H, _, _ = chemical_hessian(reg_sol_dbf, p["components"], p["phase_name"], conds)
        assert float(H[0, 0]) > 0.0
