"""Consistency tests: stability_map vs stability_at_point, and batch ordering."""

from __future__ import annotations

import numpy as np
import pytest
from pycalphad import variables as v

from spinodal.stability_map import (
    stability_at_point,
    stability_map,
    binary_spinodal_scan,
)
from spinodal.hessian import chemical_hessian
from spinodal.eigenanalysis import min_eigenvalue
from spinodal.examples.regular_solution import (
    _t_critical,
    _pressure,
    _phase_name,
    _components,
    _load_database,
)


@pytest.fixture(scope="module")
def reg_dbf():
    return _load_database()


class TestBatchConsistency:
    """stability_map results must equal individual stability_at_point calls."""

    def test_map_vs_point_min_eigenvalue(self, reg_dbf) -> None:
        T = 1500.0
        x_values = np.linspace(0.1, 0.9, 5)
        grid = [{"B": float(x)} for x in x_values]

        batch = stability_map(reg_dbf, _components, _phase_name, T, _pressure, grid)
        for k, x_b in enumerate(x_values):
            single = stability_at_point(
                reg_dbf, _components, _phase_name, T, _pressure, {"B": float(x_b)}
            )
            if single["converged"] and batch[k]["converged"]:
                assert batch[k]["min_eigenvalue"] == pytest.approx(
                    single["min_eigenvalue"], rel=1e-10
                )

    def test_map_stable_flag_consistent(self, reg_dbf) -> None:
        T = 2.0 * _t_critical  # all stable
        x_values = np.linspace(0.1, 0.9, 5)
        grid = [{"B": float(x)} for x in x_values]
        results = stability_map(reg_dbf, _components, _phase_name, T, _pressure, grid)
        for r in results:
            if r["converged"]:
                assert r["stable"] is True

    def test_binary_scan_vs_map(self, reg_dbf) -> None:
        T = 1500.0
        x_values = np.linspace(0.2, 0.8, 4)
        scan = binary_spinodal_scan(
            reg_dbf, _components, _phase_name, T, _pressure, x_values, "B"
        )
        grid = [{"B": float(x)} for x in x_values]
        grid_map = stability_map(reg_dbf, _components, _phase_name, T, _pressure, grid)
        for s, m in zip(scan, grid_map):
            if s["converged"] and m["converged"]:
                assert s["min_eigenvalue"] == pytest.approx(m["min_eigenvalue"], rel=1e-10)


class TestHessianVsEigenvalue:
    """chemical_hessian + min_eigenvalue must agree with stability_at_point."""

    @pytest.mark.parametrize("x_b", [0.2, 0.5, 0.8])
    def test_min_eig_consistent(self, reg_dbf, x_b: float) -> None:
        T = 1500.0
        conds = {v.T: T, v.P: _pressure, v.X("B"): x_b}
        H, _, _ = chemical_hessian(reg_dbf, _components, _phase_name, conds)
        lam_direct = min_eigenvalue(H)

        result = stability_at_point(
            reg_dbf, _components, _phase_name, T, _pressure, {"B": x_b}
        )
        assert result["converged"]
        assert lam_direct == pytest.approx(result["min_eigenvalue"], rel=1e-10)
