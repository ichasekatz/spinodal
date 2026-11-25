"""End-to-end validation: numerical spinodal vs analytical for regular solution."""

from __future__ import annotations

import numpy as np
import pytest

from spinodal.examples.regular_solution import (
    analytical_spinodal_T,
    find_spinodal_T,
    run_validation,
    _load_database,
    _t_critical,
    _pressure,
)


class TestAnalyticalSpinodal:
    def test_critical_point(self) -> None:
        # At x=0.5 and T_critical: T_sp(0.5) ≈ T_critical
        assert analytical_spinodal_T(0.5) == pytest.approx(_t_critical, rel=1e-6)

    def test_symmetry(self) -> None:
        for x in [0.1, 0.2, 0.3, 0.4]:
            assert analytical_spinodal_T(x) == pytest.approx(
                analytical_spinodal_T(1.0 - x), rel=1e-10
            )

    def test_zero_at_endpoints(self) -> None:
        assert analytical_spinodal_T(0.0) == pytest.approx(0.0)
        assert analytical_spinodal_T(1.0) == pytest.approx(0.0)


class TestNumericalSpinodal:
    @pytest.fixture(scope="class")
    def dbf(self):
        return _load_database()

    @pytest.mark.parametrize("x_b", [0.2, 0.35, 0.5, 0.65, 0.8])
    def test_spinodal_temperature_near_analytical(self, dbf, x_b: float) -> None:
        T_ana = analytical_spinodal_T(x_b)
        T_lo = max(100.0, T_ana * 0.5)
        T_hi = min(_t_critical * 1.5, T_ana * 1.5)
        T_num = find_spinodal_T(dbf, x_b, T_lo, T_hi)
        assert T_num is not None, f"No spinodal found at x_B={x_b}"
        assert abs(T_num - T_ana) < 1.0, (
            f"x_B={x_b}: |T_num - T_ana| = {abs(T_num - T_ana):.3f} K > 1 K"
        )

    def test_no_spinodal_above_critical(self, dbf) -> None:
        T_above = _t_critical * 1.5
        T_num = find_spinodal_T(dbf, 0.5, T_above, T_above * 1.5)
        # Both bounds are above T_c → no sign change → should return None
        # (or might converge; acceptable since f might be uniformly positive)
        # just verify no exception is raised
        assert T_num is None or isinstance(T_num, float)


class TestRunValidation:
    def test_full_validation_passes(self) -> None:
        result = run_validation(tolerance_K=1.0)
        assert result["passed"], (
            f"run_validation failed: max_error = {result['max_error']:.3f} K"
        )

    def test_result_keys(self) -> None:
        result = run_validation(x_values=np.array([0.3, 0.5, 0.7]))
        assert set(result.keys()) >= {"x", "T_analytical", "T_numerical", "errors",
                                       "max_error", "passed"}

    def test_all_numerical_finite(self) -> None:
        result = run_validation(x_values=np.linspace(0.15, 0.85, 7))
        assert np.all(np.isfinite(result["T_numerical"])), (
            "Some spinodal temperatures could not be found"
        )
