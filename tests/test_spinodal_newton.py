"""Newton-Raphson spinodal solver: validation against analytical regular solution.

For a symmetric A-B regular solution with interaction parameter Omega:
    d²G/dx² = R*T / (x*(1-x)) - 2*Omega = 0
    => x_sp(T) = (1 ± sqrt(1 - T/T_c)) / 2
    where T_c = Omega / (2*R)

Both branches are known analytically and serve as reference.
"""

from __future__ import annotations

import numpy as np
import pytest

from spinodal import spinodal_at_temperature, spinodal_curve_newton

OMEGA = 20_000.0
GAS_CONSTANT = 8.314462618
T_CRITICAL = OMEGA / (2.0 * GAS_CONSTANT)  # ≈ 1202.6 K


def _x_analytical(T: float) -> tuple[float | None, float | None]:
    """Return (x_low, x_high) analytical spinodal compositions at temperature T.

    Returns (None, None) for T >= T_critical.
    """
    disc = 1.0 - T / T_CRITICAL
    if disc <= 0.0:
        return None, None
    sq = np.sqrt(disc)
    return (1.0 - sq) / 2.0, (1.0 + sq) / 2.0


class TestSpinodaAtTemperature:
    """Tests for spinodal_at_temperature against analytical regular solution."""

    @pytest.fixture(scope="class")
    def dbf(self, reg_sol_dbf):
        return reg_sol_dbf

    @pytest.mark.parametrize("T_frac", [0.3, 0.5, 0.7, 0.9])
    def test_low_branch_matches_analytical(self, dbf, reg_sol_params, T_frac: float) -> None:
        T = T_CRITICAL * T_frac
        x_ana_low, _ = _x_analytical(T)
        assert x_ana_low is not None

        x_num = spinodal_at_temperature(
            dbf,
            reg_sol_params["components"],
            reg_sol_params["phase_name"],
            "B",
            T,
            reg_sol_params["P"],
            x_init=x_ana_low * 0.5,  # start below the spinodal
        )

        assert x_num is not None, f"Newton did not converge at T={T:.1f} K"
        assert abs(x_num - x_ana_low) < 1e-5, f"T={T:.1f} K: |x_num - x_ana| = {abs(x_num - x_ana_low):.2e} > 1e-5"

    @pytest.mark.parametrize("T_frac", [0.3, 0.5, 0.7, 0.9])
    def test_high_branch_matches_analytical(self, dbf, reg_sol_params, T_frac: float) -> None:
        T = T_CRITICAL * T_frac
        _, x_ana_high = _x_analytical(T)
        assert x_ana_high is not None

        x_num = spinodal_at_temperature(
            dbf,
            reg_sol_params["components"],
            reg_sol_params["phase_name"],
            "B",
            T,
            reg_sol_params["P"],
            x_init=(x_ana_high + 0.99) / 2.0,  # start above the spinodal
        )

        assert x_num is not None, f"Newton did not converge at T={T:.1f} K (high branch)"
        assert abs(x_num - x_ana_high) < 1e-5, f"T={T:.1f} K: |x_num - x_ana| = {abs(x_num - x_ana_high):.2e} > 1e-5"

    def test_above_critical_returns_none(self, dbf, reg_sol_params) -> None:
        T = T_CRITICAL * 1.2
        x_num = spinodal_at_temperature(
            dbf,
            reg_sol_params["components"],
            reg_sol_params["phase_name"],
            "B",
            T,
            reg_sol_params["P"],
            x_init=0.3,
        )
        assert x_num is None, f"Expected None above T_c but got x={x_num} at T={T:.1f} K"

    def test_symmetry(self, dbf, reg_sol_params) -> None:
        T = T_CRITICAL * 0.6
        x_low = spinodal_at_temperature(
            dbf,
            reg_sol_params["components"],
            reg_sol_params["phase_name"],
            "B",
            T,
            reg_sol_params["P"],
            x_init=0.1,
        )
        x_high = spinodal_at_temperature(
            dbf,
            reg_sol_params["components"],
            reg_sol_params["phase_name"],
            "B",
            T,
            reg_sol_params["P"],
            x_init=0.9,
        )
        assert x_low is not None and x_high is not None
        assert abs(x_low + x_high - 1.0) < 1e-5, f"x_low + x_high = {x_low + x_high:.8f} != 1 (broken symmetry)"


class TestSpinodaLCurveNewton:
    """Tests for spinodal_curve_newton against analytical regular solution."""

    @pytest.fixture(scope="class")
    def curve_result(self, reg_sol_dbf, reg_sol_params):
        T_arr = np.linspace(100.0, T_CRITICAL * 0.98, 25)
        return spinodal_curve_newton(
            reg_sol_dbf,
            reg_sol_params["components"],
            reg_sol_params["phase_name"],
            "B",
            T_arr,
            reg_sol_params["P"],
            x_init_low=0.05,
            x_init_high=0.95,
        )

    def test_return_keys(self, curve_result) -> None:
        assert set(curve_result.keys()) == {"T", "x_low", "x_high", "comp_name"}

    def test_all_finite_below_critical(self, curve_result) -> None:
        assert np.all(np.isfinite(curve_result["x_low"])), "x_low has NaN below T_c"
        assert np.all(np.isfinite(curve_result["x_high"])), "x_high has NaN below T_c"

    def test_low_branch_accuracy(self, curve_result) -> None:
        for T, x_num in zip(curve_result["T"], curve_result["x_low"]):
            x_ana, _ = _x_analytical(T)
            assert x_ana is not None
            assert abs(x_num - x_ana) < 1e-5, f"T={T:.1f} K: low branch |err| = {abs(x_num - x_ana):.2e}"

    def test_high_branch_accuracy(self, curve_result) -> None:
        for T, x_num in zip(curve_result["T"], curve_result["x_high"]):
            _, x_ana = _x_analytical(T)
            assert x_ana is not None
            assert abs(x_num - x_ana) < 1e-5, f"T={T:.1f} K: high branch |err| = {abs(x_num - x_ana):.2e}"

    def test_symmetry_across_T(self, curve_result) -> None:
        residuals = curve_result["x_low"] + curve_result["x_high"] - 1.0
        assert np.all(np.abs(residuals) < 1e-5), f"Max symmetry residual {np.max(np.abs(residuals)):.2e}"

    def test_above_critical_returns_nan(self, reg_sol_dbf, reg_sol_params) -> None:
        T_arr = np.array([T_CRITICAL * 1.1, T_CRITICAL * 1.5])
        result = spinodal_curve_newton(
            reg_sol_dbf,
            reg_sol_params["components"],
            reg_sol_params["phase_name"],
            "B",
            T_arr,
            reg_sol_params["P"],
        )
        assert np.all(np.isnan(result["x_low"])), "Expected NaN above T_c (x_low)"
        assert np.all(np.isnan(result["x_high"])), "Expected NaN above T_c (x_high)"

    def test_comp_name_recorded(self, curve_result) -> None:
        assert curve_result["comp_name"] == "B"
