"""Tests for spinodal boundary detection utilities."""

from __future__ import annotations

import numpy as np
import pytest
from pycalphad import variables as v

from spinodal.boundary_detection import (
    spinodal_boundary_binary,
    spinodal_temperature_binary,
    spinodal_boundary_along_path,
)
from spinodal.examples.regular_solution import (
    analytical_spinodal_T,
    _t_critical,
    _pressure,
    _load_database,
    _phase_name,
    _components,
)


@pytest.fixture(scope="module")
def reg_dbf():
    return _load_database()


class TestSpinodal_TemperatureBinary:
    @pytest.mark.parametrize("x_b", [0.2, 0.5, 0.8])
    def test_matches_analytical(self, reg_dbf, x_b: float) -> None:
        T_ana = analytical_spinodal_T(x_b)
        T_lo = max(100.0, T_ana * 0.5)
        T_hi = min(_t_critical * 1.5, T_ana * 1.5)
        T_num = spinodal_temperature_binary(
            reg_dbf, _components, _phase_name, x_b, "B",
            _pressure, T_lo, T_hi,
        )
        assert T_num is not None, f"No spinodal found at x_B={x_b}"
        assert abs(T_num - T_ana) < 1.0

    def test_returns_none_when_no_crossing(self, reg_dbf) -> None:
        # Both bounds above critical T → no crossing
        T_lo = _t_critical * 1.1
        T_hi = _t_critical * 2.0
        result = spinodal_temperature_binary(
            reg_dbf, _components, _phase_name, 0.5, "B",
            _pressure, T_lo, T_hi,
        )
        assert result is None


class TestSpinodal_BoundaryBinary:
    def test_finds_two_crossings(self, reg_dbf) -> None:
        # At T below T_c, spinodal has two boundaries in composition space
        T = 0.9 * _t_critical
        comp_values = np.linspace(0.05, 0.95, 50)
        crossings = spinodal_boundary_binary(
            reg_dbf, _components, _phase_name, T, _pressure,
            comp_values, "B",
        )
        assert len(crossings) == 2, (
            f"Expected 2 crossings, got {len(crossings)}: {crossings}"
        )

    def test_crossing_dict_keys(self, reg_dbf) -> None:
        T = 0.9 * _t_critical
        comp_values = np.linspace(0.05, 0.95, 20)
        crossings = spinodal_boundary_binary(
            reg_dbf, _components, _phase_name, T, _pressure,
            comp_values, "B",
        )
        for c in crossings:
            assert "x_boundary" in c
            assert "comp_name" in c
            assert 0.0 < c["x_boundary"] < 1.0

    def test_no_crossings_above_critical(self, reg_dbf) -> None:
        T = _t_critical * 1.5
        comp_values = np.linspace(0.05, 0.95, 20)
        crossings = spinodal_boundary_binary(
            reg_dbf, _components, _phase_name, T, _pressure,
            comp_values, "B",
        )
        assert len(crossings) == 0

    def test_symmetry_binary(self, reg_dbf) -> None:
        T = 0.85 * _t_critical
        comp_values = np.linspace(0.05, 0.95, 60)
        crossings = spinodal_boundary_binary(
            reg_dbf, _components, _phase_name, T, _pressure,
            comp_values, "B",
        )
        if len(crossings) == 2:
            x1 = crossings[0]["x_boundary"]
            x2 = crossings[1]["x_boundary"]
            # For a symmetric regular solution, crossings are symmetric about 0.5
            assert abs((x1 + x2) / 2.0 - 0.5) < 0.02


class TestSpinodal_BoundaryAlongPath:
    def test_linear_path_finds_crossing(self, reg_dbf) -> None:
        T = 0.85 * _t_critical
        path = [{"B": x} for x in np.linspace(0.05, 0.95, 40)]
        crossings = spinodal_boundary_along_path(
            reg_dbf, _components, _phase_name, T, _pressure, path
        )
        assert len(crossings) >= 1

    def test_path_crossing_keys(self, reg_dbf) -> None:
        T = 0.85 * _t_critical
        path = [{"B": x} for x in np.linspace(0.05, 0.95, 20)]
        crossings = spinodal_boundary_along_path(
            reg_dbf, _components, _phase_name, T, _pressure, path
        )
        for c in crossings:
            assert 0.0 <= c["t_boundary"] <= 1.0
            assert "composition" in c
            assert "B" in c["composition"]
