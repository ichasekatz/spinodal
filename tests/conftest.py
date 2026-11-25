"""Shared pytest fixtures for spinodal tests."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from pycalphad import Database

# ── Regular-solution TDB (A-B binary, Omega = 20 kJ/mol) ─────────────────────
_REG_SOL_TDB = """\
ELEMENT /-   ELECTRON_GAS       0 0 0 !
ELEMENT VA   VACUUM             0 0 0 !
ELEMENT A    BLANK              0 0 0 !
ELEMENT B    BLANK              0 0 0 !

PHASE LIQUID %  1  1.0  !
CONSTITUENT LIQUID : A, B : !

PARAMETER G(LIQUID,A;0) 298.15 0; 6000 N !
PARAMETER G(LIQUID,B;0) 298.15 0; 6000 N !
PARAMETER G(LIQUID,A,B;0) 298.15 +2.00000E+04; 6000 N !
"""

OMEGA = 20_000.0          # J/mol
GAS_CONSTANT = 8.314462618  # J/(mol·K)
T_CRITICAL = OMEGA / (2.0 * GAS_CONSTANT)   # ≈ 1202.6 K


@pytest.fixture(scope="session")
def reg_sol_dbf() -> Database:
    """Load the A-B regular-solution database (session scope: loaded once)."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".tdb", delete=False, dir=tempfile.gettempdir()
    ) as fh:
        fh.write(_REG_SOL_TDB)
        tdb_path = Path(fh.name)
    return Database(str(tdb_path))


@pytest.fixture
def reg_sol_params() -> dict:
    """Return scalar parameters for the regular solution."""
    return {
        "components": ["A", "B", "VA"],
        "phase_name": "LIQUID",
        "P": 101_325.0,
        "omega": OMEGA,
        "T_critical": T_CRITICAL,
        "gas_constant": GAS_CONSTANT,
    }
