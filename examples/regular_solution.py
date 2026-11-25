"""Binary regular-solution spinodal validation.

Constructs a synthetic A-B binary regular-solution TDB with known interaction
parameter Omega, then verifies that the numerically computed spinodal temperature
T_sp(x) matches the analytical result:

    T_sp(x) = 2 * Omega * x * (1 - x) / R

to within a tolerance of 1 K across the composition range [0.1, 0.9].

Usage::

    python -m spinodal.examples.regular_solution

or::

    from spinodal.examples.regular_solution import run_validation
    errors = run_validation()

References:
    Cahn, J.W. (1961). On spinodal decomposition. Acta Metall. 9, 795–801.
    Kunselman (2026), Ch. 6, Eq. 6.1.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from pycalphad import Database, variables as v
from scipy.optimize import brentq

from spinodal.hessian import chemical_hessian

# ── Physical constants ────────────────────────────────────────────────────────
gas_constant = 8.314462618  # J / (mol·K)

# ── Regular solution parameters ───────────────────────────────────────────────
omega = 20_000.0        # J/mol — symmetric interaction parameter
_t_critical = omega / (2.0 * gas_constant)   # ≈ 1202.6 K at x=0.5

# ── TDB definition for the synthetic A-B regular solution ────────────────────
_tdb_content = """\
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

_phase_name = "LIQUID"
_components = ["A", "B", "VA"]
_pressure = 101_325.0  # Pa


def _load_database() -> Database:
    """Write TDB content to a temp file and load it into pycalphad.

    Returns:
        Database: Loaded pycalphad Database object.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".tdb", delete=False, dir=Path(tempfile.gettempdir())
    ) as fh:
        fh.write(_tdb_content)
        tdb_path = Path(fh.name)
    return Database(str(tdb_path))


def analytical_spinodal_T(x_b: float) -> float:
    """Analytical spinodal temperature for a symmetric binary regular solution.

    Args:
        x_b (float): Mole fraction of component B.

    Returns:
        float: Spinodal temperature in K. Returns 0 at x_b=0 or x_b=1.
    """
    return 2.0 * omega * x_b * (1.0 - x_b) / gas_constant


def numerical_min_eigenvalue(dbf: Database, x_b: float, T: float) -> float:
    """Evaluate the minimum eigenvalue of the chemical Hessian at (x_B, T).

    For a binary system the Hessian is scalar; gauge-invariant eigenvalue =
    H / G_11 = H / 2.

    Args:
        dbf (Database): Regular-solution database loaded from ``_load_database``.
        x_b (float): Mole fraction of component B.
        T (float): Temperature in K.

    Returns:
        float: Minimum gauge-invariant eigenvalue in J/mol.
    """
    conds = {v.T: T, v.P: _pressure, v.X("B"): x_b}
    H, _, _ = chemical_hessian(dbf, _components, _phase_name, conds)
    # Binary: H is 1×1; gauge-invariant eigenvalue = H[0,0] / G_11 = H[0,0] / 2
    return float(H[0, 0]) / 2.0


def find_spinodal_T(dbf: Database, x_b: float, T_lo: float, T_hi: float) -> float | None:
    """Find spinodal temperature at x_B by bisection on the minimum eigenvalue.

    Args:
        dbf (Database): Regular-solution database.
        x_b (float): Mole fraction of component B.
        T_lo (float): Lower temperature bound in K (must be in spinodal region).
        T_hi (float): Upper temperature bound in K (must be stable).

    Returns:
        float | None: Spinodal temperature in K, or None if sign change not found.
    """
    def _f(T: float) -> float:
        return numerical_min_eigenvalue(dbf, x_b, T)

    f_lo = _f(T_lo)
    f_hi = _f(T_hi)
    if np.sign(f_lo) == np.sign(f_hi):
        return None
    return float(brentq(_f, T_lo, T_hi, xtol=0.01))


def run_validation(
    x_values: np.ndarray | None = None,
    tolerance_K: float = 1.0,
) -> dict[str, np.ndarray]:
    """Validate numerical spinodal temperatures against the analytical formula.

    Args:
        x_values (np.ndarray | None): Mole fractions of B to test. Defaults to
            nine equally spaced points from 0.1 to 0.9.
        tolerance_K (float): Acceptable error in spinodal temperature, in K.

    Returns:
        dict containing:
            - ``x`` (np.ndarray): Tested mole fractions of B.
            - ``T_analytical`` (np.ndarray): Analytical spinodal temperatures in K.
            - ``T_numerical`` (np.ndarray): Numerically found spinodal temperatures in K.
            - ``errors`` (np.ndarray): Absolute errors |T_num - T_ana| in K.
            - ``max_error`` (float): Maximum absolute error across all tested points.
            - ``passed`` (bool): True if all errors are below tolerance_K.

    Raises:
        AssertionError: If the maximum error exceeds tolerance_K.
    """
    if x_values is None:
        x_values = np.linspace(0.1, 0.9, 9)

    dbf = _load_database()

    T_analytical = np.array([analytical_spinodal_T(x) for x in x_values])
    T_numerical = np.full_like(x_values, np.nan)

    for k, x_b in enumerate(x_values):
        T_ana = T_analytical[k]
        # Search slightly below and above the critical temperature
        T_lo = max(100.0, T_ana * 0.5)
        T_hi = min(_t_critical * 1.5, T_ana * 1.5)
        T_sp = find_spinodal_T(dbf, x_b, T_lo, T_hi)
        if T_sp is not None:
            T_numerical[k] = T_sp

    errors = np.abs(T_numerical - T_analytical)
    max_err = float(np.nanmax(errors))
    passed = bool(np.all(np.isfinite(T_numerical)) and max_err <= tolerance_K)

    print(f"{'x_B':>6}  {'T_ana (K)':>10}  {'T_num (K)':>10}  {'error (K)':>10}")
    print("-" * 44)
    for x_b, t_ana, t_num, err in zip(
        x_values, T_analytical, T_numerical, errors, strict=True
    ):
        print(f"{x_b:6.2f}  {t_ana:10.2f}  {t_num:10.2f}  {err:10.4f}")

    print(f"\nMax error: {max_err:.4f} K  (tolerance: {tolerance_K:.1f} K)")
    print(f"Validation {'PASSED' if passed else 'FAILED'}")

    if not passed:
        raise AssertionError(
            f"Spinodal validation failed: max error {max_err:.3f} K "
            f"exceeds tolerance {tolerance_K:.1f} K"
        )

    return {
        "x": x_values,
        "T_analytical": T_analytical,
        "T_numerical": T_numerical,
        "errors": errors,
        "max_error": max_err,
        "passed": passed,
    }


if __name__ == "__main__":
    run_validation()
