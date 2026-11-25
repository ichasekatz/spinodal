"""Spinodal boundary detection via eigenvalue root-finding.

Locates spinodal boundaries by bracketing sign changes of the minimum
gauge-invariant eigenvalue and applying Brent's method. Works for both
binary (1-D) and multicomponent (n-D) systems.

For a binary at fixed T, the spinodal boundary is a composition x* where
min_eigenvalue(H(x*)) = 0. For a multicomponent system at fixed T, the
boundary is a surface; this module provides tools to detect it along
arbitrary 1-D paths through composition space.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.optimize import brentq

from .eigenanalysis import min_eigenvalue
from .hessian import chemical_hessian
from ._exceptions import EquilibriumConvergenceError, SpinodallHessianError

if TYPE_CHECKING:
    from pycalphad import Database
    from pycalphad import variables as v


_BRENT_XTOL = 1e-6  # composition tolerance for brentq
_BRENT_RTOL = 4 * np.finfo(float).eps  # rtol=0 triggers ValueError on some scipy versions
_BRENT_MAXITER = 100


def _min_eig_at_conds(
    dbf: Database,
    components: list[str],
    phase_name: str,
    conds: dict,
    dep_component: str | None,
) -> float:
    """Return min eigenvalue, or +inf on convergence/Hessian failure."""
    try:
        H, _, _ = chemical_hessian(dbf, components, phase_name, conds, dep_component)
        return min_eigenvalue(H)
    except (EquilibriumConvergenceError, SpinodallHessianError, IndexError, ValueError):
        return float("inf")


def spinodal_boundary_binary(
    dbf: Database,
    components: list[str],
    phase_name: str,
    T: float,
    P: float,
    comp_values: np.ndarray,
    comp_name: str,
    dep_component: str | None = None,
    xtol: float = _BRENT_XTOL,
) -> list[dict[str, Any]]:
    """Detect spinodal boundary crossings along a binary composition scan.

    Scans comp_values left-to-right, finds adjacent pairs where the minimum
    eigenvalue changes sign, and refines each crossing via Brent's method.

    Args:
        dbf (Database): Thermodynamic database.
        components (list[str]): All system components, including 'VA'.
        phase_name (str): Name of the target phase.
        T (float): Temperature in K.
        P (float): Pressure in Pa.
        comp_values (np.ndarray): 1-D array of mole fractions to scan for
            comp_name, in ascending order.
        comp_name (str): Element symbol for the scanned component.
        dep_component (str | None): Dependent component. Auto-detected if None.
        xtol (float): Composition tolerance for Brent's method.

    Returns:
        list[dict]: One dict per boundary crossing found. Each dict contains:
            - ``x_boundary`` (float): Spinodal composition (mole fraction).
            - ``comp_name`` (str): Element symbol for the boundary composition.
            - ``eigenvalue_at_boundary`` (float): Min eigenvalue (≈ 0).
            - ``sign_left`` (int): Sign of eigenvalue just left of boundary.
            - ``sign_right`` (int): Sign of eigenvalue just right of boundary.
    """
    from pycalphad import variables as v

    def _eig(x: float) -> float:
        conds = {v.T: T, v.P: P, v.X(comp_name): x}
        return _min_eig_at_conds(dbf, components, phase_name, conds, dep_component)

    eig_vals = np.array([_eig(x) for x in comp_values])
    crossings = []

    for k in range(len(comp_values) - 1):
        e_left = eig_vals[k]
        e_right = eig_vals[k + 1]
        if not (np.isfinite(e_left) and np.isfinite(e_right)):
            continue
        if np.sign(e_left) == np.sign(e_right):
            continue
        x_lo = float(comp_values[k])
        x_hi = float(comp_values[k + 1])
        try:
            x_sp = brentq(
                _eig, x_lo, x_hi,
                xtol=xtol, rtol=_BRENT_RTOL, maxiter=_BRENT_MAXITER,
            )
        except ValueError:
            continue
        crossings.append({
            "x_boundary": float(x_sp),
            "comp_name": comp_name,
            "eigenvalue_at_boundary": float(_eig(x_sp)),
            "sign_left": int(np.sign(e_left)),
            "sign_right": int(np.sign(e_right)),
        })

    return crossings


def spinodal_temperature_binary(
    dbf: Database,
    components: list[str],
    phase_name: str,
    x_b: float,
    comp_name: str,
    P: float,
    T_lo: float,
    T_hi: float,
    dep_component: str | None = None,
    xtol: float = 0.01,
) -> float | None:
    """Find the spinodal temperature at a fixed binary composition by bisection.

    Args:
        dbf (Database): Thermodynamic database.
        components (list[str]): All system components, including 'VA'.
        phase_name (str): Name of the target phase.
        x_b (float): Mole fraction of comp_name.
        comp_name (str): Element symbol for the fixed component.
        P (float): Pressure in Pa.
        T_lo (float): Lower temperature bound in K (must be in spinodal region,
            i.e. min eigenvalue < 0).
        T_hi (float): Upper temperature bound in K (must be stable,
            i.e. min eigenvalue > 0).
        dep_component (str | None): Dependent component. Auto-detected if None.
        xtol (float): Temperature tolerance for Brent's method in K.

    Returns:
        float | None: Spinodal temperature in K, or None if no sign change
        found in [T_lo, T_hi].
    """
    from pycalphad import variables as v

    def _eig(T: float) -> float:
        conds = {v.T: T, v.P: P, v.X(comp_name): x_b}
        return _min_eig_at_conds(dbf, components, phase_name, conds, dep_component)

    e_lo = _eig(T_lo)
    e_hi = _eig(T_hi)
    if not (np.isfinite(e_lo) and np.isfinite(e_hi)):
        return None
    if np.sign(e_lo) == np.sign(e_hi):
        return None
    try:
        T_sp = brentq(
            _eig, T_lo, T_hi,
            xtol=xtol, rtol=_BRENT_RTOL, maxiter=_BRENT_MAXITER,
        )
        return float(T_sp)
    except ValueError:
        return None


def spinodal_boundary_along_path(
    dbf: Database,
    components: list[str],
    phase_name: str,
    T: float,
    P: float,
    path: list[dict[str, float]],
    dep_component: str | None = None,
    xtol: float = _BRENT_XTOL,
) -> list[dict[str, Any]]:
    """Detect spinodal crossings along an arbitrary composition path.

    The path is a sequence of composition dicts. The function scans adjacent
    pairs and refines crossings by linear interpolation followed by Brent
    bisection along the path parameter t ∈ [0, 1].

    Args:
        dbf (Database): Thermodynamic database.
        components (list[str]): All system components, including 'VA'.
        phase_name (str): Name of the target phase.
        T (float): Temperature in K.
        P (float): Pressure in Pa.
        path (list[dict[str, float]]): Sequence of composition dicts, each
            mapping component name to mole fraction.
        dep_component (str | None): Dependent component. Auto-detected if None.
        xtol (float): Path-parameter tolerance for Brent's method.

    Returns:
        list[dict]: One dict per boundary crossing found. Each dict contains:
            - ``t_boundary`` (float): Path parameter ∈ [0, 1] at crossing.
            - ``composition`` (dict): Interpolated composition at crossing.
            - ``eigenvalue_at_boundary`` (float): Min eigenvalue (≈ 0).
            - ``path_index_left`` (int): Index of path point just left.
            - ``path_index_right`` (int): Index of path point just right.
    """
    from pycalphad import variables as v

    path_len = len(path)
    eig_vals = []

    for comp_dict in path:
        conds = {v.T: T, v.P: P}
        conds.update({v.X(c): f for c, f in comp_dict.items()})
        eig_vals.append(_min_eig_at_conds(dbf, components, phase_name, conds, dep_component))

    crossings = []

    for k in range(path_len - 1):
        e_left = eig_vals[k]
        e_right = eig_vals[k + 1]
        if not (np.isfinite(e_left) and np.isfinite(e_right)):
            continue
        if np.sign(e_left) == np.sign(e_right):
            continue

        c_left = path[k]
        c_right = path[k + 1]
        keys = list(c_left.keys())

        def _eig_on_segment(t: float) -> float:
            interp = {key: (1 - t) * c_left[key] + t * c_right[key] for key in keys}
            conds = {v.T: T, v.P: P}
            conds.update({v.X(c): f for c, f in interp.items()})
            return _min_eig_at_conds(dbf, components, phase_name, conds, dep_component)

        try:
            t_sp = brentq(
                _eig_on_segment, 0.0, 1.0,
                xtol=xtol, rtol=_BRENT_RTOL, maxiter=_BRENT_MAXITER,
            )
        except ValueError:
            continue

        comp_sp = {key: (1 - t_sp) * c_left[key] + t_sp * c_right[key] for key in keys}
        crossings.append({
            "t_boundary": float(t_sp),
            "composition": comp_sp,
            "eigenvalue_at_boundary": float(_eig_on_segment(t_sp)),
            "path_index_left": k,
            "path_index_right": k + 1,
        })

    return crossings
