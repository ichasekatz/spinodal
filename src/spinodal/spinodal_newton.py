r"""Analytic spinodal boundary tracing via Newton-Raphson.

Locates the spinodal boundary directly by solving H[0,0](x) = d²G/dx² = 0
using Newton-Raphson, with the third derivative d³G/dx³ as the Jacobian.

The Newton step is:

    x_{n+1} = x_n - H[0,0](x_n) / (d³G/dx³)(x_n)

H[0,0] is evaluated analytically via the Jansson derivative
(``chemical_hessian``). The third derivative is estimated by central finite
difference on H[0,0]:

    d³G/dx³ ≈ (H(x + ε) − H(x − ε)) / (2ε)

This is exact to O(ε²) and uses only the already-implemented Hessian calls.

The main advantage over grid-based methods (``spinodal_boundary_binary``) is
that the boundary is found directly — no grid scan or contour interpolation —
and the warm-start (using the previous temperature's solution as the initial
guess) reduces each temperature point to 1–3 Newton iterations.

References:
    Kunselman, C.J. (2026). Applications of Analytic Derivatives at
    Equilibrium for CALPHAD Models. PhD Dissertation, Texas A&M University.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pycalphad import variables as v

from ._exceptions import EquilibriumConvergenceError, SpinodallHessianError
from .hessian import chemical_hessian

if TYPE_CHECKING:
    from pycalphad import Database

_fd_eps_default = 1e-4   # central-difference step for d³G/dx³, in mole fraction


def _h_binary_scalar(
    dbf: Database,
    components: list[str],
    phase_name: str,
    comp_name: str,
    x: float,
    T: float,
    P: float,
    dep_component: str | None,
) -> float:
    """Return H[0,0] = d²G/dx² for a binary at composition x.

    Args:
        dbf (Database): Thermodynamic database.
        components (list[str]): All system components, including 'VA'.
        phase_name (str): Name of the target phase.
        comp_name (str): Element symbol of the independent component.
        x (float): Mole fraction of comp_name.
        T (float): Temperature in K.
        P (float): Pressure in Pa.
        dep_component (str | None): Dependent component; auto-detected if None.

    Returns:
        H[0,0] in J/mol, or NaN on solver failure.
    """
    conds = {v.T: T, v.P: P, v.X(comp_name): x}
    try:
        H, _, _ = chemical_hessian(dbf, components, phase_name, conds, dep_component)
        return float(H[0, 0])
    except (EquilibriumConvergenceError, SpinodallHessianError, IndexError, ValueError):
        return float("nan")


def spinodal_at_temperature(
    dbf: Database,
    components: list[str],
    phase_name: str,
    comp_name: str,
    T: float,
    P: float,
    x_init: float,
    dep_component: str | None = None,
    fd_eps: float = _fd_eps_default,
    tol: float = 1e-9,
    max_iter: int = 20,
    x_bounds: tuple[float, float] = (0.001, 0.999),
) -> float | None:
    """Newton-Raphson solver for the spinodal composition at fixed temperature.

    Solves H[0,0](x) = d²G/dx² = 0 using the Newton update:
        x ← x - H[0,0](x) / (d³G/dx³)(x)

    The third derivative is estimated by central finite difference on H[0,0].

    Args:
        dbf (Database): Thermodynamic database.
        components (list[str]): All system components, including 'VA'.
        phase_name (str): Name of the target phase.
        comp_name (str): Element symbol of the independent component.
        T (float): Temperature in K.
        P (float): Pressure in Pa.
        x_init (float): Initial composition guess for Newton iteration.
        dep_component (str | None): Dependent component; auto-detected if None.
        fd_eps (float): Central-difference step for d³G/dx³ in mole fraction.
            Default 1e-4 balances truncation and round-off error.
        tol (float): Convergence tolerance on |dx| and |H[0,0]| in J/mol.
        max_iter (int): Maximum Newton iterations before giving up.
        x_bounds (tuple[float, float]): Clamp x to this range after each step.

    Returns:
        Spinodal composition (mole fraction) at temperature T, or None if the
        solver did not converge.
    """
    x = float(x_init)
    x_lo, x_hi = x_bounds

    for _ in range(max_iter):
        h = _h_binary_scalar(dbf, components, phase_name, comp_name, x, T, P, dep_component)
        if not np.isfinite(h):
            return None
        if abs(h) < tol:
            return x

        h_plus = _h_binary_scalar(
            dbf, components, phase_name, comp_name, x + fd_eps, T, P, dep_component
        )
        h_minus = _h_binary_scalar(
            dbf, components, phase_name, comp_name, x - fd_eps, T, P, dep_component
        )
        if not (np.isfinite(h_plus) and np.isfinite(h_minus)):
            return None

        third_deriv = (h_plus - h_minus) / (2.0 * fd_eps)
        if abs(third_deriv) < 1e-30:
            return None

        dx = -h / third_deriv
        x = float(np.clip(x + dx, x_lo, x_hi))

        if abs(dx) < tol:
            h_final = _h_binary_scalar(
                dbf, components, phase_name, comp_name, x, T, P, dep_component
            )
            return x if (np.isfinite(h_final) and abs(h_final) < tol * 1e6) else None

    return None


def spinodal_curve_newton(
    dbf: Database,
    components: list[str],
    phase_name: str,
    comp_name: str,
    T_arr: np.ndarray,
    P: float,
    x_init_low: float = 0.05,
    x_init_high: float = 0.95,
    dep_component: str | None = None,
    fd_eps: float = _fd_eps_default,
    tol: float = 1e-9,
    max_iter: int = 20,
) -> dict[str, np.ndarray]:
    r"""Trace both branches of the binary spinodal curve over a temperature array.

    For each temperature in T_arr, Newton-Raphson finds the spinodal
    compositions on both the low-x and high-x branches.  The previous
    temperature's solution is used as the initial guess (warm start), so the
    solver typically converges in 1–3 iterations per point.

    NaN is inserted for temperatures where the solver does not converge (e.g.
    above the critical temperature or outside the phase's stability range).

    Args:
        dbf (Database): Thermodynamic database.
        components (list[str]): All system components, including 'VA'.
        phase_name (str): Name of the target phase.
        comp_name (str): Element symbol of the independent component.
        T_arr (np.ndarray): 1-D array of temperatures in K, in ascending order.
        P (float): Pressure in Pa.
        x_init_low (float): Initial composition guess for the low-x branch.
            Should lie in the low-x spinodal region at T_arr[0].
        x_init_high (float): Initial composition guess for the high-x branch.
        dep_component (str | None): Dependent component; auto-detected if None.
        fd_eps (float): Central-difference step for d³G/dx³ in mole fraction.
        tol (float): Convergence tolerance passed to spinodal_at_temperature.
        max_iter (int): Maximum Newton iterations per temperature point.

    Returns:
        dict with keys:
            - ``T`` (np.ndarray): Temperature array in K (same as T_arr).
            - ``x_low`` (np.ndarray): Low-x spinodal branch; NaN where not found.
            - ``x_high`` (np.ndarray): High-x spinodal branch; NaN where not found.
            - ``comp_name`` (str): Element symbol of the tracked component.
    """
    T_arr = np.asarray(T_arr, dtype=float)
    n = len(T_arr)
    x_low = np.full(n, np.nan)
    x_high = np.full(n, np.nan)

    x_guess_low = float(x_init_low)
    x_guess_high = float(x_init_high)

    for i, T in enumerate(T_arr):
        result = spinodal_at_temperature(
            dbf, components, phase_name, comp_name, float(T), P,
            x_guess_low, dep_component, fd_eps, tol, max_iter,
        )
        if result is not None:
            x_low[i] = result
            x_guess_low = result  # warm start: carry solution to next T

        result = spinodal_at_temperature(
            dbf, components, phase_name, comp_name, float(T), P,
            x_guess_high, dep_component, fd_eps, tol, max_iter,
        )
        if result is not None:
            x_high[i] = result
            x_guess_high = result

    return {"T": T_arr, "x_low": x_low, "x_high": x_high, "comp_name": comp_name}
