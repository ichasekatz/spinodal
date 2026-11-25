"""Stability map computation over a composition grid.

Evaluates the chemical Hessian and gauge-invariant eigenanalysis at each point
in a user-supplied composition grid to produce a spinodal stability map.

Usage example (binary system)::

    from spinodal.stability_map import binary_spinodal_scan
    results = binary_spinodal_scan(dbf, components, phase_name, T, P,
                                   comp_B_values, comp_B_name='FE')
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from pycalphad import variables as v

from .eigenanalysis import gauge_invariant_eig, min_eigenvalue, n_negative_eigenvalues
from .hessian import chemical_hessian
from ._exceptions import (
    EquilibriumConvergenceError,
    SpinodallHessianError,
)

if TYPE_CHECKING:
    from pycalphad import Database


def stability_at_point(
    dbf: Database,
    components: list[str],
    phase_name: str,
    T: float,
    P: float,
    composition_dict: dict[str, float],
    dep_component: str | None = None,
) -> dict[str, Any]:
    """Compute spinodal stability at a single composition point.

    Args:
        dbf (Database): Thermodynamic database.
        components (list[str]): All system components, including 'VA'.
        phase_name (str): Name of the target phase.
        T (float): Temperature in K.
        P (float): Pressure in Pa.
        composition_dict (dict[str, float]): Mole fractions for each independent
            component, e.g. {'FE': 0.3} for a binary system.
        dep_component (str | None): Dependent component name. Auto-detected if None.

    Returns:
        dict containing:
            - ``H`` (np.ndarray | None): Chemical Hessian in J/mol.
            - ``eigenvalues`` (np.ndarray | None): Gauge-invariant eigenvalues in J/mol.
            - ``eigenvectors`` (np.ndarray | None): Corresponding eigenvectors.
            - ``min_eigenvalue`` (float): Smallest eigenvalue in J/mol (NaN on failure).
            - ``n_negative`` (int): Number of negative eigenvalues (-1 on failure).
            - ``stable`` (bool | None): True if no negative eigenvalues.
            - ``converged`` (bool): True if the equilibrium calculation succeeded.
            - ``error`` (str | None): Error message if convergence failed.
    """
    conds = {v.T: T, v.P: P}
    conds.update({v.X(comp): frac for comp, frac in composition_dict.items()})

    try:
        H, indep_comps, dep = chemical_hessian(
            dbf, components, phase_name, conds, dep_component=dep_component
        )
        evals, evecs = gauge_invariant_eig(H)
        n_neg = n_negative_eigenvalues(H)
        return {
            "H": H,
            "eigenvalues": evals,
            "eigenvectors": evecs,
            "min_eigenvalue": float(evals[0]),
            "n_negative": n_neg,
            "stable": n_neg == 0,
            "indep_components": indep_comps,
            "dep_component": dep,
            "converged": True,
            "error": None,
        }
    except (EquilibriumConvergenceError, SpinodallHessianError) as exc:
        return {
            "H": None,
            "eigenvalues": None,
            "eigenvectors": None,
            "min_eigenvalue": np.nan,
            "n_negative": -1,
            "stable": None,
            "indep_components": None,
            "dep_component": None,
            "converged": False,
            "error": str(exc),
        }


def stability_map(
    dbf: Database,
    components: list[str],
    phase_name: str,
    T: float,
    P: float,
    composition_grid: list[dict[str, float]],
    dep_component: str | None = None,
) -> list[dict[str, Any]]:
    """Compute a spinodal stability map over a composition grid at fixed T and P.

    Args:
        dbf (Database): Thermodynamic database.
        components (list[str]): All system components, including 'VA'.
        phase_name (str): Name of the target phase.
        T (float): Temperature in K.
        P (float): Pressure in Pa.
        composition_grid (list[dict[str, float]]): Each dict maps component name
            to mole fraction for the independent components at one grid point.
        dep_component (str | None): Dependent component name. Auto-detected if None.

    Returns:
        list[dict]: One dict per grid point. Each dict contains the composition
            dict merged with the output of ``stability_at_point``.
    """
    results = []
    for comp_dict in composition_grid:
        result = stability_at_point(
            dbf, components, phase_name, T, P, comp_dict,
            dep_component=dep_component,
        )
        result.update(comp_dict)
        results.append(result)
    return results


def binary_spinodal_scan(
    dbf: Database,
    components: list[str],
    phase_name: str,
    T: float,
    P: float,
    comp_values: np.ndarray,
    comp_name: str,
    dep_component: str | None = None,
) -> list[dict[str, Any]]:
    """Convenience wrapper for a binary composition scan at fixed T.

    Args:
        dbf (Database): Thermodynamic database.
        components (list[str]): All system components, including 'VA'.
        phase_name (str): Name of the target phase.
        T (float): Temperature in K.
        P (float): Pressure in Pa.
        comp_values (np.ndarray): 1-D array of mole fractions to scan for
            comp_name.
        comp_name (str): Element symbol for the scanned component.
        dep_component (str | None): Dependent component name. Auto-detected if None.

    Returns:
        list[dict]: Stability results for each composition in comp_values.
    """
    grid = [{comp_name: float(x)} for x in comp_values]
    return stability_map(
        dbf, components, phase_name, T, P, grid,
        dep_component=dep_component,
    )
