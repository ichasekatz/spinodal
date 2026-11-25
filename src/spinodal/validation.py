"""Validation utilities: Jansson vs finite-difference chemical Hessian.

Compares the Jansson-derivative Hessian from ``chemical_hessian`` against a
finite-difference approximation built by perturbing each composition condition
individually and computing dµ via forward/central differences.

Useful for sanity-checking new databases or pycalphad builds.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .hessian import chemical_hessian, _build_single_phase_state, _component_global_index_map

if TYPE_CHECKING:
    from pycalphad import Database


def _mu_at_conds(
    dbf: Database,
    components: list[str],
    phase_name: str,
    conds: dict,
) -> np.ndarray:
    """Return the chemical potential vector at the given conditions.

    Args:
        dbf (Database): Thermodynamic database.
        components (list[str]): System components.
        phase_name (str): Target phase.
        conds (dict): Equilibrium conditions.

    Returns:
        np.ndarray: Chemical potential vector (one entry per sorted component).
    """
    spec, state, _, eff_comps = _build_single_phase_state(dbf, components, phase_name, conds)
    comp_to_global = _component_global_index_map(eff_comps)
    n = len(comp_to_global)
    mu = np.array([state.chemical_potentials[i] for i in range(n)], dtype=float)
    return mu


def finite_difference_hessian(
    dbf: Database,
    components: list[str],
    phase_name: str,
    conds: dict,
    dep_component: str | None = None,
    h: float = 1e-4,
    scheme: str = "central",
) -> tuple[np.ndarray, list[str], str]:
    r"""Compute the chemical Hessian via finite differences.

    For each independent component j, perturbs x_j by ±h and computes:

        H[i, j] ≈ [dµ_i(+h) - dµ_i(-h)] / (2h)   (central)
        H[i, j] ≈ [dµ_i(+h) - dµ_i(0)] / h         (forward)

    where dµ_i = µ_i - µ_dep. This is a numerical reference for validating
    the analytic Jansson Hessian from ``chemical_hessian``.

    Args:
        dbf (Database): Thermodynamic database.
        components (list[str]): All system components, including 'VA'.
        phase_name (str): Name of the target phase.
        conds (dict): Equilibrium conditions (T, P, and all independent X conditions).
        dep_component (str | None): Dependent component. Auto-detected if None.
        h (float): Finite-difference step size in mole-fraction units.
        scheme (str): ``'central'`` (default) or ``'forward'``.

    Returns:
        H_fd (np.ndarray): (n, n) finite-difference Hessian in J/mol.
        indep_components (list[str]): Component names for rows and columns.
        dep_component (str): Dependent component used.
    """
    from pycalphad import variables as v
    from collections import OrderedDict

    H_jan, indep_components, dep_component = chemical_hessian(
        dbf, components, phase_name, conds, dep_component
    )
    n = len(indep_components)

    _, _, _, eff_comps = _build_single_phase_state(dbf, components, phase_name, conds)
    comp_to_global = _component_global_index_map(eff_comps)
    dep_idx = comp_to_global[dep_component.upper()]

    def _dmu(x_key: object, x_val: float) -> np.ndarray:
        c = dict(conds)
        c[x_key] = x_val
        _, state, _, _ = _build_single_phase_state(dbf, components, phase_name, c)
        mu = np.array([state.chemical_potentials[i] for i in range(len(comp_to_global))])
        return mu

    x_keys = {comp: v.X(comp) for comp in indep_components}
    x_base = {comp: float(conds[x_keys[comp]]) for comp in indep_components}

    H_fd = np.zeros((n, n))

    for j, comp_j in enumerate(indep_components):
        x_key = x_keys[comp_j]
        x0 = x_base[comp_j]

        if scheme == "central":
            mu_plus = _dmu(x_key, x0 + h)
            mu_minus = _dmu(x_key, x0 - h)
            dmu = (mu_plus - mu_minus) / (2.0 * h)
        else:
            mu_plus = _dmu(x_key, x0 + h)
            mu_0 = _dmu(x_key, x0)
            dmu = (mu_plus - mu_0) / h

        for i, comp_i in enumerate(indep_components):
            i_global = comp_to_global[comp_i.upper()]
            H_fd[i, j] = dmu[i_global] - dmu[dep_idx]

    return H_fd, indep_components, dep_component


def compare_hessians(
    dbf: Database,
    components: list[str],
    phase_name: str,
    conds: dict,
    dep_component: str | None = None,
    h: float = 1e-4,
    scheme: str = "central",
) -> dict:
    """Compare Jansson and finite-difference chemical Hessians.

    Args:
        dbf (Database): Thermodynamic database.
        components (list[str]): All system components, including 'VA'.
        phase_name (str): Name of the target phase.
        conds (dict): Equilibrium conditions.
        dep_component (str | None): Dependent component. Auto-detected if None.
        h (float): Finite-difference step size in mole fractions.
        scheme (str): ``'central'`` or ``'forward'``.

    Returns:
        dict containing:
            - ``H_jansson`` (np.ndarray): Analytic Hessian (n, n).
            - ``H_fd`` (np.ndarray): Finite-difference Hessian (n, n).
            - ``abs_diff`` (np.ndarray): |H_jansson - H_fd| (n, n).
            - ``max_abs_diff`` (float): Maximum absolute difference in J/mol.
            - ``rel_diff`` (np.ndarray): |diff| / max(|H_jansson|, 1) (n, n).
            - ``max_rel_diff`` (float): Maximum relative difference.
            - ``indep_components`` (list[str]): Component names.
            - ``dep_component`` (str): Dependent component used.
    """
    H_jan, indep_components, dep_component = chemical_hessian(
        dbf, components, phase_name, conds, dep_component
    )
    H_fd, _, _ = finite_difference_hessian(
        dbf, components, phase_name, conds, dep_component, h=h, scheme=scheme
    )
    abs_diff = np.abs(H_jan - H_fd)
    denom = np.maximum(np.abs(H_jan), 1.0)
    rel_diff = abs_diff / denom

    return {
        "H_jansson": H_jan,
        "H_fd": H_fd,
        "abs_diff": abs_diff,
        "max_abs_diff": float(np.max(abs_diff)),
        "rel_diff": rel_diff,
        "max_rel_diff": float(np.max(rel_diff)),
        "indep_components": indep_components,
        "dep_component": dep_component,
    }
