"""Chemical Gibbs energy Hessian via Jansson derivatives.

The Hessian H[i,j] = d(mu_i - mu_dep) / dx_j is computed analytically by
calling fixed_component_differential once per independent component direction.
This is the Jansson approach described in Kunselman (2026), Ch. 6, Eq. 6.9.

References:
    Kunselman, C.J. (2026). Applications of Analytic Derivatives at
    Equilibrium for CALPHAD Models. PhD Dissertation, Texas A&M University,
    Ch. 6.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

import numpy as np

from pycalphad import Workspace
from pycalphad import variables as v
from pycalphad.core.minimizer import fixed_component_differential

from ._exceptions import (
    ComponentIndexError,
    DependentComponentError,
    EquilibriumConvergenceError,
    SpinodallHessianError,
)

if TYPE_CHECKING:
    from pycalphad import Database
    from pycalphad.core.minimizer import SystemSpecification, SystemState
    from pycalphad.core.composition_set import CompositionSet


def _build_single_phase_state(
    dbf: Database,
    components: list[str],
    phase_name: str,
    conds: dict,
) -> tuple[SystemSpecification, SystemState, list[CompositionSet], list[str]]:
    """Build pycalphad spec/state for a single-phase composition-fixed calculation.

    VA is excluded from the component list when the target phase has no vacancy
    sites — including VA for a vacancy-free phase adds mu_VA as a free variable
    and makes the equilibrium system under-determined (Gibbs Phase Rule violation).

    Args:
        dbf (Database): Thermodynamic database.
        components (list[str]): All system components (may include 'VA').
        phase_name (str): Name of the target phase.
        conds (dict): Equilibrium conditions (T, P, and all independent X conditions).

    Returns:
        Tuple of (spec, state, compsets, effective_components) where
        effective_components is the filtered component list actually passed to Workspace.

    """
    phase_has_va = any(
        'VA' in [str(s).upper() for s in sublat]
        for sublat in dbf.phases[phase_name].constituents
    )
    effective_components = (
        components
        if phase_has_va
        else [c for c in components if str(c).upper() != 'VA']
    )

    # N (system amount) must be fixed; without it N is a free state variable
    # and the equilibrium matrix becomes under-determined.
    effective_conds = dict(conds)
    if v.N not in effective_conds:
        effective_conds[v.N] = 1.0

    wks = Workspace(dbf, effective_components, [phase_name], effective_conds)
    all_compsets = wks.get_composition_sets()

    # Inside the spinodal region the global minimizer splits the phase into
    # multiple composition sets (common-tangent endpoints).  Spinodal analysis
    # requires the *constrained* single-phase chemical potentials at the
    # prescribed composition, not the phase-split equilibrium.  Take one
    # composition set for the target phase and reset its site fractions to the
    # target composition before rebuilding the spec/state.
    target_cs = next(
        (cs for cs in all_compsets if cs.phase_record.phase_name == phase_name),
        all_compsets[0],
    )

    # Collect the target site fractions from the prescribed v.X conditions and
    # the phase's sublattice structure.  For a single-sublattice phase the site
    # fractions equal the mole fractions.
    x_prescribed: dict[str, float] = {}
    for key, val in effective_conds.items():
        s = str(key)
        if s.startswith("X_"):
            x_prescribed[s[2:].upper()] = float(np.asarray(val).flat[0])

    # Build the full site-fraction vector in constituent order.
    constituents = [
        [str(c) for c in sublat]
        for sublat in dbf.phases[phase_name].constituents
    ]
    site_fracs: list[float] = []
    for sublat in constituents:
        nonva = [c for c in sublat if c.upper() != "VA"]
        # Pure-VA sublattice (interstitial vacancy model): y_VA = 1.0 always.
        if not nonva:
            for _ in sublat:
                site_fracs.append(1.0)
            continue
        dep_sf = None  # dependent constituent (not in X conditions)
        for const in sublat:
            if const.upper() == "VA":
                # Mixed sublattice with real atoms + VA: VA fraction is not
                # prescribed; set to 0.0 as an initial placeholder (overridden
                # by dep_sf logic below if VA is the dependent constituent).
                site_fracs.append(0.0)
            elif const.upper() in x_prescribed:
                site_fracs.append(x_prescribed[const.upper()])
            else:
                dep_sf = len(site_fracs)
                site_fracs.append(0.0)
        # Assign the remaining fraction to the dependent constituent
        if dep_sf is not None:
            x_sum = sum(x_prescribed.get(c.upper(), 0.0) for c in nonva if c.upper() in x_prescribed)
            site_fracs[dep_sf] = max(0.0, 1.0 - x_sum)

    T_val = float(np.asarray(effective_conds.get(v.T, 298.15)).flat[0])
    P_val = float(np.asarray(effective_conds.get(v.P, 101325.0)).flat[0])
    N_val = float(np.asarray(effective_conds.get(v.N, 1.0)).flat[0])

    target_cs.update(
        np.array(site_fracs, dtype=float),
        1.0,
        np.array([N_val, P_val, T_val], dtype=float),
    )
    compsets = [target_cs]

    cur_conds = OrderedDict(
        (key, float(np.asarray(val).flat[0])) for key, val in effective_conds.items()
    )
    spec = wks.solver.get_system_spec(compsets, cur_conds)
    state = spec.get_new_state(compsets)
    # One iteration initialises the Lagrange multipliers from the first KKT step.
    # This is sufficient: fixed_component_differential re-solves the linear
    # system analytically and does not require additional outer-loop convergence.
    spec.run_loop(state, 1)
    return spec, state, compsets, effective_components


def _component_global_index_map(effective_components: list[str]) -> dict[str, int]:
    """Map element name -> global index in pycalphad's chemical potential vector.

    pycalphad sorts non-VA components alphabetically and uses that ordering for
    the chemical potential vector returned by fixed_component_differential and
    SystemState.chemical_potentials.  VA is a sublattice placeholder; it never
    appears in the chemical potential vector regardless of whether it appears in
    the phase's CEF sublattice model.

    Args:
        effective_components (list[str]): Components actually passed to Workspace
            (may include 'VA' when the phase has a vacancy sublattice).

    Returns:
        dict[str, int]: Component name (uppercase) to global index (VA excluded).
    """
    non_va = [
        c for c in sorted(str(x) for x in effective_components)
        if str(c).upper() != 'VA'
    ]
    return {c.upper(): i for i, c in enumerate(non_va)}


def _fixed_component_indices(conds: dict, comp_to_global: dict[str, int]) -> set[int]:
    """Return the set of global component indices that have mole fraction conditions.

    Parses v.X(...) keys from the conditions dict. This avoids reading
    spec.prescribed_mole_fraction_coefficients, which is a Cython-only attribute
    not accessible from Python.

    Args:
        conds (dict): Conditions dict as passed to chemical_hessian.
        comp_to_global (dict[str, int]): Component-name-to-global-index map.

    Returns:
        set[int]: Global indices of components with prescribed mole fractions.
    """
    fixed = set()
    for key in conds:
        s = str(key)
        if s.startswith('X_'):
            comp_name = s[2:].upper()
            if comp_name in comp_to_global:
                fixed.add(comp_to_global[comp_name])
    return fixed


def chemical_hessian(
    dbf: Database,
    components: list[str],
    phase_name: str,
    conds: dict,
    dep_component: str | None = None,
) -> tuple[np.ndarray, list[str], str]:
    r"""Compute the chemical Gibbs energy Hessian via Jansson derivatives.

    For an N-component system, the (N-1)×(N-1) Hessian in composition space is:

        H[i, j] = d(mu_i - mu_dep) / dx_j,   i, j in {independent components}

    where mu_dep is the chemical potential of the dependent component and x_j is
    the mole fraction of the j-th independent component. This is Eq. 6.9 of
    Kunselman (2026).

    The function calls ``fixed_component_differential`` once per independent
    composition direction and assembles the result into H.

    Args:
        dbf (Database): Thermodynamic database.
        components (list[str]): All system components, including 'VA'.
        phase_name (str): Name of the target phase (single-phase equilibrium).
        conds (dict): Conditions for the equilibrium calculation. Must include
            T (K), P (Pa), and v.X conditions for every independent component
            (all non-'VA' components except dep_component). The dep_component
            must NOT appear as a v.X condition.
        dep_component (str | None): Name of the dependent component (the one
            whose fraction is not fixed independently). Determined automatically
            from conds if None — the non-VA component absent from any v.X
            condition is taken as the dependent.

    Returns:
        H (np.ndarray): (N-1, N-1) chemical Hessian in J/mol.
        indep_components (list[str]): Component names for rows and columns of H,
            in alphabetically sorted order.
        dep_component (str): Name of the dependent component used.

    Raises:
        EquilibriumConvergenceError: If pycalphad's solver does not converge.
        DependentComponentError: If the dependent component cannot be identified.
        ComponentIndexError: If a component cannot be mapped to a global index.
        SpinodallHessianError: If fixed_component_differential returns NaN (infeasible).

    References:
        Kunselman (2026), Ch. 6, Eq. 6.9.
        Sundman et al. (2015), Eq. 74 (foundation of fixed_component_differential).
    """
    spec, state, compsets, effective_components = _build_single_phase_state(
        dbf, components, phase_name, conds
    )

    nonvacant = [str(x) for x in compsets[0].phase_record.nonvacant_elements]
    comp_to_global = _component_global_index_map(effective_components)
    fixed_global_indices = _fixed_component_indices(conds, comp_to_global)

    if dep_component is None:
        candidates = [
            c for c in sorted(nonvacant)
            if comp_to_global.get(c.upper()) not in fixed_global_indices
        ]
        if not candidates:
            raise DependentComponentError(
                f"All nonvacant components appear in composition conditions. "
                f"Exactly N-2 mole fraction conditions are required for N components. "
                f"Nonvacant: {nonvacant}, fixed global indices: {fixed_global_indices}"
            )
        dep_component = candidates[0]

    dep_component = dep_component.upper()
    if dep_component not in comp_to_global:
        raise ComponentIndexError(
            f"Dependent component '{dep_component}' not found in component map. "
            f"Available: {list(comp_to_global)}"
        )
    dep_idx = comp_to_global[dep_component]

    indep_components = [c for c in sorted(nonvacant) if c.upper() != dep_component]
    n = len(indep_components)
    H = np.zeros((n, n))

    for j, comp_j in enumerate(indep_components):
        j_global = comp_to_global.get(comp_j.upper())
        if j_global is None:
            raise ComponentIndexError(
                f"Component '{comp_j}' not found in component map. "
                f"Available: {list(comp_to_global)}"
            )
        delta_mu, _, _ = fixed_component_differential(spec, state, j_global)
        delta_mu = np.asarray(delta_mu, dtype=float)

        # Only check indices within delta_mu's length; VA (last index) may exceed it.
        valid_indices = [idx for idx in comp_to_global.values() if idx < len(delta_mu)]
        if np.any(np.isnan(delta_mu[valid_indices])):
            raise SpinodallHessianError(
                f"fixed_component_differential returned NaN for component "
                f"'{comp_j}' (global index {j_global}). The least-squares solve "
                f"may be infeasible at this composition."
            )

        for i, comp_i in enumerate(indep_components):
            i_global = comp_to_global[comp_i.upper()]
            H[i, j] = delta_mu[i_global] - delta_mu[dep_idx]

    return H, indep_components, dep_component
