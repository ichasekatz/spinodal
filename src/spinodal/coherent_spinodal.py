"""Coherent Hessian for spinodal analysis with elastic contributions.

The coherent Hessian adds elastic energy penalties due to coherency strains to
the chemical Hessian (Kunselman 2026, Ch. 6, Eq. 6.12):

    H_coh[a, b] = H_chem[a, b] + 2 * eta_a * eta_b * E * V_m / (1 - nu)

where eta_i = (1/a) * da/dx_i is the linear strain coefficient for component i,
E is the isotropic elastic modulus (Pa), V_m is the molar volume (m³/mol), and
nu is the Poisson ratio.

This correction penalizes composition fluctuations involving components with
large atomic-radius mismatch, effectively shrinking the spinodal region relative
to the purely chemical prediction.

References:
    Cahn, J.W. (1961). On spinodal decomposition. Acta Metall. 9, 795–801.
    Kunselman (2026), Ch. 6, Eqs. 6.10–6.15.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .eigenanalysis import gauge_invariant_eig, min_eigenvalue, n_negative_eigenvalues
from .hessian import chemical_hessian
from ._exceptions import DependentComponentError

if TYPE_CHECKING:
    from pycalphad import Database


def elastic_correction_matrix(
    indep_components: list[str],
    dep_component: str,
    eta: dict[str, float],
    E: float,
    V_m: float,
    nu: float,
) -> np.ndarray:
    r"""Build the elastic correction matrix C for the coherent Hessian.

    For independent components a, b (excluding dep_component):

        C[a, b] = 2 * eta_a * eta_b * E * V_m / (1 - nu)

    Args:
        indep_components (list[str]): Names of independent components (rows/cols
            of the chemical Hessian), in the same order as H.
        dep_component (str): Name of the dependent component.
        eta (dict[str, float]): Linear strain coefficient eta_i = (1/a)(da/dx_i)
            for each component i (must include dep_component). Units: dimensionless.
        E (float): Isotropic elastic modulus in Pa.
        V_m (float): Molar volume in m³/mol.
        nu (float): Poisson ratio (dimensionless).

    Returns:
        np.ndarray: (n, n) elastic correction matrix C in J/mol.

    Raises:
        DependentComponentError: If a component in indep_components is missing
            from the eta dict.
    """
    n = len(indep_components)
    prefactor = 2.0 * E * V_m / (1.0 - nu)

    for comp in indep_components:
        if comp.upper() not in {k.upper() for k in eta}:
            raise DependentComponentError(
                f"eta missing for component '{comp}'. "
                f"Available: {list(eta)}"
            )

    eta_vec = np.array(
        [eta[next(k for k in eta if k.upper() == comp.upper())]
         for comp in indep_components],
        dtype=float,
    )
    C = prefactor * np.outer(eta_vec, eta_vec)
    return C


def coherent_hessian(
    dbf: Database,
    components: list[str],
    phase_name: str,
    conds: dict,
    eta: dict[str, float],
    E: float,
    V_m: float,
    nu: float,
    dep_component: str | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], str]:
    r"""Compute the coherent Gibbs energy Hessian.

    Adds an elastic correction to the chemical Hessian:

        H_coh[a, b] = H_chem[a, b] + 2 * eta_a * eta_b * E * V_m / (1 - nu)

    Args:
        dbf (Database): Thermodynamic database.
        components (list[str]): All system components, including 'VA'.
        phase_name (str): Name of the target phase.
        conds (dict): Equilibrium conditions (T, P, and all independent X conditions).
        eta (dict[str, float]): Linear strain coefficients for all components.
            Keys are element names (case-insensitive). Units: dimensionless.
        E (float): Isotropic elastic modulus in Pa.
        V_m (float): Molar volume in m³/mol.
        nu (float): Poisson ratio.
        dep_component (str | None): Dependent component. Auto-detected if None.

    Returns:
        H_chem (np.ndarray): (n, n) chemical Hessian in J/mol.
        H_coh (np.ndarray): (n, n) coherent Hessian in J/mol.
        indep_components (list[str]): Component names for rows/columns.
        dep_component (str): Dependent component used.
    """
    H_chem, indep_components, dep_component = chemical_hessian(
        dbf, components, phase_name, conds, dep_component
    )
    C = elastic_correction_matrix(indep_components, dep_component, eta, E, V_m, nu)
    H_coh = H_chem + C
    return H_chem, H_coh, indep_components, dep_component


def coherent_stability_at_point(
    dbf: Database,
    components: list[str],
    phase_name: str,
    T: float,
    P: float,
    composition_dict: dict[str, float],
    eta: dict[str, float],
    E: float,
    V_m: float,
    nu: float,
    dep_component: str | None = None,
) -> dict:
    """Compute chemical and coherent spinodal stability at a single point.

    Args:
        dbf (Database): Thermodynamic database.
        components (list[str]): All system components, including 'VA'.
        phase_name (str): Name of the target phase.
        T (float): Temperature in K.
        P (float): Pressure in Pa.
        composition_dict (dict[str, float]): Mole fractions for independent
            components.
        eta (dict[str, float]): Linear strain coefficients for all components.
        E (float): Elastic modulus in Pa.
        V_m (float): Molar volume in m³/mol.
        nu (float): Poisson ratio.
        dep_component (str | None): Dependent component. Auto-detected if None.

    Returns:
        dict containing:
            - ``H_chem`` (np.ndarray): Chemical Hessian in J/mol.
            - ``H_coh`` (np.ndarray): Coherent Hessian in J/mol.
            - ``min_eig_chem`` (float): Minimum chemical eigenvalue in J/mol.
            - ``min_eig_coh`` (float): Minimum coherent eigenvalue in J/mol.
            - ``n_neg_chem`` (int): Negative chemical eigenvalues.
            - ``n_neg_coh`` (int): Negative coherent eigenvalues.
            - ``stable_chem`` (bool): Chemical stability.
            - ``stable_coh`` (bool): Coherent stability.
            - ``indep_components`` (list[str]): Component order for H.
            - ``converged`` (bool): True if equilibrium converged.
            - ``error`` (str | None): Error message on failure.
    """
    from pycalphad import variables as v
    from ._exceptions import EquilibriumConvergenceError, SpinodallHessianError

    conds = {v.T: T, v.P: P}
    conds.update({v.X(comp): frac for comp, frac in composition_dict.items()})

    try:
        H_chem, H_coh, indep_comps, dep = coherent_hessian(
            dbf, components, phase_name, conds, eta, E, V_m, nu, dep_component
        )
        min_ec = min_eigenvalue(H_chem)
        min_ecoh = min_eigenvalue(H_coh)
        n_neg_c = n_negative_eigenvalues(H_chem)
        n_neg_coh = n_negative_eigenvalues(H_coh)
        return {
            "H_chem": H_chem,
            "H_coh": H_coh,
            "min_eig_chem": min_ec,
            "min_eig_coh": min_ecoh,
            "n_neg_chem": n_neg_c,
            "n_neg_coh": n_neg_coh,
            "stable_chem": n_neg_c == 0,
            "stable_coh": n_neg_coh == 0,
            "indep_components": indep_comps,
            "dep_component": dep,
            "converged": True,
            "error": None,
        }
    except (EquilibriumConvergenceError, SpinodallHessianError, DependentComponentError) as exc:
        import numpy as _np
        return {
            "H_chem": None,
            "H_coh": None,
            "min_eig_chem": _np.nan,
            "min_eig_coh": _np.nan,
            "n_neg_chem": -1,
            "n_neg_coh": -1,
            "stable_chem": None,
            "stable_coh": None,
            "indep_components": None,
            "dep_component": None,
            "converged": False,
            "error": str(exc),
        }
