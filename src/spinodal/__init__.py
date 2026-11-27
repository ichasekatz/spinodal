"""Spinodal decomposition analysis via Jansson derivatives.

This package computes spinodal stability maps from CALPHAD Gibbs energy models
using the analytic Jansson derivative approach. The chemical Hessian is computed
via ``fixed_component_differential`` (one Jansson solve per independent composition
direction), and gauge-invariant eigenvalues are obtained by solving the generalized
eigenvalue problem H v = lambda G v in composition simplex space.

Modules:
    hessian: Chemical Hessian via fixed_component_differential.
    eigenanalysis: Gauge-invariant eigenanalysis of the Hessian.
    stability_map: Grid scan for spinodal stability maps.

References:
    Kunselman, C.J. (2026). Applications of Analytic Derivatives at
    Equilibrium for CALPHAD Models. PhD Dissertation, Texas A&M University.
"""

from .hessian import chemical_hessian
from .eigenanalysis import (
    composition_metric,
    gauge_invariant_eig,
    min_eigenvalue,
    n_negative_eigenvalues,
)
from .stability_map import stability_at_point, stability_map, binary_spinodal_scan
from .boundary_detection import (
    spinodal_boundary_binary,
    spinodal_temperature_binary,
    spinodal_boundary_along_path,
)
from .spinodal_newton import (
    spinodal_at_temperature,
    spinodal_curve_newton,
)
from .coherent_spinodal import (
    elastic_correction_matrix,
    coherent_hessian,
    coherent_stability_at_point,
)
from ._exceptions import (
    EquilibriumConvergenceError,
    DependentComponentError,
    ComponentIndexError,
    SpinodallHessianError,
)

__all__ = [
    # hessian
    "chemical_hessian",
    # eigenanalysis
    "composition_metric",
    "gauge_invariant_eig",
    "min_eigenvalue",
    "n_negative_eigenvalues",
    # stability map
    "stability_at_point",
    "stability_map",
    "binary_spinodal_scan",
    # Newton-Raphson spinodal solver
    "spinodal_at_temperature",
    "spinodal_curve_newton",
    # boundary detection
    "spinodal_boundary_binary",
    "spinodal_temperature_binary",
    "spinodal_boundary_along_path",
    # coherent spinodal
    "elastic_correction_matrix",
    "coherent_hessian",
    "coherent_stability_at_point",
    # exceptions
    "EquilibriumConvergenceError",
    "DependentComponentError",
    "ComponentIndexError",
    "SpinodallHessianError",
]
