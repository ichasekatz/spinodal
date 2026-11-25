"""Narrow custom exceptions for the spinodal package."""

from __future__ import annotations


class EquilibriumConvergenceError(RuntimeError):
    """Raised when pycalphad's internal equilibrium solver fails to converge."""


class DependentComponentError(ValueError):
    """Raised when the dependent component cannot be determined from conditions."""


class ComponentIndexError(KeyError):
    """Raised when a component name cannot be mapped to its global index in the spec."""


class SpinodallHessianError(RuntimeError):
    """Raised when the chemical Hessian computation returns an infeasible solution."""
