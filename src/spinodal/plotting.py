"""Plotting utilities for spinodal stability maps.

Provides functions for visualizing:
- Binary spinodal curves (T vs x)
- Minimum eigenvalue vs composition at fixed T
- Eigenvalue maps over composition grids (ternary / affine projection)
- Hessian comparison (Jansson vs finite-difference)

All functions return Matplotlib Figure objects so the caller can further
customise or save them. Requires ``matplotlib``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import matplotlib.figure


def plot_min_eigenvalue_vs_composition(
    comp_values: np.ndarray,
    min_eigenvalues: list[float] | np.ndarray,
    comp_name: str,
    T: float,
    phase_name: str,
    *,
    ax: Any = None,
    color: str = "tab:blue",
) -> matplotlib.figure.Figure:
    """Plot minimum gauge-invariant eigenvalue vs composition at fixed T.

    Args:
        comp_values (np.ndarray): 1-D array of mole fractions.
        min_eigenvalues (list[float] | np.ndarray): Minimum eigenvalue at each
            composition (NaN for failed points).
        comp_name (str): Element symbol for the x-axis.
        T (float): Temperature in K (for annotation).
        phase_name (str): Phase name (for title).
        ax: Optional Matplotlib Axes to plot into.
        color (str): Line colour.

    Returns:
        matplotlib.figure.Figure: The figure containing the plot.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.get_figure()

    eigs = np.asarray(min_eigenvalues, dtype=float)
    ax.plot(comp_values, eigs, color=color, lw=1.5)
    ax.axhline(0.0, color="k", lw=0.8, ls="--")
    ax.set_xlabel(f"$x_{{{comp_name}}}$")
    ax.set_ylabel(r"$\lambda_1$ (J/mol)")
    ax.set_title(f"{phase_name}  —  T = {T:.0f} K")
    fig.tight_layout()
    return fig


def plot_binary_spinodal_curve(
    x_values: np.ndarray,
    T_spinodal: np.ndarray,
    comp_name: str,
    phase_name: str,
    *,
    T_analytical: np.ndarray | None = None,
    ax: Any = None,
) -> matplotlib.figure.Figure:
    """Plot the binary spinodal curve T_sp(x).

    Args:
        x_values (np.ndarray): 1-D array of mole fractions.
        T_spinodal (np.ndarray): Spinodal temperatures in K (NaN for missing).
        comp_name (str): Element symbol for the x-axis.
        phase_name (str): Phase name (for title).
        T_analytical (np.ndarray | None): Analytical spinodal temperatures for
            comparison (plotted as a dashed line if provided).
        ax: Optional Matplotlib Axes.

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.get_figure()

    ax.plot(x_values, T_spinodal, "o-", label="Numerical", color="tab:blue")
    if T_analytical is not None:
        ax.plot(x_values, T_analytical, "--", label="Analytical", color="tab:orange")
        ax.legend()

    ax.set_xlabel(f"$x_{{{comp_name}}}$")
    ax.set_ylabel("T (K)")
    ax.set_title(f"{phase_name} spinodal curve")
    fig.tight_layout()
    return fig


def plot_n_negative_eigenvalues_map(
    results: list[dict[str, Any]],
    comp_x: str,
    comp_y: str,
    T: float,
    phase_name: str,
    *,
    fig: Any = None,
    ax: Any = None,
    cmap: str = "RdYlGn_r",
) -> matplotlib.figure.Figure:
    """Scatter-plot the number of negative eigenvalues over a 2-D composition grid.

    Useful for ternary systems plotted in a projected binary space.

    Args:
        results (list[dict]): Output of ``stability_map``, each dict having
            keys for composition values and ``n_negative``.
        comp_x (str): Component name for the x-axis mole fraction.
        comp_y (str): Component name for the y-axis mole fraction.
        T (float): Temperature in K (for annotation).
        phase_name (str): Phase name (for title).
        fig: Optional Matplotlib Figure.
        ax: Optional Matplotlib Axes.
        cmap (str): Colour map for the scatter plot.

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    elif fig is None:
        fig = ax.get_figure()

    x_vals = np.array([r.get(comp_x, np.nan) for r in results])
    y_vals = np.array([r.get(comp_y, np.nan) for r in results])
    n_neg = np.array([r.get("n_negative", -1) for r in results])

    valid = (n_neg >= 0) & np.isfinite(x_vals) & np.isfinite(y_vals)
    n_max = int(np.max(n_neg[valid])) if valid.any() else 1
    bounds = np.arange(-0.5, n_max + 1.5, 1.0)
    norm = BoundaryNorm(bounds, plt.cm.get_cmap(cmap).N)

    sc = ax.scatter(
        x_vals[valid], y_vals[valid],
        c=n_neg[valid], cmap=cmap, norm=norm, s=40, edgecolors="none",
    )
    cbar = fig.colorbar(sc, ax=ax, ticks=range(n_max + 1))
    cbar.set_label("# negative eigenvalues")

    ax.set_xlabel(f"$x_{{{comp_x}}}$")
    ax.set_ylabel(f"$x_{{{comp_y}}}$")
    ax.set_title(f"{phase_name}  —  T = {T:.0f} K")
    fig.tight_layout()
    return fig


def plot_hessian_comparison(
    comparison: dict[str, Any],
    *,
    fig: Any = None,
) -> matplotlib.figure.Figure:
    """Side-by-side heatmaps of Jansson and finite-difference Hessians.

    Args:
        comparison (dict): Output of ``validation.compare_hessians``.
        fig: Optional Matplotlib Figure (3 subplots will be added).

    Returns:
        matplotlib.figure.Figure: The figure with 3 subplots (Jansson, FD, diff).
    """
    import matplotlib.pyplot as plt

    H_jan = comparison["H_jansson"]
    H_fd = comparison["H_fd"]
    diff = comparison["abs_diff"]
    comps = comparison["indep_components"]

    if fig is None:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    else:
        axes = fig.subplots(1, 3)

    titles = ["Jansson", "Finite diff", "|Diff|"]
    mats = [H_jan, H_fd, diff]
    for ax, title, mat in zip(axes, titles, mats):
        im = ax.imshow(mat, aspect="auto")
        ax.set_title(title)
        ax.set_xticks(range(len(comps)))
        ax.set_yticks(range(len(comps)))
        ax.set_xticklabels(comps)
        ax.set_yticklabels(comps)
        fig.colorbar(im, ax=ax, shrink=0.8, label="J/mol")

    fig.suptitle(
        f"Chemical Hessian — max |diff| = {comparison['max_abs_diff']:.2e} J/mol",
        y=1.01,
    )
    fig.tight_layout()
    return fig


def plot_eigenvalues_vs_temperature(
    T_values: np.ndarray,
    eigenvalue_data: list[np.ndarray | None],
    indep_components: list[str],
    phase_name: str,
    *,
    ax: Any = None,
) -> matplotlib.figure.Figure:
    """Plot all gauge-invariant eigenvalues vs temperature at a fixed composition.

    Args:
        T_values (np.ndarray): 1-D array of temperatures in K.
        eigenvalue_data (list[np.ndarray | None]): List of eigenvalue arrays
            (sorted ascending), one per T. Use None for failed points.
        indep_components (list[str]): Names of independent components (for legend).
        phase_name (str): Phase name (for title).
        ax: Optional Matplotlib Axes.

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.get_figure()

    n_eigs = max(
        (len(ev) for ev in eigenvalue_data if ev is not None),
        default=0,
    )
    eig_matrix = np.full((len(T_values), n_eigs), np.nan)
    for k, ev in enumerate(eigenvalue_data):
        if ev is not None:
            eig_matrix[k, : len(ev)] = ev

    colors = plt.cm.tab10.colors
    for m in range(n_eigs):
        ax.plot(
            T_values, eig_matrix[:, m],
            color=colors[m % len(colors)],
            label=f"$\\lambda_{m+1}$",
        )

    ax.axhline(0.0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("T (K)")
    ax.set_ylabel(r"$\lambda$ (J/mol)")
    ax.set_title(f"{phase_name} eigenvalues vs T")
    ax.legend()
    fig.tight_layout()
    return fig
