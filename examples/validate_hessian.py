"""Validate chemical_hessian against independent finite-difference d²G/dx².

For a binary substitutional phase (A,B)(VA) with no additional internal DOF,
the Jansson H[0,0] must equal d²G/dx² (the second derivative of the
single-phase molar Gibbs energy evaluated at the prescribed composition).

Proof: for a binary mixture G(x) with x = x_solute,
    μ_sol - μ_dep = dG/dx   (Euler theorem + Gibbs–Duhem)
    H[0,0] = d(μ_sol - μ_dep)/dx = d²G/dx²

Two grids are computed on the full (x, T) domain:
  (A) Jansson:  H[0,0] from chemical_hessian() — equals d²G/dx²
  (B) FD:       d²G/dx² via central finite differences on calculate(GM), ε=5e-4

NOTE: min_eigenvalue(H) solves the gauge-invariant problem H v = λ G v where
G_ij = δ_ij + 1 (composition simplex metric).  For binary (n=1): G=[[2]], so
λ_min = H[0,0]/2 ≠ d²G/dx².  This script compares H[0,0] directly to FD to
avoid the factor-of-2 from the metric tensor.  The spinodal boundary (zero of
both λ_min and H[0,0]) is identical.

If (A) ≈ (B): Jansson is correct.
If (A) ≠ (B): Jansson has a bug (wrong state, wrong indices, or model issue).

Usage:
    python validate_hessian.py --system Fe-Cr
    python validate_hessian.py --system Al-Zn
    python validate_hessian.py --system Cu-Co
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parent

warnings.filterwarnings("ignore")

from pycalphad import Database, calculate
from pycalphad import variables as v
from pycalphad.core.minimizer import fixed_component_differential
from spinodal.hessian import chemical_hessian, _build_single_phase_state
from spinodal._exceptions import EquilibriumConvergenceError, SpinodallHessianError

_FIG_DIR = _HERE.parent / "figures"
_TDB_CACHE = Path(__file__).resolve().parent / "_orchid_tdb_cache"
P = 101_325.0
FD_EPS = 5e-4


# ── System definitions ────────────────────────────────────────────────────────

SYSTEMS = {
    "Fe-Cr": dict(
        tdb=_TDB_CACHE / "CrFe-18Jac.tdb",
        components=["CR", "FE", "VA"],
        phase="BCC_A2",
        solute="CR",
        T_range=(400., 1100.),
        x_range=(0.10, 0.90),
        n_T=25, n_x=45,
        sigma_T_min=788.,
    ),
    "Al-Zn": dict(
        tdb=_TDB_CACHE / "AlZn-93Mey-LB.tdb",
        components=["AL", "ZN", "VA"],
        phase="FCC_A1",
        solute="ZN",
        T_range=(250., 650.),
        x_range=(0.03, 0.78),
        n_T=25, n_x=45,
        sigma_T_min=None,
    ),
    "Cu-Co": dict(
        tdb=_TDB_CACHE / "CoCu-00Kub-LB.tdb",
        components=["CO", "CU", "VA"],
        phase="FCC_A1",
        solute="CO",
        T_range=(600., 1420.),
        x_range=(0.03, 0.97),
        n_T=25, n_x=45,
        sigma_T_min=None,
    ),
}


# ── Site-fraction builder ──────────────────────────────────────────────────────

def _site_fracs(dbf: Database, phase_name: str, solute: str, x: float) -> np.ndarray:
    """Return 1-D site-fraction array for a simple substitutional phase at x."""
    constituents = [[str(c) for c in sl] for sl in dbf.phases[phase_name].constituents]
    sf: list[float] = []
    for sl in constituents:
        nonva = [c for c in sl if c.upper() != "VA"]
        if not nonva:
            sf.extend([1.0] * len(sl))
            continue
        for c in sl:
            if c.upper() == "VA":
                sf.append(0.0)
            elif c.upper() == solute.upper():
                sf.append(float(x))
            else:
                sf.append(max(0.0, 1.0 - float(x)))
    return np.array(sf, dtype=float)


# ── Method (A): Jansson chemical_hessian → H[0,0] = d²G/dx² ─────────────────

def _jansson(dbf, components, phase_name, solute, x, T):
    """H[0,0] from Jansson solve — equals d²G/dx² for binary substitutional phases."""
    conds = {v.T: T, v.P: P, v.X(solute): x}
    try:
        H, _, _ = chemical_hessian(dbf, components, phase_name, conds)
        # Return raw H[0,0] (= d²G/dx²) for direct comparison with FD.
        # Do NOT use min_eigenvalue(H) here — min_eigenvalue solves the
        # gauge-invariant problem H v = λ G v where G=[[2]] for binary,
        # giving λ = H[0,0]/2, which would cause a spurious factor-of-2 error.
        return float(H[0, 0])
    except (EquilibriumConvergenceError, SpinodallHessianError, IndexError, ValueError,
            Exception):
        return np.nan


# ── Method (B): d²G/dx² finite difference via calculate(GM) ──────────────────

def _d2G_fd(dbf, components, phase_name, solute, x, T, eps=FD_EPS):
    """d²G/dx² via central finite difference of single-phase G from calculate()."""
    try:
        xp = min(x + eps, 1.0 - 1e-6)
        xm = max(x - eps, 1e-6)
        pts = np.array([
            _site_fracs(dbf, phase_name, solute, xp),
            _site_fracs(dbf, phase_name, solute, x),
            _site_fracs(dbf, phase_name, solute, xm),
        ])
        result = calculate(dbf, components, phase_name,
                           T=T, P=P, output="GM", points=pts,
                           broadcast=False)
        G = result["GM"].values.flatten()[:3]
        if not np.all(np.isfinite(G)):
            return np.nan
        return float((G[0] - 2.0 * G[1] + G[2]) / eps**2)
    except Exception:
        return np.nan


# ── Grid computation ───────────────────────────────────────────────────────────

def compute_grids(dbf, components, phase_name, solute, x_arr, T_arr):
    """Compute two grids: (A) Jansson H[0,0], (B) FD d²G/dx². Both equal d²G/dx²."""
    nT, nX = len(T_arr), len(x_arr)
    gA = np.full((nT, nX), np.nan)    # Jansson H[0,0]
    gB = np.full((nT, nX), np.nan)    # FD d²G/dx²

    for j, T in enumerate(T_arr):
        print(f"    T={T:.0f}K", end=" ", flush=True)
        for i, x in enumerate(x_arr):
            gA[j, i] = _jansson(dbf, components, phase_name, solute, x, T)
            gB[j, i] = _d2G_fd(dbf, components, phase_name, solute, x, T)
        print()

    return gA, gB


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_validation(name, x_arr, T_arr, gA, gB, sigma_T_min=None):
    XX, TT = np.meshgrid(x_arr, T_arr)
    solute = name.split("-")[1]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=120)

    def _hmap(ax, g, title, cmap="RdBu_r"):
        gp = np.where(np.isfinite(g), g / 1000, np.nan)
        vm = max(np.nanpercentile(np.abs(gp[np.isfinite(gp)]), 95), 0.1) if np.any(np.isfinite(gp)) else 1.0
        tc = ax.contourf(XX, TT, gp, levels=20, cmap=cmap, vmin=-vm, vmax=vm)
        ax.contour(XX, TT, np.where(np.isfinite(g), g, 0.0),
                   levels=[0], colors="k", linewidths=2.0)
        fig.colorbar(tc, ax=ax).set_label("kJ/mol", fontsize=8)
        if sigma_T_min:
            ax.axhline(sigma_T_min, color="orange", lw=1.5, ls="--",
                       label=f"σ eutectoid ({sigma_T_min:.0f} K)")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(f"x_{solute}", fontsize=9)
        ax.set_ylabel("T (K)", fontsize=9)

    _hmap(axes[0], gA,
          "(A) Jansson H[0,0]\nchemical_hessian() = d²G/dx²\nRed=stable, Blue=spinodal")
    _hmap(axes[1], gB,
          "(B) d²G/dx²\ncentral FD of calculate(GM), ε=5×10⁻⁴\nRed=stable, Blue=spinodal")

    diff = gA - gB
    gp = np.where(np.isfinite(diff), diff / 1000, np.nan)
    vm = max(np.nanpercentile(np.abs(gp[np.isfinite(gp)]), 95), 0.01) if np.any(np.isfinite(gp)) else 0.1
    ax = axes[2]
    tc = ax.contourf(XX, TT, gp, levels=20, cmap="PiYG", vmin=-vm, vmax=vm)
    fig.colorbar(tc, ax=ax).set_label("kJ/mol", fontsize=8)
    ax.contour(XX, TT, np.where(np.isfinite(gA), gA, 0.0),
               levels=[0], colors="red", linewidths=1.5, label="Jansson spinodal")
    ax.contour(XX, TT, np.where(np.isfinite(gB), gB, 0.0),
               levels=[0], colors="blue", linewidths=1.5, linestyles="--", label="FD spinodal")
    ax.plot([], [], "r-", lw=1.5, label="Jansson spinodal")
    ax.plot([], [], "b--", lw=1.5, label="FD spinodal")
    if sigma_T_min:
        ax.axhline(sigma_T_min, color="orange", lw=1.5, ls="--")
    ax.legend(fontsize=7, loc="lower center")
    ax.set_title(f"(A)−(B): residual\n≈0 everywhere ⟹ Jansson correct", fontsize=9)
    ax.set_xlabel(f"x_{solute}", fontsize=9)
    ax.set_ylabel("T (K)", fontsize=9)

    d_valid = diff[np.isfinite(diff)]
    rms = np.sqrt(np.mean(d_valid**2)) / 1000 if d_valid.size else np.nan
    maxe = np.max(np.abs(d_valid)) / 1000 if d_valid.size else np.nan

    fig.suptitle(
        f"{name} — Jansson H[0,0] vs FD d²G/dx²  |  "
        f"RMS={(rms):.4f} kJ/mol  max|err|={maxe:.4f} kJ/mol  n={d_valid.size}",
        fontsize=11, y=1.01
    )
    fig.tight_layout()
    out = _FIG_DIR / f"validate_hessian_{name.replace('-','').lower()}.png"
    _FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  → {out.name}")
    print(f"  RMS(A−B)={rms:.4f} kJ/mol  max|err|={maxe:.4f} kJ/mol  n={d_valid.size}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", default="Fe-Cr", choices=list(SYSTEMS))
    args = parser.parse_args()

    cfg = SYSTEMS[args.system]
    dbf = Database(str(cfg["tdb"]))
    x_arr = np.linspace(cfg["x_range"][0], cfg["x_range"][1], cfg["n_x"])
    T_arr = np.linspace(cfg["T_range"][0], cfg["T_range"][1], cfg["n_T"])

    print(f"\n=== {args.system}: 3-method grid ({cfg['n_T']}×{cfg['n_x']}) ===")
    gA, gB = compute_grids(
        dbf, cfg["components"], cfg["phase"], cfg["solute"], x_arr, T_arr
    )
    plot_validation(args.system, x_arr, T_arr, gA, gB,
                    sigma_T_min=cfg.get("sigma_T_min"))
    print("Done.")


if __name__ == "__main__":
    main()
