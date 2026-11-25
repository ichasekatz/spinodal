# Theory

## The spinodal criterion

A homogeneous phase is **spinodally unstable** when the Gibbs energy surface has
negative curvature along some composition direction — infinitesimal composition
fluctuations lower the free energy without a nucleation barrier. The boundary
between stable and spinodally unstable regions is the **spinodal**.

For an N-component system, the spinodal criterion is:

$$\lambda_\text{min}(H) = 0$$

where H is the **chemical Hessian**

$$H_{ij} = \frac{\partial(\mu_i - \mu_\text{dep})}{\partial x_j}, \quad i, j \in \{\text{independent components}\}$$

and $\lambda_\text{min}$ is the smallest eigenvalue of H in composition simplex space
(see below). Inside the spinodal, $\lambda_\text{min} < 0$.

## Jansson analytical derivatives

`spinodal` computes H **analytically** via the Jansson derivative approach
(Kunselman 2026, Ch. 6, Eq. 6.9). For each independent component $j$, the column
$\partial \boldsymbol{\mu} / \partial x_j$ is obtained by calling
`fixed_component_differential(spec, state, j_global)` from pycalphad — a single
linear solve of the KKT system (Sundman et al. 2015, Eq. 74).

This is **exact** to machine precision at the converged equilibrium state, with no
finite-difference truncation error. Validation against $d^2 G / dx^2$ from
`calculate(GM)` shows RMS < 0.001 kJ/mol across the full (x, T) domain for
Fe-Cr BCC and Al-Zn FCC.

## Gauge-invariant eigenvalues

The eigenvalues of H depend on the choice of dependent component (gauge). To obtain
**gauge-invariant** spinodal eigenvalues, `spinodal` solves the generalized eigenvalue
problem

$$H \mathbf{v} = \lambda G \mathbf{v}$$

where $G_{ij} = \delta_{ij} + 1$ is the metric tensor of the composition simplex
(inner products of simplex basis vectors $\mathbf{e}_i - \mathbf{e}_N$). The
eigenvalues $\lambda$ are independent of which component is chosen as the dependent.

For a **binary** system (1×1 H), $G = [[2]]$, so $\lambda = H_{00}/2$. The zero
crossing — the spinodal boundary — is at the same composition regardless of the $1/2$
factor.

## Coherent spinodal

When an alloy phase-separates coherently (maintaining crystal registry across the
interface), misfit strain raises the Gibbs energy of composition fluctuations. The
**coherent Hessian** is

$$H^\text{coh}_{ij} = H_{ij} + Y_{ij}$$

where $Y$ is the elastic correction matrix computed from the molar volume, elastic
constants, and lattice parameter composition dependence. See `coherent_spinodal` for
details.

## References

- Kunselman, C.J. (2026). *Applications of Analytic Derivatives at Equilibrium for
  CALPHAD Models*. PhD Dissertation, Texas A&M University.
- Sundman, B., Kattner, U.R., Sigli, C., Stratmann, M., Le Tellier, R., Palumbo, M.,
  & Fries, S.G. (2015). The OpenCalphad thermodynamic software interface. *Computational
  Materials Science*, 125, 188–196.
- Jacob, A., Povoden-Karadeniz, E., & Kozeschnik, E. (2018). Revised thermodynamic
  description of the Fe-Cr system based on an improved sublattice model of the
  σ phase. *CALPHAD*, 60, 16–28.
