# spinodal

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![tests](https://github.com/ichasekatz/spinodal/actions/workflows/tests.yml/badge.svg)](https://github.com/ichasekatz/spinodal/actions/workflows/tests.yml)
[![lint](https://github.com/ichasekatz/spinodal/actions/workflows/lint.yml/badge.svg)](https://github.com/ichasekatz/spinodal/actions/workflows/lint.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

CALPHAD spinodal decomposition analysis via the Jansson analytical chemical Hessian.

Computes the chemical Hessian H[i,j] = d(μ_i − μ_dep)/dx_j using pycalphad's
`fixed_component_differential`, performs gauge-invariant eigenanalysis in composition
simplex space, and maps spinodal boundaries in binary and multicomponent alloy systems.
Supports both chemical and coherent spinodals (with elastic energy corrections).

[Report a Bug](https://github.com/ichasekatz/spinodal/issues/new?labels=bug) ·
[Request a Feature](https://github.com/ichasekatz/spinodal/issues/new?labels=enhancement) ·
[Documentation](https://ichasekatz.github.io/spinodal)

## Features

- **Chemical Hessian** via the Jansson analytical derivative (`fixed_component_differential`)
  — exact second derivatives at any composition and temperature, no finite differences required
- **Gauge-invariant eigenanalysis** in composition simplex space via the generalized
  eigenvalue problem H v = λ G v — eigenvalues independent of the choice of dependent component
- **Binary and multicomponent spinodal maps** on (x, T) or higher-dimensional composition grids
- **Coherent spinodal** with elastic energy correction matrix for systems where misfit strain
  shifts the instability boundary
- **Literature-validated examples**: Fe-Cr BCC (Jacob 2018), Al-Zn FCC (Mey 1993),
  Cu-Co FCC (Kubista 2000), HfNbTiV BCC (quaternary REFRACTORY HEA)
- **Independent FD validation** comparing H[0,0] to d²G/dx² from `calculate(GM)`;
  RMS < 0.001 kJ/mol for Fe-Cr and Al-Zn

## Installation

```bash
uv add spinodal
```

or with pip:

```bash
pip install spinodal
```

## Quickstart

```python
from pycalphad import Database, variables as v
from spinodal import chemical_hessian, min_eigenvalue, stability_map

dbf = Database("CrFe-18Jac.tdb")
components = ["CR", "FE", "VA"]
phase = "BCC_A2"

# Chemical Hessian at a single point
conds = {v.T: 800.0, v.P: 101_325.0, v.X("CR"): 0.4}
H, indep_comps, dep_comp = chemical_hessian(dbf, components, phase, conds)
lam = min_eigenvalue(H)
print(f"λ_min = {lam/1000:.2f} kJ/mol  ({'spinodal' if lam < 0 else 'stable'})")

# 2D spinodal map over (x, T) grid
import numpy as np
x_arr = np.linspace(0.05, 0.95, 60)
T_arr = np.linspace(400.0, 1100.0, 40)
grid = stability_map(dbf, components, phase, "CR", x_arr, T_arr)
# grid[j, i] = λ_min at (x_arr[i], T_arr[j]); spinodal where grid < 0
```

## License

Distributed under the GNU General Public License v3 or later.
See [`LICENSE`](LICENSE) for details.

## Citation

If you use `spinodal` in published work, please cite:

> Katz, C. (2026). *spinodal: CALPHAD spinodal decomposition analysis via Jansson
> chemical Hessian*. GitHub. https://github.com/ichasekatz/spinodal.
> DOI: 10.5281/zenodo.XXXXXXX

```bibtex
@software{katz_spinodal_2026,
  author    = {Katz, Chase},
  title     = {spinodal: CALPHAD spinodal decomposition analysis via Jansson chemical Hessian},
  year      = {2026},
  url       = {https://github.com/ichasekatz/spinodal},
  doi       = {10.5281/zenodo.XXXXXXX},
  version   = {0.1.0},
}
```

The analytical Jansson derivative approach is described in:

> Kunselman, C.J. (2026). *Applications of Analytic Derivatives at Equilibrium
> for CALPHAD Models*. PhD Dissertation, Texas A&M University, Ch. 6.
