# spinodal

CALPHAD spinodal decomposition analysis via the Jansson analytical chemical Hessian.

## What it does

`spinodal` computes the **chemical Hessian**

$$H_{ij} = \frac{\partial(\mu_i - \mu_\text{dep})}{\partial x_j}$$

analytically from a CALPHAD thermodynamic database using pycalphad's
`fixed_component_differential` (the Jansson derivative approach). It then finds
the spinodal boundary — where the smallest eigenvalue of H in composition simplex
space crosses zero — for binary and multicomponent alloys.

## Quickstart

```python
from pycalphad import Database, variables as v
from spinodal import chemical_hessian, min_eigenvalue

dbf = Database("CrFe-18Jac.tdb")
conds = {v.T: 800.0, v.P: 101_325.0, v.X("CR"): 0.4}
H, indep, dep = chemical_hessian(dbf, ["CR", "FE", "VA"], "BCC_A2", conds)
print(f"λ_min = {min_eigenvalue(H)/1000:.2f} kJ/mol")
```

## Installation

```bash
uv add spinodal
# or
pip install spinodal
```

## Citation

See [CITATION.cff](https://github.com/ichasekatz/spinodal/blob/main/CITATION.cff)
or the [README Citation section](https://github.com/ichasekatz/spinodal#citation).
