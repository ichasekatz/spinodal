# Quickstart

## Single-point Hessian

```python
from pycalphad import Database, variables as v
from spinodal import chemical_hessian, min_eigenvalue

dbf = Database("CrFe-18Jac.tdb")
components = ["CR", "FE", "VA"]
phase = "BCC_A2"

conds = {v.T: 800.0, v.P: 101_325.0, v.X("CR"): 0.4}
H, indep_comps, dep_comp = chemical_hessian(dbf, components, phase, conds)

lam = min_eigenvalue(H)
print(f"λ_min = {lam/1000:.2f} kJ/mol  ({'spinodal' if lam < 0 else 'stable'})")
```

## Binary spinodal map

```python
import numpy as np
from spinodal import stability_map

x_arr = np.linspace(0.05, 0.95, 60)
T_arr = np.linspace(400.0, 1100.0, 40)

# grid[j, i] = λ_min(x_arr[i], T_arr[j])
grid = stability_map(dbf, components, phase, "CR", x_arr, T_arr)

# Find spinodal boundary (zero contour)
import matplotlib.pyplot as plt
XX, TT = np.meshgrid(x_arr, T_arr)
fig, ax = plt.subplots()
ax.contour(XX, TT, grid, levels=[0], colors="red")
ax.set_xlabel("x_Cr")
ax.set_ylabel("T (K)")
plt.show()
```

## Examples

The `examples/` directory contains:

- `validate_hessian.py` — compare Jansson H[0,0] to FD d²G/dx² on a grid
- `plot_spinodal_comparison.py` — binary spinodal overlaid on phase diagram
- `regular_solution.py` — analytical benchmark for a symmetric A-B regular solution
