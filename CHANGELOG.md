# Changelog

All notable changes to `spinodal` are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.1.0] — 2026-11-25

### Added
- `chemical_hessian`: analytical Jansson chemical Hessian via `fixed_component_differential`
- `eigenanalysis`: gauge-invariant eigenvalues via generalized eigenvalue problem H v = λ G v
- `stability_map`: 2D grid scan for binary and multicomponent spinodal maps
- `boundary_detection`: 1D root-finding for spinodal boundary curves
- `coherent_spinodal`: elastic energy correction matrix for coherent spinodal
- `plotting`: utilities for spinodal and stability-map visualization
- `validation`: FD-based validation comparing Jansson H[0,0] to d²G/dx²
- Examples: Fe-Cr, Al-Zn, Cu-Co binary validation; HfNbTiV quaternary map
- Tests: regular-solution analytical benchmark, eigenanalysis, boundary detection
