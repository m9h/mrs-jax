# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added

- Exact piecewise-current phase histories with explicit SI units, signal sign,
  ideal RF sensitivity and current-delay parameters; NumPy/JAX backends.
- Additional voxel/time phase inputs to MRSI and EPSI forward models.
- Explicit EPSI ADC timestamps and independent basis sampling intervals.
- Analytical phase/gradient tests and full complex DFT reconstruction references.

### Fixed

- EPSI now uses distinct centered DFT bins and matching inverse transforms.
  Spatial outputs change from the previous incorrect grid/centering.
- EPSI reconstruction no longer mutates raw input.
- NumPy/JAX EPSI basis interpolation consistently zero-fills beyond support.
- `simulate_mrsi_jax` now computes natively in JAX, including multiple metabolites,
  instead of wrapping a NumPy result.
- MEGA-PRESS equivalence tests now compare explicit single and double precision,
  with precision-specific tolerances across CPU backends.

## [0.1.0] - 2025

### Added

- MEGA-PRESS editing pipeline: SVD coil combination, spectral registration alignment (Near et al. 2015), MAD-based outlier rejection, paired frequency/phase correction
- HERMES 4-condition Hadamard encoding and reconstruction for simultaneous GABA + GSH (Chan et al. 2016)
- Phase correction: zero-order (maximize absorption) and first-order (Nelder-Mead optimization) with GABA Gaussian fitting
- Water-referenced quantification with Gasparovic (2006) tissue correction (GM/WM/CSF fractions, 3T and 7T relaxation)
- JAX-accelerated backend: `jit`, `vmap`, `grad` equivalents of the NumPy pipeline
- Native Siemens TWIX reader via mapVBVD with automatic edit dimension detection
- Philips SDAT/SPAR reader
- LCModel `.RAW`/`.BASIS` I/O
- JAX spectral registration: jit/vmap/grad-compatible alignment
- Preprocessing: exponential/Gaussian apodization, Klose eddy current correction, frequency referencing
- Self-contained HTML QC reports with inline base64 plots
- Differentiable MRSI simulator
- Realistic WAND brain phantom with 13 metabolites
- Validated on Big GABA (S5, 12 subjects), ISMRM Fitting Challenge (28 spectra), WAND (7T, 4 VOIs)
- 70+ tests across 10 test files
