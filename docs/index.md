# mrs-jax

**MR Spectroscopy processing in JAX** -- from single-voxel GABA editing to whole-brain metabolic mapping.

mrs-jax is a complete, GPU-accelerated pipeline for processing edited Magnetic Resonance Spectroscopy data. It covers the full journey from raw scanner data to absolute metabolite concentrations, with JAX providing automatic differentiation, batch parallelism, and JIT compilation.

## Features

- **MEGA-PRESS pipeline** -- SVD coil combination, spectral registration alignment, MAD outlier rejection, paired frequency/phase correction
- **HERMES** -- 4-condition Hadamard encoding for simultaneous GABA + GSH
- **Quantification** -- Gasparovic (2006) tissue-corrected water-referenced concentrations with field-strength-specific $T_1$/$T_2$ relaxation
- **JAX backend** -- `jit`, `vmap`, `grad` for GPU-accelerated batch processing
- **Preprocessing** -- Apodization, eddy current correction (Klose), frequency referencing
- **QC reports** -- Self-contained HTML reports with inline plots and metabolite tables

## Quick start

```bash
# Install
pip install mrs-jax[all]

# Or with uv
uv pip install mrs-jax[all]
```

```python
import mrs_jax

# Load raw Siemens data
data = mrs_jax.read_twix("sub01_mega_press.dat")

# Full pipeline: coil combine -> align -> subtract -> phase -> fit -> quantify
result = mrs_jax.quantify_mega_press(
    data.data, data.dwell_time, data.centre_freq,
    water_ref=mrs_jax.read_twix("sub01_water_ref.dat").data,
    tissue_fracs={'gm': 0.6, 'wm': 0.3, 'csf': 0.1},
    te=0.068, tr=2.0,
)

print(f"GABA: {result['gaba_conc_mM']:.2f} mM")
print(f"GABA/NAA: {result['gaba_naa_ratio']:.3f}")
```

## Validated on

| Dataset | Type | N | Key result |
|---------|------|---|-----------|
| **Big GABA** S5 | Siemens MEGA-PRESS 3T | 12 subjects | GABA/NAA = 0.059 +/- 0.006 |
| **ISMRM Fitting Challenge** | Synthetic PRESS TE=30ms | 28 spectra | Known ground truth |
| **WAND** ses-05 | 7T MEGA-PRESS, 32-coil | 4 VOIs | GABA/NAA = 0.73--1.30 |

```{toctree}
:maxdepth: 2
:caption: Tutorials

tutorials/mega_press_gaba
tutorials/hermes_editing
tutorials/jax_gpu_acceleration
```

```{toctree}
:maxdepth: 2
:caption: API Reference

reference/mrs_jax
```

```{toctree}
:maxdepth: 1
:caption: Project

changelog
contributing
```
