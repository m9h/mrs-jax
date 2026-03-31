# mrs-jax

MR Spectroscopy processing in JAX — MEGA-PRESS, HERMES, and quantification.

## Features

- **MEGA-PRESS editing**: SVD coil combination, spectral registration alignment, outlier rejection, edit-ON/OFF subtraction
- **HERMES 4-condition editing**: Hadamard reconstruction for simultaneous GABA + GSH
- **Phase correction**: Zero-order and first-order, GABA Gaussian fitting
- **Water-referenced quantification**: Tissue-corrected absolute concentrations (mM)
- **JAX backend**: `jax.jit`, `jax.vmap` (batch subjects), `jax.grad` (differentiable fitting)
- **Preprocessing**: Apodization, eddy current correction, frequency referencing
- **Native I/O**: Siemens TWIX reader (no spec2nii dependency)
- **QC reports**: Self-contained HTML with inline spectral plots

## Validated on

| Dataset | Subjects | Result |
|---------|----------|--------|
| Big GABA Site S5 (Siemens) | 12 | GABA/NAA = 0.059 ± 0.006 |
| NIfTI-MRS Standard examples | 2 | MEGA-PRESS + edited |
| WAND 7T MEGA-PRESS | 4 VOIs | GABA/NAA = 0.73–1.30 |
| WAND 7T sLASER (fsl_mrs) | 4 VOIs | NAA = 15.8 mM, CRLB 2% |
| ISMRM Fitting Challenge | 28 | Synthetic spectra |

## Installation

```bash
pip install mrs-jax              # Core (NumPy only)
pip install mrs-jax[jax]         # With JAX acceleration
pip install mrs-jax[all]         # Everything including I/O
```

## Quick start

```python
import mrs_jax

# Load Siemens TWIX data
data = mrs_jax.read_twix("mega_press.dat")

# Process MEGA-PRESS
result = mrs_jax.process_mega_press(
    data.data, data.dwell_time, data.centre_freq,
    align=True, reject=True, paired_alignment=True
)

# Quantify GABA
metrics = mrs_jax.quantify_mega_press(
    data.data, data.dwell_time, data.centre_freq,
    water_ref=water_fid,
    tissue_fracs={'gm': 0.6, 'wm': 0.3, 'csf': 0.1}
)
print(f"GABA: {metrics['gaba_conc_mM']:.2f} mM")
```

## Test suite

```bash
pytest tests/ -v  # 70 tests
```

## License

MIT
