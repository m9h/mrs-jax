<p align="center">
  <h1 align="center">🧠 mrs-jax</h1>
  <p align="center">
    <strong>MR Spectroscopy processing in JAX</strong><br>
    <em>From single-voxel GABA editing to whole-brain metabolic mapping</em>
  </p>
  <p align="center">
    <a href="#features"><img src="https://img.shields.io/badge/modules-8-blue" alt="modules"></a>
    <a href="#test-suite"><img src="https://img.shields.io/badge/tests-70%20passing-brightgreen" alt="tests"></a>
    <a href="#validated-on"><img src="https://img.shields.io/badge/validated-Big%20GABA%20%7C%20ISMRM%20%7C%20WAND-orange" alt="validated"></a>
    <a href="https://github.com/m9h/mrs-jax/blob/master/LICENSE"><img src="https://img.shields.io/badge/license-MIT-lightgrey" alt="license"></a>
    <a href="#jax-backend"><img src="https://img.shields.io/badge/JAX-jit%20%7C%20vmap%20%7C%20grad-red" alt="JAX"></a>
  </p>
</p>

---

## Overview

**mrs-jax** is a complete, GPU-accelerated pipeline for processing edited Magnetic Resonance Spectroscopy data. It covers the full journey from raw scanner data to absolute metabolite concentrations — with JAX providing automatic differentiation, batch parallelism, and JIT compilation.

MRS is the only non-invasive technique that can measure neurochemical concentrations *in vivo* in the human brain. The metabolites it detects — GABA (the primary inhibitory neurotransmitter), glutamate (excitatory), NAA (neuronal integrity), creatine (energy metabolism), myo-inositol (glial marker), glutathione (antioxidant) — are the molecular substrates of brain function that structural and functional MRI cannot access.

### The MRS landscape

MRS spans a remarkable range of scales and applications:

| Scale | Technique | What it measures | mrs-jax support |
|-------|-----------|-----------------|-----------------|
| **Single voxel** | PRESS, STEAM, sLASER | Regional metabolite concentrations in a ~8 cm³ voxel | ✅ Full pipeline |
| **Spectral editing** | MEGA-PRESS | GABA, GSH — metabolites hidden under larger peaks | ✅ Full pipeline |
| **Multi-editing** | HERMES, HERCULES | Simultaneous GABA + GSH + other targets via Hadamard encoding | ✅ Hadamard reconstruction |
| **Whole-brain mapping** | EPSI / FID-MRSI | Metabolite maps at ~1 mL resolution across the entire brain (MIDAS, Maudsley) | 🔜 Planned |
| **Dynamic MRS** | fMRS, ¹³C-MRS | Time-resolved metabolic changes during tasks or after ¹³C-glucose infusion | 🔜 Planned |
| **Hyperpolarized** | ¹³C DNP | >10,000× signal enhancement for real-time metabolic flux imaging | 🔜 Planned |

The field has evolved from Andrew Maudsley's pioneering whole-brain EPSI/MIDAS work — mapping NAA, creatine, and choline across 118,000+ voxels — through the Rothman/Hyder/Shulman ¹³C-MRS studies at Yale that revealed the stoichiometric coupling between glutamate cycling and neuronal glucose oxidation, to today's edited single-voxel techniques that resolve individual neurotransmitters with clinical precision.

mrs-jax starts where the clinical need is greatest: **spectral editing for GABA and glutathione**, validated against the field's benchmark datasets, with a JAX foundation built for the whole-brain and dynamic extensions ahead.

---

## Features

### 🎯 Core editing pipeline

```
Raw TWIX → Coil combine → Align → Reject outliers → Edit subtract → Phase correct → Fit → Quantify
```

- **MEGA-PRESS**: The standard for in vivo GABA measurement. SVD coil combination, spectral registration alignment (Near et al. 2015), MAD-based outlier rejection, paired frequency/phase correction (FPC) that preserves subtraction quality
- **HERMES**: 4-condition Hadamard encoding for simultaneous GABA + GSH in a single acquisition (Chan et al. 2016)

### 📐 Quantification

- **GABA Gaussian fitting**: Automated peak detection at 3.0 ppm with amplitude, FWHM, area, and Cramér-Rao lower bounds
- **Water-referenced concentrations**: Gasparovic (2006) tissue-corrected quantification using GM/WM/CSF fractions with field-strength-specific T1/T2 relaxation (3T and 7T)
- **Phase correction**: Zero-order (maximize absorption) and first-order (Nelder-Mead optimization) to ensure accurate real-part integration

### ⚡ JAX backend

- `jax.jit` — compile the full pipeline for fast repeated execution
- `jax.vmap` — process a batch of subjects in parallel on GPU
- `jax.grad` — differentiate through the correction and fitting pipeline for gradient-based optimization

### 🔧 Preprocessing

- **Apodization**: Exponential and Gaussian line broadening with FWHM-calibrated windows
- **Eddy current correction**: Klose method using water reference phase
- **Frequency referencing**: Automatic peak-based referencing to NAA (2.01 ppm) or creatine (3.03 ppm)

### 📂 I/O

- **Native Siemens TWIX reader**: Direct `.dat` file loading via mapVBVD — no spec2nii dependency. Automatic detection of edit dimensions, multi-coil data, and header metadata (TE, TR, field strength)
- **MRSData container**: Standardized dataclass with `data`, `dwell_time`, `centre_freq`, `dim_info`, `water_ref`

### 📊 Quality control

- **Self-contained HTML reports**: Inline base64 matplotlib plots of edit-ON/OFF/difference spectra, frequency drift traces, rejection statistics, and metabolite concentration tables. No external dependencies — one `.html` file per subject.

---

## Validated on

| Dataset | Type | N | Key result |
|---------|------|---|-----------|
| **[Big GABA](https://www.nitrc.org/projects/biggaba/)** S5 | Siemens MEGA-PRESS 3T | 12 subjects | GABA/NAA = 0.059 ± 0.006 |
| **[ISMRM Fitting Challenge](https://github.com/wtclarke/mrs_fitting_challenge)** | Synthetic PRESS TE=30ms | 28 spectra | Known ground truth |
| **[NIfTI-MRS Standard](https://github.com/wtclarke/mrs_nifti_standard)** | 32-coil MEGA-PRESS + edited | 2 examples | Format validation |
| **WAND** ses-05 | 7T MEGA-PRESS, 32-coil | 4 VOIs | GABA/NAA = 0.73–1.30 |
| **WAND** ses-04 | 7T sLASER, fsl_mrs fit | 4 VOIs | NAA = 15.8 mM, CRLB 2.1% |

---

## Installation

```bash
pip install mrs-jax              # Core (NumPy only)
pip install mrs-jax[jax]         # With JAX acceleration
pip install mrs-jax[all]         # Everything including I/O + fsl_mrs
```

Development:
```bash
git clone https://github.com/m9h/mrs-jax.git
cd mrs-jax
pip install -e ".[dev,all]"
pytest tests/ -v
```

---

## Quick start

### MEGA-PRESS GABA quantification

```python
import mrs_jax

# Load raw Siemens data
data = mrs_jax.read_twix("sub01_mega_press.dat")

# Full pipeline: coil combine → align → subtract → phase → fit → quantify
result = mrs_jax.quantify_mega_press(
    data.data, data.dwell_time, data.centre_freq,
    water_ref=mrs_jax.read_twix("sub01_water_ref.dat").data,
    tissue_fracs={'gm': 0.6, 'wm': 0.3, 'csf': 0.1},
    te=0.068, tr=2.0
)

print(f"GABA:     {result['gaba_conc_mM']:.2f} mM")
print(f"GABA/NAA: {result['gaba_naa_ratio']:.3f}")
print(f"SNR:      {result['snr']:.1f}")
print(f"CRLB:     {result['crlb_percent']:.1f}%")
```

### HERMES: simultaneous GABA + GSH

```python
from mrs_jax.hermes import process_hermes

# 4-condition data: (n_spec, 4, n_dyn)
result = process_hermes(data, dwell_time, centre_freq)

# Separate difference spectra
gaba_spectrum = np.fft.fft(result.gaba_diff)  # (A+B) - (C+D)
gsh_spectrum = np.fft.fft(result.gsh_diff)    # (A+C) - (B+D)
```

### JAX: batch processing with vmap

```python
from mrs_jax.mega_press_jax import process_mega_press as process_jax
import jax

# Stack 12 subjects: (12, n_spec, 2, n_dyn)
batch_data = jax.numpy.stack(all_subjects)

# Process all subjects in parallel on GPU
batch_process = jax.vmap(lambda d: process_jax(d, dwell, cf))
results = batch_process(batch_data)
```

### QC report

```python
from mrs_jax.qc import generate_qc_report

html = generate_qc_report(
    result,
    fitting_results={'NAA': 15.8, 'GABA': 3.7, 'Cr': 5.8}
)
with open("sub01_qc.html", "w") as f:
    f.write(html)
```

---

## Architecture

```
mrs_jax/
├── io.py              # Siemens TWIX reader → MRSData
├── preproc.py         # Apodization, ECC, frequency referencing
├── mega_press.py      # MEGA-PRESS pipeline (NumPy)
├── mega_press_jax.py  # MEGA-PRESS pipeline (JAX — jit/vmap/grad)
├── hermes.py          # HERMES 4-condition Hadamard
├── phase.py           # Phase correction + GABA Gaussian fitting
├── quantify.py        # End-to-end quantification pipeline
└── qc.py              # HTML QC report generation
```

### Data flow

```
              ┌─────────┐
              │  TWIX    │ Siemens .dat
              │  SDAT    │ Philips (planned)
              │  P-file  │ GE (planned)
              └────┬─────┘
                   │ read_twix()
                   ▼
              ┌─────────┐
              │ MRSData  │ Standardized container
              └────┬─────┘
                   │
          ┌────────┼────────┐
          ▼        ▼        ▼
     ┌────────┐ ┌──────┐ ┌───────┐
     │  MEGA  │ │HERMES│ │sLASER │
     │ PRESS  │ │ 4-ed │ │ PRESS │
     └───┬────┘ └──┬───┘ └───┬───┘
         │         │         │
         ▼         ▼         │
    ┌─────────────────┐      │
    │  Difference     │      │
    │  spectrum        │      │
    └────────┬────────┘      │
             │               │
             ▼               ▼
        ┌──────────────────────┐
        │  Phase correction    │
        │  + Gaussian fitting  │
        │  + Water scaling     │
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │  [GABA] = 3.7 mM    │
        │  GABA/NAA = 0.12    │
        │  CRLB = 8.5%        │
        └──────────────────────┘
```

---

## Test suite

```bash
pytest tests/ -v  # 70 tests, ~56 seconds
```

| Test file | Tests | What it validates |
|-----------|-------|-------------------|
| `test_mega_press.py` | 13 | Core pipeline, GABA detection, Cr cancellation, drift alignment |
| `test_mrs_phase_correction.py` | 12 | Phase correction, GABA fitting, water quantification |
| `test_mrs_fpc.py` | 5 | Paired frequency/phase correction |
| `test_mrs_hermes.py` | 5 | HERMES Hadamard separation |
| `test_mrs_jax.py` | 6 | JAX equivalence, jit, vmap, grad |
| `test_mrs_qc.py` | 5 | HTML report generation |
| `test_mrs_io.py` | 6 | Siemens TWIX reader |
| `test_mrs_preproc.py` | 8 | Apodization, ECC, frequency referencing |
| `test_mrs_quantification.py` | 6 | End-to-end quantification |
| `test_mrs_integration.py` | 4 | Big GABA + WAND real data |

---

## Roadmap

| Status | Feature |
|--------|---------|
| ✅ | MEGA-PRESS editing pipeline |
| ✅ | HERMES 4-condition (GABA + GSH) |
| ✅ | Phase correction + GABA fitting |
| ✅ | Water-referenced quantification |
| ✅ | JAX backend (jit, vmap, grad) |
| ✅ | Siemens TWIX reader |
| ✅ | QC HTML reports |
| ✅ | Preprocessing (apodization, ECC, freq ref) |
| 🔜 | Philips SDAT/SPAR reader |
| 🔜 | GE P-file reader |
| 🔜 | MEGA-PRESS basis simulation |
| 🔜 | Spectral registration in JAX |
| 🔜 | LCModel .RAW/.BASIS I/O |
| 🔜 | Whole-brain MRSI (EPSI/FID-MRSI) |
| 🔜 | Dynamic fMRS time-course analysis |
| 🔜 | ¹³C-MRS metabolic flux modeling |
| 🔜 | PyGAMMA integration for J-coupling simulation |

---

## Context

MR Spectroscopy has a rich history spanning decades of methodological innovation:

**Whole-brain metabolic mapping** — Andrew Maudsley's [MIDAS](https://grantome.com/grant/NIH/R01-EB016064-01A1) system and echo-planar spectroscopic imaging (EPSI) demonstrated that metabolite distributions could be mapped across 118,000+ voxels in a single acquisition, revealing spatial patterns of NAA, creatine, and choline that structural MRI cannot see. Recent advances in compressed-sensing FID-MRSI now achieve 5mm isotropic resolution across the whole brain in 20 minutes.

**Neuroenergetics via ¹³C-MRS** — The Yale group (Rothman, Hyder, Shulman) used [¹³C-labeled glucose infusion with MRS](https://web.stanford.edu/class/rad226a/Readings/Lecture20-Rothman_2011.pdf) to trace metabolic flux in the living brain, discovering the stoichiometric relationship between glutamate-glutamine cycling and neuronal glucose oxidation — a foundational result linking neural activity to energy metabolism.

**Clinical spectral editing** — MEGA-PRESS (Mescher et al. 1998) made it possible to resolve GABA from overlapping creatine, enabling routine clinical GABA measurement. The [Big GABA project](https://www.nitrc.org/projects/biggaba/) (Mikkelsen et al. 2017) standardized this across 24 sites worldwide.

mrs-jax builds on this foundation with modern tools: JAX for differentiable computation, validated against the field's benchmark datasets, and designed to scale from single-voxel editing to the whole-brain mapping and dynamic ¹³C work that lies ahead.

---

## References

- Mikkelsen M et al. (2017) Big GABA: Edited MR spectroscopy at 24 research sites. *NeuroImage* 159:32–45
- Chan KL et al. (2016) HERMES: Hadamard encoding and reconstruction of MEGA-edited spectroscopy. *MRM* 76:11–19
- Near J et al. (2015) Frequency and phase drift correction by spectral registration in the time domain. *MRM* 73:44–50
- Gasparovic C et al. (2006) Use of tissue water as a concentration reference for proton spectroscopic imaging. *MRM* 55:1219–1226
- Clarke WT et al. (2021) FSL-MRS: An end-to-end spectroscopy analysis package. *MRM* 85:2950–2964
- Maudsley AA et al. (2009) Mapping of brain metabolite distributions by volumetric proton MR spectroscopic imaging. *MRM* 61:548–559
- Rothman DL et al. (2011) ¹³C MRS studies of neuroenergetics and neurotransmitter cycling in humans. *NMR Biomed* 24:943–957
- Marjańska M et al. (2022) Results of a fitting challenge for MR spectroscopy. *MRM* 87:2198–2211

---

## License

MIT

## Citation

If you use mrs-jax in your research, please cite the validation datasets and the key methodological references above.
