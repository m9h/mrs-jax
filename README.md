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
| 🔜 | Synthetic MRS data generation (JAX-based forward model) |
| 🔜 | Hyperpolarized ¹³C pyruvate kinetic modeling |
| 🔜 | PyGAMMA integration for J-coupling simulation |

---

## Context

MR Spectroscopy has a rich history spanning four decades of innovation — from the first in vivo spectra to FDA-cleared clinical platforms. mrs-jax is built on the shoulders of this work.

### The clinical foundation: Ross and Lin at Huntington/Caltech

Brian Ross established the [first comprehensive clinical MRS service in the United States](https://link.springer.com/article/10.1602/neurorx.2.2.197) at the Huntington Medical Research Institutes (HMRI) in Pasadena, beginning with his work alongside Sir George Radda at Oxford in 1981. Ross demonstrated that proton MRS could serve as a **"virtual biopsy"** — non-invasively measuring brain metabolites that reveal disease states invisible to structural MRI:

- **Elevated choline + reduced NAA** → tumor proliferation and neuronal loss
- **Elevated glutamine + reduced myo-inositol** → hepatic encephalopathy
- **Lactate appearance** → anaerobic metabolism (stroke, mitochondrial disease)
- **Reduced NAA alone** → neuronal dysfunction (MS, Alzheimer's, TBI)

[Alexander Lin](https://pnl.bwh.harvard.edu/index.php/alexander-lin-ph-d/), a Caltech graduate (Bioengineering and Biochemistry/Molecular Biophysics), first encountered MRS as an undergraduate in Ross's lab and went on to direct clinical services at HMRI. Lin's trajectory — from bench science to clinical translation — culminated in [BrainSpec](https://getbrainspec.com/), which received [FDA clearance in 2023](https://www.prnewswire.com/news-releases/brainspec-receives-full-fda-clearance-to-begin-using-ai-backed-solution-for-non-invasive-brain-chemistry-measurement-301997375.html) as the first MRS platform with a normative brain chemistry reference database. BrainSpec automates the delivery of MRS results from days to minutes — the clinical realization of Ross's vision from the GE MR Masters Series lecture *"Beyond MRI: MR Spectroscopy for the New Millennium."*

### Multivoxel spectroscopic imaging: Sarah Nelson at UCSF

[Sarah Nelson](https://cancer.ucsf.edu/people/profiles/nelson_sarah.3528) (1954–2019) transformed MRS from a single-voxel curiosity into a spatially resolved imaging modality for clinical oncology. Her [multivoxel MRSI program](https://radiology.ucsf.edu/research/labs/nelson) at UCSF demonstrated that 3D spectroscopic imaging could map choline, NAA, creatine, and lactate across the entire tumor volume and surrounding brain, providing precise chemical information that guided neurosurgical biopsy to the most metabolically active regions and detected infiltrating disease beyond the contrast-enhancing margin. Nelson showed that adding MRSI to conventional MRI improved glioma grading, treatment planning, and early detection of progression — work that produced over 270 publications and continuous NIH funding spanning decades. She was the driving force behind UCSF's installation of one of the world's first 7T MR scanners and the development of the [SIVIC](https://github.com/SIVICLab/sivic) software platform (with Jason Crane) that brought standardized DICOM MRSI workflows into clinical PACS.

### Hyperpolarized ¹³C MRI: Daniel Vigneron at UCSF

[Daniel Vigneron](https://radiology.ucsf.edu/people/daniel-vigneron) and colleagues at UCSF's Surbeck Laboratory pioneered the clinical translation of [hyperpolarized ¹³C MRI](https://pmc.ncbi.nlm.nih.gov/articles/PMC6490043/) — a technique that boosts the ¹³C signal by >10,000-fold through dynamic nuclear polarization (DNP), enabling real-time imaging of metabolic flux that conventional MRS cannot access. By infusing hyperpolarized [1-¹³C]pyruvate and imaging its rapid conversion to lactate (glycolysis) and alanine (transamination), Vigneron's group achieved the first human hyperpolarized ¹³C studies in prostate cancer and brain tumors. This revealed the Warburg effect *in vivo* — aggressive tumors convert pyruvate to lactate faster — providing a metabolic biomarker for treatment response within seconds of injection, long before anatomical changes are visible. The work extends Nelson's spatial metabolic mapping into the time domain: where MRSI gives a static metabolite snapshot, hyperpolarized ¹³C gives a movie of metabolism in action.

### The Oxford MRS lineage: from Radda's ³¹P to cardiac hyperpolarized ¹³C

The story of in vivo MR spectroscopy begins at Oxford. [Sir George Radda](https://www.bioch.ox.ac.uk/article/sir-george-radda-cbe-fmedsci-frs) (1936–2024) published the first paper using phosphorus-31 NMR to study tissue metabolites, creating the world's first clinical NMR unit and establishing the principle that MR could measure biochemistry — not just anatomy — in living tissue. His ³¹P studies of ATP dynamics in muscle and heart laid the foundation for everything that followed: Brian Ross trained with Radda at Oxford before bringing clinical MRS to the United States.

The Oxford MRS tradition continued through [Damian Tyler](https://www.cardiov.ox.ac.uk/team/damian-tyler) and colleagues, who translated [hyperpolarized ¹³C MRI to the human heart](https://www.ahajournals.org/doi/10.1161/CIRCRESAHA.119.316260). Building on Jan Henrik Ardenkjær-Larsen's 2003 DNP breakthrough (>10,000× signal enhancement), the Oxford cardiac group demonstrated *in vivo* assessment of pyruvate dehydrogenase flux — measuring how the heart switches between fuel sources (glucose vs fatty acids) in health, diabetes, and ischemia. This cardiac application of hyperpolarized ¹³C complements Vigneron's oncology work at UCSF: where UCSF images the Warburg effect in tumors, Oxford images metabolic flexibility in the heart.

The Oxford ecosystem also produced [FSL-MRS](https://git.fmrib.ox.ac.uk/fsl/fsl_mrs) (Will Clarke, FMRIB) — the Python-based SVS analysis package whose density-matrix simulator and NIfTI-MRS format mrs-jax builds upon — and [MMORF](https://fsl.fmrib.ox.ac.uk/fsl/docs/registration/mmorf.html) (Arthofer, Mayston, Jenkinson), the multimodal registration framework used in our WAND processing pipeline.

### Whole-brain metabolic mapping: Maudsley and MIDAS

Andrew Maudsley's [MIDAS](https://grantome.com/grant/NIH/R01-EB016064-01A1) system and echo-planar spectroscopic imaging (EPSI) demonstrated that metabolite distributions could be mapped across 118,000+ voxels in a single acquisition, revealing spatial patterns of NAA, creatine, and choline that structural MRI cannot see. Recent advances in [compressed-sensing FID-MRSI](https://pubmed.ncbi.nlm.nih.gov/34595791/) now achieve 5mm isotropic resolution across the whole brain in 20 minutes — approaching the resolution needed for cortical layer-specific metabolite mapping.

### Neuroenergetics: Rothman, Hyder, and Shulman at Yale

The Yale group used [¹³C-labeled glucose infusion with MRS](https://web.stanford.edu/class/rad226a/Readings/Lecture20-Rothman_2011.pdf) to trace metabolic flux in the living human brain, discovering the stoichiometric relationship between glutamate-glutamine cycling (V_cycle) and neuronal glucose oxidation (CMR_glc) — a foundational result linking neural activity to energy metabolism. Their work showed that ~80% of cortical energy consumption supports glutamatergic signaling, with the remainder sustaining housekeeping functions. This ¹³C-MRS methodology, combined with ¹H-[¹³C] editing to detect ¹³C label incorporation into glutamate and glutamine, opened the door to measuring neurotransmitter cycling rates *in vivo*.

### Clinical spectral editing: from MEGA-PRESS to Big GABA

MEGA-PRESS (Mescher et al. 1998) made it possible to resolve GABA — the brain's primary inhibitory neurotransmitter — from the overlapping creatine peak at 3.0 ppm, enabling routine clinical GABA measurement. The [Big GABA project](https://www.nitrc.org/projects/biggaba/) (Mikkelsen et al. 2017) standardized MEGA-PRESS across 24 sites worldwide on all three major scanner vendors, establishing reproducibility benchmarks that any new tool must meet.

### Open-source ecosystem: SIVIC, FSL-MRS, Osprey

The field's maturation is reflected in its software: [SIVIC](https://github.com/SIVICLab/sivic) (Crane, Nelson — UCSF) is the reference implementation for DICOM-standard MRSI clinical workflows, processing ~400 brain MRSI reports per year directly into PACS with standardized choline/NAA indices for tumor grading; [FSL-MRS](https://onlinelibrary.wiley.com/doi/10.1002/mrm.28630) (Clarke — Oxford) provided end-to-end SVS analysis with Python and density-matrix basis simulation; [Osprey](https://github.com/schorschinho/osprey) (Oeltzschner — Johns Hopkins) integrated preprocessing, fitting, and quantification for edited MRS with an emphasis on MEGA-PRESS/HERMES. mrs-jax complements these tools by adding JAX-based differentiable computation and GPU batch processing — learning from SIVIC's clinical workflow design, FSL-MRS's simulation engine, and Osprey's editing pipeline.

### Where mrs-jax fits

mrs-jax builds on this foundation with modern tools: JAX for differentiable computation, validated against the field's benchmark datasets (Big GABA, ISMRM Fitting Challenge), and designed to scale from single-voxel editing to the whole-brain mapping and dynamic ¹³C work that lies ahead. The immediate goal is robust, reproducible GABA and GSH quantification; the long-term vision follows Ross's original insight — making the virtual biopsy routine.

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
- Ross BD, Bluml S (2001) Magnetic resonance spectroscopy of the human brain. *Anat Rec* 265:54–84
- Ross BD, Lin AP (2005) Efficacy of proton MRS in neurological diagnosis and neurotherapeutic decision making. *Neurotherapeutics* 2:197–214
- Crane JC, Olson MP, Nelson SJ (2013) SIVIC: Open-source, standards-based software for DICOM MR spectroscopy workflows. *Int J Biomed Imaging* 2013:169526
- Ross BD, Lin AP. *Beyond MRI: MR Spectroscopy for the New Millennium.* GE MR Masters Series. GE Medical Systems.
- Oeltzschner G et al. (2020) Osprey: Open-source processing, reconstruction & estimation of MRS data. *JMRI* 52:88–105
- Nelson SJ et al. (2001) Analysis of volume MRI and MR spectroscopic imaging data for the evaluation of patients with brain tumors. *MRM* 46:228–239
- Nelson SJ (2003) Multivoxel magnetic resonance spectroscopy of brain tumors. *Mol Cancer Ther* 2:497–507
- Kurhanewicz J, Vigneron DB et al. (2019) Hyperpolarized ¹³C MRI: State of the art and future directions. *Radiology* 286:52–71
- Radda GK (1986) The use of NMR spectroscopy for the understanding of disease. *Science* 233:640–645
- Rider OJ, Tyler DJ et al. (2020) Noninvasive in vivo assessment of cardiac metabolism in the healthy and diabetic human heart using hyperpolarized ¹³C MRI. *Circ Res* 126:725–736
- Ardenkjær-Larsen JH et al. (2003) Increase in signal-to-noise ratio of >10,000 times in liquid-state NMR. *PNAS* 100:10158–10163

---

## License

MIT

## Citation

If you use mrs-jax in your research, please cite the validation datasets and the key methodological references above.
