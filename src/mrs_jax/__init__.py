"""mrs-jax: MR Spectroscopy processing in JAX.

A complete pipeline for processing edited MRS data (MEGA-PRESS, HERMES)
with JAX acceleration, validated on Big GABA, ISMRM Fitting Challenge,
and WAND multi-modal datasets.

Modules:
    mega_press — MEGA-PRESS editing pipeline (coil combine, align, subtract)
    mega_press_jax — JAX-accelerated backend (jit, vmap, grad)
    hermes — HERMES 4-condition editing (GABA + GSH)
    phase — Phase correction (zero/first-order) + GABA Gaussian fitting
    preproc — Apodization, ECC, frequency referencing
    io — Native Siemens TWIX reader
    qc — QC HTML report generation
    quantify — End-to-end quantification pipeline
"""

__version__ = "0.1.0"

from mrs_jax.mega_press import process_mega_press, MegaPressResult
from mrs_jax.hermes import process_hermes, HermesResult
from mrs_jax.quantify import quantify_mega_press
from mrs_jax.io import read_twix, MRSData
