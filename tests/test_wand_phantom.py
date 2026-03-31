"""TDD tests for realistic WAND brain phantom + MRSI simulation.

Uses WAND sub-08033 FAST tissue segmentation to create a ground-truth
digital phantom with literature metabolite concentrations, then
validates the MRSI simulator produces physiologically realistic output.
"""
import numpy as np
import pytest
from pathlib import Path

FAST_SEG = Path("/data/raw/wand/derivatives/fsl-anat/sub-08033/ses-02/sub-08033_ses-02_T1w.anat/T1_fast_seg.nii.gz")
FAST_GM = Path("/data/raw/wand/derivatives/fsl-anat/sub-08033/ses-02/sub-08033_ses-02_T1w.anat/T1_fast_pve_1.nii.gz")

skip_no_wand = pytest.mark.skipif(not FAST_SEG.exists(), reason="WAND data not available")


# ---------------------------------------------------------------------------
# Phantom construction
# ---------------------------------------------------------------------------

class TestWANDPhantom:
    @skip_no_wand
    def test_load_wand_phantom(self):
        from mrs_jax.mrsi_sim import load_wand_phantom
        model = load_wand_phantom(str(FAST_SEG))
        assert model.shape[0] > 50  # Not trivially small
        assert model.shape[1] > 50
        assert 'NAA' in model.metabolite_maps
        assert 'GABA' in model.metabolite_maps

    @skip_no_wand
    def test_phantom_has_correct_tissues(self):
        from mrs_jax.mrsi_sim import load_wand_phantom
        model = load_wand_phantom(str(FAST_SEG))
        labels = set(model.tissue_map.ravel())
        assert 0 in labels  # background
        assert 1 in labels or 2 in labels  # CSF or GM

    @skip_no_wand
    def test_phantom_downsampled(self):
        """Phantom can be downsampled for fast simulation."""
        from mrs_jax.mrsi_sim import load_wand_phantom
        model = load_wand_phantom(str(FAST_SEG), resolution_mm=8.0)
        # 192mm / 8mm = 24 voxels along x
        assert model.shape[0] < 30
        assert model.shape[1] < 40

    @skip_no_wand
    def test_naa_gm_greater_than_wm(self):
        """NAA concentration should be higher in GM than WM."""
        from mrs_jax.mrsi_sim import load_wand_phantom
        model = load_wand_phantom(str(FAST_SEG))
        naa = model.metabolite_maps['NAA']
        gm_mask = model.tissue_map == 2  # FAST: 2=GM
        wm_mask = model.tissue_map == 3  # FAST: 3=WM
        if gm_mask.any() and wm_mask.any():
            assert naa[gm_mask].mean() > naa[wm_mask].mean()


# ---------------------------------------------------------------------------
# MRSI simulation with WAND phantom
# ---------------------------------------------------------------------------

class TestWANDMRSISimulation:
    @skip_no_wand
    def test_simulate_wand_mrsi(self):
        """Full MRSI simulation on downsampled WAND phantom."""
        from mrs_jax.mrsi_sim import load_wand_phantom, make_lorentzian_basis, simulate_mrsi

        model = load_wand_phantom(str(FAST_SEG), resolution_mm=16.0)
        basis = make_lorentzian_basis(
            {'NAA': 2.01, 'Cr': 3.03, 'Cho': 3.22},
            n_points=256, dwell_time=5e-4, centre_freq=123.25e6,
        )
        kspace = simulate_mrsi(model, basis, n_spectral=256, dwell_time=5e-4)
        assert kspace.ndim == 4
        assert kspace.shape[-1] == 256
        assert np.max(np.abs(kspace)) > 0

    @skip_no_wand
    def test_reconstructed_naa_map_anatomical(self):
        """Reconstructed NAA map should follow brain anatomy."""
        from mrs_jax.mrsi_sim import load_wand_phantom, make_lorentzian_basis, simulate_mrsi

        model = load_wand_phantom(str(FAST_SEG), resolution_mm=16.0)
        basis = make_lorentzian_basis(
            {'NAA': 2.01},
            n_points=256, dwell_time=5e-4, centre_freq=123.25e6,
        )
        kspace = simulate_mrsi(model, basis, n_spectral=256, dwell_time=5e-4)
        image = np.fft.ifftn(kspace, axes=(0, 1, 2))

        # NAA peak integral per voxel
        spec = np.fft.fft(image, axis=-1)
        freq = np.fft.fftfreq(256, 5e-4)
        ppm = np.fft.fftshift(freq) / (123.25e6 / 1e6) + 4.65
        naa_mask = (ppm > 1.8) & (ppm < 2.2)
        naa_map = np.sum(np.abs(np.fft.fftshift(spec, axes=-1)[..., naa_mask]), axis=-1)

        # Brain voxels should have signal, background should not
        brain_mask = model.tissue_map > 0
        bg_mask = model.tissue_map == 0
        if brain_mask.any() and bg_mask.any():
            brain_signal = naa_map[brain_mask].mean()
            bg_signal = naa_map[bg_mask].mean()
            assert brain_signal > 5 * bg_signal, (
                f"Brain NAA ({brain_signal:.2f}) should be >> background ({bg_signal:.2f})"
            )

    @skip_no_wand
    def test_simulate_with_b0_inhomogeneity(self):
        """B0 shifts should broaden peaks / shift frequencies."""
        from mrs_jax.mrsi_sim import load_wand_phantom, make_lorentzian_basis, simulate_mrsi

        model = load_wand_phantom(str(FAST_SEG), resolution_mm=16.0)

        # Add B0 inhomogeneity: linear gradient across x
        nx = model.shape[0]
        b0_gradient = np.linspace(-20, 20, nx)  # ±20 Hz
        model.b0_shift_map = np.broadcast_to(
            b0_gradient[:, None, None], model.shape
        ).copy().astype(np.float32)

        basis = make_lorentzian_basis(
            {'NAA': 2.01},
            n_points=256, dwell_time=5e-4, centre_freq=123.25e6,
        )
        kspace = simulate_mrsi(model, basis, n_spectral=256, dwell_time=5e-4)
        assert np.max(np.abs(kspace)) > 0  # Simulation completes
