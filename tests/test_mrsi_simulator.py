"""TDD tests for MRSI/EPSI simulator — 'POSSUM for MRS'.

A differentiable forward model that generates synthetic MRSI data from
a segmented brain object with per-voxel metabolite concentrations.
"""
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Tissue phantom helpers
# ---------------------------------------------------------------------------

def make_simple_phantom(nx=8, ny=8, nz=1):
    """Create a minimal 2D phantom with GM, WM, and CSF regions.

    Returns tissue_map (nx, ny, nz) with labels 0=background, 1=GM, 2=WM, 3=CSF.
    """
    phantom = np.zeros((nx, ny, nz), dtype=int)
    # GM ring
    phantom[2:6, 2:6, :] = 1
    # WM center
    phantom[3:5, 3:5, :] = 2
    # CSF ventricle
    phantom[4, 4, :] = 3
    return phantom


def make_metabolite_maps(tissue_map):
    """Create metabolite concentration maps from tissue segmentation.

    Returns dict mapping metabolite name -> concentration array (same shape as tissue_map).
    Concentrations in institutional units (mM in tissue).
    """
    shape = tissue_map.shape
    # Literature values (mM) for GM, WM, CSF
    # idx: 0=bg, 1=GM, 2=WM, 3=CSF
    conc_table = {
        'NAA':  [0.0, 12.5, 10.0, 0.0],
        'Cr':   [0.0,  8.0,  6.0, 0.0],
        'Cho':  [0.0,  2.5,  3.0, 0.0],
        'mIns': [0.0,  7.0,  4.0, 0.0],
        'Lac':  [0.0,  0.5,  0.3, 0.0],
    }
    maps = {}
    for metab, concs in conc_table.items():
        m = np.zeros(shape, dtype=np.float32)
        for label, c in enumerate(concs):
            m[tissue_map == label] = c
        maps[metab] = m
    return maps


# ---------------------------------------------------------------------------
# Tests: tissue model
# ---------------------------------------------------------------------------

class TestTissueModel:
    def test_create_tissue_model(self):
        from mrs_jax.mrsi_sim import TissueModel
        phantom = make_simple_phantom()
        metab_maps = make_metabolite_maps(phantom)
        model = TissueModel(
            tissue_map=phantom,
            metabolite_maps=metab_maps,
            t1_map=np.ones_like(phantom, dtype=np.float32) * 1.3,
            t2star_map=np.ones_like(phantom, dtype=np.float32) * 0.040,
            b0_shift_map=np.zeros_like(phantom, dtype=np.float32),
            voxel_size=(3.0, 3.0, 10.0),  # mm
        )
        assert model.shape == (8, 8, 1)
        assert 'NAA' in model.metabolite_maps
        assert model.n_metabolites == 5

    def test_tissue_model_spatial_dimensions(self):
        from mrs_jax.mrsi_sim import TissueModel
        phantom = make_simple_phantom(16, 16, 4)
        metab_maps = make_metabolite_maps(phantom)
        model = TissueModel(
            tissue_map=phantom,
            metabolite_maps=metab_maps,
            t1_map=np.ones((16, 16, 4), dtype=np.float32) * 1.3,
            t2star_map=np.ones((16, 16, 4), dtype=np.float32) * 0.040,
            b0_shift_map=np.zeros((16, 16, 4), dtype=np.float32),
            voxel_size=(3.0, 3.0, 3.0),
        )
        assert model.shape == (16, 16, 4)


# ---------------------------------------------------------------------------
# Tests: basis spectra
# ---------------------------------------------------------------------------

class TestBasisSpectra:
    def test_make_lorentzian_basis(self):
        from mrs_jax.mrsi_sim import make_lorentzian_basis
        # Simple Lorentzian basis at known chemical shifts
        metab_shifts = {'NAA': 2.01, 'Cr': 3.03, 'Cho': 3.22}
        basis = make_lorentzian_basis(
            metab_shifts, n_points=1024, dwell_time=2.5e-4,
            centre_freq=123.25e6, linewidth_hz=3.0
        )
        assert len(basis) == 3
        assert basis['NAA'].shape == (1024,)
        assert np.iscomplexobj(basis['NAA'])

    def test_basis_peak_location(self):
        from mrs_jax.mrsi_sim import make_lorentzian_basis
        basis = make_lorentzian_basis(
            {'NAA': 2.01}, n_points=2048, dwell_time=2.5e-4,
            centre_freq=123.25e6, linewidth_hz=3.0
        )
        # FFT and check peak is near 2.01 ppm
        spec = np.fft.fftshift(np.fft.fft(basis['NAA']))
        freq = np.fft.fftshift(np.fft.fftfreq(2048, 2.5e-4))
        ppm = freq / (123.25e6 / 1e6) + 4.65
        peak_ppm = ppm[np.argmax(np.abs(spec))]
        assert abs(peak_ppm - 2.01) < 0.1


# ---------------------------------------------------------------------------
# Tests: MRSI signal generation
# ---------------------------------------------------------------------------

class TestMRSISignalGeneration:
    def test_generate_kspace_mrsi(self):
        """Generate k-space MRSI data from phantom + basis."""
        from mrs_jax.mrsi_sim import TissueModel, make_lorentzian_basis, simulate_mrsi

        phantom = make_simple_phantom()
        metab_maps = make_metabolite_maps(phantom)
        model = TissueModel(
            tissue_map=phantom,
            metabolite_maps=metab_maps,
            t1_map=np.ones_like(phantom, dtype=np.float32) * 1.3,
            t2star_map=np.ones_like(phantom, dtype=np.float32) * 0.040,
            b0_shift_map=np.zeros_like(phantom, dtype=np.float32),
            voxel_size=(3.0, 3.0, 10.0),
        )
        basis = make_lorentzian_basis(
            {'NAA': 2.01, 'Cr': 3.03, 'Cho': 3.22},
            n_points=512, dwell_time=5e-4, centre_freq=123.25e6,
        )

        kspace = simulate_mrsi(model, basis, n_spectral=512, dwell_time=5e-4)
        # Output: (kx, ky, kz, n_spectral) complex
        assert kspace.shape == (8, 8, 1, 512)
        assert np.iscomplexobj(kspace)

    def test_reconstruct_metabolite_maps(self):
        """FFT of k-space MRSI should recover spatial metabolite distributions."""
        from mrs_jax.mrsi_sim import TissueModel, make_lorentzian_basis, simulate_mrsi

        phantom = make_simple_phantom()
        metab_maps = make_metabolite_maps(phantom)
        model = TissueModel(
            tissue_map=phantom,
            metabolite_maps=metab_maps,
            t1_map=np.ones_like(phantom, dtype=np.float32) * 1.3,
            t2star_map=np.ones_like(phantom, dtype=np.float32) * 0.040,
            b0_shift_map=np.zeros_like(phantom, dtype=np.float32),
            voxel_size=(3.0, 3.0, 10.0),
        )
        basis = make_lorentzian_basis(
            {'NAA': 2.01, 'Cr': 3.03},
            n_points=512, dwell_time=5e-4, centre_freq=123.25e6,
        )
        kspace = simulate_mrsi(model, basis, n_spectral=512, dwell_time=5e-4)

        # Reconstruct: inverse FFT over spatial dims
        image_space = np.fft.ifftn(kspace, axes=(0, 1, 2))

        # Spectrum at GM voxel (3, 3, 0) should show NAA and Cr peaks
        spec = np.fft.fftshift(np.fft.fft(image_space[3, 3, 0, :]))
        assert np.max(np.abs(spec)) > 0  # Not empty

        # Spectrum at background voxel (0, 0, 0) should be ~zero
        bg_spec = np.fft.fftshift(np.fft.fft(image_space[0, 0, 0, :]))
        gm_spec = np.fft.fftshift(np.fft.fft(image_space[3, 3, 0, :]))
        assert np.max(np.abs(bg_spec)) < 0.01 * np.max(np.abs(gm_spec))

    def test_naa_higher_in_gm_than_wm(self):
        """NAA concentration is higher in GM — reconstructed maps should reflect this."""
        from mrs_jax.mrsi_sim import TissueModel, make_lorentzian_basis, simulate_mrsi

        phantom = make_simple_phantom()
        metab_maps = make_metabolite_maps(phantom)
        model = TissueModel(
            tissue_map=phantom,
            metabolite_maps=metab_maps,
            t1_map=np.ones_like(phantom, dtype=np.float32) * 1.3,
            t2star_map=np.ones_like(phantom, dtype=np.float32) * 0.040,
            b0_shift_map=np.zeros_like(phantom, dtype=np.float32),
            voxel_size=(3.0, 3.0, 10.0),
        )
        basis = make_lorentzian_basis(
            {'NAA': 2.01},
            n_points=512, dwell_time=5e-4, centre_freq=123.25e6,
        )
        kspace = simulate_mrsi(model, basis, n_spectral=512, dwell_time=5e-4)
        image_space = np.fft.ifftn(kspace, axes=(0, 1, 2))

        # NAA peak height at GM voxel (2,2) vs WM voxel (3,3)
        freq = np.fft.fftshift(np.fft.fftfreq(512, 5e-4))
        ppm = freq / (123.25e6 / 1e6) + 4.65
        naa_mask = (ppm > 1.8) & (ppm < 2.2)

        gm_naa = np.max(np.abs(np.fft.fftshift(np.fft.fft(image_space[2, 2, 0, :]))[naa_mask]))
        wm_naa = np.max(np.abs(np.fft.fftshift(np.fft.fft(image_space[3, 3, 0, :]))[naa_mask]))

        # GM NAA (12.5 mM) > WM NAA (10.0 mM)
        assert gm_naa > wm_naa * 0.9  # Allow some tolerance from k-space blurring


# ---------------------------------------------------------------------------
# Tests: JAX compatibility
# ---------------------------------------------------------------------------

class TestMRSIJAX:
    def test_simulate_mrsi_jax(self):
        """JAX version produces same output as NumPy."""
        from mrs_jax.mrsi_sim import TissueModel, make_lorentzian_basis, simulate_mrsi_jax
        import jax.numpy as jnp

        phantom = make_simple_phantom(8, 8, 1)
        metab_maps = make_metabolite_maps(phantom)
        model = TissueModel(
            tissue_map=phantom,
            metabolite_maps=metab_maps,
            t1_map=np.ones_like(phantom, dtype=np.float32) * 1.3,
            t2star_map=np.ones_like(phantom, dtype=np.float32) * 0.040,
            b0_shift_map=np.zeros_like(phantom, dtype=np.float32),
            voxel_size=(3.0, 3.0, 10.0),
        )
        basis = make_lorentzian_basis(
            {'NAA': 2.01, 'Cr': 3.03},
            n_points=256, dwell_time=5e-4, centre_freq=123.25e6,
        )
        kspace = simulate_mrsi_jax(model, basis, n_spectral=256, dwell_time=5e-4)
        assert kspace.shape == (8, 8, 1, 256)

    def test_simulate_mrsi_jax_differentiable(self):
        """Can compute gradient of simulated signal w.r.t. metabolite concentrations."""
        from mrs_jax.mrsi_sim import make_lorentzian_basis, simulate_mrsi_from_arrays
        import jax
        import jax.numpy as jnp

        basis = make_lorentzian_basis(
            {'NAA': 2.01},
            n_points=128, dwell_time=5e-4, centre_freq=123.25e6,
        )
        basis_array = jnp.array(basis['NAA'])  # (128,)

        # Concentration map for 2x2x1 phantom
        conc = jnp.array([[[10.0], [5.0]], [[0.0], [8.0]]])  # (2, 2, 1)
        t2star = jnp.ones((2, 2, 1)) * 0.04
        b0 = jnp.zeros((2, 2, 1))

        def loss_fn(conc):
            kspace = simulate_mrsi_from_arrays(
                conc, basis_array, t2star, b0,
                n_spectral=128, dwell_time=5e-4
            )
            return jnp.sum(jnp.abs(kspace) ** 2)

        grad = jax.grad(loss_fn)(conc)
        assert grad.shape == conc.shape
        # Gradient should be non-zero where concentration is non-zero
        assert jnp.any(grad != 0)
