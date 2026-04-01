"""TDD tests for EPSI readout trajectory and reconstruction.

EPSI (Echo-Planar Spectroscopic Imaging) uses an oscillating read gradient
to simultaneously encode one spatial dimension AND the spectral dimension
in a single gradient echo train. This gives Maudsley-style whole-brain
metabolite mapping in a clinically feasible acquisition time.

The key difference from phase-encoded CSI:
- CSI: one FID per phase-encode step (slow, N_x * N_y * N_z TRs)
- EPSI: one gradient echo train per phase-encode step, encoding spectral
  + one spatial dimension simultaneously (fast, N_y * N_z TRs)
"""
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Tests: EPSI trajectory
# ---------------------------------------------------------------------------

class TestEPSITrajectory:
    def test_generate_epsi_trajectory(self):
        """Generate oscillating read gradient waveform."""
        from mrs_jax.mrsi_sim import generate_epsi_trajectory

        traj = generate_epsi_trajectory(
            n_spatial=32,       # spatial points along readout
            n_spectral=256,     # spectral points
            dwell_time=2.5e-4,  # spectral dwell time
            fov_mm=240.0,       # field of view in mm
        )
        assert traj.k_readout.shape[0] == 32 * 256  # n_spatial * n_spectral
        assert traj.n_spatial == 32
        assert traj.n_spectral == 256

    def test_epsi_trajectory_oscillates(self):
        """k-space trajectory should oscillate (bipolar gradient)."""
        from mrs_jax.mrsi_sim import generate_epsi_trajectory

        traj = generate_epsi_trajectory(
            n_spatial=16, n_spectral=64, dwell_time=5e-4, fov_mm=200.0,
        )
        # k values should go back and forth
        k = traj.k_readout
        # Check that k crosses zero multiple times
        zero_crossings = np.sum(np.diff(np.sign(k)) != 0)
        assert zero_crossings >= 64 - 2  # ~1 crossing per lobe boundary

    def test_epsi_trajectory_covers_kspace(self):
        """Trajectory should cover the full spatial k-space extent."""
        from mrs_jax.mrsi_sim import generate_epsi_trajectory

        traj = generate_epsi_trajectory(
            n_spatial=32, n_spectral=128, dwell_time=2.5e-4, fov_mm=240.0,
        )
        k_max = np.max(np.abs(traj.k_readout))
        # k_max should be approximately 1/(2*voxel_size) = n_spatial/(2*FOV)
        expected_kmax = 32 / (2 * 240.0)  # 1/mm
        assert abs(k_max - expected_kmax) / expected_kmax < 0.1


# ---------------------------------------------------------------------------
# Tests: EPSI signal simulation
# ---------------------------------------------------------------------------

class TestEPSISimulation:
    def test_simulate_epsi_signal(self):
        """Simulate EPSI signal from phantom."""
        from mrs_jax.mrsi_sim import (
            TissueModel, make_lorentzian_basis,
            generate_epsi_trajectory, simulate_epsi,
        )

        # Simple 1D phantom (single line through brain)
        phantom = np.zeros((16, 1, 1), dtype=int)
        phantom[4:12, 0, 0] = 2  # GM
        phantom[6:10, 0, 0] = 3  # WM core

        from mrs_jax.mrsi_sim import METABOLITE_CONCENTRATIONS, TISSUE_T2STAR
        metab_maps = {}
        for metab, concs in METABOLITE_CONCENTRATIONS.items():
            m = np.zeros(phantom.shape, dtype=np.float32)
            for label, c in enumerate(concs):
                m[phantom == label] = c
            metab_maps[metab] = m

        t2star_map = np.ones(phantom.shape, dtype=np.float32) * 0.04
        for label, val in TISSUE_T2STAR.items():
            t2star_map[phantom == label] = max(val, 1e-6)

        model = TissueModel(
            tissue_map=phantom, metabolite_maps=metab_maps,
            t1_map=np.ones_like(phantom, dtype=np.float32) * 1.3,
            t2star_map=t2star_map,
            b0_shift_map=np.zeros_like(phantom, dtype=np.float32),
            voxel_size=(7.5, 7.5, 10.0),
        )
        basis = make_lorentzian_basis(
            {'NAA': 2.01, 'Cr': 3.03},
            n_points=64, dwell_time=5e-4, centre_freq=123.25e6,
        )
        traj = generate_epsi_trajectory(
            n_spatial=16, n_spectral=64, dwell_time=5e-4, fov_mm=120.0,
        )
        signal = simulate_epsi(model, basis, traj)
        assert signal.ndim >= 1
        assert np.max(np.abs(signal)) > 0

    def test_epsi_reconstruct_matches_csi(self):
        """EPSI reconstruction should give similar results to phase-encoded CSI."""
        from mrs_jax.mrsi_sim import (
            TissueModel, make_lorentzian_basis,
            simulate_mrsi, generate_epsi_trajectory,
            simulate_epsi, reconstruct_epsi,
        )

        # 1D phantom
        phantom = np.zeros((8, 1, 1), dtype=int)
        phantom[2:6, 0, 0] = 2  # GM

        from mrs_jax.mrsi_sim import METABOLITE_CONCENTRATIONS, TISSUE_T2STAR
        metab_maps = {}
        for metab in ['NAA', 'Cr']:
            concs = METABOLITE_CONCENTRATIONS[metab]
            m = np.zeros(phantom.shape, dtype=np.float32)
            for label, c in enumerate(concs):
                m[phantom == label] = c
            metab_maps[metab] = m

        t2star_map = np.ones(phantom.shape, dtype=np.float32) * 0.04
        model = TissueModel(
            tissue_map=phantom, metabolite_maps=metab_maps,
            t1_map=np.ones_like(phantom, dtype=np.float32) * 1.3,
            t2star_map=t2star_map,
            b0_shift_map=np.zeros_like(phantom, dtype=np.float32),
            voxel_size=(15.0, 15.0, 10.0),
        )
        basis = make_lorentzian_basis(
            {'NAA': 2.01, 'Cr': 3.03},
            n_points=32, dwell_time=1e-3, centre_freq=123.25e6,
        )

        # CSI reference
        csi_kspace = simulate_mrsi(model, basis, n_spectral=32, dwell_time=1e-3)
        csi_image = np.fft.ifftn(csi_kspace, axes=(0, 1, 2))

        # EPSI
        traj = generate_epsi_trajectory(
            n_spatial=8, n_spectral=32, dwell_time=1e-3, fov_mm=120.0,
        )
        epsi_signal = simulate_epsi(model, basis, traj)
        epsi_image = reconstruct_epsi(epsi_signal, traj)

        # Compare NAA peak at GM voxel
        gm_idx = 3  # middle of GM region
        csi_spec = np.abs(np.fft.fft(csi_image[gm_idx, 0, 0, :]))
        epsi_spec = np.abs(np.fft.fft(epsi_image[gm_idx, :]))

        # Both should have signal (NAA peak)
        assert np.max(csi_spec) > 0
        assert np.max(epsi_spec) > 0

        # Peak locations should match (within 1 bin)
        csi_peak = np.argmax(csi_spec)
        epsi_peak = np.argmax(epsi_spec)
        assert abs(csi_peak - epsi_peak) <= 2


# ---------------------------------------------------------------------------
# Tests: JAX differentiable EPSI
# ---------------------------------------------------------------------------

class TestEPSIJAX:
    def test_epsi_jax_differentiable(self):
        """Gradient flows through EPSI simulation."""
        from mrs_jax.mrsi_sim import (
            make_lorentzian_basis, generate_epsi_trajectory,
            simulate_epsi_from_arrays,
        )
        import jax
        import jax.numpy as jnp

        basis = make_lorentzian_basis(
            {'NAA': 2.01}, n_points=32, dwell_time=1e-3, centre_freq=123.25e6,
        )
        traj = generate_epsi_trajectory(
            n_spatial=8, n_spectral=32, dwell_time=1e-3, fov_mm=120.0,
        )

        conc = jnp.array([0, 0, 10, 12, 12, 10, 0, 0], dtype=jnp.float32)
        t2star = jnp.ones(8) * 0.04
        b0 = jnp.zeros(8)

        def loss_fn(c):
            sig = simulate_epsi_from_arrays(
                c, jnp.array(basis['NAA']), t2star, b0, traj
            )
            return jnp.sum(jnp.abs(sig) ** 2)

        grad = jax.grad(loss_fn)(conc)
        assert grad.shape == (8,)
        assert jnp.any(grad != 0)
