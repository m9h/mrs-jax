"""Spatial DFT and sample-time references for the 1D EPSI surrogate."""

import numpy as np
import pytest

from mrs_jax.current_encoding import CurrentEncoding
from mrs_jax.mrsi_sim import (
    EPSITrajectory, TissueModel, generate_epsi_trajectory,
    reconstruct_epsi, simulate_epsi, simulate_epsi_from_arrays,
)


def model_1d(c, t2=None, b0=None):
    c = np.asarray(c)
    shape = (len(c), 1, 1)
    return TissueModel(np.ones(shape), {"mIns": c.reshape(shape)}, np.ones(shape),
        np.asarray(t2 if t2 is not None else np.full(len(c), .1)).reshape(shape),
        np.asarray(b0 if b0 is not None else np.zeros(len(c))).reshape(shape))


@pytest.mark.parametrize("nx", [5, 8])
def test_instantaneous_lobes_recover_full_complex_spatial_phantom(nx):
    c = np.arange(nx, dtype=float) + 1
    t2 = np.linspace(.04, .1, nx)
    b0 = np.linspace(-12, 18, nx)
    traj = generate_epsi_trajectory(nx, 11, .001, 120)
    times = np.arange(11) * .001
    basis = np.exp(2j * np.pi * 33 * times)
    raw = simulate_epsi(model_1d(c, t2, b0), {"mIns": basis}, traj)
    original = raw.copy()
    raw.setflags(write=False)
    image = reconstruct_epsi(raw, traj)
    expected = (c[:, None] * basis * np.exp(-times / t2[:, None])
                * np.exp(2j * np.pi * b0[:, None] * times))
    np.testing.assert_allclose(image, expected, rtol=2e-6, atol=2e-6)
    np.testing.assert_array_equal(raw, original)
    np.testing.assert_array_equal(reconstruct_epsi(raw, traj), image)


def test_sample_times_and_direct_dft_reference_with_current():
    traj = generate_epsi_trajectory(4, 3, .001, 80, adc_dwell_time=.0001,
                                    readout_start_s=.0002)
    expected_t = np.array([.0002, .0003, .0004, .0005, .0012, .0013,
                           .0014, .0015, .0022, .0023, .0024, .0025])
    np.testing.assert_allclose(traj.sample_times_s, expected_t, atol=1e-18)
    c = np.array([1., 2., 0., 4.])
    t2 = np.array([.08, .1, .06, .04])
    b0 = np.array([10., -5., 3., 20.])
    enc = CurrentEncoding(traj.sample_times_s, [[.00035, .00135]], [[0, .003, 1]])
    phase = enc.phase(np.array([1, 2, 0, -1]) * 1e-6, [.002], phase_sign=1)
    # A constant basis isolates timing, decay, static B0 and injected phase.
    raw = simulate_epsi(model_1d(c, t2, b0), {"mIns": np.ones(32)}, traj,
                         basis_dwell_time=.0001, phase_rad=phase)
    x = np.array([-40, -20, 0, 20])
    expected = np.array([sum(c[v] * np.exp(-t / t2[v]) *
        np.exp(1j * (2 * np.pi * (b0[v] * t - k * x[v]) + phase[v, s]))
        for v in range(4)) for s, (t, k) in enumerate(zip(expected_t, traj.k_readout))])
    np.testing.assert_allclose(raw, expected, rtol=2e-6, atol=2e-6)
    pytest.importorskip("jax")
    import jax
    import jax.numpy as jnp
    actual = jax.jit(lambda p: simulate_epsi_from_arrays(
        jnp.array(c), jnp.ones(32), jnp.array(t2), jnp.array(b0), traj,
        basis_dwell_time=.0001, phase_rad=p))(jnp.array(phase))
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=1e-5)


@pytest.mark.parametrize("adc_dt", [None, .0001])
def test_numpy_jax_basis_interpolation_and_zero_padding(adc_dt):
    pytest.importorskip("jax")
    import jax.numpy as jnp
    traj = generate_epsi_trajectory(4, 4, .001, 80, adc_dwell_time=adc_dt)
    basis = np.array([1 + 2j, 2 - 1j])
    c = np.array([0., 1., 2., 0.])
    expected = simulate_epsi(model_1d(c), {"mIns": basis}, traj)
    actual = simulate_epsi_from_arrays(jnp.array(c), jnp.array(basis),
                                       jnp.full(4, .1), jnp.zeros(4), traj)
    np.testing.assert_allclose(actual, expected, atol=2e-6)
    np.testing.assert_array_equal(actual[8:], np.zeros(8))


def test_reconstruction_rejects_non_dft_trajectory():
    traj = EPSITrajectory(np.linspace(-.1, .1, 8), 4, 2, .001, 80)
    with pytest.raises(ValueError, match="DFT grid"):
        reconstruct_epsi(np.ones(8), traj)


@pytest.mark.parametrize("n_spectral, dt", [(32, .001), (64, .0005), (256, .00025)])
def test_full_basis_final_knot_agrees_across_backends(n_spectral, dt):
    pytest.importorskip("jax")
    import jax.numpy as jnp
    traj = generate_epsi_trajectory(4, n_spectral, dt, 80)
    basis = np.exp(2j * np.pi * 20 * np.arange(n_spectral) * dt)
    c = np.array([0., 1., 2., 0.])
    expected = simulate_epsi(model_1d(c), {"mIns": basis}, traj)
    actual = simulate_epsi_from_arrays(jnp.array(c), jnp.array(basis),
                                       jnp.full(4, .1), jnp.zeros(4), traj)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("kwargs", [
    {"n_spatial": 0}, {"dwell_time": 0}, {"adc_dwell_time": .001},
    {"readout_start_s": -1}, {"fov_mm": np.nan},
])
def test_reject_invalid_trajectory(kwargs):
    with pytest.raises(ValueError):
        generate_epsi_trajectory(**kwargs)
