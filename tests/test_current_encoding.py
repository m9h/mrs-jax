"""Analytical field integrals and signal/gradient checks, without scanner data."""

import numpy as np
import pytest

from mrs_jax.current_encoding import CurrentEncoding, GAMMA_RAD_S_T
from mrs_jax.mrsi_sim import TissueModel, simulate_mrsi, simulate_mrsi_jax


def test_constant_field_units_polarity_and_sign():
    times = np.array([0, .001, .004, .008])
    enc = CurrentEncoding(times, [[0, .01]], [[0, .01, 1]])
    field = np.array([2e-6, -3e-6, 0])
    expected = GAMMA_RAD_S_T * field[:, None] * .002 * times
    np.testing.assert_allclose(enc.phase(field, [.002], phase_sign=1), expected)
    np.testing.assert_allclose(enc.phase(field, [-.002], phase_sign=1), -expected)
    np.testing.assert_allclose(enc.phase(field, [.002], phase_sign=-1), -expected)
    np.testing.assert_array_equal(enc.phase(field, [0], phase_sign=1), np.zeros((3, 4)))


def test_refocusing_and_current_reversal():
    enc = CurrentEncoding([0, .01, .02, .03, .04],
                          [[0, .02], [.02, .04]], [[0, .02, 1], [.02, .04, -1]])
    steady = enc.phase(1e-6, [.001, .001], phase_sign=1)
    expected = GAMMA_RAD_S_T * 1e-9 * np.array([0, .01, .02, .01, 0])
    np.testing.assert_allclose(steady, expected, atol=1e-15)
    bipolar = enc.phase(1e-6, [.001, -.001], phase_sign=1)
    np.testing.assert_allclose(bipolar, GAMMA_RAD_S_T * 1e-9 * enc.sample_times_s)


def test_delay_gaps_and_excitation_origin():
    # Negative current start is clipped by excitation t=0; q has a gap.
    enc = CurrentEncoding([0, .001, .003, .006, .01],
                          [[-.002, .004]], [[0, .002, 1], [.004, .01, -1]])
    np.testing.assert_allclose(enc.phase(1, [1], phase_sign=1) / GAMMA_RAD_S_T,
                               [0, .001, .002, .002, .002], atol=1e-15)
    # Shift window to [0, .006]; current overlaps positive and negative q equally.
    np.testing.assert_allclose(
        enc.phase(1, [1], phase_sign=1, current_delay_s=.002) / GAMMA_RAD_S_T,
        [0, .001, .002, 0, 0], atol=1e-15)


@pytest.mark.parametrize("times, windows, sensitivity", [
    ([-1], [[0, 1]], [[0, 1, 1]]),
    ([1, 0], [[0, 1]], [[0, 1, 1]]),
    ([np.nan], [[0, 1]], [[0, 1, 1]]),
    ([0, 1], [[1, 0]], [[0, 1, 1]]),
    ([0, 1], [[0, 1], [.5, 2]], [[0, 1, 1]]),
    ([0, 1], [[0, 1]], [[0, 1, 2]]),
])
def test_reject_invalid_schedule(times, windows, sensitivity):
    with pytest.raises(ValueError):
        CurrentEncoding(times, windows, sensitivity)


def test_reject_invalid_phase_arguments():
    enc = CurrentEncoding([0, 1], [[0, 1]], [[0, 1, 1]])
    for kwargs in ({"phase_sign": 0}, {"phase_sign": 1, "backend": "torch"},
                   {"phase_sign": 1, "current_delay_s": np.nan}):
        with pytest.raises(ValueError):
            enc.phase(1e-6, [.001], **kwargs)
    with pytest.raises(ValueError):
        enc.phase(1e-6, [.001, .002], phase_sign=1)


def test_jax_jit_vmap_and_analytic_derivatives():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    enc = CurrentEncoding([.005, .02], [[0, .01]], [[0, .03, 1]])
    phase = jax.jit(lambda i, delay: enc.phase(
        2e-6, jnp.array([i]), phase_sign=-1, current_delay_s=delay, backend="jax"))
    np.testing.assert_allclose(phase(.001, .001),
        -GAMMA_RAD_S_T * 2e-9 * np.array([.004, .01]), rtol=2e-6)
    grad_i = jax.jacfwd(phase, argnums=0)(.001, .001)
    grad_delay = jax.jacfwd(phase, argnums=1)(.001, .001)
    np.testing.assert_allclose(grad_i, -GAMMA_RAD_S_T * 2e-6 * np.array([.004, .01]), rtol=2e-6)
    np.testing.assert_allclose(grad_delay, [GAMMA_RAD_S_T * 2e-9, 0], rtol=2e-6, atol=1e-7)
    result = jax.vmap(lambda i: phase(i, .001))(jnp.array([.001, -.001]))
    np.testing.assert_allclose(result[0], -result[1])


def test_phase_is_applied_before_spatial_encoding_and_preserves_magnitude():
    shape = (2, 1, 1)
    n = 16
    times = np.arange(n) * .001
    enc = CurrentEncoding(times, [[0, .02]], [[0, .02, 1]])
    field = np.array([1e-6, -2e-6]).reshape(shape)
    phase = enc.phase(field, [.001], phase_sign=1)
    model = TissueModel(np.ones(shape), {"mIns": np.full(shape, 7.)},
                        np.ones(shape), np.full(shape, .1), np.zeros(shape))
    plain = simulate_mrsi(model, {"mIns": np.ones(n)}, n, .001)
    encoded = simulate_mrsi(model, {"mIns": np.ones(n)}, n, .001, phase_rad=phase)
    plain_image = np.fft.ifftn(plain, axes=(0, 1, 2))
    image = np.fft.ifftn(encoded, axes=(0, 1, 2))
    np.testing.assert_allclose(image, plain_image * np.exp(1j * phase), atol=1e-12)
    np.testing.assert_allclose(abs(image), abs(plain_image), atol=1e-12)


def test_multimetabolite_wrapper_is_differentiable_with_current_encoding():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    n = 16
    times = np.arange(n) * .001
    enc = CurrentEncoding(times, [[0, .02]], [[0, .02, 1]])
    shape = (2, 1, 1)
    basis = {"mIns": np.ones(n), "Cr": np.exp(2j * np.pi * 40 * times)}

    def run(c, i):
        model = TissueModel(np.ones(shape), {"mIns": jnp.full(shape, c),
                            "Cr": jnp.full(shape, 3.)}, np.ones(shape),
                            jnp.full(shape, .1), jnp.zeros(shape))
        phase = enc.phase(jnp.full(shape, 1e-6), jnp.array([i]),
                          phase_sign=1, backend="jax")
        return simulate_mrsi_jax(model, basis, n, .001, phase_rad=phase)

    actual = jax.jit(run)(7., .001)
    model = TissueModel(np.ones(shape), {"mIns": np.full(shape, 7.),
                        "Cr": np.full(shape, 3.)}, np.ones(shape),
                        np.full(shape, .1), np.zeros(shape))
    expected = simulate_mrsi(model, basis, n, .001,
        phase_rad=enc.phase(np.full(shape, 1e-6), [.001], phase_sign=1))
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
    # Field/current derivative has a closed form because the field is uniform.
    jac = jax.jacfwd(run, argnums=1)(7., .001)
    np.testing.assert_allclose(jac, actual * (1j * GAMMA_RAD_S_T * 1e-6 * times),
                               rtol=3e-6, atol=2e-5)
    grad_c = jax.grad(lambda c: jnp.sum(jnp.real(run(c, .001))))(7.)
    assert np.isfinite(grad_c) and abs(grad_c) > 1
