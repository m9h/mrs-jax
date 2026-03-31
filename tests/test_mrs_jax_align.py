"""Tests for JAX spectral registration and full pipeline with alignment."""
import numpy as np
import pytest

import jax
import jax.numpy as jnp

from mrs_jax.mega_press_jax import (
    spectral_registration_jax,
    process_mega_press,
    apply_correction,
)


def make_singlet_jax(ppm, amplitude, lw, n_pts=2048, dwell=2.5e-4, cf=123.25e6):
    freq_hz = (ppm - 4.65) * (cf / 1e6)
    t = jnp.arange(n_pts) * dwell
    return amplitude * jnp.exp(2j * jnp.pi * freq_hz * t) * jnp.exp(-jnp.pi * lw * t)


def make_mega_data_jax(n_pts=2048, n_dyn=16, dwell=2.5e-4, cf=123.25e6,
                       noise=0.01, seed=42):
    rng = np.random.default_rng(seed)
    naa = np.array(make_singlet_jax(2.01, 10.0, 3.0, n_pts, dwell, cf))
    cr = np.array(make_singlet_jax(3.03, 8.0, 4.0, n_pts, dwell, cf))
    gaba = np.array(make_singlet_jax(3.01, 1.0, 8.0, n_pts, dwell, cf))
    on_sig = naa + cr + gaba
    off_sig = naa + cr - gaba
    data = np.zeros((n_pts, 2, n_dyn), dtype=complex)
    for d in range(n_dyn):
        n1 = noise * (rng.standard_normal(n_pts) + 1j * rng.standard_normal(n_pts))
        n2 = noise * (rng.standard_normal(n_pts) + 1j * rng.standard_normal(n_pts))
        data[:, 0, d] = on_sig + n1
        data[:, 1, d] = off_sig + n2
    return jnp.array(data), dwell, cf


class TestJaxSpectralRegNoShift:
    def test_jax_spectral_reg_no_shift(self):
        fid = make_singlet_jax(2.01, 1.0, 3.0)
        df, dp = spectral_registration_jax(fid, fid, 2.5e-4, 123.25e6)
        assert abs(float(df)) < 2.0
        assert abs(float(dp)) < 0.5


class TestJaxSpectralRegKnownShift:
    def test_jax_spectral_reg_known_shift(self):
        dwell = 2.5e-4
        cf = 123.25e6
        ref = make_singlet_jax(2.01, 1.0, 3.0, dwell=dwell, cf=cf)
        t = jnp.arange(2048) * dwell
        shifted = ref * jnp.exp(2j * jnp.pi * 5.0 * t)
        df, dp = spectral_registration_jax(shifted, ref, dwell, cf)
        assert abs(abs(float(df)) - 5.0) < 2.0


class TestJaxSpectralRegMatchesNumpy:
    def test_jax_spectral_reg_matches_numpy(self):
        from mrs_jax.mega_press import spectral_registration as np_reg
        dwell = 2.5e-4
        cf = 123.25e6
        ref_np = np.array(make_singlet_jax(2.01, 1.0, 3.0, dwell=dwell, cf=cf))
        t = np.arange(2048) * dwell
        shifted_np = ref_np * np.exp(2j * np.pi * 3.0 * t)

        df_np, _ = np_reg(shifted_np, ref_np, dwell, centre_freq=cf)
        df_jax, _ = spectral_registration_jax(
            jnp.array(shifted_np), jnp.array(ref_np), dwell, cf
        )
        assert abs(abs(float(df_jax)) - abs(float(df_np))) < 3.0


class TestJaxFullPipelineWithAlign:
    def test_jax_full_pipeline_with_align(self):
        data, dwell, cf = make_mega_data_jax()
        result = process_mega_press(data, dwell, cf, align=True, reject=False)
        assert result.diff.shape == (2048,)
        assert result.edit_on.shape == (2048,)


class TestJaxVmapWithAlign:
    def test_jax_vmap_with_align(self):
        subjects = []
        for seed in range(3):
            d, dw, cf = make_mega_data_jax(n_dyn=8, seed=seed)
            subjects.append(d)
        batch = jnp.stack(subjects)

        def proc(d):
            return process_mega_press(d, 2.5e-4, 123.25e6, align=True, reject=False)

        batch_proc = jax.vmap(proc)
        results = batch_proc(batch)
        assert results.diff.shape == (3, 2048)


class TestJaxJitWithAlign:
    def test_jax_jit_with_align(self):
        data, dwell, cf = make_mega_data_jax(n_dyn=8)

        @jax.jit
        def run(d):
            return process_mega_press(d, dwell, cf, align=True, reject=False)

        result = run(data)
        assert result.diff.shape == (2048,)
