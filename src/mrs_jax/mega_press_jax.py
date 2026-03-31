"""JAX-accelerated MEGA-PRESS spectral editing pipeline.

Provides JAX equivalents of the core NumPy functions in mega_press.py,
enabling jit compilation, vmap over subjects, and automatic differentiation.

The functions produce numerically identical results to their NumPy
counterparts (within floating-point tolerance).
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple


class MegaPressResult(NamedTuple):
    """Result of MEGA-PRESS processing (JAX arrays)."""
    diff: jnp.ndarray
    edit_on: jnp.ndarray
    edit_off: jnp.ndarray
    sum_spec: jnp.ndarray
    freq_shifts: jnp.ndarray
    phase_shifts: jnp.ndarray
    rejected: jnp.ndarray
    n_averages: int
    dwell_time: float
    bandwidth: float


def coil_combine_svd(data: jnp.ndarray) -> jnp.ndarray:
    """Combine multi-coil data using SVD (first singular vector).

    Parameters
    ----------
    data : jnp.ndarray, shape (n_spec, n_coils, ...)
        Multi-coil spectral data.

    Returns
    -------
    combined : jnp.ndarray, shape (n_spec, ...)
        Coil-combined data.
    """
    orig_shape = data.shape
    n_spec, n_coils = orig_shape[:2]
    extra_dims = orig_shape[2:]

    # Reshape to (n_spec, n_coils, n_extra)
    data_2d = data.reshape(n_spec, n_coils, -1)

    # Use first transient for weight estimation
    ref = data_2d[:, :, 0]  # (n_spec, n_coils)

    # SVD on coil dimension
    U, S, Vh = jnp.linalg.svd(ref.T, full_matrices=False)
    weights = U[:, 0].conj()  # First right singular vector

    # Phase-align weights to first coil
    weights = weights * jnp.exp(-1j * jnp.angle(weights[0]))

    # Apply weights
    combined = jnp.einsum('c,sc...->s...', weights, data_2d)
    return combined.reshape((n_spec,) + extra_dims)


def apply_correction(
    fid: jnp.ndarray,
    freq_shift: float,
    phase_shift: float,
    dwell_time: float,
) -> jnp.ndarray:
    """Apply frequency and phase correction to a single FID."""
    t = jnp.arange(fid.shape[0]) * dwell_time
    return fid * jnp.exp(2j * jnp.pi * freq_shift * t + 1j * phase_shift)


def reject_outliers(
    fids: jnp.ndarray,
    dwell_time: float,
    threshold: float = 3.0,
) -> jnp.ndarray:
    """Reject transients with outlier residuals (JAX version).

    Returns a boolean mask where True indicates rejected transients.
    Note: This uses a fixed-shape computation suitable for jit.
    """
    mean_fid = fids.mean(axis=1)
    residuals = jnp.sqrt(jnp.mean(jnp.abs(fids - mean_fid[:, None]) ** 2, axis=0))
    median_res = jnp.median(residuals)
    mad = jnp.median(jnp.abs(residuals - median_res))
    z_scores = 0.6745 * (residuals - median_res) / (mad + 1e-10)
    return jnp.abs(z_scores) > threshold


def spectral_registration_jax(
    fid: jnp.ndarray,
    reference: jnp.ndarray,
    dwell_time: float,
    centre_freq: float = 123.0e6,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """JAX-compatible spectral registration via FFT cross-correlation.

    Estimates frequency and phase shift of fid relative to reference
    using cross-spectral analysis. Fully jit/vmap/grad compatible.

    Parameters
    ----------
    fid : jnp.ndarray, shape (n_spec,)
    reference : jnp.ndarray, shape (n_spec,)
    dwell_time : float
    centre_freq : float

    Returns
    -------
    freq_shift : jnp.ndarray, scalar
        Frequency correction in Hz.
    phase_shift : jnp.ndarray, scalar
        Phase correction in radians.
    """
    n = fid.shape[0]
    t = jnp.arange(n) * dwell_time

    # Vectorized grid search over frequency shifts (±20 Hz, 0.2 Hz steps)
    freq_grid = jnp.linspace(-20.0, 20.0, 201)

    # For each candidate freq, apply shift and compute correlation with reference
    # Phase ramp: exp(2j*pi*df*t) for each df
    phase_ramps = jnp.exp(2j * jnp.pi * freq_grid[:, None] * t[None, :])  # (201, n)

    # Apply each shift to fid
    shifted_fids = fid[None, :] * phase_ramps  # (201, n)

    # FFT both
    spec_ref = jnp.fft.fftshift(jnp.fft.fft(reference))
    spec_shifted = jnp.fft.fftshift(jnp.fft.fft(shifted_fids, axis=1), axes=1)  # (201, n)

    # Cross-correlation: sum(ref* . shifted) for each candidate
    cross = jnp.sum(spec_ref.conj()[None, :] * spec_shifted, axis=1)  # (201,)

    # Best frequency: maximize real part of cross-correlation (= minimize phase mismatch)
    best_idx = jnp.argmax(jnp.real(cross))
    freq_shift = freq_grid[best_idx]

    # Optimal phase at best frequency
    phase_shift = -jnp.angle(cross[best_idx])

    return freq_shift, phase_shift


def _align_single_transient(carry, fid_pair):
    """Align a single ON/OFF pair. For use with jax.lax.scan."""
    ref, dwell_time, centre_freq = carry
    fid_off, fid_on = fid_pair

    df, dp = spectral_registration_jax(fid_off, ref, dwell_time, centre_freq)
    t = jnp.arange(fid_off.shape[0]) * dwell_time
    correction = jnp.exp(2j * jnp.pi * df * t + 1j * dp)

    aligned_off = fid_off * correction
    aligned_on = fid_on * correction

    return carry, (aligned_on, aligned_off, df, dp)


def process_mega_press(
    data: jnp.ndarray,
    dwell_time: float,
    centre_freq: float = 123.0e6,
    align: bool = False,
    reject: bool = False,
    reject_threshold: float = 3.0,
) -> MegaPressResult:
    """Process MEGA-PRESS data from raw multi-coil to difference spectrum.

    Parameters
    ----------
    data : jnp.ndarray, shape (n_spec, n_coils, n_edit, n_dyn) or (n_spec, n_edit, n_dyn)
        Raw MEGA-PRESS data. n_edit=2 (edit-ON, edit-OFF).
    dwell_time : float
        Dwell time in seconds.
    centre_freq : float
        Spectrometer frequency in Hz.
    align : bool
        Whether to perform frequency/phase alignment.
        NOTE: alignment is not yet implemented in the JAX version;
        pass align=False for jit/vmap compatibility.
    reject : bool
        Whether to reject outlier transients.
        NOTE: rejection with dynamic shapes is not jit-compatible;
        pass reject=False for jit/vmap compatibility.
    reject_threshold : float
        Z-score threshold for outlier rejection.

    Returns
    -------
    MegaPressResult
    """
    bw = 1.0 / dwell_time

    # Step 1: Coil combination
    if data.ndim == 4:
        combined = coil_combine_svd(data)  # (n_spec, n_edit, n_dyn)
    elif data.ndim == 3:
        combined = data
    else:
        raise ValueError(f"Expected 3D or 4D data, got {data.ndim}D")

    n_spec = combined.shape[0]
    n_edit = combined.shape[1]
    n_dyn = combined.shape[2]

    edit_on = combined[:, 0, :]   # (n_spec, n_dyn)
    edit_off = combined[:, 1, :]  # (n_spec, n_dyn)

    # Step 2: Frequency/phase alignment
    if align:
        ref = edit_off.mean(axis=1)  # Reference from mean of edit-OFF
        carry = (ref, dwell_time, centre_freq)

        # Stack (OFF, ON) pairs for scan: shape (n_dyn, n_spec) each
        fid_pairs = (edit_off.T, edit_on.T)  # Each (n_dyn, n_spec)

        _, (aligned_on_T, aligned_off_T, freq_arr, phase_arr) = jax.lax.scan(
            _align_single_transient, carry, fid_pairs
        )

        edit_on = aligned_on_T.T   # Back to (n_spec, n_dyn)
        edit_off = aligned_off_T.T
        freq_shifts = jnp.concatenate([freq_arr, freq_arr])
        phase_shifts = jnp.concatenate([phase_arr, phase_arr])
    else:
        freq_shifts = jnp.zeros(2 * n_dyn)
        phase_shifts = jnp.zeros(2 * n_dyn)

    # Step 3: Outlier rejection (not implemented for JAX jit compat)
    rejected = jnp.zeros(2 * n_dyn, dtype=bool)

    # Step 4: Average and compute difference
    avg_on = edit_on.mean(axis=1)
    avg_off = edit_off.mean(axis=1)
    diff = avg_on - avg_off
    sum_spec = avg_on + avg_off

    n_used = n_dyn

    return MegaPressResult(
        diff=diff,
        edit_on=avg_on,
        edit_off=avg_off,
        sum_spec=sum_spec,
        freq_shifts=freq_shifts,
        phase_shifts=phase_shifts,
        rejected=rejected,
        n_averages=n_used,
        dwell_time=dwell_time,
        bandwidth=bw,
    )
