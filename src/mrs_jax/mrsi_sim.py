"""MRSI/EPSI simulator — differentiable 'POSSUM for MRS'.

A forward model that generates synthetic MRSI k-space data from a
segmented brain object with per-voxel metabolite concentrations. Built
in NumPy with a JAX-compatible version for differentiable simulation.

The signal equation for each k-space point (kx, ky, kz, t) is:

    S(k, t) = Σ_r [ Σ_m C_m(r) · B_m(t) ] · exp(-t/T2*(r))
              · exp(i·2π·Δf(r)·t) · exp(-i·2π·k·r)

where:
    r = spatial position
    m = metabolite index
    C_m(r) = concentration of metabolite m at position r
    B_m(t) = basis spectrum (FID) of metabolite m
    T2*(r) = transverse relaxation at position r
    Δf(r) = B0 field inhomogeneity at position r
    k = spatial frequency encoding

For phase-encoded MRSI (CSI), the spatial encoding is a DFT —
equivalent to acquiring one FID per phase-encode step and reconstructing
with inverse FFT. The forward model is therefore:

    kspace(kx, ky, kz, t) = FFT_xyz[ voxel_signal(x, y, z, t) ]

where voxel_signal is the sum of metabolite basis spectra weighted by
concentration and modulated by T2* decay and B0 shift.

References:
    Maudsley AA et al. (2009) Mapping of brain metabolite distributions
    by volumetric proton MR spectroscopic imaging. MRM 61:548-559.

    Drobnjak I et al. (2006) POSSUM: An MRI simulator for the study of
    brain motion. MRM 56:439-448.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class TissueModel:
    """Multi-compartment tissue model for MRSI simulation.

    Each voxel has:
    - A tissue label (GM, WM, CSF, etc.)
    - Per-metabolite concentration maps
    - T1, T2*, B0 shift maps

    Attributes
    ----------
    tissue_map : ndarray, shape (nx, ny, nz)
        Integer tissue labels.
    metabolite_maps : dict[str, ndarray]
        Metabolite name -> concentration array (same spatial shape).
    t1_map : ndarray
        T1 relaxation time in seconds.
    t2star_map : ndarray
        T2* relaxation time in seconds.
    b0_shift_map : ndarray
        B0 field shift in Hz.
    voxel_size : tuple[float, float, float]
        Voxel dimensions in mm (x, y, z).
    """
    tissue_map: np.ndarray
    metabolite_maps: dict
    t1_map: np.ndarray
    t2star_map: np.ndarray
    b0_shift_map: np.ndarray
    voxel_size: tuple = (3.0, 3.0, 3.0)

    @property
    def shape(self):
        return self.tissue_map.shape

    @property
    def n_metabolites(self):
        return len(self.metabolite_maps)


def make_lorentzian_basis(
    metabolite_shifts: dict[str, float],
    n_points: int = 1024,
    dwell_time: float = 2.5e-4,
    centre_freq: float = 123.25e6,
    linewidth_hz: float = 3.0,
) -> dict[str, np.ndarray]:
    """Create Lorentzian FID basis spectra for given metabolites.

    Parameters
    ----------
    metabolite_shifts : dict
        Metabolite name -> chemical shift in ppm.
    n_points : int
        Number of spectral points.
    dwell_time : float
        Dwell time in seconds.
    centre_freq : float
        Spectrometer frequency in Hz.
    linewidth_hz : float
        Lorentzian linewidth in Hz.

    Returns
    -------
    basis : dict[str, ndarray]
        Metabolite name -> complex FID array.
    """
    t = np.arange(n_points) * dwell_time
    basis = {}
    for name, ppm in metabolite_shifts.items():
        freq_hz = (ppm - 4.65) * (centre_freq / 1e6)
        fid = np.exp(2j * np.pi * freq_hz * t) * np.exp(-np.pi * linewidth_hz * t)
        basis[name] = fid.astype(np.complex64)
    return basis


def simulate_mrsi(
    model: TissueModel,
    basis: dict[str, np.ndarray],
    n_spectral: int = 1024,
    dwell_time: float = 2.5e-4,
) -> np.ndarray:
    """Simulate phase-encoded MRSI k-space data (NumPy).

    Generates the voxel-wise signal (sum of metabolite FIDs weighted by
    concentration, modulated by T2* and B0), then FFTs over spatial dims
    to produce k-space.

    Parameters
    ----------
    model : TissueModel
        Tissue model with metabolite maps and relaxation.
    basis : dict[str, ndarray]
        Metabolite basis spectra (FIDs).
    n_spectral : int
        Number of spectral (time) points.
    dwell_time : float
        Dwell time in seconds.

    Returns
    -------
    kspace : ndarray, shape (nx, ny, nz, n_spectral), complex
    """
    nx, ny, nz = model.shape
    t = np.arange(n_spectral) * dwell_time  # (n_spectral,)

    # T2* decay: exp(-t/T2*) per voxel
    # shape: (nx, ny, nz, 1) * (1, 1, 1, n_spectral)
    t2star = model.t2star_map[..., np.newaxis]  # (nx, ny, nz, 1)
    decay = np.exp(-t[np.newaxis, np.newaxis, np.newaxis, :] / (t2star + 1e-10))

    # B0 shift: exp(i*2π*Δf*t) per voxel
    b0 = model.b0_shift_map[..., np.newaxis]  # (nx, ny, nz, 1)
    phase_mod = np.exp(2j * np.pi * b0 * t[np.newaxis, np.newaxis, np.newaxis, :])

    # Accumulate signal from all metabolites
    signal = np.zeros((nx, ny, nz, n_spectral), dtype=np.complex64)
    for name, fid in basis.items():
        if name not in model.metabolite_maps:
            continue
        conc = model.metabolite_maps[name][..., np.newaxis]  # (nx, ny, nz, 1)
        # Basis FID broadcast: (1, 1, 1, n_spectral)
        basis_fid = fid[np.newaxis, np.newaxis, np.newaxis, :n_spectral]
        signal += conc * basis_fid * decay * phase_mod

    # FFT over spatial dimensions → k-space
    kspace = np.fft.fftn(signal, axes=(0, 1, 2))
    return kspace


def simulate_mrsi_jax(
    model: TissueModel,
    basis: dict[str, np.ndarray],
    n_spectral: int = 1024,
    dwell_time: float = 2.5e-4,
):
    """JAX-compatible MRSI simulation. Same as simulate_mrsi but returns JAX arrays."""
    import jax.numpy as jnp

    kspace_np = simulate_mrsi(model, basis, n_spectral, dwell_time)
    return jnp.array(kspace_np)


def simulate_mrsi_from_arrays(
    concentrations: 'jnp.ndarray',
    basis_fid: 'jnp.ndarray',
    t2star_map: 'jnp.ndarray',
    b0_shift_map: 'jnp.ndarray',
    n_spectral: int = 1024,
    dwell_time: float = 2.5e-4,
) -> 'jnp.ndarray':
    """Fully differentiable MRSI simulation from JAX arrays.

    This version takes raw JAX arrays (not TissueModel) so it can be
    used inside jax.grad, jax.jit, and jax.vmap.

    Parameters
    ----------
    concentrations : jnp.ndarray, shape (nx, ny, nz)
        Concentration map for a SINGLE metabolite.
    basis_fid : jnp.ndarray, shape (n_spectral,)
        Basis FID for that metabolite.
    t2star_map : jnp.ndarray, shape (nx, ny, nz)
        T2* map in seconds.
    b0_shift_map : jnp.ndarray, shape (nx, ny, nz)
        B0 shift in Hz.
    n_spectral : int
        Number of spectral points.
    dwell_time : float
        Dwell time in seconds.

    Returns
    -------
    kspace : jnp.ndarray, shape (nx, ny, nz, n_spectral), complex
    """
    import jax.numpy as jnp

    t = jnp.arange(n_spectral) * dwell_time

    # Expand spatial dims for broadcasting
    t2star = t2star_map[..., jnp.newaxis]
    b0 = b0_shift_map[..., jnp.newaxis]
    conc = concentrations[..., jnp.newaxis]

    decay = jnp.exp(-t[jnp.newaxis, jnp.newaxis, jnp.newaxis, :] / (t2star + 1e-10))
    phase_mod = jnp.exp(2j * jnp.pi * b0 * t[jnp.newaxis, jnp.newaxis, jnp.newaxis, :])
    basis_broadcast = basis_fid[jnp.newaxis, jnp.newaxis, jnp.newaxis, :n_spectral]

    signal = conc * basis_broadcast * decay * phase_mod

    kspace = jnp.fft.fftn(signal, axes=(0, 1, 2))
    return kspace
