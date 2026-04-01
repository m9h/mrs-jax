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


# Literature metabolite concentrations (mM) per tissue type
# Sources: Govindaraju et al. NMR Biomed 2000, Maudsley et al. MRM 2009
# FAST labels: 0=background, 1=CSF, 2=GM, 3=WM
METABOLITE_CONCENTRATIONS = {
    #             BG    CSF    GM     WM
    'NAA':      [0.0,  0.0,  12.5,  10.0],
    'NAAG':     [0.0,  0.0,   2.0,   1.5],
    'Cr':       [0.0,  0.0,   8.0,   6.0],
    'PCr':      [0.0,  0.0,   4.5,   4.0],
    'Cho':      [0.0,  0.0,   2.5,   3.0],
    'mIns':     [0.0,  0.0,   7.0,   4.0],
    'Glu':      [0.0,  0.0,  11.0,   6.0],
    'Gln':      [0.0,  0.0,   4.0,   3.0],
    'GABA':     [0.0,  0.0,   1.5,   1.0],
    'GSH':      [0.0,  0.0,   2.0,   1.5],
    'Tau':      [0.0,  0.0,   1.5,   1.0],
    'Lac':      [0.0,  0.0,   0.5,   0.3],
    'Asp':      [0.0,  0.0,   2.0,   1.5],
}

# T1 and T2* per tissue at 3T (seconds)
TISSUE_T1 = {0: 0.0, 1: 4.16, 2: 1.33, 3: 0.83}
TISSUE_T2STAR = {0: 0.0, 1: 1.65, 2: 0.040, 3: 0.045}


def load_wand_phantom(
    segmentation_path: str,
    resolution_mm: float = None,
) -> TissueModel:
    """Load a realistic brain phantom from WAND FAST segmentation.

    Creates a TissueModel with per-voxel metabolite concentrations
    based on tissue type (GM/WM/CSF) and literature values.

    Parameters
    ----------
    segmentation_path : str
        Path to FAST tissue segmentation NIfTI (labels: 0=bg, 1=CSF, 2=GM, 3=WM).
    resolution_mm : float, optional
        If given, downsample the phantom to this isotropic resolution
        (in mm) for faster simulation.

    Returns
    -------
    TissueModel
    """
    import nibabel as nib
    from scipy.ndimage import zoom

    img = nib.load(segmentation_path)
    seg = img.get_fdata().astype(int)
    voxel_size = img.header.get_zooms()[:3]

    # Downsample if requested
    if resolution_mm is not None:
        scale = tuple(v / resolution_mm for v in voxel_size)
        seg = zoom(seg, scale, order=0)  # Nearest-neighbor for labels
        voxel_size = (resolution_mm, resolution_mm, resolution_mm)

    # Build metabolite concentration maps
    metab_maps = {}
    for metab, concs in METABOLITE_CONCENTRATIONS.items():
        m = np.zeros(seg.shape, dtype=np.float32)
        for label, c in enumerate(concs):
            m[seg == label] = c
        metab_maps[metab] = m

    # Build T1 and T2* maps
    t1_map = np.zeros(seg.shape, dtype=np.float32)
    t2star_map = np.zeros(seg.shape, dtype=np.float32)
    for label in TISSUE_T1:
        t1_map[seg == label] = TISSUE_T1[label]
        t2star_map[seg == label] = TISSUE_T2STAR[label]

    # Set minimum T2* to avoid division by zero
    t2star_map = np.maximum(t2star_map, 1e-6)

    return TissueModel(
        tissue_map=seg,
        metabolite_maps=metab_maps,
        t1_map=t1_map,
        t2star_map=t2star_map,
        b0_shift_map=np.zeros(seg.shape, dtype=np.float32),
        voxel_size=voxel_size,
    )


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


# =========================================================================
# EPSI trajectory and simulation
# =========================================================================

@dataclass
class EPSITrajectory:
    """EPSI readout trajectory.

    EPSI uses an oscillating read gradient to encode one spatial dimension
    and the spectral dimension simultaneously. Each oscillation lobe
    traverses k-space in the read direction; successive lobes alternate
    direction (bipolar). The spectral dimension is encoded by the time
    between corresponding k-space positions across lobes.

    Attributes
    ----------
    k_readout : ndarray, shape (n_spatial * n_spectral,)
        k-space position (1/mm) at each sample point along the readout.
    n_spatial : int
        Number of spatial encoding points per lobe.
    n_spectral : int
        Number of spectral points (= number of gradient lobes).
    dwell_time : float
        Spectral dwell time in seconds (time between lobes).
    fov_mm : float
        Field of view in mm.
    """
    k_readout: np.ndarray
    n_spatial: int
    n_spectral: int
    dwell_time: float
    fov_mm: float


def generate_epsi_trajectory(
    n_spatial: int = 32,
    n_spectral: int = 256,
    dwell_time: float = 2.5e-4,
    fov_mm: float = 240.0,
) -> EPSITrajectory:
    """Generate an EPSI oscillating readout trajectory.

    The trajectory consists of n_spectral bipolar gradient lobes,
    each traversing n_spatial points in k-space. Odd lobes go left→right,
    even lobes go right→left.

    Parameters
    ----------
    n_spatial : int
        Spatial resolution (points per gradient lobe).
    n_spectral : int
        Number of spectral points (gradient lobes).
    dwell_time : float
        Time between corresponding k-space positions across lobes (seconds).
    fov_mm : float
        Field of view in mm.

    Returns
    -------
    EPSITrajectory
    """
    # k-space extent: -k_max to +k_max
    k_max = n_spatial / (2.0 * fov_mm)  # 1/mm

    # One lobe: linear ramp from -k_max to +k_max (or reversed)
    k_lobe_forward = np.linspace(-k_max, k_max, n_spatial)
    k_lobe_reverse = np.linspace(k_max, -k_max, n_spatial)

    # Stack lobes: alternating forward/reverse
    k_all = []
    for i in range(n_spectral):
        if i % 2 == 0:
            k_all.append(k_lobe_forward)
        else:
            k_all.append(k_lobe_reverse)

    k_readout = np.concatenate(k_all)

    return EPSITrajectory(
        k_readout=k_readout,
        n_spatial=n_spatial,
        n_spectral=n_spectral,
        dwell_time=dwell_time,
        fov_mm=fov_mm,
    )


def simulate_epsi(
    model: TissueModel,
    basis: dict,
    traj: EPSITrajectory,
) -> np.ndarray:
    """Simulate EPSI signal from a tissue model along the readout direction.

    Computes signal for a 1D readout (x-direction) at each time point,
    using the EPSI trajectory to determine k-space position.

    Parameters
    ----------
    model : TissueModel
        Tissue model (uses first spatial dimension as readout).
    basis : dict[str, ndarray]
        Metabolite basis spectra.
    traj : EPSITrajectory

    Returns
    -------
    signal : ndarray, shape (n_spatial * n_spectral,)
        Raw EPSI signal along the oscillating readout.
    """
    nx = model.shape[0]
    n_total = traj.n_spatial * traj.n_spectral

    # Spatial positions along readout (x direction), in mm
    voxel_size_x = traj.fov_mm / nx if nx > 0 else 1.0
    x_positions = (np.arange(nx) - nx / 2) * voxel_size_x  # mm, centered

    # Time array: each sample has a time = lobe_index * dwell_time + intra-lobe offset
    # For simplicity, spectral time = lobe index * dwell_time
    lobe_indices = np.repeat(np.arange(traj.n_spectral), traj.n_spatial)
    t_spectral = lobe_indices * traj.dwell_time

    # Build voxel signals: for each x position, compute FID at each spectral time
    # Average over y, z dimensions (projection onto readout)
    signal = np.zeros(n_total, dtype=np.complex64)

    for ix in range(nx):
        # Sum metabolite contributions for this x position
        voxel_fid = np.zeros(n_total, dtype=np.complex64)

        for name, basis_fid in basis.items():
            if name not in model.metabolite_maps:
                continue
            # Concentration averaged over y, z at this x
            conc = model.metabolite_maps[name][ix, :, :].mean()
            if conc == 0:
                continue

            # Interpolate basis FID at spectral times
            n_basis = len(basis_fid)
            basis_times = np.arange(n_basis) * traj.dwell_time
            basis_real = np.interp(t_spectral, basis_times, np.real(basis_fid), right=0)
            basis_imag = np.interp(t_spectral, basis_times, np.imag(basis_fid), right=0)
            basis_interp = basis_real + 1j * basis_imag

            voxel_fid += conc * basis_interp

        # T2* decay
        t2star = model.t2star_map[ix, :, :].mean()
        t2star = max(t2star, 1e-6)
        decay = np.exp(-t_spectral / t2star)

        # B0 shift
        b0 = model.b0_shift_map[ix, :, :].mean()
        phase_mod = np.exp(2j * np.pi * b0 * t_spectral)

        # Spatial encoding: exp(-i * 2π * k * x)
        spatial_phase = np.exp(-2j * np.pi * traj.k_readout * x_positions[ix])

        signal += voxel_fid * decay * phase_mod * spatial_phase

    return signal


def reconstruct_epsi(
    signal: np.ndarray,
    traj: EPSITrajectory,
) -> np.ndarray:
    """Reconstruct EPSI signal to spatial-spectral image.

    Separates the interleaved spatial-spectral data, reverses even lobes,
    and applies FFT along the spatial dimension.

    Parameters
    ----------
    signal : ndarray, shape (n_spatial * n_spectral,)
        Raw EPSI signal.
    traj : EPSITrajectory

    Returns
    -------
    image : ndarray, shape (n_spatial, n_spectral)
        Reconstructed spatial-spectral image.
    """
    ns = traj.n_spatial
    nf = traj.n_spectral

    # Reshape to (n_spectral, n_spatial) — one row per lobe
    data = signal.reshape(nf, ns)

    # Reverse even-numbered lobes (bipolar correction)
    data[1::2, :] = data[1::2, ::-1]

    # Now each column is a spatial position, each row is a spectral time point
    # Transpose to (n_spatial, n_spectral)
    data = data.T

    # FFT along spatial dimension → image space
    # (Already in k-space along spatial dim from the readout encoding)
    image = np.fft.ifft(data, axis=0)

    return image


def simulate_epsi_from_arrays(
    concentrations,
    basis_fid,
    t2star,
    b0,
    traj: EPSITrajectory,
):
    """Differentiable EPSI simulation from JAX arrays (1D readout).

    Parameters
    ----------
    concentrations : jnp.ndarray, shape (nx,)
        Concentration per voxel along readout.
    basis_fid : jnp.ndarray, shape (n_basis,)
        Basis FID for one metabolite.
    t2star : jnp.ndarray, shape (nx,)
        T2* per voxel.
    b0 : jnp.ndarray, shape (nx,)
        B0 shift per voxel (Hz).
    traj : EPSITrajectory

    Returns
    -------
    signal : jnp.ndarray, shape (n_spatial * n_spectral,)
    """
    import jax.numpy as jnp

    nx = concentrations.shape[0]
    ns = traj.n_spatial
    nf = traj.n_spectral
    n_total = ns * nf

    voxel_size_x = traj.fov_mm / nx
    x_pos = (jnp.arange(nx) - nx / 2) * voxel_size_x

    lobe_indices = jnp.repeat(jnp.arange(nf), ns)
    t_spectral = lobe_indices * traj.dwell_time
    k = jnp.array(traj.k_readout)

    # Interpolate basis at spectral times
    n_basis = basis_fid.shape[0]
    basis_times = jnp.arange(n_basis) * traj.dwell_time
    # Simple: use the lobe index to sample the basis
    lobe_idx_clipped = jnp.clip(lobe_indices, 0, n_basis - 1)
    basis_samples = basis_fid[lobe_idx_clipped]

    # Vectorized over voxels
    # (nx, 1) * (1, n_total) broadcasting
    conc = concentrations[:, None]
    t2s = t2star[:, None]
    b0v = b0[:, None]
    xp = x_pos[:, None]

    decay = jnp.exp(-t_spectral[None, :] / (t2s + 1e-10))
    phase_b0 = jnp.exp(2j * jnp.pi * b0v * t_spectral[None, :])
    spatial = jnp.exp(-2j * jnp.pi * k[None, :] * xp)

    voxel_signals = conc * basis_samples[None, :] * decay * phase_b0 * spatial
    signal = jnp.sum(voxel_signals, axis=0)

    return signal
