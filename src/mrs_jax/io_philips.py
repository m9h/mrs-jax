"""Philips SDAT/SPAR MRS reader.

Reads Philips MRS data from paired .SDAT (binary) and .SPAR (text parameter)
files, returning a standardized MRSData object.
"""

import numpy as np
from pathlib import Path
from mrs_jax.io import MRSData


def parse_spar(filepath: str) -> dict:
    """Parse a Philips .SPAR parameter file.

    SPAR files are text files with key : value pairs.
    Lines starting with ! are comments.

    Returns
    -------
    meta : dict
        Parameter dictionary with typed values (float/int/str).
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"SPAR file not found: {filepath}")

    meta = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('!'):
                continue
            if ':' in line:
                key, _, val = line.partition(':')
                key = key.strip()
                val = val.strip()
                # Try to convert to number
                try:
                    if '.' in val:
                        meta[key] = float(val)
                    else:
                        meta[key] = int(val)
                except ValueError:
                    meta[key] = val
    return meta


def read_sdat(filepath: str, n_points: int, n_rows: int) -> np.ndarray:
    """Read Philips .SDAT binary file.

    SDAT stores data as interleaved float32 (real, imag) pairs,
    organized as (n_rows, n_points) complex values.

    Parameters
    ----------
    filepath : str
        Path to .SDAT file.
    n_points : int
        Number of spectral points per row.
    n_rows : int
        Number of rows (averages/dynamics).

    Returns
    -------
    data : ndarray, shape (n_rows, n_points), complex64
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"SDAT file not found: {filepath}")

    raw = np.fromfile(str(filepath), dtype=np.float32)
    expected = n_points * n_rows * 2  # real + imag

    if len(raw) < expected:
        raise ValueError(
            f"SDAT file too small: expected {expected} float32 values, "
            f"got {len(raw)}"
        )

    raw = raw[:expected]
    complex_data = raw[0::2] + 1j * raw[1::2]
    return complex_data.reshape(n_rows, n_points)


def read_philips(sdat_path: str) -> MRSData:
    """Read Philips MRS from paired SDAT/SPAR files.

    Automatically finds the .SPAR file by replacing the .SDAT extension.

    Parameters
    ----------
    sdat_path : str
        Path to the .SDAT file.

    Returns
    -------
    MRSData
        Standardized MRS data container.

    Raises
    ------
    FileNotFoundError
        If SDAT or SPAR file not found.
    """
    sdat_path = Path(sdat_path)
    if not sdat_path.exists():
        raise FileNotFoundError(f"SDAT file not found: {sdat_path}")

    # Find SPAR file (try multiple case variants)
    spar_path = None
    for ext in ['.SPAR', '.spar', '.Spar']:
        candidate = sdat_path.with_suffix(ext)
        if candidate.exists():
            spar_path = candidate
            break

    # Also try replacing full extension
    if spar_path is None:
        stem = sdat_path.stem
        for ext in ['.SPAR', '.spar']:
            candidate = sdat_path.parent / (stem + ext)
            if candidate.exists():
                spar_path = candidate
                break

    if spar_path is None:
        raise FileNotFoundError(
            f"Cannot find SPAR file for {sdat_path}. "
            f"Tried: {sdat_path.with_suffix('.SPAR')}, {sdat_path.with_suffix('.spar')}"
        )

    # Parse parameters
    meta = parse_spar(str(spar_path))

    n_points = int(meta.get('samples', 2048))
    n_rows = int(meta.get('rows', 1))

    # Read binary data
    data = read_sdat(str(sdat_path), n_points, n_rows)

    # Transpose to (n_spec, n_rows) — mrs-jax convention: spectral dim first
    data = data.T  # (n_points, n_rows)

    # Extract metadata
    centre_freq = float(meta.get('synthesizer_frequency', 127.8e6))
    te = float(meta.get('echo_time', 0))
    tr = float(meta.get('repetition_time', 0))
    sample_freq = float(meta.get('sample_frequency', 2000))
    dwell_time = 1.0 / sample_freq if sample_freq > 0 else 2.5e-4

    # Field strength from frequency
    field_strength = centre_freq / (42.576e6) if centre_freq > 1e6 else 3.0

    return MRSData(
        data=data,
        dwell_time=dwell_time,
        centre_freq=centre_freq,
        te=te,
        tr=tr,
        field_strength=field_strength,
        n_coils=1,  # SDAT is already coil-combined
        n_averages=n_rows,
        dim_info={'spec': 0, 'dyn': 1},
    )
