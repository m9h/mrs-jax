"""LCModel .RAW and .BASIS file I/O.

Supports reading and writing LCModel formats for interoperability
with the most widely used MRS fitting tool.

LCModel .RAW format:
    $SEQPAR block with sequence parameters (ECHOT, SEQ)
    $NMID block with dataset metadata (ID, FMTDAT, TRTEFACT, VOLUME)
    Complex data as paired real/imag values in FORTRAN format
"""

import re
import numpy as np
from pathlib import Path


def write_raw(
    filepath: str,
    fid: np.ndarray,
    dwell_time: float,
    te: float = 30.0,
    seq: str = 'PRESS',
    metab_id: str = 'unknown',
    volume: float = 1.0,
):
    """Write a complex FID as LCModel .RAW format.

    Parameters
    ----------
    filepath : str
        Output file path.
    fid : ndarray
        Complex FID array.
    dwell_time : float
        Dwell time in seconds.
    te : float
        Echo time in milliseconds.
    seq : str
        Sequence name.
    metab_id : str
        Metabolite or dataset ID.
    volume : float
        Voxel volume scaling factor.
    """
    with open(filepath, 'w') as f:
        f.write(f" $SEQPAR\n")
        f.write(f" ECHOT = {te:.1f}\n")
        f.write(f" SEQ = '{seq}'\n")
        f.write(f" $END\n")
        f.write(f" $NMID\n")
        f.write(f" ID = '{metab_id}'\n")
        f.write(f" FMTDAT = '(2E15.6)'\n")
        f.write(f" TRTEFACT = {dwell_time:.10E}\n")
        f.write(f" VOLUME = {volume:.1f}\n")
        f.write(f" $END\n")
        for i in range(len(fid)):
            f.write(f"  {fid[i].real:15.6E}{fid[i].imag:15.6E}\n")


def _parse_inline_block(line: str, meta: dict):
    """Parse a single-line $BLOCK key=val, key=val $END."""
    # Remove $SEQPAR, $NMID, $END markers
    cleaned = re.sub(r'\$\w+', '', line).strip()
    # Split on commas or spaces, parse key=value pairs
    for part in re.split(r'[,\s]+', cleaned):
        if '=' in part:
            key, _, val = part.partition('=')
            key = key.strip()
            val = val.strip().strip("'\"")
            if not key:
                continue
            try:
                meta[key.lower()] = float(val)
            except ValueError:
                meta[key.lower()] = val


def read_raw(filepath: str) -> tuple[np.ndarray, dict]:
    """Read an LCModel .RAW file.

    Returns
    -------
    fid : ndarray
        Complex FID.
    meta : dict
        Metadata extracted from $SEQPAR and $NMID blocks.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"RAW file not found: {filepath}")

    with open(filepath, 'r') as f:
        lines = f.readlines()

    meta = {}
    data_lines = []
    in_block = False
    block_name = None

    past_header = False
    for line in lines:
        stripped = line.strip()

        # Detect header blocks
        if '$SEQPAR' in stripped:
            in_block = True
            # Check for inline params: $SEQPAR ... $END on one line
            if '$END' in stripped:
                _parse_inline_block(stripped, meta)
                in_block = False
            continue
        if '$NMID' in stripped:
            in_block = True
            if '$END' in stripped:
                _parse_inline_block(stripped, meta)
                in_block = False
                past_header = True
            continue
        if '$END' in stripped:
            in_block = False
            past_header = True
            continue

        if in_block:
            # Parse key = value
            match = re.match(r'\s*(\w+)\s*=\s*(.*)', stripped)
            if match:
                key = match.group(1).strip()
                val = match.group(2).strip().strip("'\"").rstrip(',')
                try:
                    val = float(val)
                except ValueError:
                    pass
                meta[key.lower()] = val
        elif past_header and stripped and not stripped.startswith('$'):
            data_lines.append(stripped)

    # Parse data
    values = []
    for line in data_lines:
        # Split by whitespace, parse as float pairs
        parts = line.split()
        for p in parts:
            try:
                values.append(float(p))
            except ValueError:
                pass

    # Pair up as complex
    n = len(values) // 2
    fid = np.array(values[:2*n:2]) + 1j * np.array(values[1:2*n:2])

    # Map common LCModel keys
    if 'echot' in meta:
        meta['te'] = meta['echot']
    if 'trtefact' in meta:
        meta['dwell_time'] = meta['trtefact']

    return fid, meta


def write_basis(
    filepath: str,
    basis_dict: dict[str, np.ndarray],
    dwell_time: float,
    te: float = 30.0,
    seq: str = 'PRESS',
):
    """Write a set of basis spectra as concatenated LCModel .RAW entries.

    Parameters
    ----------
    filepath : str
        Output file path.
    basis_dict : dict
        Mapping of metabolite name -> complex FID array.
    dwell_time : float
        Dwell time in seconds.
    te : float
        Echo time in milliseconds.
    seq : str
        Sequence name.
    """
    with open(filepath, 'w') as f:
        for name, fid in basis_dict.items():
            f.write(f" $SEQPAR\n")
            f.write(f" ECHOT = {te:.1f}\n")
            f.write(f" SEQ = '{seq}'\n")
            f.write(f" $END\n")
            f.write(f" $NMID\n")
            f.write(f" ID = '{name}'\n")
            f.write(f" FMTDAT = '(2E15.6)'\n")
            f.write(f" TRTEFACT = {dwell_time:.10E}\n")
            f.write(f" VOLUME = 1.0\n")
            f.write(f" $END\n")
            for i in range(len(fid)):
                f.write(f"  {fid[i].real:15.6E}{fid[i].imag:15.6E}\n")


def read_basis(filepath: str) -> dict[str, np.ndarray]:
    """Read a concatenated LCModel .BASIS file containing multiple metabolites.

    Returns
    -------
    basis : dict
        Mapping of metabolite name -> complex FID.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"BASIS file not found: {filepath}")

    with open(filepath, 'r') as f:
        content = f.read()

    # Split on $SEQPAR blocks
    entries = re.split(r'\s*\$SEQPAR\b', content)

    basis = {}
    for entry in entries:
        if not entry.strip():
            continue

        # Extract ID
        id_match = re.search(r"ID\s*=\s*'([^']+)'", entry)
        if not id_match:
            continue
        name = id_match.group(1)

        # Find data after $END of $NMID block
        parts = re.split(r'\$END', entry)
        if len(parts) < 3:
            # Only one $END — data is after it
            data_text = parts[-1]
        else:
            # Two $END blocks ($SEQPAR and $NMID) — data is after the last
            data_text = parts[-1]

        # Parse floats
        values = []
        for line in data_text.strip().split('\n'):
            line = line.strip()
            if not line or line.startswith('$'):
                continue
            for p in line.split():
                try:
                    values.append(float(p))
                except ValueError:
                    pass

        if len(values) >= 2:
            n = len(values) // 2
            fid = np.array(values[:2*n:2]) + 1j * np.array(values[1:2*n:2])
            basis[name] = fid

    return basis
