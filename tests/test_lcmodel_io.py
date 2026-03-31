"""TDD tests for LCModel .RAW/.BASIS I/O."""
import numpy as np
import pytest
from pathlib import Path
import tempfile


class TestWriteRaw:
    def test_write_raw(self):
        from mrs_jax.io_lcmodel import write_raw
        fid = np.exp(2j * np.pi * 50 * np.arange(1024) * 2.5e-4)
        with tempfile.NamedTemporaryFile(suffix='.RAW', delete=False) as f:
            write_raw(f.name, fid, dwell_time=2.5e-4, te=30.0)
            assert Path(f.name).stat().st_size > 0


class TestReadRaw:
    def test_read_raw_roundtrip(self):
        from mrs_jax.io_lcmodel import write_raw, read_raw
        fid = np.exp(2j * np.pi * 50 * np.arange(512) * 2.5e-4) * 100
        with tempfile.NamedTemporaryFile(suffix='.RAW', delete=False) as f:
            write_raw(f.name, fid, dwell_time=2.5e-4, te=30.0)
            fid_back, meta = read_raw(f.name)
        np.testing.assert_allclose(np.real(fid), np.real(fid_back), atol=1e-3)
        np.testing.assert_allclose(np.imag(fid), np.imag(fid_back), atol=1e-3)
        assert abs(meta['te'] - 30.0) < 0.1


class TestWriteBasis:
    def test_write_basis(self):
        from mrs_jax.io_lcmodel import write_basis
        n = 512
        t = np.arange(n) * 2.5e-4
        basis = {
            'NAA': np.exp(2j * np.pi * -130 * t) * np.exp(-3 * t),
            'Cr': np.exp(2j * np.pi * -70 * t) * np.exp(-4 * t),
        }
        with tempfile.NamedTemporaryFile(suffix='.BASIS', delete=False) as f:
            write_basis(f.name, basis, dwell_time=2.5e-4, te=30.0)
            content = Path(f.name).read_text()
        assert 'NAA' in content
        assert 'Cr' in content


class TestReadBasis:
    def test_read_basis_roundtrip(self):
        from mrs_jax.io_lcmodel import write_basis, read_basis
        n = 256
        t = np.arange(n) * 2.5e-4
        basis_orig = {
            'NAA': np.exp(2j * np.pi * -130 * t) * np.exp(-3 * t) * 50,
            'Cr': np.exp(2j * np.pi * -70 * t) * np.exp(-4 * t) * 40,
        }
        with tempfile.NamedTemporaryFile(suffix='.BASIS', delete=False) as f:
            write_basis(f.name, basis_orig, dwell_time=2.5e-4, te=30.0)
            basis_back = read_basis(f.name)
        assert set(basis_back.keys()) == {'NAA', 'Cr'}
        for name in basis_orig:
            np.testing.assert_allclose(
                np.real(basis_orig[name]), np.real(basis_back[name]), atol=1e-3
            )


ISMRM_RAW = Path("/home/mhough/dev/neurojax/tests/data/mrs/ismrm_fitting_challenge/repo/datasets_LCModel")
ISMRM_BASIS = Path("/home/mhough/dev/neurojax/tests/data/mrs/ismrm_fitting_challenge/repo/basisset_LCModel")


@pytest.mark.skipif(not ISMRM_RAW.exists(), reason="ISMRM data not available")
class TestReadISMRMRaw:
    def test_read_ismrm_raw(self):
        from mrs_jax.io_lcmodel import read_raw
        raw_file = list(ISMRM_RAW.glob("*.RAW"))[0]
        fid, meta = read_raw(str(raw_file))
        assert fid.shape[0] > 100
        assert np.iscomplexobj(fid)


@pytest.mark.skipif(not ISMRM_BASIS.exists(), reason="ISMRM basis not available")
class TestReadISMRMBasis:
    def test_read_ismrm_basis(self):
        from mrs_jax.io_lcmodel import read_raw
        basis_files = list(ISMRM_BASIS.glob("*.RAW"))
        assert len(basis_files) > 0
        fid, meta = read_raw(str(basis_files[0]))
        assert fid.shape[0] > 100
