"""TDD tests for Philips SDAT/SPAR reader."""
import numpy as np
import pytest
from pathlib import Path
import tempfile


def make_spar_content(te=68.0, tr=2000.0, samples=2048, rows=320, frequency=127.8e6):
    """Create synthetic SPAR file content."""
    return f"""! Philips SPAR file
! Generated for testing
examination_name : test_mega
scan_id : 1
samples : {samples}
rows : {rows}
synthesizer_frequency : {frequency}
echo_time : {te}
repetition_time : {tr}
sample_frequency : 2000
nucleus : 1H
"""


def make_sdat_binary(n_points=2048, n_rows=320):
    """Create synthetic SDAT binary data (VAX float32 pairs)."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal(n_points * n_rows) + 1j * rng.standard_normal(n_points * n_rows)
    # SDAT stores as interleaved float32 (real, imag)
    interleaved = np.zeros(n_points * n_rows * 2, dtype=np.float32)
    interleaved[0::2] = np.real(data).astype(np.float32)
    interleaved[1::2] = np.imag(data).astype(np.float32)
    return interleaved.tobytes(), data.reshape(n_rows, n_points)


class TestSparParser:
    def test_read_spar_metadata(self):
        from mrs_jax.io_philips import parse_spar
        with tempfile.NamedTemporaryFile(suffix='.SPAR', mode='w', delete=False) as f:
            f.write(make_spar_content(te=68.0, tr=2000.0, samples=2048, rows=320))
            f.flush()
            meta = parse_spar(f.name)
        assert meta['samples'] == 2048
        assert meta['rows'] == 320
        assert abs(meta['echo_time'] - 68.0) < 0.1
        assert abs(meta['repetition_time'] - 2000.0) < 0.1

    def test_spar_parser_handles_comments(self):
        from mrs_jax.io_philips import parse_spar
        content = "! This is a comment\nsamples : 1024\n! Another comment\nrows : 64\n"
        with tempfile.NamedTemporaryFile(suffix='.SPAR', mode='w', delete=False) as f:
            f.write(content)
            f.flush()
            meta = parse_spar(f.name)
        assert meta['samples'] == 1024
        assert meta['rows'] == 64


class TestSdatReader:
    def test_read_sdat_complex(self):
        from mrs_jax.io_philips import read_sdat
        binary, expected = make_sdat_binary(n_points=1024, n_rows=16)
        with tempfile.NamedTemporaryFile(suffix='.SDAT', delete=False) as f:
            f.write(binary)
            f.flush()
            data = read_sdat(f.name, n_points=1024, n_rows=16)
        assert data.shape == (16, 1024)
        assert np.iscomplexobj(data)
        np.testing.assert_allclose(data, expected, atol=1e-6)


class TestReadPhilips:
    def test_read_philips_returns_mrsdata(self):
        from mrs_jax.io_philips import read_philips
        # Create paired SDAT + SPAR
        binary, _ = make_sdat_binary(n_points=2048, n_rows=64)
        with tempfile.TemporaryDirectory() as tmpdir:
            sdat_path = Path(tmpdir) / "test.SDAT"
            spar_path = Path(tmpdir) / "test.SPAR"
            sdat_path.write_bytes(binary)
            spar_path.write_text(make_spar_content(samples=2048, rows=64))
            mrd = read_philips(str(sdat_path))
        assert mrd.data.shape[0] == 2048  # n_spec first
        assert mrd.dwell_time > 0
        assert mrd.centre_freq > 0

    def test_read_philips_nonexistent_raises(self):
        from mrs_jax.io_philips import read_philips
        with pytest.raises(FileNotFoundError):
            read_philips("/nonexistent/path/data.SDAT")
