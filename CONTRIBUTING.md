# Contributing

Thank you for your interest in contributing to mrs-jax. This guide covers everything you need to get started.

## 1. Getting Started

We use [uv](https://docs.astral.sh/uv/) for Python package management.

```bash
git clone https://github.com/m9h/mrs-jax.git
cd mrs-jax
uv venv
source .venv/bin/activate
uv pip install -e ".[dev,all,doc]"
```

## 2. Development Workflow

1. Create a feature branch from `master`:
   ```bash
   git checkout -b feature/your-feature master
   ```
2. Make your changes, ensuring tests pass:
   ```bash
   pytest tests/ -v
   ```
3. Open a pull request against `master`.

### Branch naming

- `feature/` -- new features or modules
- `fix/` -- bug fixes
- `doc/` -- documentation improvements
- `refactor/` -- code restructuring without behavior changes

## 3. Code Style

### Docstrings

This project uses **NumPy-style docstrings** throughout. Every public function must have a docstring with at minimum:

- A one-line summary
- `Parameters` section with types
- `Returns` section with types

Example:

```python
def exponential_apodization(
    fid: np.ndarray,
    dwell_time: float,
    broadening_hz: float,
) -> np.ndarray:
    """Apply exponential (Lorentzian) apodization to an FID.

    Multiplies the FID by exp(-pi * broadening_hz * t), which adds
    *broadening_hz* Hz to the Lorentzian linewidth of every peak.

    Parameters
    ----------
    fid : ndarray
        Time-domain FID. The first axis is the spectral (time) dimension.
    dwell_time : float
        Dwell time in seconds.
    broadening_hz : float
        Additional line broadening in Hz.

    Returns
    -------
    ndarray
        Apodized FID with the same shape as input.

    References
    ----------
    de Graaf (2019) In Vivo NMR Spectroscopy, 3rd ed. Wiley.
    """
```

### Module docstrings

Every module must begin with a docstring that describes its purpose and lists key references.

### General conventions

- Type hints on all public function signatures
- `from __future__ import annotations` at the top of each module
- Constants in UPPER_CASE
- Private helpers prefixed with `_`

## 4. Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=mrs_jax --cov-report=term-missing
```

### Test categories

- **Unit tests** (`test_mrs_*.py`) -- isolated function-level tests with synthetic data
- **Integration tests** (`test_mrs_integration.py`) -- end-to-end tests against Big GABA and WAND datasets (skipped if data not available)

### Writing tests

- Synthetic data preferred for unit tests (no external data dependencies)
- Use `pytest.approx` for floating-point comparisons
- Integration tests should be decorated with `@pytest.mark.skipif` when benchmark data is not available
- New modules must have corresponding test files

## 5. Documentation

Documentation is built with Sphinx using the [furo](https://pradyunsg.me/furo/) theme.

```bash
# Install doc dependencies
uv pip install -e ".[doc]"

# Build docs
cd docs
make html

# View locally
open _build/html/index.html
```

### Adding documentation

- API docs are auto-generated from docstrings via `sphinx-apidoc`
- Use MyST Markdown (`.md`) for narrative docs
- Math: use `$...$` for inline and `$$...$$` for display equations
- Cross-reference modules with `` {py:mod}`mrs_jax.phase` ``

### Regenerating API docs

```bash
sphinx-apidoc -o docs/reference src/mrs_jax --separate --module-first --force
```

## 6. AI-Assisted Development

AI-assisted contributions are welcome. If using an LLM for code generation:

- Verify all generated code against MRS domain knowledge
- Validate numerical results against published benchmarks (Big GABA, ISMRM Fitting Challenge)
- Review docstrings for scientific accuracy -- LLMs can hallucinate references
- Run the full test suite before submitting

## 7. Scientific Standards

MRS processing code has clinical implications. Contributions must:

- **Cite primary sources** -- reference the original paper for each algorithm (e.g., Near et al. 2015 for spectral registration, Gasparovic et al. 2006 for water-referenced quantification)
- **Validate against benchmarks** -- new processing steps should be tested against Big GABA, ISMRM Fitting Challenge, or WAND datasets where applicable
- **Report units** -- always document physical units (Hz, ppm, seconds, mM) in docstrings and variable names
- **Preserve numerical precision** -- MRS signals span many orders of magnitude; be careful with floating-point operations and document expected precision
- **Follow consensus recommendations** -- Near et al. (2021) "Preprocessing, analysis and quantification in single-voxel MRS: experts' consensus recommendations" is the guiding reference
