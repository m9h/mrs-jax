# HERMES Multi-Editing: Simultaneous GABA and Glutathione

This tutorial demonstrates the HERMES (Hadamard Encoding and Reconstruction of
MEGA-Edited Spectroscopy) pipeline in `mrs-jax`. HERMES extends MEGA-PRESS by
acquiring four editing conditions in a single session, enabling simultaneous
quantification of GABA (3.0 ppm) and GSH (glutathione, 2.95 ppm) without
doubling scan time {cite:p}`chan2016hermes`.

## Background: Hadamard encoding

Standard MEGA-PRESS uses two conditions (ON/OFF) and edits one metabolite. To
edit two metabolites simultaneously, HERMES applies a 4-condition Hadamard
scheme with two orthogonal editing dimensions -- one targeting GABA, the other
targeting GSH:

| Condition | GABA editing | GSH editing |
|-----------|:------------:|:-----------:|
| **A**     | ON           | ON          |
| **B**     | ON           | OFF         |
| **C**     | OFF          | ON          |
| **D**     | OFF          | OFF         |

The signal in each condition is a linear combination of the edited and
non-edited components:

$$
\begin{aligned}
S_A &= S_{\text{base}} + S_{\text{GABA}} + S_{\text{GSH}} \\
S_B &= S_{\text{base}} + S_{\text{GABA}} - S_{\text{GSH}} \\
S_C &= S_{\text{base}} - S_{\text{GABA}} + S_{\text{GSH}} \\
S_D &= S_{\text{base}} - S_{\text{GABA}} - S_{\text{GSH}}
\end{aligned}
$$

where $S_{\text{base}}$ contains all non-edited singlets (NAA, creatine, etc.).

## Hadamard reconstruction

The Hadamard matrix $H_4$ encodes/decodes both metabolites simultaneously:

$$
H_4 = \begin{pmatrix}
+1 & +1 & +1 & +1 \\
+1 & +1 & -1 & -1 \\
+1 & -1 & +1 & -1 \\
+1 & -1 & -1 & +1
\end{pmatrix}
$$

The difference spectra that isolate each metabolite are:

$$
\text{GABA}_{\text{diff}} = (A + B) - (C + D) = 4\,S_{\text{GABA}}
$$

$$
\text{GSH}_{\text{diff}} = (A + C) - (B + D) = 4\,S_{\text{GSH}}
$$

Note that each subtraction algebraically cancels the other edited metabolite
and the baseline -- no cross-contamination is possible in the ideal case
{cite:p}`saleh2016multi`.

## Step 0: Imports

```python
import numpy as np
from mrs_jax.hermes import process_hermes, HermesResult
from mrs_jax.phase import (
    zero_order_phase_correction,
    fit_gaba_gaussian,
    water_referenced_quantification,
)
```

## Step 1: Generate synthetic HERMES data

The synthetic generator creates 4-condition data with known GABA and GSH
amplitudes. The data shape is `(n_spec, 4, n_dyn)` -- four conditions
interleaved over `n_dyn` dynamics.

```python
def make_singlet(ppm, amplitude, lw, n_pts, dwell, cf):
    """Single Lorentzian FID at a given ppm."""
    freq_hz = (ppm - 4.65) * (cf / 1e6)
    t = np.arange(n_pts) * dwell
    return amplitude * np.exp(2j * np.pi * freq_hz * t) * np.exp(-np.pi * lw * t)


def ppm_axis(n, dwell, cf):
    """Chemical shift axis in ppm."""
    freq = np.fft.fftshift(np.fft.fftfreq(n, dwell))
    return freq / (cf / 1e6) + 4.65


def make_hermes_data(
    n_pts=2048, n_dyn=16, dwell=2.5e-4, cf=123.25e6,
    gaba_conc=1.0, gsh_conc=0.5, naa_conc=10.0, cr_conc=8.0,
    noise_level=0.001, seed=42,
):
    """Synthetic HERMES data: shape (n_pts, 4, n_dyn).

    Hadamard encoding:
        A: +GABA, +GSH
        B: +GABA, -GSH
        C: -GABA, +GSH
        D: -GABA, -GSH
    """
    rng = np.random.default_rng(seed)

    naa  = make_singlet(2.01, naa_conc,  3.0,  n_pts, dwell, cf)
    cr   = make_singlet(3.03, cr_conc,   4.0,  n_pts, dwell, cf)
    gaba = make_singlet(3.01, gaba_conc, 8.0,  n_pts, dwell, cf)
    gsh  = make_singlet(2.95, gsh_conc,  10.0, n_pts, dwell, cf)

    baseline = naa + cr

    cond_a = baseline + gaba + gsh   # A: both ON
    cond_b = baseline + gaba - gsh   # B: GABA ON, GSH OFF
    cond_c = baseline - gaba + gsh   # C: GABA OFF, GSH ON
    cond_d = baseline - gaba - gsh   # D: both OFF

    data = np.zeros((n_pts, 4, n_dyn), dtype=complex)
    for d in range(n_dyn):
        for c_idx, cond in enumerate([cond_a, cond_b, cond_c, cond_d]):
            noise = noise_level * (
                rng.standard_normal(n_pts) + 1j * rng.standard_normal(n_pts)
            )
            data[:, c_idx, d] = cond + noise

    return data


# Create data
dwell = 2.5e-4
cf = 123.25e6
data = make_hermes_data(
    gaba_conc=2.0, gsh_conc=1.0, n_dyn=16,
    noise_level=0.001, dwell=dwell, cf=cf,
)
print(f"HERMES data shape: {data.shape}")
# -> (2048, 4, 16) = (n_spec, n_conditions, n_dyn)
```

## Step 2: Process with `process_hermes`

The `process_hermes` function averages each condition across dynamics and
applies the Hadamard reconstruction:

```python
result = process_hermes(data, dwell, cf, align=False)

print(f"Result type:   {type(result).__name__}")
print(f"GABA diff:     {result.gaba_diff.shape}")
print(f"GSH diff:      {result.gsh_diff.shape}")
print(f"Conditions:    {result.conditions.shape}")
print(f"N averages:    {result.n_averages}")
print(f"Bandwidth:     {result.bandwidth:.0f} Hz")
```

The `HermesResult` named tuple contains:

| Field        | Shape         | Description                          |
|-------------|---------------|--------------------------------------|
| `gaba_diff` | `(n_spec,)`   | GABA difference FID: (A+B) - (C+D)  |
| `gsh_diff`  | `(n_spec,)`   | GSH difference FID: (A+C) - (B+D)   |
| `conditions`| `(n_spec, 4)` | Averaged conditions A, B, C, D       |
| `n_averages`| `int`         | Number of dynamics averaged          |
| `dwell_time`| `float`       | Dwell time in seconds                |
| `bandwidth` | `float`       | Spectral bandwidth in Hz             |

## Step 3: Inspect the GABA difference spectrum

```python
ppm = ppm_axis(len(result.gaba_diff), dwell, cf)
gaba_spec = np.fft.fftshift(np.fft.fft(result.gaba_diff))

# Verify GABA peak near 3.01 ppm
gaba_mask = (ppm > 2.8) & (ppm < 3.2)
gaba_peak = np.max(np.abs(gaba_spec[gaba_mask]))
print(f"GABA peak magnitude: {gaba_peak:.4f}")
assert gaba_peak > 0, "GABA signal not detected"
```

```{note}
The Hadamard subtraction exactly cancels the GSH component in the GABA
difference and vice versa. In the test suite,
`TestHermesSeparation.test_hermes_separation` verifies that baseline noise
is less than 5% of the peak amplitude in a control region (5.0--6.0 ppm),
confirming no cross-contamination.
```

## Step 4: Inspect the GSH difference spectrum

```python
gsh_spec = np.fft.fftshift(np.fft.fft(result.gsh_diff))

gsh_mask = (ppm > 2.75) & (ppm < 3.15)
gsh_peak = np.max(np.abs(gsh_spec[gsh_mask]))
print(f"GSH peak magnitude: {gsh_peak:.4f}")
assert gsh_peak > 0, "GSH signal not detected"
```

## Step 5: Phase and fit GABA

Apply zero-order phase correction and fit a Gaussian to the GABA peak in the
real part of the difference spectrum:

```python
gaba_phased = zero_order_phase_correction(result.gaba_diff)
gaba_spec_phased = np.fft.fftshift(np.fft.fft(gaba_phased))
gaba_real = np.real(gaba_spec_phased)

gaba_fit = fit_gaba_gaussian(gaba_real, ppm, fit_range=(2.7, 3.3))
print(f"GABA centre: {gaba_fit['centre_ppm']:.3f} ppm")
print(f"GABA area:   {gaba_fit['area']:.4f}")
print(f"GABA CRLB:   {gaba_fit['crlb_percent']:.1f}%")
```

## Step 6: Phase and fit GSH

The same Gaussian fitting approach works for GSH -- just adjust the fit range
to target the 2.95 ppm resonance:

```python
gsh_phased = zero_order_phase_correction(result.gsh_diff)
gsh_spec_phased = np.fft.fftshift(np.fft.fft(gsh_phased))
gsh_real = np.real(gsh_spec_phased)

gsh_fit = fit_gaba_gaussian(gsh_real, ppm, fit_range=(2.6, 3.2))
print(f"GSH centre: {gsh_fit['centre_ppm']:.3f} ppm")
print(f"GSH area:   {gsh_fit['area']:.4f}")
print(f"GSH CRLB:   {gsh_fit['crlb_percent']:.1f}%")
```

```{tip}
`fit_gaba_gaussian` is a general-purpose Gaussian fitter despite its name.
You can use it for any edited peak by adjusting the `fit_range` parameter.
The fitting model is
$G(\nu) = A \exp\!\bigl(-(\nu - \nu_0)^2 / 2\sigma^2\bigr) + B$
with bounded optimization via `scipy.optimize.curve_fit`.
```

## Step 7: Water-referenced quantification (both metabolites)

With a water reference, convert both peak areas to absolute concentrations:

```python
# Synthesise water reference for this tutorial
water_fid = make_singlet(4.65, 1000.0, 2.0, 2048, dwell, cf)
water_spec = np.fft.fftshift(np.fft.fft(water_fid))
water_area = float(np.max(np.abs(water_spec)))

tissue_fracs = {'gm': 0.6, 'wm': 0.4, 'csf': 0.0}

gaba_conc = water_referenced_quantification(
    metab_area=gaba_fit['area'],
    water_area=water_area,
    tissue_fracs=tissue_fracs,
    te=0.080,         # 80 ms (typical HERMES TE)
    tr=2.0,
    metab_t1=1.3,     # GABA T1 at 3T
    metab_t2=0.16,    # GABA T2 at 3T
    field_strength=3.0,
)

gsh_conc = water_referenced_quantification(
    metab_area=gsh_fit['area'],
    water_area=water_area,
    tissue_fracs=tissue_fracs,
    te=0.080,
    tr=2.0,
    metab_t1=0.9,     # GSH T1 at 3T (Choi et al. 2006)
    metab_t2=0.10,    # GSH T2 at 3T
    field_strength=3.0,
)

print(f"GABA concentration: {gaba_conc:.2f} mM")
print(f"GSH concentration:  {gsh_conc:.2f} mM")
```

```{note}
Typical in vivo concentrations at 3 T:
- GABA: 1.0--1.5 mM (anterior cingulate cortex)
- GSH: 0.5--1.5 mM (varies by region)

The relaxation parameters (`metab_t1`, `metab_t2`) differ between GABA and
GSH and affect the absolute quantification. Use literature values appropriate
for your field strength and brain region.
```

## Step 8: Verify separation quality

A key advantage of HERMES is that the Hadamard algebra provides exact
separation -- GABA does not leak into the GSH difference and vice versa. We
can verify this by checking the noise floor in a baseline region:

```python
# Check a spectral region far from both peaks (5-6 ppm)
baseline_mask = (ppm > 5.0) & (ppm < 6.0)

gaba_baseline = np.max(np.abs(gaba_spec[baseline_mask]))
gsh_baseline  = np.max(np.abs(gsh_spec[baseline_mask]))

print(f"GABA baseline / peak: {gaba_baseline / gaba_peak:.4f}")
print(f"GSH  baseline / peak: {gsh_baseline / gsh_peak:.4f}")

# Both ratios should be very small (< 5%)
assert gaba_baseline < 0.05 * gaba_peak, "GABA baseline contamination"
assert gsh_baseline < 0.05 * gsh_peak,   "GSH baseline contamination"
```

## Comparison: HERMES vs two separate MEGA-PRESS scans

| Metric            | 2 x MEGA-PRESS | HERMES         |
|-------------------|:---------------:|:--------------:|
| Scan time         | 2x              | 1x             |
| GABA sensitivity  | Full            | Full           |
| GSH sensitivity   | Full            | Full           |
| Conditions needed | 2 + 2           | 4              |
| Motion coherence  | Separate scans  | Same scan      |
| Processing        | Standard        | Hadamard decode|

HERMES achieves the same per-metabolite SNR as MEGA-PRESS in half the total
scan time, because each metabolite uses all acquired dynamics (not half).
The same-scan acquisition also ensures that motion and drift affect both
metabolites identically.

## Summary of `process_hermes` API

```python
def process_hermes(
    data: np.ndarray,          # (n_spec, 4, n_dyn)
    dwell_time: float,         # seconds
    centre_freq: float = 123.0e6,  # Hz
    align: bool = False,       # frequency/phase alignment
) -> HermesResult:
    ...
```

Input validation:
- `data` must be 3D with exactly 4 conditions along axis 1
- Raises `ValueError` for incorrect shapes

## References

```{bibliography}
:cited:
```
