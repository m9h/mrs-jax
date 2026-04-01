# MEGA-PRESS GABA Quantification: From Raw Data to Concentrations

This tutorial walks through the complete MEGA-PRESS processing pipeline in
`mrs-jax`, from loading raw Siemens TWIX data to absolute GABA concentrations
in millimolar. Each step is demonstrated with synthetic data so you can run
everything without scanner data, then swap in your own TWIX files at the end.

## Background

MEGA-PRESS (MEscher-GArwood Point RESolved Spectroscopy) is a J-difference
editing technique for detecting low-concentration metabolites such as GABA
(gamma-aminobutyric acid) at 3.0 ppm. The experiment acquires interleaved
**edit-ON** and **edit-OFF** transients. In the edit-ON condition, a
frequency-selective pulse at 1.9 ppm refocuses the GABA C3-H3 resonance at
3.0 ppm via J-coupling. Subtracting edit-OFF from edit-ON cancels all
unedited singlets (NAA, creatine) and reveals the GABA signal
{cite:p}`mescher1998simultaneous`.

The FID signal model for a single Lorentzian resonance at chemical shift
$\delta$ (ppm) is:

$$
s(t) = A \, e^{i \phi_0} \, \exp\!\Bigl(2\pi i \, f(\delta) \, t\Bigr)
       \exp\!\bigl(-\pi \, \mathrm{LW} \, t\bigr)
$$

where $f(\delta) = (\delta - 4.65) \times (B_0 \cdot \gamma / 10^6)$ converts
ppm to Hz, $A$ is the amplitude, $\phi_0$ is the zero-order phase, and
$\mathrm{LW}$ is the Lorentzian linewidth in Hz.

## Pipeline overview

```text
TWIX .dat
  |
  v
read_twix()              -- load raw data + metadata
  |
  v
exponential_apodization() -- matched-filter SNR boost
eddy_current_correction() -- remove phase distortions (Klose 1990)
frequency_reference()     -- lock NAA to 2.01 ppm
  |
  v
process_mega_press()      -- coil combine (SVD) -> spectral registration
                             -> outlier rejection -> ON-OFF subtraction
  |
  v
quantify_mega_press()     -- phase correction -> Gaussian fit -> water ref
  |
  v
generate_qc_report()      -- HTML report with inline plots
```

## Step 0: Installation and imports

```python
import numpy as np
from mrs_jax.io import read_twix, MRSData
from mrs_jax.preproc import (
    exponential_apodization,
    gaussian_apodization,
    eddy_current_correction,
    frequency_reference,
)
from mrs_jax.mega_press import (
    coil_combine_svd,
    spectral_registration,
    apply_correction,
    reject_outliers,
    process_mega_press,
    MegaPressResult,
)
from mrs_jax.phase import (
    zero_order_phase_correction,
    fit_gaba_gaussian,
    water_referenced_quantification,
)
from mrs_jax.quantify import quantify_mega_press
from mrs_jax.qc import generate_qc_report
```

## Step 1: Generate synthetic MEGA-PRESS data

For reproducibility we synthesise multi-coil MEGA-PRESS data with known ground
truth. The helper below creates a 4-coil, 32-dynamic dataset at 3 T with NAA
(2.01 ppm), creatine (3.03 ppm), and GABA (3.01 ppm).

```python
def make_singlet(ppm, amplitude, lw, n_pts, dwell, cf):
    """Create a single Lorentzian FID at a given chemical shift."""
    freq_hz = (ppm - 4.65) * (cf / 1e6)
    t = np.arange(n_pts) * dwell
    return amplitude * np.exp(2j * np.pi * freq_hz * t) * np.exp(-np.pi * lw * t)


def make_mega_data(
    n_pts=2048, n_coils=4, n_dyn=32, dwell=2.5e-4, cf=123.25e6,
    gaba_conc=1.0, naa_conc=10.0, cr_conc=8.0, noise_level=0.01, seed=42,
):
    """Synthetic MEGA-PRESS: shape (n_pts, n_coils, 2, n_dyn)."""
    rng = np.random.default_rng(seed)

    naa = make_singlet(2.01, naa_conc, 3.0, n_pts, dwell, cf)
    cr  = make_singlet(3.03, cr_conc,  4.0, n_pts, dwell, cf)
    gaba = make_singlet(3.01, gaba_conc, 8.0, n_pts, dwell, cf)

    edit_on  = naa + cr + gaba     # GABA refocused
    edit_off = naa + cr - gaba     # GABA inverted (cancels in subtraction)

    coil_weights = rng.standard_normal(n_coils) + 1j * rng.standard_normal(n_coils)
    coil_weights /= np.max(np.abs(coil_weights))

    data = np.zeros((n_pts, n_coils, 2, n_dyn), dtype=complex)
    for d in range(n_dyn):
        for c in range(n_coils):
            noise_on  = noise_level * (rng.standard_normal(n_pts) + 1j * rng.standard_normal(n_pts))
            noise_off = noise_level * (rng.standard_normal(n_pts) + 1j * rng.standard_normal(n_pts))
            data[:, c, 0, d] = coil_weights[c] * edit_on  + noise_on
            data[:, c, 1, d] = coil_weights[c] * edit_off + noise_off

    return data

# Create synthetic data
dwell = 2.5e-4          # 4 kHz bandwidth
cf    = 123.25e6        # 3 T proton Larmor frequency
data  = make_mega_data(dwell=dwell, cf=cf, noise_level=0.01)
print(f"Raw data shape: {data.shape}")
# -> (2048, 4, 2, 32)  = (n_spec, n_coils, n_edit, n_dyn)
```

````{tip}
To use real scanner data instead, replace the synthetic block with:
```python
mrd = read_twix("/path/to/your_mega_press.dat")
data = mrd.data
dwell = mrd.dwell_time
cf = mrd.centre_freq
```
The `MRSData` object also exposes `te`, `tr`, `field_strength`, and an
optional `water_ref` array.
````

## Step 2: Preprocessing

### 2a. Exponential apodization

A matched filter (exponential window) trades spectral resolution for SNR.
Adding $b$ Hz of line broadening multiplies the FID by
$\exp(-\pi \, b \, t)$:

```python
from mrs_jax.preproc import exponential_apodization

# Apply 3 Hz exponential line broadening to each coil/dynamic
data_apod = exponential_apodization(data, dwell, broadening_hz=3.0)
```

```{note}
`exponential_apodization` broadcasts across all trailing dimensions, so you
can pass the full `(n_spec, n_coils, n_edit, n_dyn)` array directly.
At $t = 0$ the window equals 1, preserving the first point of the FID.
Zero broadening returns an unmodified copy.
```

### 2b. Eddy current correction (ECC)

Gradient switching induces time-varying phase distortions (eddy currents) in
the FID. The Klose method {cite:p}`klose1990vivo` removes them by subtracting
the instantaneous phase of an unsuppressed water reference, point by point:

$$
s_{\text{corrected}}(t) = s(t) \cdot \exp\!\bigl(-i\,\phi_{\text{water}}(t)\bigr)
$$

```python
from mrs_jax.preproc import eddy_current_correction

# If you have a water reference FID:
# water_fid = mrd.water_ref  # from read_twix(..., load_water_ref=True)
# data_ecc = eddy_current_correction(data_apod, water_fid)

# For this tutorial we skip ECC (no water ref in synthetic data)
data_ecc = data_apod
```

```{note}
ECC only corrects phase -- the signal magnitude is preserved exactly. The
water reference must have the same number of spectral points along axis 0.
```

### 2c. Frequency referencing

Scanner drift or B0 inhomogeneity can shift peaks away from their canonical
positions. `frequency_reference` finds the tallest peak near a target ppm and
applies a time-domain frequency shift to place it correctly:

```python
from mrs_jax.preproc import frequency_reference

# Reference a single FID to NAA at 2.01 ppm
fid_example = data_ecc[:, 0, 0, 0]  # first coil, edit-ON, first dynamic
fid_ref = frequency_reference(
    fid_example, dwell, cf,
    target_ppm=2.01,
    target_peak_ppm=2.01,
    search_window_ppm=0.3,
)
```

## Step 3: Coil combination (SVD)

Multi-coil data must be combined into a single channel before averaging.
`coil_combine_svd` uses the first right singular vector of the coil
correlation matrix as weights, phase-aligned to the first coil element:

```python
from mrs_jax.mega_press import coil_combine_svd

# Input:  (n_spec, n_coils, n_edit, n_dyn)
# Output: (n_spec, n_edit, n_dyn)
combined = coil_combine_svd(data_ecc)
print(f"After coil combine: {combined.shape}")
# -> (2048, 2, 32)
```

```{tip}
SVD coil combination yields an SNR improvement proportional to
$\sqrt{N_{\text{coils}}}$ relative to the best single coil. The test suite
verifies this: `TestCoilCombineSVD.test_snr_improvement` creates 8-coil data
and confirms the combined SNR exceeds the single-coil SNR.
```

## Step 4: Spectral registration (frequency/phase alignment)

Scanner frequency drift during the ~10-minute acquisition broadens averaged
peaks. Spectral registration {cite:p}`near2015frequency` aligns each transient
to a reference by minimising:

$$
\min_{\Delta f,\,\phi}\;
\bigl\lVert S_{\text{ref}}(\nu) - S(\nu)\,e^{i(2\pi\Delta f\, t + \phi)} \bigr\rVert^2
$$

over a metabolite-rich ppm window (typically 1.8--4.2 ppm).

```python
from mrs_jax.mega_press import spectral_registration, apply_correction

edit_off = combined[:, 1, :]  # (n_spec, n_dyn) -- more stable peaks
edit_on  = combined[:, 0, :]

# Build reference from the mean of all edit-OFF transients
ref = edit_off.mean(axis=1)

freq_shifts = np.zeros(32)
phase_shifts = np.zeros(32)

for i in range(32):
    df, dp = spectral_registration(
        edit_off[:, i], ref, dwell, centre_freq=cf,
    )
    freq_shifts[i] = df
    phase_shifts[i] = dp
    edit_off[:, i] = apply_correction(edit_off[:, i], df, dp, dwell)
    edit_on[:, i]  = apply_correction(edit_on[:, i],  df, dp, dwell)

print(f"Mean freq drift: {freq_shifts.mean():.2f} Hz")
print(f"Max |freq shift|: {np.max(np.abs(freq_shifts)):.2f} Hz")
```

```{note}
The pipeline supports **paired alignment** via
`process_mega_press(..., paired_alignment=True)`. In this mode, corrections
are estimated from the edit-OFF transients and the same correction is applied
to both ON and OFF for each dynamic, preserving their relative phase
relationship for clean subtraction.
```

## Step 5: Outlier rejection

Motion-corrupted transients are identified by their residual from the mean
FID. The z-score is computed using the median absolute deviation (MAD) for
robustness:

$$
z_i = \frac{0.6745 \cdot (r_i - \tilde{r})}{\text{MAD}(r)}
$$

Transients with $|z_i| > \theta$ (default $\theta = 3$) are rejected.

```python
from mrs_jax.mega_press import reject_outliers

rejected_off = reject_outliers(edit_off, dwell, threshold=3.0)
rejected_on  = reject_outliers(edit_on,  dwell, threshold=3.0)

print(f"Rejected edit-OFF: {rejected_off.sum()}/{len(rejected_off)}")
print(f"Rejected edit-ON:  {rejected_on.sum()}/{len(rejected_on)}")

# Remove outliers before averaging
edit_off_clean = edit_off[:, ~rejected_off]
edit_on_clean  = edit_on[:, ~rejected_on]
```

## Step 6: Difference spectrum

Average each condition and subtract to isolate the edited GABA signal:

```python
avg_on  = edit_on_clean.mean(axis=1)
avg_off = edit_off_clean.mean(axis=1)
diff    = avg_on - avg_off

# Compute frequency-domain spectrum
n = len(diff)
freq = np.fft.fftshift(np.fft.fftfreq(n, dwell))
ppm  = freq / (cf / 1e6) + 4.65
diff_spec = np.fft.fftshift(np.fft.fft(diff))
```

In the difference spectrum, creatine (present equally in both conditions)
cancels to zero while the GABA peak at 3.0 ppm survives. The test
`TestProcessMegaPress.test_cr_cancels_in_difference` verifies that creatine
residual in the difference is less than 1% of its edit-OFF amplitude.

## Step 7: Phase correction

Before fitting, the difference spectrum must be phased so the GABA peak
appears as a pure absorption-mode (real) lineshape. Zero-order phase
correction maximises $\sum \mathrm{Re}\bigl[S(\nu)\,e^{i\phi_0}\bigr]$:

```python
from mrs_jax.phase import zero_order_phase_correction

diff_phased, phi0 = zero_order_phase_correction(diff, return_phase=True)
print(f"Applied phase correction: {np.degrees(phi0):.1f} degrees")

diff_spec_phased = np.fft.fftshift(np.fft.fft(diff_phased))
diff_real = np.real(diff_spec_phased)
```

## Step 8: GABA Gaussian fit

The GABA+ peak at ~3.0 ppm is fitted with a Gaussian model:

$$
G(\nu) = A \exp\!\biggl(-\frac{(\nu - \nu_0)^2}{2\sigma^2}\biggr) + B
$$

where $A$ is the amplitude, $\nu_0$ is the centre frequency (ppm), $\sigma$
is the width, and $B$ is a local baseline offset. The FWHM is
$2\sqrt{2\ln 2}\,\sigma \approx 2.355\,\sigma$. The Cramer-Rao lower bound
(CRLB) is estimated from the covariance matrix of the fit.

```python
from mrs_jax.phase import fit_gaba_gaussian

gaba_fit = fit_gaba_gaussian(diff_real, ppm, fit_range=(2.7, 3.3))

print(f"GABA centre:   {gaba_fit['centre_ppm']:.3f} ppm")
print(f"GABA FWHM:     {gaba_fit['fwhm_ppm']:.3f} ppm")
print(f"GABA area:     {gaba_fit['area']:.4f}")
print(f"GABA CRLB:     {gaba_fit['crlb_percent']:.1f}%")
print(f"Fit residual:  {gaba_fit['residual']:.6f}")
```

```{tip}
A CRLB below 20% is generally considered acceptable for GABA quantification
{cite:p}`edden2014gannet`. Values above 50% indicate unreliable fits. The
`crlb_percent` field comes directly from the diagonal of the fit covariance
matrix: $\text{CRLB} = 100 \times |\sigma_A / A|$.
```

## Step 9: Water-referenced quantification

Absolute concentrations require an unsuppressed water reference. The Gasparovic
formula {cite:p}`gasparovic2006use` converts the metabolite-to-water signal
ratio to millimolar concentration:

$$
[M] = \frac{S_M}{S_W} \cdot \frac{f_W}{R_M} \cdot \frac{[W]}{f_{\text{tissue}}}
$$

where $f_W$ is the relaxation-weighted water content summed over grey matter,
white matter, and CSF compartments, $R_M$ accounts for metabolite relaxation
($T_1$, $T_2$), and $[W] = 55{,}556$ mM is the molar concentration of
pure water.

```python
from mrs_jax.phase import water_referenced_quantification

# Synthesise a water reference for this tutorial
water_fid = make_singlet(4.65, 1000.0, 2.0, 2048, dwell, cf)
water_spec = np.fft.fftshift(np.fft.fft(water_fid))
water_area = float(np.max(np.abs(water_spec)))

gaba_conc = water_referenced_quantification(
    metab_area=gaba_fit['area'],
    water_area=water_area,
    tissue_fracs={'gm': 0.6, 'wm': 0.4, 'csf': 0.0},
    te=0.068,        # 68 ms (typical MEGA-PRESS TE)
    tr=2.0,          # 2 s repetition time
    metab_t1=1.3,    # GABA T1 at 3T (s)
    metab_t2=0.16,   # GABA T2 at 3T (s)
    field_strength=3.0,
)
print(f"GABA concentration: {gaba_conc:.2f} mM")
```

```{note}
The tissue fractions (`gm`, `wm`, `csf`) come from segmented structural MRI.
CSF correction is important: metabolites are concentrated in tissue, so a
voxel with 40% CSF will have a higher tissue-corrected concentration than
the raw ratio suggests. The test
`TestQuantifyWithTissueFracs.test_quantify_with_tissue_fracs` verifies this.
```

## Step 10: All-in-one `quantify_mega_press`

For convenience, `quantify_mega_press` chains all the above steps into a
single call and returns a comprehensive result dictionary:

```python
from mrs_jax.quantify import quantify_mega_press

result = quantify_mega_press(
    data,
    dwell_time=dwell,
    centre_freq=cf,
    water_ref=water_fid,
    tissue_fracs={'gm': 0.6, 'wm': 0.4, 'csf': 0.0},
    te=0.068,
    tr=2.0,
    align=True,
    reject=True,
    reject_threshold=3.0,
    paired_alignment=False,
    gaba_fit_range=(2.7, 3.3),
)

print(f"GABA concentration: {result['gaba_conc_mM']:.2f} mM")
print(f"GABA/NAA ratio:     {result['gaba_naa_ratio']:.4f}")
print(f"GABA area:          {result['gaba_area']:.4f}")
print(f"NAA area:           {result['naa_area']:.4f}")
print(f"SNR:                {result['snr']:.1f}")
print(f"CRLB:               {result['crlb_percent']:.1f}%")
print(f"Averages used:      {result['n_averages']}")
```

The returned dictionary also contains the processed FIDs (`diff_fid`,
`edit_on_fid`, `edit_off_fid`), per-transient alignment parameters
(`freq_shifts`, `phase_shifts`), and a boolean rejection mask (`rejected`),
all suitable for downstream QC.

## Step 11: QC report

Generate a self-contained HTML report with spectra plots, alignment metrics,
and metabolite tables:

```python
from mrs_jax.qc import generate_qc_report

html = generate_qc_report(
    result,
    fitting_results={
        'GABA+': {
            'concentration_mM': result['gaba_conc_mM'],
            'crlb_percent': result['crlb_percent'],
        },
    },
    title="MEGA-PRESS GABA QC Report",
)

with open("mega_press_qc.html", "w") as f:
    f.write(html)
```

The report includes:
- Edit-ON, edit-OFF, and difference spectra
- Per-transient frequency and phase drift plots
- Outlier rejection summary
- Metabolite concentration table

## Using real scanner data

To process actual Big GABA {cite:p}`mikkelsen2017big` or WAND MEGA-PRESS
data, replace the synthetic data block with the TWIX reader:

```python
mrd = read_twix("/path/to/mega_press.dat", load_water_ref=True)

result = quantify_mega_press(
    mrd.data,
    dwell_time=mrd.dwell_time,
    centre_freq=mrd.centre_freq,
    water_ref=mrd.water_ref,
    tissue_fracs={'gm': 0.6, 'wm': 0.3, 'csf': 0.1},
    te=mrd.te / 1000,     # MRSData stores TE in ms
    tr=mrd.tr / 1000,     # MRSData stores TR in ms
)
```

The `MRSData` object provides:
- `data`: complex FID array, shape `(n_spec, n_coils, n_edit, n_dyn)`
- `dwell_time`: dwell time in seconds
- `centre_freq`: spectrometer frequency in Hz
- `te`, `tr`: echo time and repetition time in milliseconds
- `field_strength`: B0 in Tesla
- `water_ref`: unsuppressed water FID (if `load_water_ref=True`)
- `n_coils`, `n_averages`: integer counts
- `dim_info`: axis label mapping

## References

```{bibliography}
:cited:
```
