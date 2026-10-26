# Current-induced phase in MRSI and EPSI

This is the first implementation step in the [neuroelectric-mri program](https://github.com/m9h/neuroelectric-mri). It connects a spatial field per ampere and an explicitly timed current waveform to complex spectroscopy data. It does not estimate conductivity or operate a scanner.

## Phase and clock contract

`CurrentEncoding` integrates

```text
phi(r, t) = phase_sign * gamma * b_parallel(r) * integral_0^t I(u) q(u) du
```

- `b_parallel` is in **T/A**, already transformed into scanner coordinates and projected along B0. Include the full outgoing/return circuit in its field calculation.
- Current amplitudes are in **A**; all times are in **s** relative to transverse excitation at zero; phase is in **rad**.
- Current windows are nonoverlapping, piecewise constant; gaps mean zero. Positive `current_delay_s` moves current later while keeping RF/ADC times fixed.
- Sensitivity intervals contain ideal `q = -1, 0, +1` weights. They describe extra field phase only. They do not simulate RF pulses, chemical-shift refocusing or J evolution.
- `phase_sign` is mandatory. `+1` matches this package's positive-Hz phase evolution; `-1` matches the `Mx+iMy` analytical references in neuroelectric-mri. The acquisition-to-simulation convention still needs calibration.
- Integration clips the waveform at excitation zero and every ADC sample. A field of the separable form `b(r) I(t)` is assumed. Sum phase histories for independent current paths; do not interpret a separable field as a general time-dependent tissue solver.

```python
import numpy as np
from mrs_jax.current_encoding import CurrentEncoding
from mrs_jax.mrsi_sim import TissueModel, simulate_mrsi

times = np.arange(128) * 0.0005
shape = (4, 4, 1)
encoding = CurrentEncoding(
    sample_times_s=times,
    current_windows_s=[[0.005, 0.025], [0.030, 0.050]],
    sensitivity_intervals=[[0, 0.064, 1]],
)
phase = encoding.phase(
    np.full(shape, 1e-6),       # illustrative field coefficient, T/A
    [0.001, -0.001],           # synthetic currents, A
    phase_sign=1,
)
model = TissueModel(
    np.ones(shape), {"test": np.ones(shape)}, np.ones(shape),
    np.full(shape, 0.08), np.zeros(shape),
)
kspace = simulate_mrsi(
    model, {"test": np.ones(128, dtype=complex)},
    n_spectral=128, dwell_time=0.0005, phase_rad=phase,
)
```

The waveform is synthetic, not a proposed hardware operating setting. A phase-encoded MRSI call assumes the same excitation-relative schedule for every spatial encode. Shot-to-shot drift, motion and different waveforms per encode need a more general acquisition model.

## Differentiation

Construct the schedule outside JAX transformations, then use `encoding.phase(..., backend="jax")` within `jit`, `grad` or `vmap`. Derivatives propagate through field coefficients, current amplitudes and delay; delay derivatives have kinks when interval boundaries coincide. Schedule arrays are copied and validated before tracing. Values of traced parameters must be validated by the caller; the JAX path cannot raise ordinary Python errors for nonfinite traced values.

Both `simulate_mrsi_from_arrays` and `simulate_mrsi_jax` now use native JAX operations. The latter sums multiple metabolite contributions. `TissueModel` is not a JAX pytree: construct it within the transformed function or pass arrays to the array API. The rest of the processing/quantification package is not thereby made end-to-end differentiable.

Pass `phase_rad` to any MRSI/EPSI forward kernel to multiply the voxel signal by `exp(+i*phase_rad)` **before** spatial encoding. Do not add the static B0 phase twice. Preserve the raw complex data: registration or phase correction may otherwise remove the effect under study.

## EPSI timing and reconstruction

`generate_epsi_trajectory(..., adc_dwell_time=..., readout_start_s=...)` creates per-sample timestamps, including intra-lobe evolution. `dwell_time` means time between **lobe starts**; alternating readout directions give corresponding k positions different intra-lobe offsets. When `adc_dwell_time` is omitted, the old instantaneous-lobe approximation remains available as an explicit reference. Supply actual `k_readout` and `sample_times_s` through `EPSITrajectory` for an imported trajectory.

Basis FIDs use their own `basis_dwell_time` (default: lobe interval). Both backends interpolate the real and imaginary components linearly and zero-fill outside the basis support. The supplied basis must span the entire ADC train, including the last intra-lobe sample, and be dense enough to avoid interpolation error. No extrapolated metabolite signal is invented after a short basis ends.

The generated k grid now contains distinct centered DFT bins, with matching centered voxel coordinates. `reconstruct_epsi` reverses alternate lobes on a copy and performs a centered inverse FFT. **This corrects earlier spatial-grid and raw-input mutation bugs; outputs differ from the previous implementation.** It rejects non-DFT grids. This reconstruction is exact for the instantaneous-lobe reference, but does not correct time-shear from intra-lobe spectral evolution. General sample-timed inversion, coils, gradient ramps and hardware limits remain future work.

The current NumPy EPSI model averages concentration, T2* and B0 over y/z. Use `(nx, 1, 1)` phantoms for its independent 1D checks; it is not a validated projection through heterogeneous 3D tissue.

## Verification and next gates

The new tests cover analytical constant-field units/signs, current reversal, ideal refocusing, timing gaps/delays, independent phase derivatives, native-JAX multimetabolite gradients, full complex spatial recovery on odd/even DFT grids, raw-input preservation, short-basis padding and explicit sample-time direct-DFT comparison. The cross-repository benchmark additionally compares the JAX-capable integrator to neuroelectric-mri's independent scalar interval implementation.

The next gates are realistic sequence-specific coupled-spin mI bases, nuisance/noise sensitivity and inference that does not know the injected phase a priori. The default Lorentzian helper supplies one resonance per metabolite; its `mIns` label is not evidence of quantitative myo-inositol accuracy. T1 saturation, RF preparation and scanner validation remain outside this kernel.
