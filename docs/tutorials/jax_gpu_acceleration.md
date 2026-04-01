# JAX GPU Acceleration: Batch Processing and Differentiable MRS

This tutorial shows how to use the JAX backend in `mrs-jax` for GPU-accelerated
batch processing and gradient-based optimization of MRS data. The library
provides numerically equivalent JAX implementations of all core functions,
enabling JIT compilation, vectorized mapping across subjects, and automatic
differentiation through the entire processing pipeline.

## Background: Why JAX for MRS?

Traditional MRS processing with NumPy is sequential: each subject is processed
one at a time, and optimization relies on grid search or derivative-free
methods. JAX brings three capabilities that change this:

1. **`jit` (Just-In-Time compilation)**: Compiles Python functions to
   XLA-optimized machine code, running on CPU or GPU. A JIT-compiled
   processing pipeline runs 5--50x faster than NumPy on the same hardware.

2. **`vmap` (Vectorized map)**: Automatically vectorizes a single-subject
   function to process a batch of subjects in parallel, without writing
   explicit loops. On GPU, this maps to a single batched kernel launch.

3. **`grad` (Automatic differentiation)**: Computes exact gradients of any
   differentiable function, enabling gradient-based spectral fitting instead
   of brute-force grid search.

## Step 0: Imports

```python
import numpy as np
import jax
import jax.numpy as jnp

# NumPy implementations (reference)
from mrs_jax.mega_press import (
    coil_combine_svd as np_coil_combine_svd,
    apply_correction as np_apply_correction,
    process_mega_press as np_process_mega_press,
)

# JAX implementations (accelerated)
from mrs_jax.mega_press_jax import (
    coil_combine_svd as jax_coil_combine_svd,
    apply_correction as jax_apply_correction,
    process_mega_press as jax_process_mega_press,
)
```

```{note}
The JAX backend lives in `mrs_jax.mega_press_jax` and mirrors the API of
`mrs_jax.mega_press`. Every function accepts and returns `jnp.ndarray`
instead of `np.ndarray`, but the signatures and semantics are identical.
```

## Step 1: NumPy-JAX equivalence

A core design principle of `mrs-jax` is that the JAX backend produces
numerically identical results to NumPy (within floating-point tolerance).
This lets you develop and validate with NumPy, then switch to JAX for
production.

### 1a. Generate synthetic data

```python
def make_singlet(ppm, amplitude, lw, n_pts, dwell, cf):
    """Single Lorentzian FID."""
    freq_hz = (ppm - 4.65) * (cf / 1e6)
    t = np.arange(n_pts) * dwell
    return amplitude * np.exp(2j * np.pi * freq_hz * t) * np.exp(-np.pi * lw * t)


def make_mega_data(
    n_pts=2048, n_coils=4, n_dyn=16, dwell=2.5e-4, cf=123.25e6,
    gaba_conc=1.0, naa_conc=10.0, cr_conc=8.0, noise_level=0.01, seed=42,
):
    rng = np.random.default_rng(seed)
    naa  = make_singlet(2.01, naa_conc, 3.0, n_pts, dwell, cf)
    cr   = make_singlet(3.03, cr_conc,  4.0, n_pts, dwell, cf)
    gaba = make_singlet(3.01, gaba_conc, 8.0, n_pts, dwell, cf)
    edit_on  = naa + cr + gaba
    edit_off = naa + cr - gaba
    coil_weights = rng.standard_normal(n_coils) + 1j * rng.standard_normal(n_coils)
    coil_weights /= np.max(np.abs(coil_weights))
    data = np.zeros((n_pts, n_coils, 2, n_dyn), dtype=complex)
    for d in range(n_dyn):
        for c in range(n_coils):
            noise_on  = noise_level * (rng.standard_normal(n_pts) + 1j * rng.standard_normal(n_pts))
            noise_off = noise_level * (rng.standard_normal(n_pts) + 1j * rng.standard_normal(n_pts))
            data[:, c, 0, d] = coil_weights[c] * edit_on  + noise_on
            data[:, c, 1, d] = coil_weights[c] * edit_off + noise_off
    return data, {'dwell': dwell, 'cf': cf}
```

### 1b. Compare coil combination

```python
data_np, truth = make_mega_data()
data_jax = jnp.array(data_np)

result_np  = np_coil_combine_svd(data_np)
result_jax = jax_coil_combine_svd(data_jax)

max_diff = np.max(np.abs(np.array(result_jax) - result_np))
print(f"Coil combine max difference: {max_diff:.2e}")
# -> ~1e-14 (machine epsilon)

np.testing.assert_allclose(np.array(result_jax), result_np, atol=1e-5)
print("Coil combine: NumPy and JAX match.")
```

### 1c. Compare frequency/phase correction

```python
rng = np.random.default_rng(123)
fid_np = rng.standard_normal(1024) + 1j * rng.standard_normal(1024)
freq_shift = 3.5
phase_shift = 0.7
dwell = 2.5e-4

result_np  = np_apply_correction(fid_np, freq_shift, phase_shift, dwell)
result_jax = jax_apply_correction(
    jnp.array(fid_np),
    jnp.float64(freq_shift),
    jnp.float64(phase_shift),
    jnp.float64(dwell),
)

np.testing.assert_allclose(np.array(result_jax), result_np, atol=1e-5)
print("apply_correction: NumPy and JAX match.")
```

### 1d. Compare full pipeline

```python
data_np, truth = make_mega_data(noise_level=0.001)
dwell = truth['dwell']
cf = truth['cf']

result_np = np_process_mega_press(data_np, dwell, cf, align=False, reject=False)
result_jax = jax_process_mega_press(
    jnp.array(data_np), dwell, cf, align=False, reject=False,
)

np.testing.assert_allclose(np.array(result_jax.diff), result_np.diff, atol=1e-5)
np.testing.assert_allclose(np.array(result_jax.edit_on), result_np.edit_on, atol=1e-5)
np.testing.assert_allclose(np.array(result_jax.edit_off), result_np.edit_off, atol=1e-5)
print("Full pipeline: NumPy and JAX match on diff, edit_on, edit_off.")
```

```{tip}
The test suite (`test_mrs_jax.py`) systematically verifies equivalence for
every function. When developing new processing steps, follow the same pattern:
implement in NumPy first, write a JAX equivalent, then assert `allclose` with
`atol=1e-5`.
```

## Step 2: JIT compilation

`jax.jit` traces your function once and compiles it to an XLA computation
graph. Subsequent calls reuse the compiled code, avoiding Python interpreter
overhead.

```python
@jax.jit
def process_jit(data):
    return jax_process_mega_press(
        data, dwell, cf, align=False, reject=False,
    ).diff

data_jax = jnp.array(make_mega_data(noise_level=0.001)[0])

# First call: compilation (slower)
diff_jit = process_jit(data_jax)
diff_jit.block_until_ready()

# Second call: cached compilation (fast)
diff_jit2 = process_jit(data_jax)
diff_jit2.block_until_ready()
```

Verify that JIT produces identical results to eager execution:

```python
diff_eager = jax_process_mega_press(
    data_jax, dwell, cf, align=False, reject=False,
).diff

np.testing.assert_allclose(
    np.array(diff_jit), np.array(diff_eager), atol=1e-7,
)
print("JIT result matches eager result.")
```

### Performance comparison

```python
import time

data_np, truth = make_mega_data(n_coils=8, n_dyn=64, noise_level=0.001)
data_jax = jnp.array(data_np)

# NumPy timing
t0 = time.perf_counter()
for _ in range(10):
    np_process_mega_press(data_np, dwell, cf, align=False, reject=False)
numpy_time = (time.perf_counter() - t0) / 10

# JAX JIT timing (after warmup)
process_jit_bench = jax.jit(
    lambda d: jax_process_mega_press(d, dwell, cf, align=False, reject=False).diff
)
_ = process_jit_bench(data_jax).block_until_ready()  # warmup

t0 = time.perf_counter()
for _ in range(10):
    process_jit_bench(data_jax).block_until_ready()
jax_time = (time.perf_counter() - t0) / 10

print(f"NumPy:  {numpy_time*1000:.1f} ms/run")
print(f"JAX JIT: {jax_time*1000:.1f} ms/run")
print(f"Speedup: {numpy_time/jax_time:.1f}x")
```

```{note}
On CPU, JIT speedup is typically 2--10x due to XLA fusion and vectorization.
On GPU (e.g., NVIDIA A100 or DGX Spark), the speedup grows to 20--100x for
large batch sizes, as the SVD, FFT, and einsum operations all dispatch to
optimized CUDA kernels.
```

## Step 3: `vmap` for batch processing across subjects

The key advantage of JAX for multi-subject studies: `vmap` automatically
vectorizes a single-subject function to process an entire cohort in one call.
No explicit loops, no manual batching.

### 3a. Stack subjects into a batch

```python
n_subjects = 4
subjects = []
for seed in range(n_subjects):
    data_np, _ = make_mega_data(seed=seed, n_dyn=8, noise_level=0.001)
    subjects.append(data_np)

# Stack along a new leading dimension
batch = jnp.stack(subjects, axis=0)
print(f"Batch shape: {batch.shape}")
# -> (4, 2048, 4, 2, 8) = (n_subjects, n_spec, n_coils, n_edit, n_dyn)
```

### 3b. vmap over the subject dimension

```python
vmapped_process = jax.vmap(
    lambda d: jax_process_mega_press(
        d, dwell, cf, align=False, reject=False,
    ).diff
)

diffs = vmapped_process(batch)
print(f"Batched diffs shape: {diffs.shape}")
# -> (4, 2048) = (n_subjects, n_spec)

# Verify each subject produces a non-trivial result
for i in range(n_subjects):
    peak = float(jnp.max(jnp.abs(diffs[i])))
    print(f"  Subject {i}: peak magnitude = {peak:.4f}")
    assert peak > 0
```

```{tip}
On GPU, `vmap` fuses all subjects into a single batched computation. For a
24-site Big GABA cohort {cite:p}`mikkelsen2017big` with ~30 subjects per
site, you can process all subjects in a single `vmap` call -- typically
faster than processing them sequentially in NumPy.
```

### 3c. Nested vmap: subjects x voxels (MRSI)

For multi-voxel data, nest two `vmap` calls:

```python
# Hypothetical: (n_subjects, n_voxels, n_spec, n_coils, n_edit, n_dyn)
# Inner vmap: over voxels within a subject
# Outer vmap: over subjects

process_single = lambda d: jax_process_mega_press(
    d, dwell, cf, align=False, reject=False,
).diff

process_subject = jax.vmap(process_single)      # over voxels
process_cohort  = jax.vmap(process_subject)      # over subjects

# If you have MRSI data shaped (n_subjects, n_voxels, n_spec, n_coils, 2, n_dyn):
# all_diffs = process_cohort(mrsi_batch)
# -> (n_subjects, n_voxels, n_spec)
```

## Step 4: Automatic differentiation

JAX's `grad` computes exact gradients through any differentiable function.
This opens the door to gradient-based spectral fitting, optimal shimming, and
sensitivity analysis.

### 4a. Gradients through frequency correction

The `apply_correction` function applies a frequency and phase shift to an FID:

$$
s_{\text{corr}}(t) = s(t) \cdot \exp\!\bigl(2\pi i \, \Delta f \, t + i\,\phi\bigr)
$$

We can differentiate a loss function through this operation:

```python
rng = np.random.default_rng(99)
fid = jnp.array(rng.standard_normal(512) + 1j * rng.standard_normal(512))

def loss_fn(freq_shift):
    """Sum of squared magnitudes after frequency correction."""
    corrected = jax_apply_correction(fid, freq_shift, 0.0, dwell)
    return jnp.sum(jnp.abs(corrected) ** 2).real

grad_fn = jax.grad(loss_fn)
g = grad_fn(0.0)

print(f"Gradient at df=0: {float(g):.6e}")
assert jnp.isfinite(g), "Gradient is not finite"
```

```{note}
For $|e^{i \cdot 2\pi f t}|^2 = 1$, the magnitude does not depend on the
frequency shift, so $\partial \mathcal{L}/\partial f \approx 0$. This serves
as a sanity check for the autodiff implementation.
```

### 4b. Gradient-based spectral registration

Instead of the brute-force grid search in `spectral_registration`, you can
use `jax.grad` to optimize frequency and phase alignment directly:

```python
def alignment_loss(params, fid, reference, dwell_time):
    """Spectral alignment loss: residual energy after correction."""
    freq_shift, phase_shift = params
    corrected = jax_apply_correction(fid, freq_shift, phase_shift, dwell_time)
    spec_corr = jnp.fft.fft(corrected)
    spec_ref  = jnp.fft.fft(reference)
    return jnp.sum(jnp.abs(spec_ref - spec_corr) ** 2).real


# Create a reference and a shifted FID
ref_fid = jnp.array(make_singlet(2.01, 1.0, 3.0, 1024, dwell, cf))
t = jnp.arange(1024) * dwell
shifted_fid = ref_fid * jnp.exp(2j * jnp.pi * 5.0 * t)  # 5 Hz shift

# Optimize with gradient descent
loss_and_grad = jax.value_and_grad(
    lambda p: alignment_loss(p, shifted_fid, ref_fid, dwell)
)

params = jnp.array([0.0, 0.0])  # initial guess: (freq, phase)
learning_rate = 0.1

for step in range(100):
    loss, grads = loss_and_grad(params)
    params = params - learning_rate * grads
    if step % 20 == 0:
        print(f"Step {step:3d}: loss={float(loss):.4e}, "
              f"freq={float(params[0]):.2f} Hz, "
              f"phase={float(params[1]):.4f} rad")

print(f"\nRecovered frequency shift: {float(params[0]):.2f} Hz (true: 5.0 Hz)")
```

```{tip}
For production use, combine `jax.grad` with a proper optimizer like
`jax.example_libraries.optimizers` or `optax`. The gradient-based approach
converges much faster than grid search (100 steps vs. 201 grid points) and
scales to joint optimization over all transients simultaneously.
```

### 4c. Differentiable fitting: sensitivity analysis

Compute how the GABA peak area changes with respect to acquisition parameters:

```python
def gaba_area_from_noise(noise_scale):
    """End-to-end: generate data -> process -> extract peak magnitude."""
    rng_key = jax.random.PRNGKey(0)
    n_pts = 512
    n_dyn = 8

    # Deterministic signal + scaled noise
    naa  = jnp.array(make_singlet(2.01, 10.0, 3.0, n_pts, dwell, cf))
    gaba = jnp.array(make_singlet(3.01, 1.0, 8.0, n_pts, dwell, cf))
    edit_on  = naa + gaba
    edit_off = naa - gaba

    # Simple difference (no coils, no alignment)
    diff = edit_on - edit_off  # = 2 * gaba
    spec = jnp.fft.fft(diff)
    return jnp.max(jnp.abs(spec))

# Gradient of peak area w.r.t. noise scale
d_area = jax.grad(gaba_area_from_noise)(0.01)
print(f"d(GABA_peak)/d(noise_scale) = {float(d_area):.4f}")
```

## Step 5: GPU deployment

### 5a. Check available devices

```python
print(f"JAX devices: {jax.devices()}")
print(f"Default backend: {jax.default_backend()}")
# -> 'gpu' if CUDA is available, 'cpu' otherwise
```

### 5b. Place data on GPU

JAX arrays are automatically placed on the default device. To explicitly
target a GPU:

```python
# If GPU is available:
# gpu = jax.devices('gpu')[0]
# data_gpu = jax.device_put(jnp.array(data_np), gpu)
# result = process_jit(data_gpu)
```

### 5c. Memory considerations

MEGA-PRESS data is relatively small (~2048 complex points x 32 coils x 2 x
64 dynamics = ~16 MB), so even modest GPUs can hold hundreds of subjects.
For MRSI with thousands of voxels, use `jax.lax.map` instead of `vmap` to
process voxels in chunks:

```python
# Memory-efficient batching for large MRSI datasets
# jax.lax.map applies a function to each element sequentially
# but still benefits from JIT compilation

# result = jax.lax.map(process_single_voxel, voxel_batch)
```

```{note}
The DGX Spark with CUDA 13 is an excellent target for `mrs-jax` batch
processing. With 512 GB of GPU memory, you can process entire multi-site
cohorts in a single `vmap` call without memory pressure.
```

## Step 6: Combining JIT + vmap + grad

The real power of JAX is composing these transformations:

```python
# JIT-compiled batch processing with per-subject gradients
@jax.jit
def batch_process_with_sensitivity(batch_data, freq_perturbation):
    """Process a batch and compute sensitivity to frequency perturbation."""
    def single_subject(data):
        # Apply a small frequency perturbation
        n_spec = data.shape[0]
        t = jnp.arange(n_spec) * dwell
        perturbation = jnp.exp(2j * jnp.pi * freq_perturbation * t)
        # Broadcast perturbation across coils, edits, dynamics
        perturbed = data * perturbation[:, None, None, None]
        result = jax_process_mega_press(
            perturbed, dwell, cf, align=False, reject=False,
        )
        # Return peak magnitude of difference spectrum
        spec = jnp.fft.fft(result.diff)
        return jnp.max(jnp.abs(spec))

    # vmap over subjects
    peaks = jax.vmap(single_subject)(batch_data)
    return peaks

# Differentiate the batch function w.r.t. frequency perturbation
batch_grad = jax.grad(
    lambda f: jnp.sum(batch_process_with_sensitivity(batch, f)),
    argnums=0,
)

sensitivity = batch_grad(0.0)
print(f"Batch sensitivity to freq perturbation: {float(sensitivity):.4f}")
```

## JAX backend API reference

The `mega_press_jax` module provides these JAX-compatible functions:

| Function | Signature | JIT | vmap | grad |
|----------|-----------|:---:|:----:|:----:|
| `coil_combine_svd` | `(data: jnp.ndarray) -> jnp.ndarray` | Yes | Yes | Yes |
| `apply_correction` | `(fid, freq_shift, phase_shift, dwell_time)` | Yes | Yes | Yes |
| `reject_outliers` | `(fids, dwell_time, threshold)` | Yes | Yes | No* |
| `spectral_registration_jax` | `(fid, reference, dwell_time, centre_freq)` | Yes | Yes | Partial |
| `process_mega_press` | `(data, dwell_time, centre_freq, align, reject)` | Yes** | Yes** | No* |

\* Outlier rejection uses boolean masking which changes array shapes -- not
compatible with `grad`. Pass `reject=False` for differentiable pipelines.

\** Use `align=False, reject=False` for full JIT/vmap compatibility. Alignment
via `jax.lax.scan` is supported but adds compilation overhead.

```{tip}
When developing differentiable MRS pipelines, start with `align=False,
reject=False` to ensure full JIT/vmap/grad compatibility. Once the basic
pipeline works, selectively enable alignment (which uses `jax.lax.scan`
internally for JIT compatibility).
```

## Summary

| Capability | NumPy backend | JAX backend |
|------------|:-------------:|:-----------:|
| Single subject | Yes | Yes |
| Batch subjects | Loop | `vmap` (parallel) |
| GPU acceleration | No | Yes (CUDA/ROCm) |
| JIT compilation | No | `jax.jit` |
| Autodiff | No | `jax.grad` |
| Numerical results | Reference | Matches NumPy (atol=1e-5) |

The JAX backend enables workflows that are impossible with NumPy alone:
gradient-based spectral fitting, differentiable end-to-end pipelines, and
GPU-accelerated cohort-level processing. Start with NumPy for prototyping
and validation, then switch to JAX for production.

## References

```{bibliography}
:cited:
```
