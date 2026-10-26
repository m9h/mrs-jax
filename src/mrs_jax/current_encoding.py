"""Ideal current-induced phase histories for spectroscopy forward models.

Reference equation: phi(r, t) = sign * gamma * b(r) * integral_0^t I(u) q(u) du.
The integration is exact for piecewise-constant current and signed RF
sensitivity; this is not an RF/Bloch solver or a hardware controller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

GAMMA_RAD_S_T = 2 * np.pi * 42.57747892e6


def _windows(value: Any, columns: int, name: str) -> np.ndarray:
    result = np.array(value, dtype=float, copy=True)
    if result.ndim != 2 or result.shape[1] != columns or not len(result):
        raise ValueError(f"{name} must have shape (n, {columns}), n > 0")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    if np.any(result[:, 1] <= result[:, 0]):
        raise ValueError(f"{name} must contain positive-duration intervals")
    if np.any(result[1:, 0] < result[:-1, 1]):
        raise ValueError(f"{name} must be ordered and nonoverlapping")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class CurrentEncoding:
    """Validated timing schedule, constructed outside JAX transformations.

    Parameters
    ----------
    sample_times_s : array_like, shape (n_samples,)
        Nonnegative, nondecreasing ADC times relative to transverse excitation
        at zero. Repeated times allow ideal instantaneous spatial encoding.
    current_windows_s : array_like, shape (n_current, 2)
        Ordered, nonoverlapping start/end times in seconds. Amplitudes are
        supplied to :meth:`phase`; gaps and times outside windows have zero
        current. Negative start times are allowed.
    sensitivity_intervals : array_like, shape (n_sensitivity, 3)
        Ordered start/end times in seconds and ideal toggling weights q in
        {-1, 0, +1}. Gaps have zero sensitivity. These weights encode only the
        additional field's phase response, not metabolite spin evolution.

    Returns
    -------
    CurrentEncoding
        Immutable schedule with copied, read-only arrays.
    """

    sample_times_s: np.ndarray
    current_windows_s: np.ndarray
    sensitivity_intervals: np.ndarray

    def __post_init__(self) -> None:
        times = np.array(self.sample_times_s, dtype=float, copy=True)
        if (times.ndim != 1 or not times.size or not np.all(np.isfinite(times))
                or np.any(times < 0) or np.any(np.diff(times) < 0)):
            raise ValueError("sample_times_s must be finite, nonnegative and nondecreasing")
        times.setflags(write=False)
        currents = _windows(self.current_windows_s, 2, "current_windows_s")
        sensitivity = _windows(self.sensitivity_intervals, 3, "sensitivity_intervals")
        if not np.all(np.isin(sensitivity[:, 2], [-1, 0, 1])):
            raise ValueError("sensitivity weights must be -1, 0 or +1")
        object.__setattr__(self, "sample_times_s", times)
        object.__setattr__(self, "current_windows_s", currents)
        object.__setattr__(self, "sensitivity_intervals", sensitivity)

    def phase(
        self,
        field_per_amp_t: Any,
        current_a: Any,
        *,
        phase_sign: int,
        current_delay_s: Any = 0.0,
        backend: str = "numpy",
    ) -> Any:
        """Integrate the additional phase at every ADC sample.

        Parameters
        ----------
        field_per_amp_t : array_like, shape (...,)
            Spatial magnetic field per ampere (T/A), already projected along
            scanner B0. A scalar represents one voxel. Multiple independent
            current paths can be modeled by summing their phase histories.
        current_a : array_like, shape (n_current,)
            Current in amperes in each current window.
        phase_sign : {-1, +1}
            Required signal convention. +1 matches mrs-jax's positive-Hz
            phase evolution; -1 matches the Mx+iMy convention used by the
            neuroelectric-mri analytical references. Calibrate the mapping
            to acquired data rather than inferring it from a file format.
        current_delay_s : scalar
            Delay of current windows relative to the fixed RF/ADC clock.
            Positive values move current later. Phase is zero at t=0.
        backend : {'numpy', 'jax'}
            JAX supports differentiation through field, amplitudes and delay
            away from interval boundaries, where delay derivatives can kink.
            Validate traced parameter values before JIT; only their shapes
            can be checked inside transformed code.

        Returns
        -------
        array, shape field_per_amp_t.shape + (n_samples,)
            Unwrapped additional phase in radians. No baseline B0 phase,
            gradient phase, chemical shift or hardware response is included.
        """
        if phase_sign not in (-1, 1):
            raise ValueError("phase_sign must explicitly be -1 or +1")
        if backend == "numpy":
            xp = np
        elif backend == "jax":
            import jax.numpy as xp
        else:
            raise ValueError("backend must be 'numpy' or 'jax'")
        field = xp.asarray(field_per_amp_t)
        current = xp.asarray(current_a)
        delay = xp.asarray(current_delay_s)
        if current.shape != (len(self.current_windows_s),) or delay.shape != ():
            raise ValueError("current_a needs one amplitude per window; delay must be scalar")
        if any(xp.iscomplexobj(x) for x in (field, current, delay)):
            raise ValueError("field, current and delay must be real")
        if backend == "numpy" and not all(
                np.all(np.isfinite(x)) for x in (field, current, delay)):
            raise ValueError("field, current and delay must be finite")
        windows = xp.asarray(self.current_windows_s) + delay
        sensitivity = xp.asarray(self.sensitivity_intervals)
        # Each overlap is clipped by excitation zero and each ADC endpoint.
        starts = xp.maximum(0, xp.maximum(windows[:, 0, None], sensitivity[None, :, 0]))
        ends = xp.minimum(windows[:, 1, None], sensitivity[None, :, 1])
        overlaps = xp.maximum(
            0, xp.minimum(xp.asarray(self.sample_times_s)[:, None, None], ends) - starts
        )
        area_as = xp.sum(overlaps * current[None, :, None]
                         * sensitivity[None, None, :, 2], axis=(1, 2))
        return phase_sign * GAMMA_RAD_S_T * field[..., None] * area_as
