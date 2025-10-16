# Power Spectral Density (PSD) Guide

This guide explains how PSD is computed in this project: what signal we analyze, two PSD signal choices (amplitude vs intensity), the two estimation strategies (Welch vs multi-trajectory averaging), and how to configure and interpret results.

## What signal do we analyze? Two options

The state is a complex vector $z(t) = [z_0(t), z_1(t), \dots]^T$. For each mode/index $i$, you can choose one of two PSD signals via `plotting.psd_signal`:

1) Amplitude spectrum (`"amplitude"`, default)

- Signal: $x_i(t) = |z_i(t)| = \sqrt{\operatorname{Re}(z_i)^2 + \operatorname{Im}(z_i)^2}$
- Use when you're interested in the modulation of the magnitude (envelope dynamics) of each complex mode.
- Frequency axis is two-sided in code for consistency, but practical content is nonnegative; plotting can be linear/semilog as configured.

2) Intensity spectrum (`"intensity"`)

- Signal: the complex process itself, $x_i(t) = z_i(t)$, and PSD is based on $|Z_i(f)|^2$ with a two-sided spectrum (using `return_onesided=False` or full FFT), then `fftshift` for display.
- Use when literature defines PSD on the complex variable (e.g., modal intensity in linear response analysis) and negative frequencies carry symmetric information.

In code:
- `compute_modes_complex(sol)` returns complex arrays for all modes.
- `compute_modes_amplitudes(sol)` returns amplitude arrays for all modes.
- The plotting pipeline accepts `signal='amplitude'|'intensity'` and computes PSD accordingly.

## PSD definition and discrete computation

For a continuous-time process $x(t)$, the PSD can be defined via the Wiener–Khinchin theorem:
$$
S_{xx}(f) = \int_{-\infty}^{\infty} R_{xx}(\tau)\, e^{-j 2\pi f \tau}\, d\tau,
$$
where $R_{xx}(\tau)$ is the autocorrelation. For a finite-length signal of duration $T$, a common approximation is the periodogram:
$$
\hat{S}_{xx}(f) \approx \frac{1}{T}\,\bigg|\int_0^T x(t)\, e^{-j 2\pi f t}\, dt\bigg|^2.
$$

In discrete time with sampling interval $\Delta t$, length $N$, and $x[n] = x(n\,\Delta t)$, the discrete frequencies are $f_k = \tfrac{k}{N\,\Delta t}$. With a window $w[n]$ the single-record periodogram is
$$
\hat{S}_{xx}(f_k) = \frac{\Delta t}{N}\,\Big|\sum_{n=0}^{N-1} x[n]\, w[n] \, e^{-j 2\pi k n / N}\Big|^2.
$$

- In the vectorized path we use `'numpy.fft.rfft'` to implement this, with density-type scaling so that $\int S_{xx}(f)\, df \approx \operatorname{Var}(x)$.
- In the non-aligned path and in the Welch method we call SciPy’s `'periodogram(..., scaling="density")'` and `'welch(...)'` to keep consistent units and scaling.

## Strategy 1: Welch method (segment averaging)

File: `src/phase_diagram/plotting.py`, function `compute_psd_modes_average_welch`.

Idea: split a time series into (possibly overlapping) segments, apply a window and compute a periodogram per segment, then average across segments to reduce variance. In this project:

- For each solution/trajectory and each selected signal (amplitude or complex), we call `scipy.signal.welch`.
- Then we average across trajectories (e.g., multiple ICs).
- Key parameters (pass in JSON under `'psd.params'`):
  - `'nperseg'`: segment length (auto-truncated if longer than the series).
  - `'window'`: window name or tuple (e.g., `'hann'`, `'boxcar'`).
  - `'noverlap'`: number of overlapping samples.
  - `'detrend'`: `'constant'` or `'linear'` (default `'constant'`).
- Sampling frequency is $f_s = 1/\Delta t$, with $\Delta t$ from `'sol.t'` (uniform by default).

Returns a shared frequency axis `'f'` and a PSD array of shape `'(num_modes, len(f))'`.

## Strategy 2: Multi-trajectory periodogram averaging (replicates)

File: `src/phase_diagram/plotting.py`, function `compute_psd_modes_average_multi_traj`.

Use case: for one IC, simulate multiple independent stochastic replicates. Compute a periodogram for each replicate’s selected signal and average across replicates (ensemble averaging).

Implementation notes:

- If all replicates share an identical time grid (same length and step), a vectorized FFT path is used:
  1) Optional detrending `'constant'` (remove mean) or `'linear'` (remove LS line).
  2) Apply a `'window'` (default `'boxcar'`, i.e., none).
  3) Use FFT per mode along time and average power across replicates. For `signal='intensity'`, use full complex FFT; for `signal='amplitude'`, FFT of the real, nonnegative amplitude.
- If time grids differ, we fall back to per-replicate `'periodogram'` and interpolate to a common frequency axis before averaging.
- Key parameters (under `'psd.params'`):
  - `'replicates'` (alias `'trajectories'`): number of replicates to include.
  - `'window'`: window function.
  - `'detrend'`: `'constant'`, `'linear'`, or `null`.

Returns a shared frequency axis `'f'` and a PSD array of shape `'(num_modes, len(f))'`.

Note: in the per-IC replicate workflow, each IC typically gets its own PSD figure.

## Plotting and masking

- Use `plot_psd_modes(...)` to draw PSDs. Supported `method`: `welch` or `multi-trajectory` (aliases `multi_traj`, `multi`). The `signal` parameter selects amplitude vs intensity.
- A display-only numeric mask is applied: any points with $S < 10^{-10}$ are omitted from the plotted curve. This does not affect the computed PSD values, only their visualization.

Axis scaling and frequency range:
- `plotting.psd_axis_scale`: `linear|semilogy|semilogx|loglog`.
- `plotting.psd_freq_range`: `[fmin, fmax]` to limit the displayed two-sided frequency range.
- For `semilogx|loglog`, non-positive frequencies are dropped automatically (only positive frequencies are shown).

## Configuration examples

Welch method (amplitude spectrum):

```jsonc
{
  "psd": {
    "enabled": true,
    "method": "welch",
    "params": {
      "nperseg": 256,
      "window": "hann",
      "noverlap": 128,
      "detrend": "constant"
    }
  },
  "plotting": { "plot_psd": true, "psd_signal": "amplitude", "psd_axis_scale": "semilogy" }
}
```

Multi-trajectory averaging (replicates for one IC, intensity spectrum):

```jsonc
{
  "psd": {
    "enabled": true,
    "method": "multi-trajectory",
    "params": {
      "replicates": 100,       // alias: trajectories
      "window": "boxcar",
      "detrend": null
    }
  },
  "plotting": { "plot_psd": true, "psd_signal": "intensity", "psd_freq_range": [-2.0, 2.0] },
  "noise": {
    "type": "gaussian-white",
    "seed": 1234,
    "expectation": {
      "type": "multi-trajectory", // or "instant" / "time-window"
      "params": { "var_index": 0, "tau": 10.0 }
    }
  }
}
```

Tip: `'replicates'` controls the number of ensemble members. With identical time grids, the implementation uses a fast vectorized FFT.

## Relation to noise expectation policies

Expectation policies (`'instant'`, `'time-window'`, `'multi-trajectory'`) affect the diffusion and thus the statistics of the simulated time series, but do not change the PSD definition:

- `'instant'`: update diffusion from instantaneous $|z_k|^2$.
- `'time-window'`: exponential moving average (via `'tau'` or `'alpha'`).
- `'multi-trajectory'`: use ensemble mean across trajectories/replicates per time step.

These policies change the resulting statistics of $x_i(t)$ and therefore the observed PSD.

## Outputs and shapes

- Returned data: frequency vector `f` and PSD matrix `S` with `S.shape = (num_modes, len(f))`.
- Plotting: `plot_psd_modes(...)` draws one curve per mode. In per-IC replicate flows, each IC yields its own PSD figure. The y-label reflects the chosen signal (`PSD of |z|` for amplitude; `PSD |Z(f)|^2` for intensity).
- Save locations: governed by the run’s output folder (e.g., `'runs/.../figs'`).

## Practical tips

- Resolution: frequency resolution is roughly $\Delta f = f_s / N$. For finer resolution, use longer signals or higher sample rates (smaller $\Delta t$).
- Parameter choices:
  - In Welch, `'nperseg'` around 1/8–1/4 of the series length is a common starting point.
  - A tapered window (e.g., `'hann'`) reduces leakage; 50% overlap is typical.
  - Consider detrending if slow drifts are present to avoid excessive low-frequency power.
- Grid consistency: aim to use the same time grid across trajectories/replicates to enable vectorization and avoid interpolation.
- Reproducibility: set `'noise.seed'`. Different seeds change ensemble statistics.

## Functions at a glance

- `compute_modes_amplitudes(sol)`: extract $|z_i|$ and `sol.t` from a solution.
- `compute_psd_modes_average_welch(...)`: PSD via `scipy.signal.welch`, averaged across trajectories; supports `signal`.
- `compute_psd_modes_average_multi_traj(...)`: replicate periodogram averaging; vectorized when time grids align; supports `signal`.
- `plot_psd_modes(...)`: unified plotting entry with display masking and options (`signal`, `axis_scale`, `freq_range`).

For configuration structure and complete examples, see `'docs/config_guide.md'` and the JSONs in `'notebooks/'`.