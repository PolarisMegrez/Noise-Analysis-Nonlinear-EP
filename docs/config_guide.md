# Configuration Guide

This project is controlled by a single JSON file. This guide explains each section with clearly quoted names so they render in monospace.

## Top-level layout

Top-level keys:
- `'system'`: Model file and simulation settings (time span, sampling, parameters, output base).
- `'ic'`: Inline initial conditions (alternative to `'ic_json'`).
- `'ic_json'`: Path to a JSON file with initial conditions.
- `'plotting'`: Plot controls.
- `'noise'`: Optional stochastic noise configuration.
- `'psd'`: Power spectral density configuration.

Either `'ic'` or `'ic_json'` must be provided.

## 'system'

- `'py'`: Path to the Python file under `'src/models'` (resolved relative to this config file).
- `'t_span'`: `[t0, t1]` total time interval for simulation.
- `'t_points'` (optional): Total samples to create a uniform time grid.
- `'out'` (optional): Base output directory (default `'../runs'`).
- `'params'`: Arbitrary parameters forwarded to the system function.

Notes:
- The system function is `'ODEs'` in the given Python file; `'func'` 参数不使用。

Example:
```json
"system": { "py": "../src/models/vanderpol.py", "t_span": [0.0, 120.0], "t_points": 5000, "out": "../runs", "params": {"omega_a": 1.0} }
```

## 'ic' (inline)

- `'n_vars'`: Number of complex variables.
- `'count'`: Number of initial conditions.
- `'initial_conditions'`: List of length `'count'`, each with `'n_vars'` complex values (as `[real, imag]` or string like "1+0j").
- `'params'`: Optional; normally put model parameters in `'system.params'` (the runner accepts either; `'system.params'` overrides).

Example:
```json
"ic": {
  "n_vars": 2,
  "count": 3,
  "initial_conditions": [
    [[0.5, 0.0], [0.0, 0.0]],
    [[0.6, 0.1], [0.0, 0.0]],
    [[0.4, -0.2], [0.0, 0.0]]
  ]
}
```

Alternatively, provide `'ic_json'` pointing to a separate file with the same structure.

## 'plotting'

- `'var_index'`: Index to use in the complex-plane phase plot.
- `'mod_i'`, `'mod_j'`: Indices for the `'|z_i| vs |z_j|'` plot.
- `'acquire_interval'` (optional): `[t_start, t_end]` window within `'system.t_span'` used for plotting and PSD; defaults to the full span if omitted.
- `'plot_phase'` (default `true`): draw phase plots.
- `'plot_psd'` (default `true`): draw PSD per IC.
- `'show'`: Whether to show figures interactively.
- `'psd_axis_scale'` (optional): `'linear' | 'semilogy' | 'semilogx' | 'loglog'` axis scaling for PSD plots (default `'linear'`). Note: `'semilogx'`/`'loglog'` drop non-positive frequencies automatically.
- `'psd_freq_range'` (optional): `[fmin, fmax]` to limit the plotted frequency range (two-sided frequencies after fftshift). If omitted, the full range is plotted.
- `'psd_signal'` (optional): `'amplitude'` or `'intensity'` to choose PSD computed on $|z(t)|$ (amplitude spectrum, default) or on complex $z(t)$ (two-sided intensity spectrum with $|Z(f)|^2$).

Notes:
- 不区分“预热/采集”阶段：仿真按 `'system.t_span'` 进行一次，绘图和 PSD 在结束后按 `'plotting.acquire_interval'` 对保存的时间序列进行切片。
- 在绘制之前会校验采集窗口：
  - If `acquire_interval[0] >= system.t_span[1]`, an error is raised (no overlap).
  - If `acquire_interval[1] > system.t_span[1]`, it is clamped to `system.t_span[1]` with a warning that includes the original and clamped values.
- Progress is printed as `[IC k/N] solving...` with a percentage and ETA per IC.

Example:
```json
"plotting": {
  "var_index": 0,
  "mod_i": 0,
  "mod_j": 1,
  "acquire_interval": [20.0, 120.0],
  "psd_axis_scale": "semilogy",
  "psd_freq_range": [-2.0, 2.0],
  "psd_signal": "intensity",
  "show": false
}
```

## 'noise' (optional)

- `'type'`: `'none'` or `'gaussian-white'`.
- `'D'`: Constant diffusion matrix in real-expanded coordinates (size `'2n x 2n'`).
- `'seed'`: RNG seed (optional).
- `'params'`: Forwarded to a model-provided `'diffusion_matrix(t, z, **params)'` if present.
- `'expectation'`: expectation policy for quantities like `'E|z_k|^2'`:
  - `'type'`: `'instant'` | `'time-window'` | `'multi-trajectory'` | `'average'` (alias mapped at runtime)
  - `'params'`: `{ "var_index": <int>, "tau": <float>, "alpha": <0..1> }`
- `'model'` (optional): `{ "py": <path>, "func": <name>, "params": {...} }` to dynamically load a diffusion generator.
- `'solver'` (optional): options for the SDE solver:
  - `'adaptive'`: `true|false` to enable adaptive Euler–Maruyama with step-doubling (default `false`).
  - `'atol'`, `'rtol'`, `'h_init'`, `'h_min'`, `'h_max'`, `'safety'`: error control and step size hints.

Example (constant `'D'`):
```json
"noise": {
  "type": "gaussian-white",
  "D": [[1e-4,0,0,0],[0,1e-4,0,0],[0,0,1e-4,0],[0,0,0,1e-4]],
  "seed": 42,
  "solver": { "adaptive": false, "rtol": 1e-3, "atol": 1e-4 }
}
```

Notes:
- If the system module defines `'diffusion_matrix(t, z, **params)'`, it is auto-used when `'noise.type'` is `'gaussian-white'`.
- The diffusion matrix must be real and of shape `'(2n, 2n)'` in the real-expanded variables.
- Default: `'solver.adaptive'` is `false` if not specified.
- When `'psd.method'` is `'multi-trajectory'` (replicates) the solver uses a synchronous lock-step Euler–Maruyama. If `'solver.adaptive'` is `true`, it will be automatically disabled with an informational message.

## 'psd' (optional)

Choose how to compute the PSD of `'|z_i|'` for each mode. Two methods are available:

1) Welch (default)
- `'method'`: `'welch'`
- `'params'`: `{ "nperseg": <int>, "window": <name|tuple>, "noverlap": <int>, "detrend": "constant|linear" }`

Example:
```json
"psd": { "method": "welch", "params": { "nperseg": 512, "window": "hann", "noverlap": 128, "detrend": "constant" } }
```

2) Multi-trajectory (replicates per IC)
- `'method'`: `'multi-trajectory'`
- `'params'`: `{ "replicates": <int>, "window": <name|tuple>, "detrend": "constant|linear|null" }`（支持 `'trajectories'` 作为等价键）

Example:
```json
"psd": { "method": "multi-trajectory", "params": { "replicates": 100, "window": "boxcar", "detrend": null } }
```

Notes:
- Both methods assume a uniform time grid. Provide `'system.t_points'` to ensure this. If an adaptive solver is enabled with Welch, the code will interpolate to a uniform grid before computing PSD and print a warning.
- PSD figures are generated per IC: `'figs/psd_modes_ic{k}.png'`.
- Toggle with `'plotting.plot_psd'`.

The data used for PSD is taken from the `'plotting.acquire_interval'` window if provided; otherwise the full `'system.t_span'` is used.

## Replot from a saved run

You can regenerate figures from previously saved data (no re-simulation) and adjust the acquisition window on the fly:

Optional command (Windows PowerShell):

```
python -m phase_diagram.main --replot "..\runs\run-20250101-120000" --acquire_interval "20,120"
```

This reads `solutions.npz` and `metadata.json` in that run folder, slices by the given `'acquire_interval'`, and writes new figures into `figs/`.

## Minimal end-to-end example

```json
{
  "system": { "py": "../src/models/vanderpol.py", "t_span": [0.0, 120.0], "t_points": 5000, "out": "../runs" },
  "ic_json": "./vdp_ic.json",
  "plotting": { "var_index": 0, "mod_i": 0, "mod_j": 1, "acquire_interval": [20.0, 120.0], "show": false },
  "noise": {
    "type": "gaussian-white", "seed": 42,
    "solver": { "adaptive": false, "rtol": 1e-3, "atol": 1e-4 }
  },
  "psd": { "method": "welch", "params": { "nperseg": 512 } }
}
```