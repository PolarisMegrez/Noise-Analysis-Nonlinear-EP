# Configuration Guide

This project is driven by a single JSON configuration file. This guide explains each section and shows examples, including how to choose the PSD computation method.

## File layout

Top-level keys:
- system: Points to the Python file and function that define the dynamical system.
- ic: Inline initial condition block (optional if ic_json is used).
- ic_json: Path to a JSON with initial conditions (alternative to inline ic).
- run: Execution options (what to plot, sample count, etc.).
- noise: Optional stochastic noise configuration.
- psd: Optional power spectral density configuration.

You can mix `ic` (inline) or `ic_json` (external file). At least one must be provided.

## system

- py: Path to the Python file defining the model (under `src/models`). Relative paths are resolved against this config’s directory.
- func: Function name inside that file. Recommended unified name: `ODEs(t, z, **params)` returning a complex vector dz/dt.

Example:
```json
"system": {
  "py": "../src/models/vanderpol.py",
  "func": "ODEs"
}
```

## ic (inline)

- n_vars: Number of complex variables in the system.
- count: Number of initial conditions.
- t_span: `[t0, t1]` time interval.
- initial_conditions: List of length `count`, each a list of `n_vars` complex numbers (as `[real, imag]` or string like "1+0j").
- params: Arbitrary parameter dictionary passed to the system function.

Example inline IC block:
```json
"ic": {
  "n_vars": 2,
  "count": 3,
  "t_span": [0.0, 200.0],
  "initial_conditions": [
    [[0.5, 0.0], [0.0, 0.0]],
    [[0.6, 0.1], [0.0, 0.0]],
    [[0.4, -0.2], [0.0, 0.0]]
  ],
  "params": {"omega_a": 1.0, "omega_b": 1.2, "gamma_a": 0.04, "Gamma": 0.1, "gamma_b": 0.1, "g": 0.4}
}
```

Alternatively, supply `ic_json` to a separate file with the same structure.

## run

- var_index: Which complex variable index to use in the complex-plane trajectory figure.
- mod_i, mod_j: Indices for the |z_i| vs |z_j| figure.
- t_points: Number of sample points in t_eval (uniform). If omitted, the integrator chooses.
- out: Base output directory for run folders. Defaults to `../runs`.
- show: Whether to show figures interactively. Typically `false` for batch runs.

Example:
```json
"run": {
  "var_index": 0,
  "mod_i": 0,
  "mod_j": 1,
  "t_points": 2000,
  "out": "../runs",
  "show": false
}
```

## noise (optional)

- type: "none" or "gaussian-white".
- D: Constant diffusion matrix in real-expanded coordinates (size 2n x 2n). For n=2, a 4x4 matrix.
- seed: RNG seed (optional) for reproducibility.
- params: Parameters passed into the model’s `diffusion_matrix(t, z, **params)`. Typical keys include `D_scale`, `amp_smoothing`, etc.
- expectation: Optional policy controlling how expectation-like quantities (e.g., E|α|^2) are estimated:
  - type: one of `instant`, `time-window`, `multi-trajectory`.
  - params:
    - var_index: which complex variable index to use (default 0).
    - For time-window: `tau` (seconds) or `alpha` (smoothing factor in (0,1)).
    - For multi-trajectory: no extra params required (uses synchronous averaging across ICs per step).

Example (constant D):
```json
"noise": {
  "type": "gaussian-white",
  "D": [[1e-4,0,0,0], [0,1e-4,0,0], [0,0,1e-4,0], [0,0,0,1e-4]],
  "seed": 42
}
```

Notes:
- If the system module defines a `diffusion_matrix(t, z, **params)`, it is used automatically when `noise.type` is `gaussian-white`.
- The function receives the current time `t` and complex state vector `z` and must return a real (2n x 2n) diffusion matrix for the real-expanded system.

## psd (optional)

Choose how to compute the power spectral density (PSD) of |z_i| for each mode. Two methods are available:

1) Welch method (default)
- method: "welch"
- params:
  - nperseg: Segment length (default 256).
  - window: Window name or tuple (default "hann").
  - noverlap: Overlap between segments (optional; must be < nperseg if provided).
  - detrend: Detrend option (default "constant").

Example:
```json
"psd": {
  "method": "welch",
  "params": {
    "nperseg": 512,
    "window": "hann",
    "noverlap": 128,
    "detrend": "constant"
  }
}
```

2) Multi-trajectory average of periodograms
- method: "multi-trajectory"
- params:
  - trajectories: Number of trajectories (ICs) to average (default: all).
  - window: Window applied to each trace before FFT (default: "boxcar" i.e., none).
  - detrend: Optional detrending per trace ("constant" or "linear").

Example:
```json
"psd": {
  "method": "multi-trajectory",
  "params": {
    "trajectories": 3,
    "window": "boxcar",
    "detrend": "constant"
  }
}
```

Notes:
- Both methods assume a uniform time grid (t_eval is uniform when `t_points` is set).
- Figures are saved under `run-YYYYmmdd-HHMMSS/figs/psd_modes.png`.
- No parallelization is used for the multi-trajectory method at this time.

## Putting it together

A minimal full config:
```json
{
  "system": {"py": "../src/systems/vanderpol.py", "func": "vanderpol"},
  // becomes
  "system": {"py": "../src/models/vanderpol.py", "func": "ODEs"},
  "ic_json": "./vdp_ic.json",
  "run": {"var_index": 0, "mod_i": 0, "mod_j": 1, "t_points": 2000, "out": "../runs", "show": false},
  "noise": {"type": "gaussian-white", "params": {"D_scale": 1e-4}, "seed": 42},
  "psd": {"method": "welch", "params": {"nperseg": 512}}
}
```