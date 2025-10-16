# Noise Analysis for Nonlinear Exceptional Point Sensing — Supplementary Code

This repository contains supplementary code accompanying a research paper under preparation on nonlinear exceptional-point (EP) sensing and noise analysis. The codebase provides a reproducible pipeline to simulate user-defined complex ODE systems, sweep multiple initial conditions, and generate phase-space visualizations and data artifacts per run.

Authors: Yu Xue-Hao and Qiao Cong-Feng (University of Chinese Academy of Sciences, UCAS)

## Overview

- Language: Python (tested on Python 3.12, Windows; should work on recent Python 3.x on macOS/Linux).
- Purpose: numerically integrate first-order ODE systems defined in complex form, over multiple initial conditions, and save results and plots for noise/EP-sensing analysis workflows.
- Key modules:
	- `phase_diagram.dynamics`: solver utilities for complex ODEs (Solve IVPs; handle multiple ICs)
	- `phase_diagram.plotting`: phase-plane and modulus–modulus trajectory plots (start/end markers included)
	- `phase_diagram.io`: I/O helpers (read JSON inputs, create run folders, save results/metadata)
		- `systems.vanderpol`: example two-mode system (α, β) used in our experiments

## System model used in examples

The example two-mode complex ODE (state z = [α, β]):

	dα/dt = [ -i·ω_a + (γ_a/2) + Γ·(1/2 − |α|^2) ]·α − i·g·β

	dβ/dt = [ -i·ω_b − (γ_b/2) ]·β − i·g·α

Defined in `src/systems/vanderpol.py` as function `vanderpol(t, z, **params)`.

## Installation

Install dependencies:

```
pip install -r requirements.txt
```

Optional editable install (enables `python -m phase_diagram.main`):

```
python -m pip install -e .
```

## Configuration (single input “port”)

Edit `notebooks/alpha_beta_example.json` to control the system, initial conditions, and run options. Example:

```
{
	"system": {
		"py": "../src/systems/vanderpol.py",
		"func": "vanderpol"
	},
	"ic": {
		"count": 3,
		"n_vars": 2,
		"t_span": [0.0, 100.0],
		"initial_conditions": [
			[[0.8, 0.0], [0.0, 0.0]],
			[[0.5, 0.5], [0.2, -0.2]],
			[[-0.2, 0.1], [0.6, 0.0]]
		],
		"params": {
			"omega_a": 1.0,
			"omega_b": 1.2,
			"gamma_a": 0.04,
			"Gamma": 0.1,
			"gamma_b": 0.1,
			"g": 0.4
		}
	},
	"run": {
		"var_index": 0,
		"mod_i": 0,
		"mod_j": 1,
		"t_points": 2000,
		"out": "../runs",
		"show": false
	}
}
```

Notes:
- A system function must have signature `f(t: float, z: np.ndarray[complex], **params) -> np.ndarray[complex]` and return the complex derivative vector dz/dt.
- ICs can be provided inline (as above) or via an external JSON (use `ic_json`/`ic_file` keys instead of `ic`).

### Noise configuration (optional)

Stochastic differential equations (c-number Langevin) are supported via Euler–Maruyama with Gaussian white noise. Add a `noise` block to the config:

```
"noise": {
	"type": "gaussian-white",      # or "none"
	"D": [[1e-4, 0, 0, 0],          # diffusion matrix in real-expanded space (2n x 2n)
				[0, 1e-4, 0, 0],
				[0, 0, 1e-4, 0],
				[0, 0, 0, 1e-4]],
	"seed": 12345                    # optional RNG seed for reproducibility
}
```

Notes:
- For n complex variables, the real-expanded dimension is 2n; `D` must be (2n x 2n).
- Currently implemented: `type = none` (no noise) or `gaussian-white`.
- Future noise models can be added without breaking the interface.

## How to run (Windows PowerShell)

From repository root:

Option A — Run as a script (no install):
```
$env:PYTHONPATH = "$PWD\src"
python .\src\phase_diagram\main.py --config .\notebooks\alpha_beta_example.json
```

Option B — Run as a module (after editable install):
```
python -m phase_diagram.main --config .\notebooks\alpha_beta_example.json
```

Each run saves to `runs/run-YYYYmmdd-HHMMSS/` (by default placed at the parent of the workspace, i.e., `../runs`):
- input.json (copied)
- metadata.json (paths, parameters, time grid, etc.)
- solutions.npz (arrays `t_i`, `y_i` per IC)
- figs/phase_trajectories.png (complex plane for selected variable)
- figs/modulus_phase_trajectories.png (|z_i| vs |z_j|)

Plot markers:
- Start of each trajectory: hollow triangle (same color as line)
- End of each trajectory: hollow cross (same color as line)

## Define your own system

Create a Python file exporting a function:

```
def my_system(t: float, z: np.ndarray, **params) -> np.ndarray:
		# z is a complex vector; return dz/dt as a complex vector of same shape
		...
```

Point the config to that file via `system.py` and `system.func`.

## Reproducibility

The runner copies the used configuration JSON (and IC JSON if provided) into each run folder and saves both metadata and raw solution arrays to enable full offline reproduction.

## Project structure

```
src/phase_diagram/
	dynamics.py     # complex ODE solvers and helpers
	io.py           # JSON I/O, run-folder creation, saving artifacts
	main.py         # CLI entrypoint and run() orchestration
	plotting.py     # plotting utilities (phase + modulus)
src/systems/
	vanderpol.py    # example two-mode (α, β) system used in the paper
notebooks/
	alpha_beta_example.json  # example config
	Phase_Diagram.ipynb      # demo notebook
runs/ (or ../runs)       # per-run outputs generated automatically
```

## Troubleshooting

- Module not found (phase_diagram): ensure `$env:PYTHONPATH = "$PWD\src"` or install with `pip install -e .`.
- PowerShell quoting: prefer double-quotes for `-Command` strings and single-quotes inside them to avoid escaping issues.
- Empty figures or no outputs: confirm `input.json` paths are correct relative to the file’s location.