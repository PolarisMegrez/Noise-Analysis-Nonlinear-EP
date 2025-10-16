# Noise Analysis for Nonlinear Exceptional Point Sensing — Supplementary Code

Supplementary code for nonlinear exceptional-point (EP) sensing and noise analysis. It provides a reproducible, config-driven pipeline to simulate user-defined complex ODE/SDE systems, run multiple initial conditions and/or stochastic replicates, and generate phase plots and PSDs per run.

Authors: Yu Xue-Hao and Qiao Cong-Feng (University of Chinese Academy of Sciences, UCAS)

## What it does (current)

- Config-driven runs (single JSON) with:
  - System: model file, parameters, time span, sampling density
  - ICs: multiple initial conditions (inline or external JSON)
  - Plotting: acquisition window、phase plots、PSD
  - Noise: Gaussian white noise via Euler–Maruyama (deterministic when disabled)
- PSD for each mode with two choices of analyzed signal:
  - amplitude: on |z(t)| (默认)
  - intensity: on complex z(t) (two-sided |Z(f)|^2)
- PSD options: method (Welch or multi-trajectory averaging), axis scale (linear/semilogy/semilogx/loglog), frequency range [fmin,fmax]
- Acquisition interval post-slicing: simulate once over t_span, then slice [t0,t1] for plotting/PSD
- Replot from saved results: no re-simulation; can override acquisition window and plotting options
- Progress with ETA per IC; metadata + raw arrays saved for reproducibility

See docs for details:
- Configuration: `docs/config_guide.md`
- PSD concepts/options: `docs/psd_guide.md`

## Install

```
pip install -r requirements.txt
```

Optional editable install (enables `python -m phase_diagram.main`):

```
python -m pip install -e .
```

## Run (Windows PowerShell)

From repository root:

- As a script (no install):
```
$env:PYTHONPATH = "$PWD\src"
python .\src\phase_diagram\main.py --config .\notebooks\input.json
```

- As a module (after editable install):
```
python -m phase_diagram.main --config .\notebooks\input.json
```

Each run saves to `runs/run-YYYYmmdd-HHMMSS/` (default base is `../runs`); key artifacts:
- `metadata.json` (paths, parameters, plotting/PSD options)
- `solutions.npz` (saved time grids and states)
- `figs/phase_trajectories.png`, `figs/modulus_phase_trajectories.png`, `figs/psd_modes_ic*.png`

Replot from saved results (no simulation):
```
python -m phase_diagram.main --replot .\runs\run-YYYYmmdd-HHMMSS --acquire_interval "20,120"
```

The demo notebook `notebooks/Phase_Diagram.ipynb` includes a ready-to-use cell to replot from a run directory, with interactive inputs (acquire interval, PSD signal/axes/range).

## Configure briefly

Top-level keys in the JSON config:
- `system`: `{ py, t_span, t_points, out, params }`
- `ic` or `ic_json`: initial conditions (inline or external file)
- `plotting`: `{ var_index, mod_i, mod_j, acquire_interval, plot_phase, plot_psd, psd_signal, psd_axis_scale, psd_freq_range, show }`
- `noise`: optional Gaussian-white noise (`type`, `D` or model, `solver`)
- `psd`: `{ method: "welch" | "multi-trajectory", params: {...} }`

Refer to `docs/config_guide.md` for field definitions, validation rules (e.g., acquire_interval clamping), and examples.

## Project layout

```
src/phase_diagram/
  dynamics.py   # ODE/SDE solvers and helpers
  io.py         # JSON I/O, run-folder creation, save/load artifacts
  main.py       # CLI entry and orchestration (run/replot)
  plotting.py   # phase/PSD plotting and PSD computation
src/models/
  vanderpol.py  # example two-mode model used by sample configs
docs/
  config_guide.md  # configuration guide
  psd_guide.md     # PSD guide
notebooks/
  input.json, vdp_welch.json, vdp_multi_traj.json  # example configs
  Phase_Diagram.ipynb  # demo + replot cell
runs/ (or ../runs)  # per-run outputs (created automatically)
```

## Notes on provenance and license

This project was authored by the team with assistance from GitHub Copilot (AI pair programming).

License: MIT. See the `LICENSE` file at the repository root.