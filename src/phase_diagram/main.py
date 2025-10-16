import argparse
import json
import os
import sys
import shutil
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import numpy as np

# Support running as a script (python src/phase_diagram/main.py) and as a module
if __package__ is None or __package__ == "":
    # Add <repo_root>/src to sys.path for absolute imports
    repo_src = Path(__file__).resolve().parents[2] / "src"
    if str(repo_src) not in sys.path:
        sys.path.insert(0, str(repo_src))
    from phase_diagram.io import (
        read_initial_conditions_json,
        parse_ic_block,
        create_run_directory,
        copy_json_config,
        save_solutions_npz,
        save_metadata_json,
    )
    from phase_diagram.dynamics import load_system_function, solve_multiple_ics, solve_replicates_for_ic
    from phase_diagram.plotting import (
        plot_phase_trajectories,
        plot_modulus_phase_trajectories,
        plot_psd_modes,
    )
else:
    from .io import (
        read_initial_conditions_json,
        parse_ic_block,
        create_run_directory,
        copy_json_config,
        save_solutions_npz,
        save_metadata_json,
    )
    from .dynamics import load_system_function, solve_multiple_ics, solve_replicates_for_ic
    from .plotting import (
        plot_phase_trajectories,
        plot_modulus_phase_trajectories,
        plot_psd_modes,
    )


def run(
    ic_json: Optional[str] = None,
    system_py: Optional[str] = None,
    func: Optional[str] = None,
    var_index: int = 0,
    mod_i: int = 0,
    mod_j: int = 0,
    base_out_dir: str = "../runs",
    t_points: Optional[int] = None,
    show: bool = False,
    *,
    ic_inline: Optional[Dict[str, Any]] = None,
    noise: Optional[Dict[str, Any]] = None,
    psd: Optional[Dict[str, Any]] = None,
    plot_phase: bool = True,
    plot_psd: bool = True,
    # Acquisition interval for plotting/PSD (optional): [t_start, t_end]
    acquire_interval: Optional[Tuple[float, float]] = None,
    # PSD plot axis scaling and frequency range options
    psd_axis_scale: str = 'linear',
    psd_freq_range: Optional[Tuple[float, float]] = None,
    # PSD signal selection: 'amplitude' (|z|) or 'intensity' (complex z two-sided)
    psd_signal: str = 'amplitude',
):
    """
    Execute one full experiment run: load ICs/params, solve for all ICs, save data and figures.
    - ic_json: path to JSON input file
    - system_py: path to python file defining the system function
    - func: function name inside the system_py
    - var_index: which complex variable to plot in complex phase plane
    - mod_i, mod_j: indices for |z_i| vs |z_j| plot
    - base_out_dir: base directory to store runs
    - t_points: override number of t_eval points (if provided)
    - show: whether to display figures interactively in addition to saving
    Returns the created run directory path.
    """
    # Create output directory
    run_dir = create_run_directory(base_out_dir)

    # Load inputs from inline IC block or from ic_json path
    if ic_inline is not None:
        ics, t_span, params = parse_ic_block(ic_inline)
    else:
        if not ic_json:
            raise ValueError("ic_json must be provided when ic_inline is None")
        ics, t_span, params = read_initial_conditions_json(ic_json)
        # Copy IC JSON for reproducibility
        copy_json_config(ic_json, run_dir)

    # Default function name is 'ODEs' if not provided by config
    if func is None:
        func = "ODEs"
    if not system_py:
        raise ValueError("system_py must be provided")
    system_func = load_system_function(system_py, func)
    # Attempt to auto-load noise diffusion_matrix from the same system module
    auto_D_func = None
    try:
        from importlib.util import spec_from_file_location, module_from_spec
        spec = spec_from_file_location("user_system_module_auto", system_py)
        if spec is not None and spec.loader is not None:
            mod = module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore[attr-defined]
            if hasattr(mod, "diffusion_matrix") and callable(getattr(mod, "diffusion_matrix")):
                base_D = getattr(mod, "diffusion_matrix")
                # Bind system params to diffusion if they share names
                noise_params = {}
                # Note: 'noise' is an argument to run(), may be None
                if isinstance(noise, dict) and isinstance(noise.get("params"), dict):
                    noise_params = dict(noise.get("params"))
                def auto_D_func(ti, zc, **kw):
                    merged = {}
                    merged.update(params or {})
                    merged.update(noise_params)
                    merged.update(kw)
                    return base_D(ti, zc, **merged)
    except Exception:
        auto_D_func = None

    # Decide PSD strategy to steer simulation count and expectation mapping
    psd_cfg = psd or {}
    psd_method = (psd_cfg.get("method") or "welch").lower() if isinstance(psd_cfg, dict) else "welch"
    psd_params = psd_cfg.get("params", {}) if isinstance(psd_cfg, dict) else {}
    # PSD enable flag (default on)
    psd_enabled = True
    if isinstance(psd_cfg, dict) and "enabled" in psd_cfg:
        psd_enabled = bool(psd_cfg.get("enabled", True))

    # Use all ICs; replicates are applied per IC when method is multi-trajectory
    ics_to_use = list(ics)
    # Replicate count for multi-trajectory averaging (default 4). Support legacy key 'trajectories'.
    replicates = 1
    if psd_method in ("multi-trajectory", "multi", "multi_traj"):
        replicates = int(psd_params.get("replicates", psd_params.get("trajectories", 4)))

    # (3) Check noise model and map expectation policy if needed
    noise_effective = noise
    if isinstance(noise_effective, dict):
        ntype = (noise_effective.get("type") or "none").lower()
        if ntype in ("none", "off", "disabled"):
            # (4a) no noise
            noise_effective = {"type": "none"}
        elif ntype in ("gaussian-white", "gaussian"):
            # (4b) gaussian white noise
            exp_cfg = noise_effective.get("expectation") if isinstance(noise_effective.get("expectation"), dict) else None
            if exp_cfg:
                etype = (exp_cfg.get("type") or "").lower()
                # (4b-1) instant: no change
                # (4b-2) average: map depending on PSD method
                if etype in ("average", "avg"):
                    if psd_method == "welch":
                        exp_cfg["type"] = "time-window"
                    elif psd_method in ("multi-trajectory", "multi", "multi_traj"):
                        exp_cfg["type"] = "multi-trajectory"
                    noise_effective["expectation"] = exp_cfg
    # Build time grid from t_span and t_points
    if t_points is None:
        t_eval = None
    else:
        t_eval = np.linspace(t_span[0], t_span[1], int(t_points))
    sols_any = None
    # Normalize solver.adaptive default to False if solver present
    if isinstance(noise_effective, dict) and isinstance(noise_effective.get("solver"), dict):
        solver_opts = dict(noise_effective.get("solver", {}))
        if "adaptive" not in solver_opts:
            solver_opts["adaptive"] = False
            noise_effective = dict(noise_effective)
            noise_effective["solver"] = solver_opts

    # Warn if Welch PSD with adaptive steps: interpolation will be applied before Welch
    if psd_method == "welch" and isinstance(noise_effective, dict) and isinstance(noise_effective.get("solver"), dict) and noise_effective["solver"].get("adaptive"):
        print("[warn] Adaptive time stepping is enabled; Welch PSD requires uniform sampling. We'll interpolate |z_i|(t) to a uniform grid before computing PSD.")

    if psd_method in ("multi-trajectory", "multi", "multi_traj") and replicates > 1 and isinstance(noise_effective, dict) and (noise_effective.get("type", "").lower() in ("gaussian-white", "gaussian")):
        # If adaptive solver requested, inform and disable (multi-trajectory requires lock-step)
        if isinstance(noise_effective.get("solver"), dict) and noise_effective["solver"].get("adaptive"):
            print("[info] Adaptive solver disabled for multi-trajectory mode; using lock-step EM.")
            noise_effective = dict(noise_effective)
            noise_effective["solver"] = dict(noise_effective.get("solver", {}))
            noise_effective["solver"]["adaptive"] = False
        # Per-IC replicates with synchronous vectorized EM
        sols_nested = []  # List[List[Result]]
        # Simple textual progress for ICs
        total_ics = len(ics_to_use)
        for idx_ic, y0 in enumerate(ics_to_use, start=1):
            print(f"[IC {idx_ic}/{total_ics}] solving {replicates} replicates...")
            # inline progress for time steps
            start_time_ic = time.time()
            def cb(step: int, total: int):
                pct = 100.0 * step / max(total, 1)
                if step > 0 and total > 0:
                    elapsed = max(time.time() - start_time_ic, 1e-6)
                    remaining = elapsed * (total - step) / step
                    eta_m = int(remaining // 60)
                    eta_s = int(remaining % 60)
                    eta_str = f"ETA {eta_m:02d}:{eta_s:02d}"
                else:
                    eta_str = "ETA --:--"
                print(f"  progress: {pct:5.1f}% ({step}/{total})  {eta_str}", end="\r")
            reps = solve_replicates_for_ic(
                system_func,
                y0,
                t_span,
                params=params,
                t_eval=t_eval,
                noise=noise_effective,
                D_func_complex=auto_D_func,
                replicates=replicates,
                progress_cb=cb,
            )
            print("")
            sols_nested.append(reps)
        sols_any = sols_nested
    else:
        # Single trajectory per IC (deterministic ODE if noise none; SDE otherwise)
        def pf(idx_ic: int, total_ics: int):
            print(f"[IC {idx_ic}/{total_ics}] solving...")
            start_time_ic = time.time()
            def cb(step: int, total: int):
                pct = 100.0 * step / max(total, 1)
                if step > 0 and total > 0:
                    elapsed = max(time.time() - start_time_ic, 1e-6)
                    remaining = elapsed * (total - step) / step
                    eta_m = int(remaining // 60)
                    eta_s = int(remaining % 60)
                    eta_str = f"ETA {eta_m:02d}:{eta_s:02d}"
                else:
                    eta_str = "ETA --:--"
                print(f"  progress: {pct:5.1f}% ({step}/{total})  {eta_str}", end="\r")
            return cb
        sols_any = solve_multiple_ics(
            system_func,
            ics_to_use,
            t_span,
            params=params,
            t_eval=t_eval,
            noise=noise_effective,
            D_func_complex=auto_D_func,
            progress_factory=pf,
        )

    # Save raw solutions
    data_path = str(Path(run_dir) / "solutions.npz")
    # Flatten nested replicate results for saving
    if isinstance(sols_any, list) and len(sols_any) > 0 and isinstance(sols_any[0], list):
        flat = []
        for res_list in sols_any:
            flat.extend(res_list)
        save_solutions_npz(data_path, flat)
    else:
        save_solutions_npz(data_path, sols_any)  # type: ignore[arg-type]

    # Save metadata (new structured layout)
    meta = {
        "ic_json": str(Path(ic_json).resolve()) if ic_json else None,
        "system": {
            "py": str(Path(system_py).resolve()),
            "func": func,
            "t_span": list(t_span),
            "t_points": int(t_points) if t_points is not None else None,
            "params": params,
            "out": base_out_dir,
        },
        "num_ics": len(ics_to_use),
        "plotting": {
            "var_index": var_index,
            "mod_i": mod_i,
            "mod_j": mod_j,
            "acquire_interval": list(acquire_interval) if acquire_interval else None,
            "psd_axis_scale": psd_axis_scale,
            "psd_freq_range": list(psd_freq_range) if psd_freq_range is not None else None,
            "psd_signal": psd_signal,
        },
        "noise": noise_effective or {"type": "none"},
        "psd": psd_cfg or {"method": "welch", "params": {"nperseg": 256}},
    }
    save_metadata_json(str(Path(run_dir) / "metadata.json"), meta)

    # Save figures
    figs_dir = Path(run_dir) / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    # Helper to slice solutions by time interval
    def slice_solutions_by_time(solutions_list, t_start: float, t_end: float):
        out = []
        for sol in solutions_list:
            t = sol.t
            idx = (t >= t_start - 1e-12) & (t <= t_end + 1e-12)
            class R: pass
            r = R()
            r.t = t[idx]
            r.y = sol.y[:, idx]
            r.success = sol.success
            r.message = getattr(sol, "message", "") + f" [slice {t_start:.3g},{t_end:.3g}]"
            out.append(r)
        return out

    # Prepare solutions for phase plotting (combine ICs in one figure)
    if isinstance(sols_any, list) and len(sols_any) > 0 and isinstance(sols_any[0], list):
        # Average per IC across replicates for phase plots
        plot_solutions = []
        for res_list in sols_any:
            Ys = [r.y for r in res_list]
            Yavg = sum(Ys) / float(len(Ys))
            class Result: pass
            avg_res = Result()
            avg_res.t = res_list[0].t
            avg_res.y = Yavg
            avg_res.success = True
            avg_res.message = "Averaged trajectory over replicates"
            plot_solutions.append(avg_res)
    else:
        plot_solutions = sols_any if isinstance(sols_any, list) else [sols_any]

    # Determine acquisition interval (defaults to full span) with overlap checks
    t0, t1 = float(t_span[0]), float(t_span[1])
    if acquire_interval:
        a0_req = float(acquire_interval[0])
        a1_req = float(acquire_interval[1]) if len(acquire_interval) > 1 else t1
        # Check basic overlap: lower bound must be < t_span upper bound
        if not (a0_req < t1):
            raise ValueError(f"acquire_interval lower bound {a0_req} is not less than t_span upper bound {t1}; no overlap")
        # Clamp to t_span
        a0_eff = max(a0_req, t0)
        a1_eff = min(a1_req, t1)
        if a1_req > t1:
            print(f"[warn] acquire_interval upper bound {a1_req} exceeds t_span upper {t1}; clamping to {a1_eff}.")
        if a0_eff >= a1_eff:
            raise ValueError(f"acquire_interval [{a0_req}, {a1_req}] does not overlap t_span [{t0}, {t1}]")
        t_acq0, t_acq1 = a0_eff, a1_eff
    else:
        t_acq0, t_acq1 = t0, t1

    if plot_phase:
        sols_a = slice_solutions_by_time(plot_solutions, t_acq0, t_acq1)
        plot_phase_trajectories(
            sols_a,
            var_index=var_index,
            title="Phase trajectories (acquisition)",
            save_path=str(figs_dir / "phase_trajectories.png"),
            show=show,
        )
        plot_modulus_phase_trajectories(
            sols_a,
            i=mod_i,
            j=mod_j,
            title="|z_i| vs |z_j| (acquisition)",
            save_path=str(figs_dir / "modulus_phase_trajectories.png"),
            show=show,
        )

    # PSD plots
    try:
        if plot_psd and psd_enabled:
            # Determine PSD settings
            psd_method = meta.get("psd", {}).get("method", "welch") if isinstance(meta.get("psd"), dict) else "welch"
            psd_params = meta.get("psd", {}).get("params", {}) if isinstance(meta.get("psd"), dict) else {}
            # Determine number of variables
            if isinstance(sols_any, list) and len(sols_any) > 0 and isinstance(sols_any[0], list):
                n_vars = sols_any[0][0].y.shape[0] // 2 if sols_any[0] else 0
            else:
                n_vars = sols_any[0].y.shape[0] // 2 if sols_any else 0
            if n_vars > 0:
                if isinstance(sols_any, list) and len(sols_any) > 0 and isinstance(sols_any[0], list):
                    # Per-IC PSD figures using replicates
                    for idx_ic, reps in enumerate(sols_any):
                        # Slice each replicate to acquisition interval only
                        reps_acq = []
                        for r in reps:
                            t = r.t
                            idx = (t >= t_acq0 - 1e-12) & (t <= t_acq1 + 1e-12)
                            class R: pass
                            ra = R()
                            ra.t = t[idx]
                            ra.y = r.y[:, idx]
                            reps_acq.append(ra)
                        # Default labels depend on PSD signal type
                        if (psd_signal or 'amplitude').lower() == 'intensity':
                            psd_labels = [f"z[{k}]" for k in range(n_vars)]
                        else:
                            psd_labels = [f"|z[{k}]|" for k in range(n_vars)]
                        plot_psd_modes(
                            reps_acq,
                            mode_indices=list(range(n_vars)),
                            labels=psd_labels,
                            title=f"PSD (IC {idx_ic+1}) - {psd_method} (acq)",
                            method=psd_method,
                            params=psd_params,
                            signal=psd_signal,
                            axis_scale=psd_axis_scale,
                            freq_range=psd_freq_range,
                            save_path=str(figs_dir / f"psd_modes_ic{idx_ic+1}.png"),
                            show=show,
                        )
                else:
                    # Welch or single-trajectory: produce per-IC figures as well
                    for idx_ic, sol_ic in enumerate(sols_any):
                        # Slice to acquisition interval only
                        t = sol_ic.t
                        idx = (t >= t_acq0 - 1e-12) & (t <= t_acq1 + 1e-12)
                        class R: pass
                        ra = R()
                        ra.t = t[idx]
                        ra.y = sol_ic.y[:, idx]
                        if (psd_signal or 'amplitude').lower() == 'intensity':
                            psd_labels = [f"z[{k}]" for k in range(n_vars)]
                        else:
                            psd_labels = [f"|z[{k}]|" for k in range(n_vars)]
                        plot_psd_modes(
                            [ra],
                            mode_indices=list(range(n_vars)),
                            labels=psd_labels,
                            title=f"PSD (IC {idx_ic+1}) - {psd_method} (acq)",
                            method=psd_method,
                            params=psd_params,
                            signal=psd_signal,
                            axis_scale=psd_axis_scale,
                            freq_range=psd_freq_range,
                            save_path=str(figs_dir / f"psd_modes_ic{idx_ic+1}.png"),
                            show=show,
                        )
    except Exception as e:
        print(f"[warn] PSD plotting skipped due to error: {e}")

    return run_dir


def _load_solutions_from_npz(npz_path: str):
    """Load solutions saved by save_solutions_npz into a flat list of simple objects with .t and .y."""
    import numpy as _np
    data = _np.load(npz_path)
    # find indices by scanning keys like t_0, y_0
    idxs: List[int] = []
    for k in data.files:
        if k.startswith("t_"):
            try:
                idxs.append(int(k.split("_")[1]))
            except Exception:
                pass
    idxs = sorted(set(idxs))
    sols = []
    for i in idxs:
        t = data[f"t_{i}"]
        y = data[f"y_{i}"]
        class R: pass
        r = R()
        r.t = _np.asarray(t)
        r.y = _np.asarray(y)
        r.success = True
        r.message = "loaded from npz"
        sols.append(r)
    return sols


def replot_run(
    run_dir: str,
    *,
    acquire_interval: Optional[Tuple[float, float]] = None,
    plot_phase: bool = True,
    plot_psd: bool = True,
    show: bool = False,
):
    """
    Re-generate phase and PSD figures from a saved run directory.
    If acquire_interval is not provided, uses the one recorded in metadata (if any),
    otherwise falls back to the original full t_span.
    """
    run_path = Path(run_dir)
    meta_path = run_path / "metadata.json"
    npz_path = run_path / "solutions.npz"
    if not meta_path.exists() or not npz_path.exists():
        raise FileNotFoundError("run_dir must contain metadata.json and solutions.npz")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    sols_flat = _load_solutions_from_npz(str(npz_path))

    # Determine grouping: replicates per IC if any
    num_ics = int(meta.get("num_ics", 1))
    total = len(sols_flat)
    reps_per_ic = max(1, total // max(num_ics, 1)) if total % max(num_ics, 1) == 0 else 1
    if reps_per_ic > 1:
        nested = [sols_flat[i*reps_per_ic:(i+1)*reps_per_ic] for i in range(num_ics)]
        sols_any = nested
    else:
        sols_any = sols_flat

    # Extract plotting metadata
    # Plotting/meta extraction supports new structured metadata with fallback to legacy
    plotting_meta = meta.get("plotting", {}) if isinstance(meta.get("plotting"), dict) else {}
    var_index = int(plotting_meta.get("var_index", meta.get("var_index", 0)))
    mod_i = int(plotting_meta.get("mod_i", meta.get("mod_i", 0)))
    mod_j = int(plotting_meta.get("mod_j", meta.get("mod_j", 1 if (mod_i == 0) else 0)))
    psd_axis_scale = plotting_meta.get("psd_axis_scale", "linear")
    psd_freq_range = plotting_meta.get("psd_freq_range")
    psd_signal = plotting_meta.get("psd_signal", "amplitude")
    psd_cfg = meta.get("psd", {}) if isinstance(meta.get("psd"), dict) else {"method": "welch", "params": {}}
    psd_method = (psd_cfg.get("method") or "welch").lower()
    psd_params = psd_cfg.get("params", {})
    if isinstance(meta.get("system"), dict) and isinstance(meta["system"].get("t_span"), list):
        t_span = tuple(meta["system"]["t_span"])  # type: ignore[assignment]
    else:
        t_span = tuple(meta.get("t_span", (0.0, 1.0)))  # legacy fallback
    # Determine acquisition interval
    # Resolve acquisition interval with overlap checks (prefer explicit arg, fallback to recorded plotting.acquire_interval)
    recorded_acq = None
    try:
        recorded_acq = plotting_meta.get("acquire_interval") or meta.get("run", {}).get("acquire_interval")
        if isinstance(recorded_acq, list) and len(recorded_acq) >= 2:
            recorded_acq = (float(recorded_acq[0]), float(recorded_acq[1]))
        else:
            recorded_acq = None
    except Exception:
        recorded_acq = None
    t0, t1 = float(t_span[0]), float(t_span[1])
    if acquire_interval:
        a0_req = float(acquire_interval[0])
        a1_req = float(acquire_interval[1]) if len(acquire_interval) > 1 else t1
    elif recorded_acq:
        a0_req = float(recorded_acq[0])
        a1_req = float(recorded_acq[1])
    else:
        a0_req, a1_req = t0, t1
    if not (a0_req < t1):
        raise ValueError(f"acquire_interval lower bound {a0_req} is not less than t_span upper bound {t1}; no overlap")
    a0_eff = max(a0_req, t0)
    a1_eff = min(a1_req, t1)
    if a1_req > t1:
        print(f"[warn] acquire_interval upper bound {a1_req} exceeds t_span upper {t1}; clamping to {a1_eff}.")
    if a0_eff >= a1_eff:
        raise ValueError(f"acquire_interval [{a0_req}, {a1_req}] does not overlap t_span [{t0}, {t1}]")
    t_acq0, t_acq1 = a0_eff, a1_eff

    # Helper to slice solutions by time interval
    def slice_solutions_by_time(solutions_list, t_start: float, t_end: float):
        out = []
        for sol in solutions_list:
            t = sol.t
            idx = (t >= t_start - 1e-12) & (t <= t_end + 1e-12)
            class R: pass
            r = R()
            r.t = t[idx]
            r.y = sol.y[:, idx]
            r.success = True
            r.message = getattr(sol, "message", "") + f" [slice {t_start:.3g},{t_end:.3g}]"
            out.append(r)
        return out

    # Prepare averaged solutions for phase plots
    if isinstance(sols_any, list) and len(sols_any) > 0 and isinstance(sols_any[0], list):
        plot_solutions = []
        for res_list in sols_any:
            Ys = [r.y for r in res_list]
            Yavg = sum(Ys) / float(len(Ys))
            class Result: pass
            avg_res = Result()
            avg_res.t = res_list[0].t
            avg_res.y = Yavg
            avg_res.success = True
            avg_res.message = "Averaged trajectory over replicates"
            plot_solutions.append(avg_res)
    else:
        plot_solutions = sols_any if isinstance(sols_any, list) else [sols_any]

    figs_dir = run_path / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    if plot_phase:
        sols_a = slice_solutions_by_time(plot_solutions, t_acq0, t_acq1)
        plot_phase_trajectories(
            sols_a,
            var_index=var_index,
            title="Phase trajectories (acquisition)",
            save_path=str(figs_dir / "phase_trajectories.png"),
            show=show,
        )
        plot_modulus_phase_trajectories(
            sols_a,
            i=mod_i,
            j=mod_j,
            title="|z_i| vs |z_j| (acquisition)",
            save_path=str(figs_dir / "modulus_phase_trajectories.png"),
            show=show,
        )

    if plot_psd:
        try:
            # Determine number of variables
            if isinstance(sols_any, list) and len(sols_any) > 0 and isinstance(sols_any[0], list):
                n_vars = sols_any[0][0].y.shape[0] // 2 if sols_any[0] else 0
            else:
                n_vars = sols_any[0].y.shape[0] // 2 if sols_any else 0
            if n_vars > 0:
                if isinstance(sols_any, list) and len(sols_any) > 0 and isinstance(sols_any[0], list):
                    # Per-IC PSD figures using replicates
                    for idx_ic, reps in enumerate(sols_any):
                        reps_acq = []
                        for r in reps:
                            t = r.t
                            idx = (t >= t_acq0 - 1e-12) & (t <= t_acq1 + 1e-12)
                            class R: pass
                            ra = R()
                            ra.t = t[idx]
                            ra.y = r.y[:, idx]
                            reps_acq.append(ra)
                        if (psd_signal or 'amplitude').lower() == 'intensity':
                            psd_labels = [f"z[{k}]" for k in range(n_vars)]
                        else:
                            psd_labels = [f"|z[{k}]|" for k in range(n_vars)]
                        plot_psd_modes(
                            reps_acq,
                            mode_indices=list(range(n_vars)),
                            labels=psd_labels,
                            title=f"PSD (IC {idx_ic+1}) - {psd_method} (acq)",
                            method=psd_method,
                            params=psd_params,
                            signal=psd_signal,
                            axis_scale=psd_axis_scale,
                            freq_range=psd_freq_range,
                            save_path=str(figs_dir / f"psd_modes_ic{idx_ic+1}.png"),
                            show=show,
                        )
                else:
                    # Welch or single-trajectory: produce per-IC figures as well
                    for idx_ic, sol_ic in enumerate(sols_any):
                        t = sol_ic.t
                        idx = (t >= t_acq0 - 1e-12) & (t <= t_acq1 + 1e-12)
                        class R: pass
                        ra = R()
                        ra.t = t[idx]
                        ra.y = sol_ic.y[:, idx]
                        if (psd_signal or 'amplitude').lower() == 'intensity':
                            psd_labels = [f"z[{k}]" for k in range(n_vars)]
                        else:
                            psd_labels = [f"|z[{k}]|" for k in range(n_vars)]
                        plot_psd_modes(
                            [ra],
                            mode_indices=list(range(n_vars)),
                            labels=psd_labels,
                            title=f"PSD (IC {idx_ic+1}) - {psd_method} (acq)",
                            method=psd_method,
                            params=psd_params,
                            signal=psd_signal,
                            axis_scale=psd_axis_scale,
                            freq_range=psd_freq_range,
                            save_path=str(figs_dir / f"psd_modes_ic{idx_ic+1}.png"),
                            show=show,
                        )
        except Exception as e:
            print(f"[warn] PSD replot skipped due to error: {e}")

    print(f"Replotted figures saved to: {figs_dir}")
    return str(figs_dir)


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg


def run_from_config(config_path: str) -> str:
    cfg = load_config(config_path)

    # Resolve paths relative to config file location
    cfg_dir = Path(config_path).resolve().parent

    # System definition (new schema): always use ODEs, pull t_span/params/t_points/out from system
    sys_block = cfg.get("system", {})
    sys_py = sys_block.get("py") or cfg.get("system_py")
    if sys_py:
        sys_py = str((cfg_dir / sys_py).resolve()) if not os.path.isabs(sys_py) else sys_py
    func = "ODEs"
    # ICs: inline block or external file path
    ic_block_src = cfg.get("ic")
    ic_json = cfg.get("ic_json") or cfg.get("ic_file")
    if ic_json:
        ic_json = str((cfg_dir / ic_json).resolve()) if not os.path.isabs(ic_json) else ic_json
    # Plotting options (renamed from 'run')
    plotting = cfg.get("plotting", {})
    var_index = int(plotting.get("var_index", cfg.get("var_index", 0)))
    mod_i = int(plotting.get("mod_i", cfg.get("mod_i", 0)))
    mod_j = int(plotting.get("mod_j", cfg.get("mod_j", 0)))
    show = bool(plotting.get("show", cfg.get("show", False)))
    plot_phase = bool(plotting.get("plot_phase", cfg.get("plot_phase", True)))
    plot_psd = bool(plotting.get("plot_psd", cfg.get("plot_psd", True)))
    multi_traj_plot = plotting.get("multi_traj_plot", "average")
    acq_interval = plotting.get("acquire_interval")
    psd_axis_scale = plotting.get("psd_axis_scale", "linear")
    psd_freq_range = plotting.get("psd_freq_range")
    psd_signal = plotting.get("psd_signal", "amplitude")
    # System-level t_points/out
    base_out_dir = sys_block.get("out", cfg.get("out", "runs"))
    t_points = sys_block.get("t_points", cfg.get("t_points"))
    if t_points is not None:
        t_points = int(t_points)
    # t_span and params from system
    t_span = tuple(map(float, (sys_block.get("t_span") or cfg.get("t_span") or [0.0, 1.0])))
    sys_params = dict(sys_block.get("params", {}))

    # Build an ic block compatible with parse_ic_block for downstream run()
    if ic_block_src is None:
        raise ValueError("Missing 'ic' block in config")
    ic_block = dict(ic_block_src)
    ic_block["t_span"] = list(t_span)
    ic_block["params"] = sys_params

    # Noise options (optional)
    noise_cfg = cfg.get("noise")
    noise: Optional[Dict[str, Any]] = None
    if noise_cfg:
        noise = dict(noise_cfg)
        # Resolve model.py path if present
        model = noise.get("model")
        if isinstance(model, dict):
            n_py = model.get("py")
            if n_py:
                model["py"] = str((cfg_dir / n_py).resolve()) if not os.path.isabs(n_py) else n_py
                noise["model"] = model
        # If D is provided as nested lists, keep as-is; solver will convert/validate later

    # PSD options (optional)
    psd_cfg = cfg.get("psd")

    run_dir = run(
        ic_json=ic_json,
        system_py=sys_py,
        func=func,
        var_index=var_index,
        mod_i=mod_i,
        mod_j=mod_j,
        base_out_dir=base_out_dir,
        t_points=t_points,
        show=show,
        plot_phase=plot_phase,
        plot_psd=plot_psd,
        acquire_interval=tuple(acq_interval) if isinstance(acq_interval, list) else None,
        psd_axis_scale=str(psd_axis_scale),
        psd_freq_range=tuple(psd_freq_range) if isinstance(psd_freq_range, list) and len(psd_freq_range) >= 2 else None,
        psd_signal=str(psd_signal),
        ic_inline=ic_block,
        noise=noise,
        psd=psd_cfg,
    )
    # If PSD config exists, append to metadata for traceability
    if psd_cfg:
        try:
            meta_path = Path(run_dir) / "metadata.json"
            with open(meta_path, "r", encoding="utf-8") as f:
                meta_json = json.load(f)
            meta_json["psd"] = psd_cfg
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta_json, f, indent=2)
        except Exception as e:
            print(f"Warning: failed to record PSD config in metadata: {e}")
    # Copy the full configuration JSON used for this run
    try:
        dst = Path(run_dir) / Path(config_path).name
        shutil.copy2(config_path, dst)
    except Exception as e:
        # Non-fatal: continue without blocking the run
        print(f"Warning: failed to copy config file: {e}")
    # Persist run options used (e.g., multi_traj_plot) into metadata
    try:
        meta_path = Path(run_dir) / "metadata.json"
        with open(meta_path, "r", encoding="utf-8") as f:
            meta_json = json.load(f)
        meta_json["plotting"] = {**(meta_json.get("plotting", {})), "multi_traj_plot": multi_traj_plot, "plot_phase": plot_phase, "plot_psd": plot_psd, "acquire_interval": acq_interval, "var_index": var_index, "mod_i": mod_i, "mod_j": mod_j, "psd_axis_scale": psd_axis_scale, "psd_freq_range": psd_freq_range, "psd_signal": psd_signal}
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_json, f, indent=2)
    except Exception as e:
        print(f"Warning: failed to update metadata with run options: {e}")
    return run_dir


def main():
    parser = argparse.ArgumentParser(description="Phase diagram runner (config-driven)")
    parser.add_argument("--config", type=str, default=None, help="Path to unified configuration JSON")
    parser.add_argument("--replot", type=str, default=None, help="Existing run directory to replot from saved data")
    parser.add_argument("--acquire_interval", type=str, default=None, help="Optional acquisition window 't0,t1' for plotting/PSD")
    # Backward-compatible options (will be ignored if --config is provided)
    parser.add_argument("--ic_json", type=str, default=None, help="Path to JSON file of initial conditions")
    parser.add_argument("--system_py", type=str, default=None, help="Path to python file defining the system function")
    parser.add_argument("--func", type=str, default=None, help="Function name inside system_py")
    parser.add_argument("--var_index", type=int, default=0, help="Complex variable index for phase plane plot")
    parser.add_argument("--mod_i", type=int, default=0, help="Index of |z_i| for modulus plot")
    parser.add_argument("--mod_j", type=int, default=0, help="Index of |z_j| for modulus plot")
    parser.add_argument("--out", type=str, default="runs", help="Base output directory")
    parser.add_argument("--t_points", type=int, default=None, help="Override number of time samples for t_eval")
    parser.add_argument("--show", action="store_true", help="Show figures interactively")
    args = parser.parse_args()

    # Replot mode takes precedence
    if args.replot:
        acq = None
        if args.acquire_interval:
            try:
                parts = [float(x) for x in args.acquire_interval.split(",")]
                if len(parts) >= 2:
                    acq = (parts[0], parts[1])
            except Exception:
                acq = None
        replot_run(args.replot, acquire_interval=acq, plot_phase=True, plot_psd=True, show=args.show)
        run_dir = args.replot
    elif args.config:
        run_dir = run_from_config(args.config)
    else:
        run_dir = run(
            ic_json=args.ic_json,
            system_py=args.system_py,
            func=args.func,
            var_index=args.var_index,
            mod_i=args.mod_i,
            mod_j=args.mod_j,
            base_out_dir=args.out,
            t_points=args.t_points,
            show=args.show,
        )
    print(f"Saved outputs to: {run_dir}")


if __name__ == "__main__":
    main()
