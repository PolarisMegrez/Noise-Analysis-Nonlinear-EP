import argparse
import json
import os
import sys
import shutil
from pathlib import Path
from typing import Optional, Dict, Any

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
    from phase_diagram.dynamics import load_system_function, solve_multiple_ics
    from phase_diagram.plotting import (
        plot_phase_trajectories,
        plot_modulus_phase_trajectories,
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
    from .dynamics import load_system_function, solve_multiple_ics
    from .plotting import (
        plot_phase_trajectories,
        plot_modulus_phase_trajectories,
    )


def run(
    ic_json: Optional[str] = None,
    system_py: Optional[str] = None,
    func: Optional[str] = None,
    var_index: int = 0,
    mod_i: int = 0,
    mod_j: int = 0,
    base_out_dir: str = "runs",
    t_points: Optional[int] = None,
    show: bool = False,
    *,
    ic_inline: Optional[Dict[str, Any]] = None,
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

    if not (system_py and func):
        raise ValueError("system_py and func must be provided")
    system_func = load_system_function(system_py, func)

    # Solve
    if t_points is None:
        t_eval = None
    else:
        t_eval = np.linspace(t_span[0], t_span[1], int(t_points))
    sols = solve_multiple_ics(system_func, ics, t_span, params=params, t_eval=t_eval)

    # Save raw solutions
    data_path = str(Path(run_dir) / "solutions.npz")
    save_solutions_npz(data_path, sols)

    # Save metadata
    meta = {
        "ic_json": str(Path(ic_json).resolve()) if ic_json else None,
        "system_py": str(Path(system_py).resolve()),
        "func": func,
        "t_span": list(t_span),
        "t_points": int(t_points) if t_points is not None else None,
        "num_ics": len(ics),
        "params": params,
        "var_index": var_index,
        "mod_i": mod_i,
        "mod_j": mod_j,
    }
    save_metadata_json(str(Path(run_dir) / "metadata.json"), meta)

    # Save figures
    figs_dir = Path(run_dir) / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    plot_phase_trajectories(
        sols,
        var_index=var_index,
        title="Phase trajectories of selected complex variable",
        save_path=str(figs_dir / "phase_trajectories.png"),
        show=show,
    )

    plot_modulus_phase_trajectories(
        sols,
        i=mod_i,
        j=mod_j,
        title="|z_i| vs |z_j| trajectories",
        save_path=str(figs_dir / "modulus_phase_trajectories.png"),
        show=show,
    )

    return run_dir


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg


def run_from_config(config_path: str) -> str:
    cfg = load_config(config_path)

    # Resolve paths relative to config file location
    cfg_dir = Path(config_path).resolve().parent

    # System definition
    sys_py = cfg.get("system", {}).get("py") or cfg.get("system_py")
    if sys_py:
        sys_py = str((cfg_dir / sys_py).resolve()) if not os.path.isabs(sys_py) else sys_py
    func = cfg.get("system", {}).get("func") or cfg.get("func")

    # ICs: inline block or external file path
    ic_block = cfg.get("ic")
    ic_json = cfg.get("ic_json") or cfg.get("ic_file")
    if ic_json:
        ic_json = str((cfg_dir / ic_json).resolve()) if not os.path.isabs(ic_json) else ic_json

    # Run options
    run_opts = cfg.get("run", {})
    var_index = int(run_opts.get("var_index", cfg.get("var_index", 0)))
    mod_i = int(run_opts.get("mod_i", cfg.get("mod_i", 0)))
    mod_j = int(run_opts.get("mod_j", cfg.get("mod_j", 0)))
    base_out_dir = run_opts.get("out", cfg.get("out", "runs"))
    t_points = run_opts.get("t_points", cfg.get("t_points"))
    if t_points is not None:
        t_points = int(t_points)
    show = bool(run_opts.get("show", cfg.get("show", False)))

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
        ic_inline=ic_block,
    )
    # Copy the full configuration JSON used for this run
    try:
        dst = Path(run_dir) / Path(config_path).name
        shutil.copy2(config_path, dst)
    except Exception as e:
        # Non-fatal: continue without blocking the run
        print(f"Warning: failed to copy config file: {e}")
    return run_dir


def main():
    parser = argparse.ArgumentParser(description="Phase diagram runner (config-driven)")
    parser.add_argument("--config", type=str, default=None, help="Path to unified configuration JSON")
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

    if args.config:
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
