import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np


def read_initial_conditions_json(path: str):
    """Read an IC JSON file and return (list of complex IC arrays, (t0, t1), params dict).
    Supports older schema where t_span/params live in the same file; for new unified config, run_from_config supplies t_span/params via system.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    count = int(data["count"])
    n_vars = int(data["n_vars"])
    # Prefer local t_span for standalone IC files; run_from_config overrides with system.t_span
    t_span = tuple(map(float, data.get("t_span", [0.0, 1.0])))
    ic_raw = data["initial_conditions"]
    if len(ic_raw) != count:
        raise ValueError(f"count={count} but got {len(ic_raw)} initial condition sets")
    initial_conditions: List[np.ndarray] = []
    for ic in ic_raw:
        if len(ic) != n_vars:
            raise ValueError(f"Each initial condition must contain {n_vars} complex variables")
        z = []
        for pair in ic:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError("Each complex variable must be [Re, Im]")
            z.append(pair[0] + 1j * pair[1])
        initial_conditions.append(np.array(z, dtype=complex))
    params = dict(data.get("params", {}))
    return initial_conditions, t_span, params


def parse_ic_block(ic_block: Dict[str, Any]) -> Tuple[List[np.ndarray], Tuple[float, float], Dict[str, Any]]:
    """Parse an inline IC block and return (IC list, (t0, t1), params dict)."""
    count = int(ic_block["count"])
    n_vars = int(ic_block["n_vars"])
    t_span = tuple(map(float, ic_block.get("t_span", [0.0, 1.0])))
    ic_raw = ic_block["initial_conditions"]
    if len(ic_raw) != count:
        raise ValueError(f"count={count} but got {len(ic_raw)} initial condition sets")
    initial_conditions: List[np.ndarray] = []
    for ic in ic_raw:
        if len(ic) != n_vars:
            raise ValueError(f"Each initial condition must contain {n_vars} complex variables")
        z = []
        for pair in ic:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError("Each complex variable must be [Re, Im]")
            z.append(pair[0] + 1j * pair[1])
        initial_conditions.append(np.array(z, dtype=complex))
    params = dict(ic_block.get("params", {}))
    return initial_conditions, t_span, params


def create_run_directory(base_dir: str = "runs", prefix: str = "run") -> str:
    """Create a timestamped output folder and return its absolute path."""
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(base_dir) / f"{prefix}-{ts}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return str(run_dir.resolve())


def copy_json_config(src_json_path: str, dst_dir: str, dst_name: Optional[str] = None) -> str:
    """Copy a JSON config file into dst_dir and return the destination path."""
    dst_dir_path = Path(dst_dir)
    dst_dir_path.mkdir(parents=True, exist_ok=True)
    dst_name = dst_name or Path(src_json_path).name
    dst_path = dst_dir_path / dst_name
    shutil.copy2(src_json_path, dst_path)
    return str(dst_path)


def save_solutions_npz(npz_path: str, solutions: List[Any]) -> None:
    """Save OdeResult sequences into a compressed .npz (keys: t_i, y_i)."""
    arrs: Dict[str, Any] = {}
    for i, sol in enumerate(solutions):
        arrs[f"t_{i}"] = np.asarray(sol.t)
        arrs[f"y_{i}"] = np.asarray(sol.y)
    np.savez_compressed(npz_path, **arrs)


def save_metadata_json(path: str, metadata: Dict[str, Any]) -> None:
    """Write a small JSON metadata file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)