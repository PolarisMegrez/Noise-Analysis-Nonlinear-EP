import numpy as np
from importlib.util import spec_from_file_location, module_from_spec
from typing import Callable, List, Dict, Any, Tuple, Optional

def load_system_function(py_file: str, func_name: str) -> Callable:
    """
    Load a callable named 'func_name' from the Python file at 'py_file'.
    Expected signature: f(t: float, z: np.ndarray[complex], **params) -> np.ndarray[complex].
    """
    spec = spec_from_file_location("user_system_module", py_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {py_file}")
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    fn = getattr(mod, func_name, None)
    if fn is None or not callable(fn):
        raise AttributeError(f"Function {func_name} not found in {py_file}")
    return fn

def load_optional_function(py_file: str, func_name: str) -> Optional[Callable]:
    """Load a callable if present in the given module, else return None."""
    spec = spec_from_file_location("user_system_module", py_file)
    if spec is None or spec.loader is None:
        return None
    mod = module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    except Exception:
        return None
    fn = getattr(mod, func_name, None)
    return fn if callable(fn) else None

def _complex_to_real(z: np.ndarray) -> np.ndarray:
    out = np.empty(2 * len(z), dtype=float)
    out[0::2] = z.real
    out[1::2] = z.imag
    return out

def _real_to_complex(y: np.ndarray) -> np.ndarray:
    return y[0::2] + 1j * y[1::2]


def _abs2_stable(zc: complex) -> float:
    """Compute |zc|^2 robustly, avoiding overflow, and clip to a large finite cap."""
    r = float(np.real(zc))
    i = float(np.imag(zc))
    ar = abs(r)
    ai = abs(i)
    if ar == 0.0 and ai == 0.0:
        return 0.0
    m = max(ar, ai)
    xr = r / m
    xi = i / m
    val = (xr * xr + xi * xi) * (m * m)
    if not np.isfinite(val):
        return 1e300
    return min(val, 1e300)


def _validate_noise_matrix(D: np.ndarray, dim_real: int) -> np.ndarray:
    D = np.asarray(D, dtype=float)
    if D.shape != (dim_real, dim_real):
        raise ValueError(f"Noise matrix D must be of shape ({dim_real}, {dim_real}), got {D.shape}")
    # Symmetry/PSD is not enforced here; Cholesky is attempted with jitter fallback at usage time.
    return D


def _euler_maruyama(
    f_real: Callable[[float, np.ndarray], np.ndarray],
    y0_real: np.ndarray,
    t_eval: np.ndarray,
    *,
    D: Optional[np.ndarray] = None,
    D_func: Optional[Callable[[float, np.ndarray], np.ndarray]] = None,
    noise_type: str = "none",
    rng: Optional[np.random.Generator] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
):
    """
    Euler–Maruyama for real-valued SDEs: dy = f dt + G dW with covariance D = G G^T.
    - noise_type: "none" or "gaussian-white".
    - D: constant (m x m) diffusion matrix; if both D and D_func are None or noise_type=="none", the drift-only ODE is integrated.
    - D_func: state/time dependent diffusion matrix generator D(t, y).
    """
    y0_real = np.asarray(y0_real, dtype=float)
    t = np.asarray(t_eval, dtype=float)
    m = y0_real.size
    Y = np.zeros((m, t.size), dtype=float)
    Y[:, 0] = y0_real
    if noise_type == "none" or (D is None and D_func is None):
        for k in range(1, t.size):
            dt = t[k] - t[k - 1]
            Y[:, k] = Y[:, k - 1] + dt * f_real(t[k - 1], Y[:, k - 1])
        return t, Y

    rng = rng or np.random.default_rng()
    G_const = None
    if D_func is None and D is not None:
        D = _validate_noise_matrix(D, m)
        # Compute matrix square root via Cholesky with small jitter fallback
        jitter = 1e-12
        try:
            G_const = np.linalg.cholesky(D)
        except np.linalg.LinAlgError:
            G_const = np.linalg.cholesky(D + jitter * np.eye(m))
    jitter = 1e-12
    for k in range(1, t.size):
        dt = t[k] - t[k - 1]
        drift = f_real(t[k - 1], Y[:, k - 1])
        dW = rng.standard_normal(m) * np.sqrt(max(dt, 0.0))
        if D_func is not None:
            Dk = _validate_noise_matrix(D_func(t[k - 1], Y[:, k - 1]), m)
            try:
                Gk = np.linalg.cholesky(Dk)
            except np.linalg.LinAlgError:
                Gk = np.linalg.cholesky(Dk + jitter * np.eye(m))
            noise_step = Gk @ dW
        else:
            noise_step = G_const @ dW  # type: ignore[operator]
        Y[:, k] = Y[:, k - 1] + dt * drift + noise_step
        if progress_cb is not None:
            try:
                progress_cb(k, t.size - 1)
            except Exception:
                pass
    return t, Y


def _euler_maruyama_adaptive_sampled(
    f_real: Callable[[float, np.ndarray], np.ndarray],
    y0_real: np.ndarray,
    t_eval: np.ndarray,
    *,
    D: Optional[np.ndarray] = None,
    D_func: Optional[Callable[[float, np.ndarray], np.ndarray]] = None,
    rng: Optional[np.random.Generator] = None,
    atol: float = 1e-4,
    rtol: float = 1e-3,
    h_init: Optional[float] = None,
    h_min: Optional[float] = None,
    h_max: Optional[float] = None,
    safety: float = 0.9,
    progress_cb: Optional[Callable[[int, int], None]] = None,
):
    """
    Adaptive Euler–Maruyama with step-doubling error control and sampled outputs on given t_eval.
    Uses one full step (h) vs two half steps (h/2 + h/2) with consistent Brownian increments to estimate strong error.
    Only supports single-trajectory integration (no coupling across trajectories).
    """
    y = np.asarray(y0_real, dtype=float).copy()
    t_grid = np.asarray(t_eval, dtype=float)
    m = y.size
    Y = np.zeros((m, t_grid.size), dtype=float)
    Y[:, 0] = y

    # Handle deterministic fallback
    if D is None and D_func is None:
        # Simple adaptive Euler using drift-only step-doubling
        def step_drift(ti, yi, h):
            return yi + h * f_real(ti, yi)
        t_curr = t_grid[0]
        # Initial step guess
        if h_init is None:
            h = max(1e-3, float(t_grid[1] - t_grid[0]) if len(t_grid) > 1 else 1e-2)
        else:
            h = float(h_init)
        hmin = float(h_min) if h_min is not None else 1e-8
        hmax = float(h_max) if h_max is not None else max(h, 1e-1)
        for k in range(1, t_grid.size):
            t_target = t_grid[k]
            while t_curr < t_target - 1e-15:
                h = min(h, t_target - t_curr, hmax)
                y_full = step_drift(t_curr, y, h)
                y_half = step_drift(t_curr, y, h * 0.5)
                y_half = step_drift(t_curr + 0.5 * h, y_half, h * 0.5)
                err = np.linalg.norm(y_half - y_full, ord=2)
                tol = float(atol + rtol * max(1.0, np.linalg.norm(y, ord=2)))
                if err <= tol or h <= hmin * 1.001:
                    # accept half-step solution (more accurate)
                    y = y_half
                    t_curr += h
                    # increase step
                    if err > 1e-30:
                        h = min(hmax, max(hmin, safety * h * (tol / err) ** 0.5))
                    else:
                        h = min(hmax, 2.0 * h)
                else:
                    # reject and shrink
                    h = max(hmin, safety * h * (tol / max(err, 1e-30)) ** 0.5)
            Y[:, k] = y
            if progress_cb is not None:
                try:
                    progress_cb(k, t_grid.size - 1)
                except Exception:
                    pass
        return t_grid, Y

    # Stochastic path
    rng = rng or np.random.default_rng()
    # Precompute constant G if available
    G_const = None
    if D_func is None and D is not None:
        D = _validate_noise_matrix(D, m)
        jitter = 1e-12
        try:
            G_const = np.linalg.cholesky(D)
        except np.linalg.LinAlgError:
            G_const = np.linalg.cholesky(D + jitter * np.eye(m))
    jitter = 1e-12
    t_curr = float(t_grid[0])
    if h_init is None:
        h = max(1e-3, float(t_grid[1] - t_grid[0]) if len(t_grid) > 1 else 1e-2)
    else:
        h = float(h_init)
    hmin = float(h_min) if h_min is not None else 1e-8
    hmax = float(h_max) if h_max is not None else max(h, 1e-1)

    for k in range(1, t_grid.size):
        t_target = float(t_grid[k])
        while t_curr < t_target - 1e-15:
            h = min(h, t_target - t_curr, hmax)
            # Draw consistent Brownian increments
            dW = rng.standard_normal(m) * np.sqrt(max(h, 0.0))
            dW1 = rng.standard_normal(m) * np.sqrt(max(0.5 * h, 0.0))
            dW2 = dW - dW1  # ensures dW1 + dW2 = dW in distributional sense

            # Build G at current state if needed
            if D_func is not None:
                Dk = _validate_noise_matrix(D_func(t_curr, y), m)
                try:
                    G = np.linalg.cholesky(Dk)
                except np.linalg.LinAlgError:
                    G = np.linalg.cholesky(Dk + jitter * np.eye(m))
            else:
                G = G_const  # type: ignore[assignment]

            # Full step
            drift = f_real(t_curr, y)
            y_full = y + h * drift + G @ dW  # type: ignore[operator]

            # Two half steps
            # step 1
            drift1 = f_real(t_curr, y)
            if D_func is not None:
                D1 = _validate_noise_matrix(D_func(t_curr, y), m)
                try:
                    G1 = np.linalg.cholesky(D1)
                except np.linalg.LinAlgError:
                    G1 = np.linalg.cholesky(D1 + jitter * np.eye(m))
            else:
                G1 = G_const  # type: ignore[assignment]
            y_mid = y + 0.5 * h * drift1 + G1 @ dW1  # type: ignore[operator]
            # step 2 (use updated state)
            drift2 = f_real(t_curr + 0.5 * h, y_mid)
            if D_func is not None:
                D2 = _validate_noise_matrix(D_func(t_curr + 0.5 * h, y_mid), m)
                try:
                    G2 = np.linalg.cholesky(D2)
                except np.linalg.LinAlgError:
                    G2 = np.linalg.cholesky(D2 + jitter * np.eye(m))
            else:
                G2 = G_const  # type: ignore[assignment]
            y_half = y_mid + 0.5 * h * drift2 + G2 @ dW2  # type: ignore[operator]

            # Error estimate and accept/reject
            err = np.linalg.norm(y_half - y_full, ord=2)
            tol = float(atol + rtol * max(1.0, np.linalg.norm(y, ord=2)))
            if err <= tol or h <= hmin * 1.001:
                # accept half-step result
                y = y_half
                t_curr += h
                # adapt step
                if err > 1e-30:
                    h = min(hmax, max(hmin, safety * h * (tol / err) ** 0.5))
                else:
                    h = min(hmax, 2.0 * h)
            else:
                # reject and shrink
                h = max(hmin, safety * h * (tol / max(err, 1e-30)) ** 0.5)
        Y[:, k] = y
        if progress_cb is not None:
            try:
                progress_cb(k, t_grid.size - 1)
            except Exception:
                pass
    return t_grid, Y

def _euler_maruyama_multi(
    f_real: Callable[[float, np.ndarray], np.ndarray],
    y0_list: List[np.ndarray],
    t_eval: np.ndarray,
    *,
    D_list_func: Optional[Callable[[float, List[np.ndarray]], List[np.ndarray]]] = None,
    noise_type: str = "gaussian-white",
    rng: Optional[np.random.Generator] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
):
    """
    Synchronous Euler–Maruyama for multiple trajectories advanced in lock-step.
    - D_list_func(t, states) must return a list of diffusion matrices, one per trajectory.
    - Use this to implement expectation policies that couple trajectories (e.g., ensemble averages).
    """
    t = np.asarray(t_eval, dtype=float)
    n_traj = len(y0_list)
    m = y0_list[0].size if n_traj > 0 else 0
    Y_list = [np.zeros((m, t.size), dtype=float) for _ in range(n_traj)]
    for i in range(n_traj):
        Y_list[i][:, 0] = np.asarray(y0_list[i], dtype=float)
    if noise_type == "none" or D_list_func is None:
        for k in range(1, t.size):
            dt = t[k] - t[k - 1]
            for i in range(n_traj):
                Y_list[i][:, k] = Y_list[i][:, k - 1] + dt * f_real(t[k - 1], Y_list[i][:, k - 1])
        return t, Y_list
    rng = rng or np.random.default_rng()
    jitter = 1e-12
    for k in range(1, t.size):
        dt = t[k] - t[k - 1]
        states = [Y_list[i][:, k - 1] for i in range(n_traj)]
        D_list = D_list_func(t[k - 1], states)
        G_list = []
        for Di in D_list:
            Di = _validate_noise_matrix(Di, m)
            Di = np.nan_to_num(Di, nan=0.0, posinf=1e12, neginf=0.0)
            Di = 0.5 * (Di + Di.T)
            try:
                Gi = np.linalg.cholesky(Di)
            except np.linalg.LinAlgError:
                Gi = np.linalg.cholesky(Di + jitter * np.eye(m))
            G_list.append(Gi)
        for i in range(n_traj):
            drift = f_real(t[k - 1], Y_list[i][:, k - 1])
            dW = rng.standard_normal(m) * np.sqrt(max(dt, 0.0))
            Y_list[i][:, k] = Y_list[i][:, k - 1] + dt * drift + G_list[i] @ dW
        if progress_cb is not None:
            try:
                progress_cb(k, t.size - 1)
            except Exception:
                pass
    return t, Y_list

def solve_complex_ode(
    system_func: Callable,
    y0_complex: np.ndarray,
    t_span: Tuple[float, float],
    params: Dict[str, Any] = None,
    t_eval: np.ndarray = None,
    atol: float = 1e-9,
    rtol: float = 1e-8
):
    """
    Solve a first-order complex ODE dz/dt = system_func(t, z, **params) by expanding to real coordinates
    and passing to scipy.integrate.solve_ivp.
    """
    from scipy.integrate import solve_ivp
    params = params or {}

    def real_rhs(t, y_real):
        z = _real_to_complex(np.asarray(y_real))
        dz = np.asarray(system_func(t, z, **params))
        out = np.empty_like(y_real, dtype=float)
        out[0::2] = dz.real
        out[1::2] = dz.imag
        return out

    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 2000)

    y0_real = _complex_to_real(np.asarray(y0_complex, dtype=complex))
    sol = solve_ivp(real_rhs, t_span, y0_real, t_eval=t_eval, atol=atol, rtol=rtol)
    return sol


def solve_complex_sde(
    system_func: Callable,
    y0_complex: np.ndarray,
    t_span: Tuple[float, float],
    params: Dict[str, Any] = None,
    t_eval: Optional[np.ndarray] = None,
    *,
    noise_type: str = "gaussian-white",
    D: Optional[np.ndarray] = None,
    D_func_complex: Optional[Callable[[float, np.ndarray], np.ndarray]] = None,
    seed: Optional[int] = None,
    solver_opts: Optional[Dict[str, Any]] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
):
    """
    Integrate a complex SDE by expanding to real coordinates and applying Euler–Maruyama.
    - D: constant diffusion matrix in real-expanded space (2n x 2n).
    - D_func_complex(t, z): optional complex-space diffusion generator; will be wrapped to real space.
    """
    params = params or {}

    def f_real(t, y_real):
        z = _real_to_complex(np.asarray(y_real))
        dz = np.asarray(system_func(t, z, **params))
        out = np.empty_like(y_real, dtype=float)
        out[0::2] = dz.real
        out[1::2] = dz.imag
        return out

    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 2000)

    y0_real = _complex_to_real(np.asarray(y0_complex, dtype=complex))
    rng = np.random.default_rng(seed) if seed is not None else None
    # Build a real-space diffusion function if a complex-space function is provided
    D_func_real = None
    if D_func_complex is not None:
        def D_func_real(ti: float, y_real: np.ndarray) -> np.ndarray:
            z = _real_to_complex(np.asarray(y_real))
            return np.asarray(D_func_complex(ti, z))
    # Choose solver: adaptive EM if requested, else fixed-step EM on provided grid
    use_adaptive = bool(solver_opts.get("adaptive", False)) if isinstance(solver_opts, dict) else False
    if use_adaptive:
        atol = float(solver_opts.get("atol", 1e-4))
        rtol = float(solver_opts.get("rtol", 1e-3))
        h_init = solver_opts.get("h_init")
        h_min = solver_opts.get("h_min")
        h_max = solver_opts.get("h_max")
        safety = float(solver_opts.get("safety", 0.9))
        t, Y = _euler_maruyama_adaptive_sampled(
            f_real,
            y0_real,
            np.asarray(t_eval, dtype=float),
            D=D,
            D_func=D_func_real,
            rng=rng,
            atol=atol,
            rtol=rtol,
            h_init=h_init,
            h_min=h_min,
            h_max=h_max,
            safety=safety,
            progress_cb=progress_cb,
        )
    else:
        t, Y = _euler_maruyama(f_real, y0_real, t_eval, D=D, D_func=D_func_real, noise_type=noise_type, rng=rng, progress_cb=progress_cb)

    # Wrap result in a minimal object with attributes like solve_ivp output
    class Result:
        pass

    res = Result()
    res.t = t
    res.y = Y
    res.success = True
    res.message = "Euler-Maruyama completed"
    return res

def solve_multiple_ics(
    system_func: Callable,
    initial_conditions: List[np.ndarray],
    t_span: Tuple[float, float],
    params: Dict[str, Any] = None,
    t_eval: np.ndarray = None,
    atol: float = 1e-9,
    rtol: float = 1e-8,
    *,
    noise: Optional[Dict[str, Any]] = None,
    D_func_complex: Optional[Callable[[float, np.ndarray], np.ndarray]] = None,
    solver_opts: Optional[Dict[str, Any]] = None,
    progress_factory: Optional[Callable[[int, int], Optional[Callable[[int, int], None]]]] = None,
):
    """
    Solve for multiple initial conditions.
    - If noise is None or noise.type in {none, off}, integrate deterministic ODE per IC.
    - If noise.type is gaussian-white, use Euler–Maruyama with either:
      (a) constant D, or
      (b) state-dependent diffusion via D_func_complex (possibly loaded from model as diffusion_matrix).
    Expectation policy (noise.expectation.type):
      - instant: per-trajectory instantaneous |z[var_index]|^2.
      - time-window: per-trajectory exponential moving average (tau or alpha).
      - multi-trajectory: synchronous integration where D for each trajectory uses the ensemble mean at each time.
    """
    sols = []
    use_sde = False
    sde_type = None
    D = None
    D_func_complex_local = D_func_complex
    seed = None
    expectation_cfg = None
    if noise:
        sde_type = (noise.get("type") or "none").lower()
        if sde_type in ("gaussian", "gaussian-white"):
            use_sde = True
            D = noise.get("D")
            seed = noise.get("seed")
            expectation_cfg = noise.get("expectation") if isinstance(noise.get("expectation"), dict) else None
            # Optional state-dependent diffusion specified via a function at noise["model"]
            model_cfg = noise.get("model") if isinstance(noise.get("model"), dict) else None
            if model_cfg and D_func_complex_local is None:
                n_py = model_cfg.get("py")
                n_fn = model_cfg.get("func")
                n_params = model_cfg.get("params") if isinstance(model_cfg.get("params"), dict) else {}
                if n_py and n_fn:
                    base_fn = load_system_function(n_py, n_fn)
                    def D_func_wrapped(ti: float, z_complex: np.ndarray):
                        # Combine system params and noise model params; noise overrides
                        all_params = {**(params or {}), **n_params}
                        return base_fn(ti, z_complex, **all_params)
                    D_func_complex_local = D_func_wrapped
        elif sde_type in ("none", "off", "disabled"):
            use_sde = False
        else:
            # reserved for future extensions; fallback to ODE
            use_sde = False
    # If no noise or not SDE, just ODE for all ICs
    if not use_sde:
        for y0 in initial_conditions:
            sols.append(solve_complex_ode(system_func, y0, t_span, params=params, t_eval=t_eval, atol=atol, rtol=rtol))
        return sols

    # Build real-space RHS
    def f_real(ti, y_real):
        z = _real_to_complex(np.asarray(y_real))
        dz = np.asarray(system_func(ti, z, **(params or {})))
        out = np.empty_like(y_real, dtype=float)
        out[0::2] = dz.real
        out[1::2] = dz.imag
        return out

    # Build expectation wrapper for D_func if present
    def make_traj_wrapper(base_fn, policy: Optional[dict]):
        if base_fn is None:
            return None
        pol_type = (policy.get("type") if policy else None) or "instant"
        p = policy.get("params", {}) if policy else {}
        var_index = int(p.get("var_index", 0))
        # per-trajectory EMA state
        ema_val = None
        last_t = None
        tau = p.get("tau")  # time constant for EMA
        alpha_fixed = p.get("alpha")  # direct smoothing factor
        def wrapped(ti: float, z_complex: np.ndarray):
            nonlocal ema_val, last_t
            amp2 = np.abs(z_complex[var_index])**2
            if pol_type == "instant":
                mean_val = float(amp2)
            elif pol_type in ("time-window", "time_window"):
                if ema_val is None:
                    ema_val = float(amp2)
                    last_t = ti
                else:
                    dt = float(ti - (last_t if last_t is not None else ti))
                    if alpha_fixed is not None:
                        a = float(alpha_fixed)
                    elif tau is not None and dt > 0:
                        a = 1.0 - np.exp(-dt / float(tau))
                    else:
                        a = 0.1
                    ema_val = (1 - a) * ema_val + a * float(amp2)
                    last_t = ti
                mean_val = float(ema_val)
            else:
                # default fallback
                mean_val = float(amp2)
            return base_fn(ti, z_complex, alpha_amp2_mean=mean_val)
        return wrapped

    # Multi-trajectory synchronous policy (shared expectation across different IC trajectories)
    if expectation_cfg and ((expectation_cfg.get("type") or "").lower() in ("multi-trajectory", "multi", "multi_traj")) and D_func_complex_local is not None:
        # Prepare initial states
        if t_eval is None:
            t_eval = np.linspace(t_span[0], t_span[1], 2000)
        y0_list = [_complex_to_real(np.asarray(y0, dtype=complex)) for y0 in initial_conditions]
        p = expectation_cfg.get("params", {}) if expectation_cfg else {}
        var_index = int(p.get("var_index", 0))
        # Merge params for diffusion (system + noise model params if any were in wrapper); here assume base_fn already merged
        base_fn = D_func_complex_local
        def D_list_func(ti: float, states: List[np.ndarray]) -> List[np.ndarray]:
            # compute average |z[var_index]|^2 across trajectories at current states
            zs = [_real_to_complex(s) for s in states]
            amp2_vals = [_abs2_stable(z[var_index]) for z in zs]
            mean_val = float(np.mean(amp2_vals)) if len(amp2_vals) > 0 else 0.0
            return [np.asarray(base_fn(ti, zs[i], alpha_amp2_mean=mean_val)) for i in range(len(zs))]
        rng = np.random.default_rng(seed) if seed is not None else None
        # For coupled multi-trajectory expectation, we require lock-step time grid; adaptive EM is not supported here.
        # Provide group-level progress if available
        group_cb = None
        if callable(progress_factory):
            group_cb = progress_factory(1, 1)
        t_grid, Y_list = _euler_maruyama_multi(
            f_real,
            y0_list,
            np.asarray(t_eval, dtype=float),
            D_list_func=D_list_func,
            noise_type="gaussian-white",
            rng=rng,
            progress_cb=group_cb,
        )
        # Package results
        class Result: pass
        for Y in Y_list:
            r = Result()
            r.t = t_grid
            r.y = Y
            r.success = True
            r.message = "Euler-Maruyama (multi) completed"
            sols.append(r)
        return sols

    # Otherwise, integrate independently with per-trajectory policy (instant or time-window)
    D_func_wrapped = make_traj_wrapper(D_func_complex_local, expectation_cfg)
    for idx_ic, y0 in enumerate(initial_conditions, start=1):
        cb = None
        if callable(progress_factory):
            cb = progress_factory(idx_ic, len(initial_conditions))
        sols.append(
            solve_complex_sde(
                system_func,
                y0,
                t_span,
                params=params,
                t_eval=t_eval,
                noise_type="gaussian-white",
                D=D,
                D_func_complex=D_func_wrapped,
                seed=seed,
                solver_opts=(noise.get("solver") if isinstance(noise, dict) else None),
                progress_cb=cb,
            )
        )
    return sols


def solve_replicates_for_ic(
    system_func: Callable,
    y0_complex: np.ndarray,
    t_span: Tuple[float, float],
    params: Dict[str, Any] = None,
    t_eval: Optional[np.ndarray] = None,
    *,
    noise: Optional[Dict[str, Any]] = None,
    D_func_complex: Optional[Callable[[float, np.ndarray], np.ndarray]] = None,
    replicates: int = 1,
    progress_cb: Optional[Callable[[int, int], None]] = None,
):
    """
    For one IC, run N independent stochastic trajectories in lock-step (vectorized EM).
    Expectation policy within a replicate set:
      - instant: per-replicate instantaneous |z[var_index]|^2.
      - time-window: per-replicate EMA with tau or alpha.
      - multi-trajectory: ensemble mean across replicates at each step.
    If noise is None/off, falls back to a single deterministic ODE solution.
    Returns a list of Result-like objects (one per replicate) each with (t, y).
    """
    params = params or {}
    # If no noise, just do ODE once
    if not noise or (str(noise.get("type", "none")).lower() in ("none", "off", "disabled")):
        sol = solve_complex_ode(system_func, y0_complex, t_span, params=params, t_eval=t_eval)
        return [sol]

    ntype = str(noise.get("type", "gaussian-white")).lower()
    seed = noise.get("seed")
    expectation_cfg = noise.get("expectation") if isinstance(noise.get("expectation"), dict) else None

    # Build real-space RHS
    def f_real(ti, y_real):
        z = _real_to_complex(np.asarray(y_real))
        dz = np.asarray(system_func(ti, z, **params))
        out = np.empty_like(y_real, dtype=float)
        out[0::2] = dz.real
        out[1::2] = dz.imag
        return out

    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 2000)

    # Prepare replicated initial states in real-expanded space
    y0_real = _complex_to_real(np.asarray(y0_complex, dtype=complex))
    y0_list = [y0_real.copy() for _ in range(max(1, int(replicates))) ]

    # Build base diffusion function in complex space
    base_fn = D_func_complex
    # Support optional noise model loader (noise.model)
    if base_fn is None and isinstance(noise.get("model"), dict):
        m = noise["model"]
        n_py = m.get("py")
        n_fn = m.get("func")
        n_params = m.get("params") if isinstance(m.get("params"), dict) else {}
        if n_py and n_fn:
            base_loaded = load_system_function(n_py, n_fn)
            def base_fn(ti: float, z_complex: np.ndarray, **kw):  # type: ignore[assignment]
                all_params = {**params, **n_params, **kw}
                return base_loaded(ti, z_complex, **all_params)

    # If still None, build D_list_func that returns zeros to effectively ignore noise
    if base_fn is None and ntype in ("gaussian-white", "gaussian"):
        def D_list_func_zero(ti: float, states: List[np.ndarray]) -> List[np.ndarray]:
            m = states[0].size if states else 0
            return [np.zeros((m, m), dtype=float) for _ in states]
        D_list_func = D_list_func_zero
    elif ntype in ("gaussian-white", "gaussian"):
        # Prepare expectation policy
        pol_type = (expectation_cfg.get("type") if expectation_cfg else None) or "instant"
        pol_type = str(pol_type).lower()
        p = expectation_cfg.get("params", {}) if expectation_cfg else {}
        var_index = int(p.get("var_index", 0))
        tau = p.get("tau")
        alpha_fixed = p.get("alpha")

        # Per-trajectory EMA state
        ema_vals: List[Optional[float]] = [None] * len(y0_list)
        last_ts: List[Optional[float]] = [None] * len(y0_list)

        def D_list_func(ti: float, states: List[np.ndarray]) -> List[np.ndarray]:
            zs = [_real_to_complex(s) for s in states]
            # Ensemble mean if multi-trajectory
            if pol_type in ("multi-trajectory", "multi", "multi_traj"):
                amp2_vals = [np.abs(z[var_index])**2 for z in zs]
                mean_val = float(np.mean(amp2_vals)) if len(amp2_vals) > 0 else 0.0
                return [np.asarray(base_fn(ti, zs[i], alpha_amp2_mean=mean_val)) for i in range(len(zs))]
            # Instant or time-window per trajectory
            out = []
            for i, z in enumerate(zs):
                amp2 = float(np.abs(z[var_index])**2)
                if pol_type in ("time-window", "time_window"):
                    if ema_vals[i] is None:
                        ema_vals[i] = amp2
                        last_ts[i] = ti
                    else:
                        dt = float(ti - (last_ts[i] if last_ts[i] is not None else ti))
                        if alpha_fixed is not None:
                            a = float(alpha_fixed)
                        elif tau is not None and dt > 0:
                            a = 1.0 - np.exp(-dt / float(tau))
                        else:
                            a = 0.1
                        ema_vals[i] = (1 - a) * float(ema_vals[i]) + a * amp2
                        last_ts[i] = ti
                    mean_val = float(ema_vals[i])
                else:
                    mean_val = amp2
                out.append(np.asarray(base_fn(ti, z, alpha_amp2_mean=mean_val)))
            return out
    else:
        # Unsupported noise types: default to deterministic
        def D_list_func(ti: float, states: List[np.ndarray]) -> List[np.ndarray]:
            m = states[0].size if states else 0
            return [np.zeros((m, m), dtype=float) for _ in states]

    rng = np.random.default_rng(seed) if seed is not None else None
    t_grid, Y_list = _euler_maruyama_multi(
        f_real,
        y0_list,
        np.asarray(t_eval, dtype=float),
        D_list_func=D_list_func,
        noise_type="gaussian-white" if ntype in ("gaussian-white", "gaussian") else "none",
        rng=rng,
        progress_cb=progress_cb,
    )

    class Result: pass
    sols: List[Any] = []
    for Y in Y_list:
        r = Result()
        r.t = t_grid
        r.y = Y
        r.success = True
        r.message = "Euler-Maruyama (replicates) completed"
        sols.append(r)
    return sols