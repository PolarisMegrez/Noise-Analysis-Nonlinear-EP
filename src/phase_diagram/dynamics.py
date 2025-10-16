import numpy as np
from importlib.util import spec_from_file_location, module_from_spec
from typing import Callable, List, Dict, Any, Tuple, Optional

def load_system_function(py_file: str, func_name: str) -> Callable:
    """
    从指定 .py 文件加载名为 func_name 的用户方程函数。
    约定签名：f(t: float, z: np.ndarray[complex], **params) -> np.ndarray[complex]
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


def _validate_noise_matrix(D: np.ndarray, dim_real: int) -> np.ndarray:
    D = np.asarray(D, dtype=float)
    if D.shape != (dim_real, dim_real):
        raise ValueError(f"Noise matrix D must be of shape ({dim_real}, {dim_real}), got {D.shape}")
    # ensure symmetric positive semidefinite: we will Cholesky with jitter if needed
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
):
    """
    Euler–Maruyama integration for real-valued SDE dy = f dt + G dW, with covariance D = G G^T.
    - noise_type: "none" or "gaussian-white" (others reserved)
    - D: (m x m) diffusion matrix for the real-expanded system; if None or noise_type==none, no noise is added
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
    return t, Y


def _euler_maruyama_multi(
    f_real: Callable[[float, np.ndarray], np.ndarray],
    y0_list: List[np.ndarray],
    t_eval: np.ndarray,
    *,
    D_list_func: Optional[Callable[[float, List[np.ndarray]], List[np.ndarray]]] = None,
    noise_type: str = "gaussian-white",
    rng: Optional[np.random.Generator] = None,
):
    """Synchronous Euler–Maruyama for multiple trajectories with shared expectation.
    D_list_func returns a list of (m x m) diffusion matrices, one per trajectory, at (t, states).
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
            try:
                Gi = np.linalg.cholesky(Di)
            except np.linalg.LinAlgError:
                Gi = np.linalg.cholesky(Di + jitter * np.eye(m))
            G_list.append(Gi)
        for i in range(n_traj):
            drift = f_real(t[k - 1], Y_list[i][:, k - 1])
            dW = rng.standard_normal(m) * np.sqrt(max(dt, 0.0))
            Y_list[i][:, k] = Y_list[i][:, k - 1] + dt * drift + G_list[i] @ dW
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
    通用复变量一阶 ODE 求解器。system_func 接受复向量并返回复导数。
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
):
    """
    Integrate complex SDE by expanding into real coordinates and using Euler–Maruyama.
    D is the diffusion matrix in real-expanded space of shape (2n, 2n).
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
    t, Y = _euler_maruyama(f_real, y0_real, t_eval, D=D, D_func=D_func_real, noise_type=noise_type, rng=rng)

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
):
    """
    Solve for multiple ICs (ODE by default). If noise is provided (dict), use SDE path.
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

    # Multi-trajectory synchronous policy
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
            amp2_vals = [np.abs(z[var_index])**2 for z in zs]
            mean_val = float(np.mean(amp2_vals)) if len(amp2_vals) > 0 else 0.0
            return [np.asarray(base_fn(ti, zs[i], alpha_amp2_mean=mean_val)) for i in range(len(zs))]
        rng = np.random.default_rng(seed) if seed is not None else None
        t_grid, Y_list = _euler_maruyama_multi(f_real, y0_list, np.asarray(t_eval, dtype=float), D_list_func=D_list_func, noise_type="gaussian-white", rng=rng)
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

    # Otherwise, integrate independently with per-trajectory policy
    D_func_wrapped = make_traj_wrapper(D_func_complex_local, expectation_cfg)
    for y0 in initial_conditions:
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
            )
        )
    return sols