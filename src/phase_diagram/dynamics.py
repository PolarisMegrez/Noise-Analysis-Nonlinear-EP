import numpy as np
from importlib.util import spec_from_file_location, module_from_spec
from typing import Callable, List, Dict, Any, Tuple

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

def _complex_to_real(z: np.ndarray) -> np.ndarray:
    out = np.empty(2 * len(z), dtype=float)
    out[0::2] = z.real
    out[1::2] = z.imag
    return out

def _real_to_complex(y: np.ndarray) -> np.ndarray:
    return y[0::2] + 1j * y[1::2]

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

def solve_multiple_ics(
    system_func: Callable,
    initial_conditions: List[np.ndarray],
    t_span: Tuple[float, float],
    params: Dict[str, Any] = None,
    t_eval: np.ndarray = None,
    atol: float = 1e-9,
    rtol: float = 1e-8
):
    """
    多组初值分别求解，返回解列表。
    """
    sols = []
    for y0 in initial_conditions:
        sols.append(
            solve_complex_ode(system_func, y0, t_span, params=params, t_eval=t_eval, atol=atol, rtol=rtol)
        )
    return sols