import numpy as np

def ODEs(
    t: float,
    z: np.ndarray,
    *,
    omega_a: float = 1.0,
    omega_b: float = 1.2,
    gamma_a: float = 0.05,
    Gamma: float = 0.0,
    gamma_b: float = 0.1,
    g: float = 0.5,
) -> np.ndarray:
    alpha = complex(z[0])
    beta = complex(z[1])
    abs_a2 = (alpha.real**2 + alpha.imag**2)

    growth_alpha = (gamma_a / 2.0) + Gamma * (0.5 - abs_a2)
    dalpha = (-1j * omega_a + growth_alpha) * alpha - 1j * g * beta
    dbeta = (-1j * omega_b - 0.5 * gamma_b) * beta - 1j * g * alpha

    return np.array([dalpha, dbeta], dtype=complex)


def diffusion_matrix(
    t: float,
    z: np.ndarray,
    *,
    D_scale: float = 1.0,
    gamma_a: float = 0.05,
    Gamma: float = 0.0,
    gamma_b: float = 0.1,
    **kwargs,
) -> np.ndarray:
    if "D" in kwargs and kwargs["D"] is not None:
        try:
            D_scale = float(kwargs["D"])
        except Exception:
            pass
    z = np.asarray(z)
    alpha = complex(z[0])
    n = z.size
    assert n == 2, "vanderpol diffusion assumes two-mode system (alpha, beta)"

    # expectation source: prefer injected alpha_amp2_mean; fallback to instantaneous |alpha|^2
    # compute |alpha|^2 robustly
    ar = float(alpha.real)
    ai = float(alpha.imag)
    am = max(abs(ar), abs(ai))
    if am == 0.0:
        abs_a2_inst = 0.0
    else:
        xr = ar / am
        xi = ai / am
        abs_a2_inst = (xr * xr + xi * xi) * (am * am)
        if not np.isfinite(abs_a2_inst):
            abs_a2_inst = 1e300
    abs_a2_mean = float(kwargs.get("alpha_amp2_mean", abs_a2_inst))
    # Clip expectation to keep diffusion finite
    abs_a2_mean = float(np.clip(abs_a2_mean, 0.0, 1e12))

    # Sanitize D_scale and compute diagonal diffusion terms
    try:
        D_scale = float(D_scale)
    except Exception:
        D_scale = 1.0
    D_scale = float(np.clip(D_scale, 0.0, 1e6))
    D_alpha = D_scale * max(0.0, (gamma_a / 2.0) + Gamma * (2.0 * abs_a2_mean - 1.0))
    D_beta = D_scale * max(0.0, (gamma_b / 2.0))
    # Clip to finite bounds
    D_alpha = float(np.clip(D_alpha, 0.0, 1e12))
    D_beta = float(np.clip(D_beta, 0.0, 1e12))

    Dr = np.zeros((2 * n, 2 * n), dtype=float)
    Dr[0, 0] = D_alpha
    Dr[1, 1] = D_alpha
    Dr[2, 2] = D_beta
    Dr[3, 3] = D_beta
    return Dr