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
    amp_smoothing: float = 0.05,
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

    abs_a2 = (alpha.real**2 + alpha.imag**2)
    abs_a2_mean = (1.0 - amp_smoothing) * abs_a2 + amp_smoothing * 0.5

    D_alpha = D_scale * max(0.0, (gamma_a / 2.0) + Gamma * (2.0 * abs_a2_mean - 1.0))
    D_beta = D_scale * max(0.0, (gamma_b / 2.0))

    Dr = np.zeros((2 * n, 2 * n), dtype=float)
    Dr[0, 0] = D_alpha
    Dr[1, 1] = D_alpha
    Dr[2, 2] = D_beta
    Dr[3, 3] = D_beta
    return Dr