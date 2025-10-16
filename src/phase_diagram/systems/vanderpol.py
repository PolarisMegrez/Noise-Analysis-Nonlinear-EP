import numpy as np

def vanderpol(
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
  """
  Two-mode complex ODE (alpha, beta):
    dα/dt = [ -i*ω_a + (γ_a/2) + Γ*(1/2 - |α|^2) ] * α - i*g*β
    dβ/dt = [ -i*ω_b - (γ_b/2) ] * β - i*g*α

  z is a complex state vector with two entries: z[0]=α, z[1]=β.
  Parameters are passed via **params from the solver.
  """
  alpha = complex(z[0])
  beta = complex(z[1])
  abs_a2 = (alpha.real**2 + alpha.imag**2)

  growth_alpha = (gamma_a / 2.0) + Gamma * (0.5 - abs_a2)
  dalpha = (-1j * omega_a + growth_alpha) * alpha - 1j * g * beta
  dbeta = (-1j * omega_b - 0.5 * gamma_b) * beta - 1j * g * alpha

  return np.array([dalpha, dbeta], dtype=complex)
