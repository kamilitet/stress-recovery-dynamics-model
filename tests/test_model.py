import numpy as np

from src.stress_recovery import (
    ModelParameters,
    constant_stressor,
    equilibrium,
    is_asymptotically_stable,
    rhs,
    simulate_euler,
    simulate_ivp,
    stability_eigenvalues,
)


def test_equilibrium_has_zero_derivative():
    p = ModelParameters(alpha=1.2, beta=0.6, gamma=0.9)
    e0 = 1.5
    state = np.array(equilibrium(p, e0))
    deriv = rhs(0.0, state, p, lambda _: e0)
    assert np.allclose(deriv, 0.0)


def test_positive_beta_gamma_give_asymptotic_stability():
    p = ModelParameters(alpha=1.0, beta=0.5, gamma=0.8)
    eig = stability_eigenvalues(p)
    assert np.all(np.real(eig) < 0.0)
    assert is_asymptotically_stable(p)


def test_euler_approaches_reference_solution():
    p = ModelParameters()
    time = np.linspace(0.0, 20.0, 2001)
    _, s_euler, r_euler = simulate_euler(time, p)
    _, s_ref, r_ref = simulate_ivp(time, p)
    rmse = np.sqrt(np.mean((s_euler - s_ref) ** 2 + (r_euler - r_ref) ** 2))
    assert rmse < 0.02


def test_constant_forcing_relaxes_toward_equilibrium():
    p = ModelParameters(alpha=1.0, beta=0.5, gamma=0.8)
    time = np.linspace(0.0, 80.0, 4001)
    _, s, r = simulate_ivp(time, p, stressor=lambda t: constant_stressor(t, 1.0))
    s_star, r_star = equilibrium(p, 1.0)
    assert abs(s[-1] - s_star) < 1e-3
    assert abs(r[-1] - r_star) < 1e-3
