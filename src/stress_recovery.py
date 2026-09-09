"""Numerical tools for a stylized stress--recovery dynamical system.

The model is educational and abstract. It is not a medical or psychological
assessment tool and its state variables are dimensionless model quantities.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp

ArrayLike = np.ndarray | float
Stressor = Callable[[ArrayLike], ArrayLike]


@dataclass(frozen=True)
class ModelParameters:
    """Parameters of the coupled ODE model."""

    alpha: float = 1.0  # response strength to external forcing
    beta: float = 0.5   # recovery feedback strength
    gamma: float = 0.8  # adaptation rate of the recovery state

    def __post_init__(self) -> None:
        if self.alpha <= 0 or self.beta <= 0 or self.gamma <= 0:
            raise ValueError("alpha, beta, and gamma must all be positive")


def constant_stressor(t: ArrayLike, level: float = 1.0) -> ArrayLike:
    """Constant external forcing E(t)."""
    arr = np.asarray(t, dtype=float)
    out = np.full_like(arr, level, dtype=float)
    return float(out) if np.ndim(t) == 0 else out


def periodic_stressor(
    t: ArrayLike,
    baseline: float = 1.0,
    amplitude: float = 0.5,
    omega: float = 0.5,
) -> ArrayLike:
    """Sinusoidally varying external forcing E(t)."""
    arr = np.asarray(t, dtype=float)
    out = baseline + amplitude * np.sin(omega * arr)
    return float(out) if np.ndim(t) == 0 else out


def pulse_stressor(
    t: ArrayLike,
    centers: tuple[float, ...] = (10.0, 20.0, 30.0),
    amplitude: float = 2.0,
    width: float = 1.5,
    baseline: float = 0.5,
) -> ArrayLike:
    """Baseline forcing with Gaussian pulses."""
    arr = np.asarray(t, dtype=float)
    out = np.full_like(arr, baseline, dtype=float)
    for center in centers:
        out += amplitude * np.exp(-0.5 * ((arr - center) / width) ** 2)
    return float(out) if np.ndim(t) == 0 else out


def rhs(
    t: float,
    state: np.ndarray,
    parameters: ModelParameters,
    stressor: Stressor,
) -> np.ndarray:
    """Right-hand side of the coupled ODE system."""
    s, r = state
    forcing = float(stressor(t))
    ds = parameters.alpha * forcing - parameters.beta * r
    dr = parameters.gamma * (s - r)
    return np.array([ds, dr], dtype=float)


def simulate_euler(
    time: np.ndarray,
    parameters: ModelParameters = ModelParameters(),
    stressor: Stressor = periodic_stressor,
    initial_state: tuple[float, float] = (0.5, 0.2),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate the model with explicit Euler integration.

    The forcing is evaluated at the beginning of each time step, which is
    consistent with the explicit Euler update from t[i-1] to t[i].
    """
    time = np.asarray(time, dtype=float)
    if time.ndim != 1 or len(time) < 2:
        raise ValueError("time must be a one-dimensional array with >= 2 points")
    if np.any(np.diff(time) <= 0):
        raise ValueError("time values must be strictly increasing")

    forcing = np.asarray(stressor(time), dtype=float)
    if forcing.shape != time.shape:
        raise ValueError("stressor(time) must return an array with the same shape")

    s = np.empty_like(time)
    r = np.empty_like(time)
    s[0], r[0] = initial_state

    for i in range(1, len(time)):
        dt = time[i] - time[i - 1]
        ds = parameters.alpha * forcing[i - 1] - parameters.beta * r[i - 1]
        dr = parameters.gamma * (s[i - 1] - r[i - 1])
        s[i] = s[i - 1] + dt * ds
        r[i] = r[i - 1] + dt * dr

    return forcing, s, r


def simulate_ivp(
    time: np.ndarray,
    parameters: ModelParameters = ModelParameters(),
    stressor: Stressor = periodic_stressor,
    initial_state: tuple[float, float] = (0.5, 0.2),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reference simulation using SciPy's adaptive RK solver."""
    time = np.asarray(time, dtype=float)
    result = solve_ivp(
        lambda tt, yy: rhs(tt, yy, parameters, stressor),
        (float(time[0]), float(time[-1])),
        np.asarray(initial_state, dtype=float),
        t_eval=time,
        rtol=1e-9,
        atol=1e-11,
    )
    if not result.success:
        raise RuntimeError(result.message)
    forcing = np.asarray(stressor(time), dtype=float)
    return forcing, result.y[0], result.y[1]


def equilibrium(parameters: ModelParameters, forcing_level: float) -> tuple[float, float]:
    """Equilibrium for a constant forcing E(t)=forcing_level."""
    value = parameters.alpha * forcing_level / parameters.beta
    return value, value


def jacobian(parameters: ModelParameters) -> np.ndarray:
    """Jacobian matrix of the autonomous state dynamics."""
    return np.array(
        [[0.0, -parameters.beta], [parameters.gamma, -parameters.gamma]],
        dtype=float,
    )


def stability_eigenvalues(parameters: ModelParameters) -> np.ndarray:
    """Eigenvalues of the Jacobian."""
    return np.linalg.eigvals(jacobian(parameters))


def is_asymptotically_stable(parameters: ModelParameters) -> bool:
    """Return True when every Jacobian eigenvalue has negative real part."""
    return bool(np.all(np.real(stability_eigenvalues(parameters)) < 0.0))


def time_above_threshold(time: np.ndarray, values: np.ndarray, threshold: float) -> float:
    """Approximate time spent above an illustrative threshold.

    The threshold is a model-analysis device, not a clinical cutoff.
    """
    time = np.asarray(time, dtype=float)
    values = np.asarray(values, dtype=float)
    if time.shape != values.shape:
        raise ValueError("time and values must have the same shape")
    if len(time) < 2:
        return 0.0
    mask = values > threshold
    return float(np.sum(np.diff(time) * mask[:-1]))
