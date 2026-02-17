"""
Core physics engine for the N-body gravitational problem.
Handles the ODE system, numerical integration, and collision detection in 2D and 3D.
"""

import numpy as np
from scipy.integrate import solve_ivp

# --- Constants ---
# Gravitational constant adapted to AU and years
G = ((6.674e-11) * (6.684e-12) ** 3) * (365.25 * 24 * 3600) ** 2
# Softening parameter to avoid division by zero
SOFTENING = 1e-2
# Default volumetric mass density (kg/m^3) for radius estimation
DEFAULT_DENSITY = 5000.0
# AU in meters
AU_METERS = 1.5e11


# =============================================================================
# ODE systems
# =============================================================================

def _ode_2d(t, state, n_bodies, masses):
    """
    Right-hand side of the 2D N-body ODE system.

    State vector layout: [x1, y1, x2, y2, ..., vx1, vy1, vx2, vy2, ...]
    """
    positions = state[: 2 * n_bodies]
    velocities = state[2 * n_bodies: 4 * n_bodies]
    accelerations = np.zeros(2 * n_bodies)

    for i in range(n_bodies):
        xi, yi = positions[2 * i], positions[2 * i + 1]
        ax, ay = 0.0, 0.0
        for j in range(n_bodies):
            if j == i:
                continue
            xj, yj = positions[2 * j], positions[2 * j + 1]
            dx, dy = xi - xj, yi - yj
            dist_cubed = (dx ** 2 + dy ** 2) ** 1.5 + SOFTENING
            ax -= G * masses[j] * dx / dist_cubed
            ay -= G * masses[j] * dy / dist_cubed
        accelerations[2 * i] = ax
        accelerations[2 * i + 1] = ay

    return np.concatenate([velocities, accelerations])


def _ode_3d(t, state, n_bodies, masses):
    """
    Right-hand side of the 3D N-body ODE system.

    State vector layout: [x1, y1, z1, x2, y2, z2, ..., vx1, vy1, vz1, ...]
    """
    positions = state[: 3 * n_bodies]
    velocities = state[3 * n_bodies: 6 * n_bodies]
    accelerations = np.zeros(3 * n_bodies)

    for i in range(n_bodies):
        xi = positions[3 * i]
        yi = positions[3 * i + 1]
        zi = positions[3 * i + 2]
        ax, ay, az = 0.0, 0.0, 0.0
        for j in range(n_bodies):
            if j == i:
                continue
            xj = positions[3 * j]
            yj = positions[3 * j + 1]
            zj = positions[3 * j + 2]
            dx, dy, dz = xi - xj, yi - yj, zi - zj
            dist_cubed = (dx ** 2 + dy ** 2 + dz ** 2) ** 1.5 + SOFTENING
            ax -= G * masses[j] * dx / dist_cubed
            ay -= G * masses[j] * dy / dist_cubed
            az -= G * masses[j] * dz / dist_cubed
        accelerations[3 * i] = ax
        accelerations[3 * i + 1] = ay
        accelerations[3 * i + 2] = az

    return np.concatenate([velocities, accelerations])


# =============================================================================
# Solver
# =============================================================================

def solve(state0, masses, t_max, n_steps, dim=2, method="RK45"):
    """
    Integrate the N-body system.

    Parameters
    ----------
    state0 : array-like
        Initial state vector [positions..., velocities...].
    masses : array-like
        Mass of each body (kg).
    t_max : float
        Integration end time (years).
    n_steps : int
        Number of time evaluation points.
    dim : int
        2 or 3.
    method : str
        Integration method passed to solve_ivp.

    Returns
    -------
    scipy.integrate.OdeSolution
    """
    n_bodies = len(masses)
    t_eval = np.linspace(0, t_max, n_steps)
    ode_func = _ode_2d if dim == 2 else _ode_3d
    solution = solve_ivp(
        ode_func,
        [0, t_max],
        state0,
        t_eval=t_eval,
        method=method,
        args=(n_bodies, masses),
    )
    return solution


# =============================================================================
# Collision detection
# =============================================================================

def body_radius(mass, density=DEFAULT_DENSITY):
    """Compute body radius in AU from mass (kg) and density (kg/m^3)."""
    r_meters = ((3 * mass) / (4 * np.pi * density)) ** (1 / 3)
    return r_meters / AU_METERS


def detect_collisions(solution, masses, n_bodies, dim=2, radius_scale=100.0):
    """
    Detect collisions over the full trajectory.

    Parameters
    ----------
    solution : OdeSolution
        Output of `solve`.
    masses : array-like
    n_bodies : int
    dim : int
        2 or 3.
    radius_scale : float
        Multiplier on the sum of radii for collision threshold.

    Returns
    -------
    list of dict
        Each entry: {"t_index": int, "body_i": int, "body_j": int, "time": float}
    """
    radii = np.array([body_radius(m) for m in masses])
    n_steps = solution.y.shape[1]
    collisions = []
    active_pairs = set()

    for t in range(n_steps):
        for i in range(n_bodies):
            for j in range(i + 1, n_bodies):
                if dim == 2:
                    dx = solution.y[2 * i, t] - solution.y[2 * j, t]
                    dy = solution.y[2 * i + 1, t] - solution.y[2 * j + 1, t]
                    dist_sq = dx ** 2 + dy ** 2
                else:
                    dx = solution.y[3 * i, t] - solution.y[3 * j, t]
                    dy = solution.y[3 * i + 1, t] - solution.y[3 * j + 1, t]
                    dz = solution.y[3 * i + 2, t] - solution.y[3 * j + 2, t]
                    dist_sq = dx ** 2 + dy ** 2 + dz ** 2

                threshold_sq = ((radii[i] + radii[j]) * radius_scale) ** 2

                if dist_sq <= threshold_sq:
                    if (i, j) not in active_pairs:
                        collisions.append({
                            "t_index": t,
                            "body_i": i,
                            "body_j": j,
                            "time": solution.t[t],
                        })
                        active_pairs.add((i, j))
                else:
                    active_pairs.discard((i, j))

    return collisions


# =============================================================================
# Initial condition helpers
# =============================================================================

def random_initial_conditions(n_bodies, dim=2, pos_range=5.0, vel_range=1.0,
                              mass_low=1e27, mass_high=1e30):
    """
    Generate random initial conditions.

    Returns
    -------
    state0 : ndarray
    masses : ndarray
    """
    masses = np.random.uniform(mass_low, mass_high, n_bodies)
    positions = np.random.uniform(-pos_range, pos_range, dim * n_bodies)
    velocities = np.random.uniform(-vel_range, vel_range, dim * n_bodies)
    state0 = np.concatenate([positions, velocities])
    return state0, masses


def body_display_size(masses, density=DEFAULT_DENSITY, min_size=2):
    """Compute relative marker sizes for plotting."""
    radii = np.array([body_radius(m, density) for m in masses])
    log_radii = np.log(radii + 1e-30)
    sizes = log_radii - log_radii.min() + min_size
    return sizes
