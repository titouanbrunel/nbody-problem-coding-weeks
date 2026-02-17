"""
Experiment framework for the N-body simulation.

Each experiment is a class inheriting from Experiment.
Run all or a specific one via CLI:
    python research.py --experiment all
    python research.py -e energy_conservation
"""

import os
import abc
import argparse
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from physics import (
    solve,
    detect_collisions,
    random_initial_conditions,
    body_display_size,
    G,
    SOFTENING,
)

RESULTS_DIR = "results"


# =============================================================================
# Abstract base class
# =============================================================================

class Experiment(abc.ABC):
    """Base class for all experiments."""

    name: str = "base"

    def __init__(self):
        self.output_dir = os.path.join(RESULTS_DIR, self.name)
        os.makedirs(self.output_dir, exist_ok=True)

    @abc.abstractmethod
    def run(self):
        ...

    def save_csv(self, filename, data, header=""):
        path = os.path.join(self.output_dir, filename)
        np.savetxt(path, data, delimiter=",", header=header, comments="")
        print(f"  Saved {path}")

    def save_fig(self, filename):
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved {path}")


# =============================================================================
# Experiment: Energy conservation
# =============================================================================

class EnergyConservation(Experiment):
    """Check total energy (kinetic + potential) drift over time."""

    name = "energy_conservation"

    def _total_energy_2d(self, sol, masses, n):
        n_steps = sol.y.shape[1]
        energy = np.zeros(n_steps)
        for t in range(n_steps):
            ke = 0.0
            pe = 0.0
            for i in range(n):
                vx = sol.y[2 * n + 2 * i, t]
                vy = sol.y[2 * n + 2 * i + 1, t]
                ke += 0.5 * masses[i] * (vx ** 2 + vy ** 2)
                for j in range(i + 1, n):
                    dx = sol.y[2 * i, t] - sol.y[2 * j, t]
                    dy = sol.y[2 * i + 1, t] - sol.y[2 * j + 1, t]
                    r = np.sqrt(dx ** 2 + dy ** 2) + SOFTENING ** (1 / 1.5)
                    pe -= G * masses[i] * masses[j] / r
            energy[t] = ke + pe
        return energy

    def run(self):
        print(f"[{self.name}] Running energy conservation check...")
        n = 5
        state0, masses = random_initial_conditions(n, dim=2, vel_range=0.5)
        sol = solve(state0, masses, t_max=10, n_steps=1000, dim=2)
        energy = self._total_energy_2d(sol, masses, n)

        self.save_csv("energy.csv", np.column_stack([sol.t, energy]),
                      header="time,total_energy")

        plt.figure(figsize=(8, 4))
        plt.plot(sol.t, energy)
        plt.xlabel("Time (yr)")
        plt.ylabel("Total energy")
        plt.title("Energy conservation (2D, 5 bodies)")
        self.save_fig("energy.png")

        drift = abs(energy[-1] - energy[0]) / (abs(energy[0]) + 1e-30)
        print(f"  Relative energy drift: {drift:.4e}")


# =============================================================================
# Experiment: Solver comparison (RK45 vs RK23 vs DOP853)
# =============================================================================

class SolverComparison(Experiment):
    """Compare integration methods on the same initial conditions."""

    name = "solver_comparison"

    def run(self):
        print(f"[{self.name}] Comparing ODE solvers...")
        n = 4
        state0, masses = random_initial_conditions(n, dim=2, vel_range=0.5)
        methods = ["RK23", "RK45", "DOP853"]
        results = {}

        for m in methods:
            sol = solve(state0, masses, t_max=10, n_steps=500, dim=2, method=m)
            results[m] = sol
            print(f"  {m}: {sol.y.shape[1]} steps, success={sol.success}")

        # Plot trajectory of body 0 for each method
        plt.figure(figsize=(8, 8))
        for m in methods:
            sol = results[m]
            plt.plot(sol.y[0], sol.y[1], label=m)
        plt.xlabel("x (AU)")
        plt.ylabel("y (AU)")
        plt.title("Body 0 trajectory — solver comparison")
        plt.legend()
        plt.axis("equal")
        self.save_fig("solver_comparison.png")


# =============================================================================
# Experiment: Collision detection stats
# =============================================================================

class CollisionStats(Experiment):
    """Run simulations and collect collision statistics vs. number of bodies."""

    name = "collision_stats"

    def run(self):
        print(f"[{self.name}] Collecting collision statistics...")
        body_counts = [3, 5, 8, 10, 15]
        n_trials = 5
        rows = []

        for n in body_counts:
            counts = []
            for trial in range(n_trials):
                state0, masses = random_initial_conditions(n, dim=2, vel_range=0.5)
                sol = solve(state0, masses, t_max=10, n_steps=500, dim=2)
                collisions = detect_collisions(sol, masses, n, dim=2)
                counts.append(len(collisions))
            mean_c = np.mean(counts)
            std_c = np.std(counts)
            rows.append([n, mean_c, std_c])
            print(f"  N={n}: {mean_c:.1f} +/- {std_c:.1f} collisions")

        data = np.array(rows)
        self.save_csv("collision_stats.csv", data,
                      header="n_bodies,mean_collisions,std_collisions")

        plt.figure(figsize=(8, 4))
        plt.errorbar(data[:, 0], data[:, 1], yerr=data[:, 2], marker="o", capsize=4)
        plt.xlabel("Number of bodies")
        plt.ylabel("Number of collisions")
        plt.title("Collision count vs. N (2D)")
        self.save_fig("collision_stats.png")


# =============================================================================
# Experiment: 2D vs 3D trajectory comparison
# =============================================================================

class Dim2Dvs3D(Experiment):
    """Compare 2D and 3D simulations with equivalent initial conditions."""

    name = "2d_vs_3d"

    def run(self):
        print(f"[{self.name}] Comparing 2D and 3D dynamics...")
        n = 4
        state0_2d, masses = random_initial_conditions(n, dim=2, vel_range=0.5)
        # Build a 3D state by adding z=0, vz=0
        pos_2d = state0_2d[: 2 * n].reshape(n, 2)
        vel_2d = state0_2d[2 * n:].reshape(n, 2)
        pos_3d = np.hstack([pos_2d, np.zeros((n, 1))]).flatten()
        vel_3d = np.hstack([vel_2d, np.zeros((n, 1))]).flatten()
        state0_3d = np.concatenate([pos_3d, vel_3d])

        sol_2d = solve(state0_2d, masses, t_max=10, n_steps=500, dim=2)
        sol_3d = solve(state0_3d, masses, t_max=10, n_steps=500, dim=3)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for i in range(n):
            axes[0].plot(sol_2d.y[2 * i], sol_2d.y[2 * i + 1], label=f"Body {i}")
            axes[1].plot(sol_3d.y[3 * i], sol_3d.y[3 * i + 1], label=f"Body {i}")
        axes[0].set_title("2D simulation")
        axes[1].set_title("3D simulation (z₀=0)")
        for ax in axes:
            ax.set_xlabel("x (AU)")
            ax.set_ylabel("y (AU)")
            ax.legend(fontsize=7)
            ax.set_aspect("equal")
        self.save_fig("2d_vs_3d.png")

        # Quantify divergence of body 0
        diff_x = sol_2d.y[0] - sol_3d.y[0]
        diff_y = sol_2d.y[1] - sol_3d.y[1]
        self.save_csv("divergence_body0.csv",
                      np.column_stack([sol_2d.t, diff_x, diff_y]),
                      header="time,dx,dy")


# =============================================================================
# Experiment: Softening parameter sensitivity
# =============================================================================

class SofteningSensitivity(Experiment):
    """Study the effect of the softening parameter on trajectories."""

    name = "softening_sensitivity"

    def run(self):
        print(f"[{self.name}] Testing softening parameter sensitivity...")
        import physics

        n = 3
        state0, masses = random_initial_conditions(n, dim=2, vel_range=0.3)
        epsilons = [1e-4, 1e-2, 1e-1, 5e-1]

        plt.figure(figsize=(8, 8))
        for eps in epsilons:
            original = physics.SOFTENING
            physics.SOFTENING = eps
            # Need to re-import won't work; directly patch the module-level constant
            # Instead we re-solve; the ODE function reads SOFTENING at call time
            sol = solve(state0, masses, t_max=10, n_steps=500, dim=2)
            physics.SOFTENING = original
            plt.plot(sol.y[0], sol.y[1], label=f"ε={eps:.0e}")

        plt.xlabel("x (AU)")
        plt.ylabel("y (AU)")
        plt.title("Body 0 trajectory — softening sensitivity")
        plt.legend()
        plt.axis("equal")
        self.save_fig("softening_sensitivity.png")


# =============================================================================
# Experiment: Time step convergence
# =============================================================================

class TimeStepConvergence(Experiment):
    """Check solution convergence as the number of time steps increases."""

    name = "timestep_convergence"

    def run(self):
        print(f"[{self.name}] Checking time-step convergence...")
        n = 3
        state0, masses = random_initial_conditions(n, dim=2, vel_range=0.3)
        step_counts = [100, 250, 500, 1000, 2000, 5000]

        # Use the finest as reference
        ref = solve(state0, masses, t_max=10, n_steps=10000, dim=2)
        ref_final = ref.y[:, -1]

        errors = []
        for ns in step_counts:
            sol = solve(state0, masses, t_max=10, n_steps=ns, dim=2)
            err = np.linalg.norm(sol.y[:, -1] - ref_final)
            errors.append(err)
            print(f"  n_steps={ns}: error={err:.4e}")

        data = np.column_stack([step_counts, errors])
        self.save_csv("convergence.csv", data, header="n_steps,final_state_error")

        plt.figure(figsize=(8, 4))
        plt.loglog(step_counts, errors, "o-")
        plt.xlabel("Number of time steps")
        plt.ylabel("Final state error (L2)")
        plt.title("Time-step convergence")
        self.save_fig("convergence.png")


# =============================================================================
# Experiment: Mass distribution effect
# =============================================================================

class MassDistributionEffect(Experiment):
    """Compare dynamics for uniform vs. heavy-center mass distributions."""

    name = "mass_distribution"

    def run(self):
        print(f"[{self.name}] Comparing mass distributions...")
        n = 6
        np.random.seed(42)

        # Uniform masses
        state0, _ = random_initial_conditions(n, dim=2, vel_range=0.3)
        masses_uniform = np.full(n, 1e29)

        # One heavy central body
        masses_central = np.full(n, 1e27)
        masses_central[0] = 1e30
        state0_c = state0.copy()
        state0_c[0], state0_c[1] = 0.0, 0.0  # center body at origin
        state0_c[2 * n], state0_c[2 * n + 1] = 0.0, 0.0  # zero velocity

        sol_u = solve(state0, masses_uniform, t_max=10, n_steps=500, dim=2)
        sol_c = solve(state0_c, masses_central, t_max=10, n_steps=500, dim=2)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for i in range(n):
            axes[0].plot(sol_u.y[2 * i], sol_u.y[2 * i + 1])
            axes[1].plot(sol_c.y[2 * i], sol_c.y[2 * i + 1])
        axes[0].set_title("Uniform masses")
        axes[1].set_title("Heavy central body")
        for ax in axes:
            ax.set_xlabel("x (AU)")
            ax.set_ylabel("y (AU)")
            ax.set_aspect("equal")
        self.save_fig("mass_distribution.png")


# =============================================================================
# Experiment: Angular momentum conservation
# =============================================================================

class AngularMomentumConservation(Experiment):
    """Track total angular momentum over time (should be conserved)."""

    name = "angular_momentum"

    def run(self):
        print(f"[{self.name}] Checking angular momentum conservation...")
        n = 5
        state0, masses = random_initial_conditions(n, dim=2, vel_range=0.5)
        sol = solve(state0, masses, t_max=10, n_steps=1000, dim=2)

        n_steps = sol.y.shape[1]
        L_total = np.zeros(n_steps)
        for t in range(n_steps):
            for i in range(n):
                x = sol.y[2 * i, t]
                y = sol.y[2 * i + 1, t]
                vx = sol.y[2 * n + 2 * i, t]
                vy = sol.y[2 * n + 2 * i + 1, t]
                L_total[t] += masses[i] * (x * vy - y * vx)

        self.save_csv("angular_momentum.csv",
                      np.column_stack([sol.t, L_total]),
                      header="time,angular_momentum")

        plt.figure(figsize=(8, 4))
        plt.plot(sol.t, L_total)
        plt.xlabel("Time (yr)")
        plt.ylabel("Total angular momentum")
        plt.title("Angular momentum conservation (2D, 5 bodies)")
        self.save_fig("angular_momentum.png")

        drift = abs(L_total[-1] - L_total[0]) / (abs(L_total[0]) + 1e-30)
        print(f"  Relative angular momentum drift: {drift:.4e}")


# =============================================================================
# Experiment: Scaling benchmark
# =============================================================================

class ScalingBenchmark(Experiment):
    """Benchmark wall-clock time vs. number of bodies."""

    name = "scaling_benchmark"

    def run(self):
        import time

        print(f"[{self.name}] Benchmarking computation time...")
        body_counts = [2, 4, 8, 12, 16, 20]
        times_list = []

        for n in body_counts:
            state0, masses = random_initial_conditions(n, dim=2, vel_range=0.5)
            t0 = time.perf_counter()
            solve(state0, masses, t_max=5, n_steps=300, dim=2)
            elapsed = time.perf_counter() - t0
            times_list.append(elapsed)
            print(f"  N={n}: {elapsed:.3f}s")

        data = np.column_stack([body_counts, times_list])
        self.save_csv("scaling.csv", data, header="n_bodies,wall_time_s")

        plt.figure(figsize=(8, 4))
        plt.plot(body_counts, times_list, "o-")
        plt.xlabel("Number of bodies")
        plt.ylabel("Wall-clock time (s)")
        plt.title("Solver scaling")
        self.save_fig("scaling.png")


# =============================================================================
# Experiment: Trajectory stability (Lyapunov-like)
# =============================================================================

class TrajectoryStability(Experiment):
    """Perturb initial conditions slightly and measure divergence."""

    name = "trajectory_stability"

    def run(self):
        print(f"[{self.name}] Measuring trajectory sensitivity to initial conditions...")
        n = 4
        state0, masses = random_initial_conditions(n, dim=2, vel_range=0.5)
        perturbations = [0, 1e-8, 1e-6, 1e-4, 1e-2]

        sol_ref = solve(state0, masses, t_max=10, n_steps=500, dim=2)
        plt.figure(figsize=(8, 5))

        for p in perturbations:
            perturbed = state0 + np.random.normal(0, p, len(state0)) if p > 0 else state0
            sol = solve(perturbed, masses, t_max=10, n_steps=500, dim=2)
            divergence = np.sqrt(
                (sol.y[0] - sol_ref.y[0]) ** 2 + (sol.y[1] - sol_ref.y[1]) ** 2
            )
            plt.plot(sol.t, divergence, label=f"δ={p:.0e}")

        plt.xlabel("Time (yr)")
        plt.ylabel("Position divergence (AU)")
        plt.title("Sensitivity to initial conditions (body 0)")
        plt.legend()
        plt.yscale("log")
        self.save_fig("stability.png")


# =============================================================================
# Registry
# =============================================================================

EXPERIMENTS = {
    "energy_conservation": EnergyConservation,
    "solver_comparison": SolverComparison,
    "collision_stats": CollisionStats,
    "2d_vs_3d": Dim2Dvs3D,
    "softening_sensitivity": SofteningSensitivity,
    "timestep_convergence": TimeStepConvergence,
    "mass_distribution": MassDistributionEffect,
    "angular_momentum": AngularMomentumConservation,
    "scaling_benchmark": ScalingBenchmark,
    "trajectory_stability": TrajectoryStability,
}


def main():
    parser = argparse.ArgumentParser(description="N-body simulation experiments")
    parser.add_argument(
        "-e", "--experiment",
        type=str,
        default="all",
        help=f"Experiment to run. Options: all, {', '.join(EXPERIMENTS.keys())}",
    )
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.experiment == "all":
        for name, cls in EXPERIMENTS.items():
            exp = cls()
            exp.run()
            print()
    else:
        if args.experiment not in EXPERIMENTS:
            print(f"Unknown experiment: {args.experiment}")
            print(f"Available: {', '.join(EXPERIMENTS.keys())}")
            return
        exp = EXPERIMENTS[args.experiment]()
        exp.run()


if __name__ == "__main__":
    main()
