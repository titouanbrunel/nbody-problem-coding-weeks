"""
N-body gravitational simulation.

Usage:
    python main.py                          # Default 2D, 5 bodies
    python main.py --dim 3 --bodies 10
    python main.py --experiment all          # Run all experiments
    python main.py -e energy_conservation    # Run one experiment
"""

import argparse
import numpy as np

from physics import solve, random_initial_conditions, detect_collisions
from visualization import animate_2d, animate_3d


def run_simulation(args):
    """Run an interactive N-body simulation with animation."""
    np.random.seed(args.seed if args.seed >= 0 else None)

    state0, masses = random_initial_conditions(
        n_bodies=args.bodies,
        dim=args.dim,
        pos_range=args.pos_range,
        vel_range=args.vel_range,
        mass_low=args.mass_low,
        mass_high=args.mass_high,
    )

    print(f"Solving {args.bodies}-body problem in {args.dim}D...")
    print(f"  t_max={args.t_max} yr, n_steps={args.n_steps}, method={args.method}")

    sol = solve(state0, masses, t_max=args.t_max, n_steps=args.n_steps,
                dim=args.dim, method=args.method)

    if not sol.success:
        print(f"Warning: solver reported failure — {sol.message}")

    # Collision detection
    collisions = detect_collisions(sol, masses, args.bodies, dim=args.dim)
    print(f"  Detected {len(collisions)} collision event(s)")

    # Animation
    if args.dim == 2:
        animate_2d(
            sol, args.bodies, masses,
            trail_length=args.trail,
            save_path=args.save,
        )
    else:
        animate_3d(
            sol, args.bodies, masses, args.n_steps,
            trail_length=args.trail,
            save_path=args.save,
        )


def run_experiments(experiment_name):
    """Delegate to the research module."""
    from research import EXPERIMENTS, RESULTS_DIR
    import os

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if experiment_name == "all":
        for name, cls in EXPERIMENTS.items():
            exp = cls()
            exp.run()
            print()
    else:
        if experiment_name not in EXPERIMENTS:
            print(f"Unknown experiment: {experiment_name}")
            print(f"Available: {', '.join(EXPERIMENTS.keys())}")
            return
        exp = EXPERIMENTS[experiment_name]()
        exp.run()


def main():
    parser = argparse.ArgumentParser(description="N-body gravitational simulation")
    parser.add_argument("-e", "--experiment", type=str, default=None,
                        help="Run experiment(s). Use 'all' or a specific name.")
    parser.add_argument("--dim", type=int, default=2, choices=[2, 3])
    parser.add_argument("--bodies", type=int, default=5)
    parser.add_argument("--t_max", type=float, default=10.0)
    parser.add_argument("--n_steps", type=int, default=500)
    parser.add_argument("--method", type=str, default="RK45")
    parser.add_argument("--pos_range", type=float, default=5.0)
    parser.add_argument("--vel_range", type=float, default=1.0)
    parser.add_argument("--mass_low", type=float, default=1e27)
    parser.add_argument("--mass_high", type=float, default=1e30)
    parser.add_argument("--trail", type=int, default=10)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--save", type=str, default=None,
                        help="Save animation to GIF (provide path).")
    args = parser.parse_args()

    if args.experiment is not None:
        run_experiments(args.experiment)
    else:
        run_simulation(args)


if __name__ == "__main__":
    main()
