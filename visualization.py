"""
Visualization: 2D and 3D animations for the N-body simulation.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.ticker import MultipleLocator


def mass_colormap(masses):
    """Map masses to colors using the coolwarm colormap."""
    normed = masses / masses.max()
    return plt.cm.coolwarm(normed)


def animate_2d(solution, n_bodies, masses, trail_length=10, interval=20,
               pos_range=None, save_path=None):
    """
    Animate the 2D N-body solution.

    Parameters
    ----------
    solution : OdeSolution
    n_bodies : int
    masses : array-like
    trail_length : int
    interval : int
        Milliseconds between frames.
    pos_range : float or None
        Axis limits [-pos_range, pos_range]. Auto-computed if None.
    save_path : str or None
        If provided, save the animation as a GIF.
    """
    colors = mass_colormap(masses)

    from physics import body_display_size
    sizes = body_display_size(masses)

    plt.style.use("dark_background")
    fig, ax = plt.subplots()
    ax.set_aspect("equal")

    if pos_range is None:
        all_pos = solution.y[: 2 * n_bodies, :]
        pos_range = np.abs(all_pos).max() * 1.3

    ax.set_xlim(-pos_range, pos_range)
    ax.set_ylim(-pos_range, pos_range)
    ax.set_xlabel("x (AU)")
    ax.set_ylabel("y (AU)")

    lines = [
        ax.plot([], [], "o-", markersize=max(2, sizes[i]), markevery=10000,
                color=colors[i], lw=1)[0]
        for i in range(n_bodies)
    ]

    def _update(k):
        for j in range(n_bodies):
            start = max(1, k - trail_length)
            lines[j].set_data(
                solution.y[2 * j, k:start:-1],
                solution.y[2 * j + 1, k:start:-1],
            )
        ax.set_title(
            f"N-body problem ({n_bodies} bodies)  t = {solution.t[k]:.1f} yr"
        )
        return lines

    anim = animation.FuncAnimation(
        fig, _update, frames=len(solution.t), interval=interval, blit=False
    )

    if save_path:
        writer = animation.PillowWriter(fps=30)
        anim.save(save_path, writer=writer)
        print(f"Saved animation to {save_path}")

    plt.tight_layout()
    plt.show()


def animate_3d(solution, n_bodies, masses, n_steps, trail_length=10,
               interval=20, pos_range=None, save_path=None):
    """
    Animate the 3D N-body solution with a slowly rotating camera.

    Parameters
    ----------
    solution : OdeSolution
    n_bodies : int
    masses : array-like
    n_steps : int
        Total number of time steps (used for camera rotation speed).
    trail_length : int
    interval : int
    pos_range : float or None
    save_path : str or None
    """
    colors = mass_colormap(masses)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    if pos_range is None:
        all_pos = solution.y[: 3 * n_bodies, :]
        pos_range = np.abs(all_pos).max() * 1.3

    ax.set_xlim3d(-pos_range, pos_range)
    ax.set_ylim3d(-pos_range, pos_range)
    ax.set_zlim3d(-pos_range, pos_range)
    ax.set_xlabel("x (AU)")
    ax.set_ylabel("y (AU)")
    ax.set_zlabel("z (AU)")
    ax.grid(False)

    for axis_pane in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis_pane.set_major_locator(MultipleLocator(max(1, int(pos_range / 3))))

    pane_color = (0, 52 / 255, 76 / 255, 1)
    ax.xaxis.set_pane_color(pane_color)
    ax.yaxis.set_pane_color(pane_color)
    ax.zaxis.set_pane_color(pane_color)

    lines = [
        ax.plot([], [], "o-", markersize=4, markevery=10000,
                color=colors[i], lw=1)[0]
        for i in range(n_bodies)
    ]

    def _update(k):
        for j in range(n_bodies):
            start = max(1, k - trail_length)
            lines[j].set_data(
                solution.y[3 * j, k:start:-1],
                solution.y[3 * j + 1, k:start:-1],
            )
            lines[j].set_3d_properties(
                solution.y[3 * j + 2, k:start:-1]
            )
        ax.view_init(azim=k * 90 / n_steps)
        ax.set_title(
            f"N-body problem ({n_bodies} bodies)  t = {solution.t[k]:.1f} yr"
        )
        return lines

    anim = animation.FuncAnimation(
        fig, _update, frames=len(solution.t), interval=interval, blit=False
    )

    if save_path:
        writer = animation.PillowWriter(fps=30)
        anim.save(save_path, writer=writer)
        print(f"Saved animation to {save_path}")

    plt.tight_layout()
    plt.show()
