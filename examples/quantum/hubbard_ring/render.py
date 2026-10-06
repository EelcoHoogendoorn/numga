"""The low levels of the Hubbard ring against the Heisenberg ring, and a flipped spin walking the ring."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.quantum.hubbard_ring import core

UP, DOWN, LEVELS = "#c0392b", "#2e86c1", "#7d3c98"
# The sites on a circle, the first at the bottom left, in the order the hops go round.
PLACES = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]) * 0.7


# --- plumbing -------------------------------------------------------------------------
def draw_levels(ratios: np.ndarray, excitations: core.Scalar, heisenberg: np.ndarray) -> plt.Figure:
    """The lowest levels above the ground state, in units of the exchange, against the repulsion; the
    Heisenberg ring's levels dashed."""
    figure, ax = plt.subplots(figsize=(8, 3.6), layout="constrained")
    ax.plot(ratios, excitations.to_array(), color=LEVELS, linewidth=1.0)
    for level in heisenberg:
        ax.axhline(level, color="0.5", linestyle="--", linewidth=0.8)
    ax.set(xlim=(ratios[0], ratios[-1]), ylim=(-0.1, heisenberg[-1] + 0.6),
           xlabel=r"repulsion / hopping, $U/t$", ylabel=r"above the ground state, $\Delta E\,U/4t^2$")
    return figure


def animate_ring(labels: list[str], occupation: core.Scalar) -> list[np.ndarray]:
    """Each site's spin-up count as an arrow up and its spin-down count as an arrow down, one ring per
    case, frame by frame."""
    counts = occupation.to_array()                                            # [cases, times, sites, spins]
    figure = plt.figure(figsize=(3.3 * len(labels), 3.6), dpi=80)
    turn = np.linspace(0, 2 * np.pi, 120)
    arrows = []
    for case, label in enumerate(labels):
        ax = figure.add_axes((case / len(labels), 0.08, 1 / len(labels), 0.9))
        ax.plot(0.99 * np.cos(turn), 0.99 * np.sin(turn), color="0.85", linewidth=0.8)
        ax.scatter(*PLACES.T, color="0.3", s=12, zorder=3)
        ups = [ax.plot([], [], color=UP, linewidth=4.0, solid_capstyle="butt")[0] for _ in PLACES]
        downs = [ax.plot([], [], color=DOWN, linewidth=4.0, solid_capstyle="butt")[0] for _ in PLACES]
        arrows.append((ups, downs))
        ax.set(xlim=(-1.3, 1.3), ylim=(-1.3, 1.3), xticks=[], yticks=[])
        ax.set_xlabel(label, fontsize=10)
        ax.set_aspect("equal")
        for side in ax.spines.values():
            side.set_visible(False)
    frames = []
    for index in range(counts.shape[1]):
        for (ups, downs), case in zip(arrows, counts[:, index]):
            for up, down, (x, y), (up_count, down_count) in zip(ups, downs, PLACES, case):
                up.set_data([x, x], [y, y + 0.5 * up_count])
                down.set_data([x, x], [y, y - 0.5 * down_count])
        figure.canvas.draw()
        frames.append(np.asarray(figure.canvas.buffer_rgba())[..., :3].copy())
    plt.close(figure)
    return frames


def inline(frames: list[np.ndarray], duration_ms: int):
    """Frames as a looping GIF to show in a notebook, kept in memory."""
    from io import BytesIO
    from IPython.display import Image as Shown
    from PIL import Image
    images = [Image.fromarray(pixels) for pixels in frames]
    buffer = BytesIO()
    images[0].save(buffer, format="GIF", save_all=True, append_images=images[1:], duration=duration_ms, loop=0)
    return Shown(data=buffer.getvalue(), format="gif")
