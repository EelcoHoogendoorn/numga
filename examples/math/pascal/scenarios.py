"""Scene parameters and animation entry point for Pascal's theorem.

Five points sampled on an ellipse fix the quadric, and a line turning through a half-turn about the first point sweeps a sixth point once around the curve while the three opposite-side crossings remain on the Pascal line.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from examples.math.pascal import core
from numga import stack

# Semi-axes and centre of the ellipse from which the five fixed corners are sampled.
semi_axes = np.array([1.6, 1.0])
centre = np.array([0.3, -0.1])
five_angles = np.array([0.4, 1.44, 4.12, 4.77, 6.08])

# Frame count for the half-turn, representative still index, and frame duration in milliseconds.
n_frames = 180
still = 3 * n_frames // 5
frame_duration_ms = 40

# Off-quadric probe count and RNG seed for comparing `pascal_form` against `quadric`.
n_probes = 8
seed = 0


# --- plumbing -------------------------------------------------------------------------
def ellipse_points(angles: np.ndarray) -> core.Point:
    """Angles (`[...]`) around the reference ellipse map to Euclidean points (`[...] Point-`)."""
    coordinates = centre + semi_axes * np.stack([np.cos(angles), np.sin(angles)], axis=-1)
    return core.mv.point(coordinates)                                          # [...] Point-


def turning_angles(n_steps: int) -> np.ndarray:
    """Midpoint angles (`[n_steps]`) across a half-turn avoid the tangent direction where the line meets `five[0]` twice."""
    return np.linspace(0.0, np.pi, n_steps, endpoint=False) + np.pi / (2 * n_steps)


def main() -> None:
    import matplotlib.pyplot as plt

    from examples.math.pascal import render
    from examples.util.animation import save_animation, save_figure

    five = ellipse_points(five_angles)                                         # [5] Point-
    shape = core.quadric(five)                                                 # [] Plane+ <- Point-
    states: list[tuple[core.Point, core.Line, core.Crossing, core.Line]] = []

    def recorded() -> Iterator[tuple[core.Point, core.Line, core.Crossing, core.Line]]:
        for state in core.hexagon_configurations(five, turning_angles(n_frames)):
            states.append(state)
            yield state

    save_animation(render.animate(shape, recorded()), "pascal_turn", frame_duration_ms)
    figure = render.draw(shape, *states[still])
    save_figure(figure, "pascal_hexagon")
    plt.close(figure)

    # checks
    hexagons = stack([state[0] for state in states])                           # [n_frames, 6] Point-
    crossings = stack([state[2] for state in states])                          # [n_frames, 3] Crossing+
    joins = crossings[:, 0] & crossings[:, 1] & crossings[:, 2]                # [n_frames] Scalar+
    sizes = crossings[:, 0].norm() * crossings[:, 1].norm() * crossings[:, 2].norm()   # [n_frames] Scalar+
    assert ((shape(five) & five).abs() < 1e-11).all()
    assert ((shape(hexagons) & hexagons).abs() < 1e-11).all()
    assert ((joins / sizes).abs() < 1e-10).all()
    probes = core.mv.point(np.random.default_rng(seed).normal(size=(n_probes, 2)))   # [n_probes] Point-
    ratios = core.pascal_form(five)(probes, probes) / (shape(probes) & probes)       # [n_probes] Scalar+
    assert ((ratios - ratios[0]).abs() < 1e-11 * ratios[0].abs()).all()


if __name__ == "__main__":
    main()
