"""Scenes for Poncelet's porism: an ellipsoid, and inside it the quadric touching the sides of a pentagon
inscribed in it, beside the quadric touching the sides of the same pentagon shrunk a little towards
the centre. A start runs once around the ellipsoid; from each start the path of five sides closes
for the first pair and misses for the second."""

from __future__ import annotations

from collections.abc import Iterator
from itertools import islice

import numpy as np

from numga import stack
from examples.math.poncelet import core

# The ellipsoid: its centre, its semi-axes, and how far its first axis is turned from x.
centre = np.array([0.4, -0.2])
semi = np.array([2.0, 1.2])
tilt = 0.5
# The corners of the pentagon, as angles around the ellipsoid, bunched to one side, and the next corner
# of each; and how far the second pentagon is shrunk towards the centre.
corner_angles = np.array([0.1, 0.8, 1.9, 3.0, 5.2])
following = np.array([1, 2, 3, 4, 0])
shrink = 0.97
# The sides of each path.
n_sides = 5
# The frames of the turn, the angle of the first start, the frame shown as a still, the duration of a
# frame in milliseconds, and the colours of the animation's palette.
n_frames = 120
first_start = 0.05
still = 10
frame_duration = 60
n_colours = 32
# The ellipsoid turned about the origin by the tilt and carried to its centre.
shift_to_centre = ((core.mv.xw * centre[0] + core.mv.yw * centre[1]) * 0.5).exp()   # [] Motor+
tilt_turn = (core.mv.xy * (-tilt / 2)).exp()                                   # [] Motor+
placement = shift_to_centre * tilt_turn                                        # [] Motor+


# --- math -----------------------------------------------------------------------------
def run(angles: np.ndarray, n_sides: int) -> tuple[core.Quadric, core.Envelope, core.Point]:
    """The ellipsoid, the two inner quadrics held by their tangents, and from the point of the ellipsoid at each
    angle the vertices of the path of the given number of sides about each inner quadric."""
    outer = core.ellipsoid(semi, placement)                                    # [] Plane+ <- Point-
    corners = stack([on_ellipsoid(corner_angles, 1.0), on_ellipsoid(corner_angles, shrink)])   # [inners, corners] Point-
    inner = core.envelope(corners & corners[..., following])                   # [inners] Pole+ <- Chord-
    starts = on_ellipsoid(angles, 1.0)                                         # [starts] Point-
    first_heading = core.tangents(inner[:, None], starts[None, :])[..., 0]     # [inners, starts] Point-
    # The ellipsoid's centre, inside both inner quadrics: no tangent of either passes through it.
    inside = core.mv.point(centre)                                             # [] Point-
    paths = core.path(outer, inner[:, None], starts[None, :], first_heading, inside)
    vertices = stack(list(islice(paths, n_sides + 1)))                         # [sides + 1, inners, starts] Point-
    return outer, inner, vertices


def turn(n_frames: int) -> Iterator[tuple[core.Quadric, core.Envelope, core.Point]]:
    """A start once around the ellipsoid: per frame the ellipsoid, the inner quadrics, and the two paths."""
    start_angles = np.linspace(0.0, 2 * np.pi, n_frames, endpoint=False) + first_start
    outer, inner, vertices = run(start_angles, n_sides)
    for frame in range(n_frames):
        yield outer, inner, vertices[:, :, frame]


# --- plumbing -------------------------------------------------------------------------
def on_ellipsoid(angles: np.ndarray, scale: float) -> core.Point:
    """Points at the given angles on the ellipsoid, drawn in towards its centre by the scale."""
    return placement >> core.mv.point(scale * semi * np.stack([np.cos(angles), np.sin(angles)], axis=-1))


def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.math.poncelet import render

    scenes = list(turn(n_frames))
    save_figure(render.draw(*scenes[still]), "poncelet_pentagons")
    save_animation(render.animate(scenes), "poncelet_turn", frame_duration, n_colours)

    # checks
    # Every vertex lies on the ellipsoid, and every side touches its inner quadric, passing through its own
    # pole, measured against the envelope's size; the path about the first inner quadric closes after five
    # sides from every start, the path about the second does not.
    outer, inner, vertices = run(np.linspace(0.0, 2 * np.pi, n_frames, endpoint=False) + first_start, n_sides)
    np.testing.assert_allclose((outer(vertices) & vertices).to_array(), 0.0, atol=1e-12)
    sides = (vertices[:-1] & vertices[1:]).normalized()                        # [sides, inners, starts] Chord-
    touch = (sides & inner[:, None](sides)).to_array() / np.abs(inner.kernel).max(axis=(-1, -2))[:, None]
    np.testing.assert_allclose(touch, 0.0, atol=1e-10)
    gap = np.abs((vertices[n_sides] - vertices[0]).kernel).max(axis=-1)        # [inners, starts]
    np.testing.assert_allclose(gap[0], 0.0, atol=1e-11)
    assert gap[1].min() > 1e-2


if __name__ == "__main__":
    main()
