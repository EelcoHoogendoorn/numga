"""Scenes for Poncelet's porism: an ellipse, and inside it the conic touching the sides of a pentagon
inscribed in it, beside the conic touching the sides of the same pentagon shrunk a little towards
the centre. A start runs once around the ellipse; from each start the path of five sides closes
for the first pair and misses for the second."""

from __future__ import annotations

from collections.abc import Iterator
from itertools import islice

import numpy as np

from numga import stack
from examples.math.poncelet import core

# The ellipse: its centre, its semi-axes, and how far its first axis is turned from x.
CENTRE = np.array([0.4, -0.2])
SEMI = np.array([2.0, 1.2])
TILT = 0.5
# The corners of the pentagon, as angles around the ellipse, bunched to one side, and the next corner
# of each; and how far the second pentagon is shrunk towards the centre.
CORNERS = np.array([0.1, 0.8, 1.9, 3.0, 5.2])
NEXT = np.array([1, 2, 3, 4, 0])
SHRINK = 0.97
# The sides of each path.
N_SIDES = 5
# The frames of the turn, the angle of the first start, the frame shown as a still, and the duration of
# a frame in milliseconds.
N_FRAMES = 120
FIRST_START = 0.05
STILL = 10
FRAME_DURATION = 60


# --- math -----------------------------------------------------------------------------
def run(angles: np.ndarray, n_sides: int) -> tuple[core.Conic, core.Envelope, core.Point]:
    """The ellipse, the two inner conics held by their tangents, and from the point of the ellipse at each
    angle the vertices of the path of the given number of sides about each inner conic."""
    outer = core.ellipse(SEMI, placement())                                    # [] Plane <- Point
    corners = stack([on_ellipse(CORNERS, 1.0), on_ellipse(CORNERS, SHRINK)])   # [inners, corners] Point
    inner = core.envelope(corners & corners[..., NEXT])                        # [inners] Pole <- Chord
    starts = on_ellipse(angles, 1.0)                                           # [starts] Point
    first = core.tangents(inner[:, None], starts[None, :])[..., 0]             # [inners, starts] Point
    # The ellipse's centre, inside both inner conics: no tangent of either passes through it.
    inside = core.mv.point(CENTRE)                                             # [] Point
    vertices = stack(list(islice(core.path(outer, inner[:, None], starts[None, :], first, inside), n_sides + 1)))   # [sides + 1, inners, starts] Point
    return outer, inner, vertices


def turn(n_frames: int) -> Iterator[tuple[core.Conic, core.Envelope, core.Point]]:
    """A start once around the ellipse: per frame the ellipse, the inner conics, and the two paths."""
    outer, inner, vertices = run(np.linspace(0.0, 2 * np.pi, n_frames, endpoint=False) + FIRST_START, N_SIDES)
    for frame in range(n_frames):
        yield outer, inner, vertices[:, :, frame]


# --- plumbing -------------------------------------------------------------------------
def placement() -> core.Motor:
    """The motor that turns the ellipse about the origin by the tilt and carries it to the centre."""
    mv = core.mv
    return ((mv.xw * CENTRE[0] + mv.yw * CENTRE[1]) * 0.5).exp() * (mv.xy * (-TILT / 2)).exp()


def on_ellipse(angles: np.ndarray, scale: float) -> core.Point:
    """Points at the given angles on the ellipse, drawn in towards its centre by the scale."""
    return placement() >> core.mv.point(scale * SEMI * np.stack([np.cos(angles), np.sin(angles)], axis=-1))


def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.math.poncelet import render

    scenes = list(turn(N_FRAMES))
    save_figure(render.draw(*scenes[STILL]), "poncelet_pentagons")
    save_animation(render.animate(scenes), "poncelet_turn", FRAME_DURATION)

    # checks
    # Every vertex lies on the ellipse, and every side touches its inner conic, passing through its own
    # pole, measured against the envelope's size; the path about the first inner conic closes after five
    # sides from every start, the path about the second does not.
    outer, inner, vertices = run(np.linspace(0.0, 2 * np.pi, N_FRAMES, endpoint=False) + FIRST_START, N_SIDES)
    np.testing.assert_allclose((outer(vertices) & vertices).to_array(), 0.0, atol=1e-12)
    sides = (vertices[:-1] & vertices[1:]).normalized()                        # [sides, inners, starts] Chord
    touch = (sides & inner[:, None](sides)).to_array() / np.abs(inner.kernel).max(axis=(-1, -2))[:, None]
    np.testing.assert_allclose(touch, 0.0, atol=1e-10)
    gap = np.abs((vertices[N_SIDES] - vertices[0]).kernel).max(axis=-1)        # [inners, starts]
    np.testing.assert_allclose(gap[0], 0.0, atol=1e-11)
    assert gap[1].min() > 1e-2


if __name__ == "__main__":
    main()
