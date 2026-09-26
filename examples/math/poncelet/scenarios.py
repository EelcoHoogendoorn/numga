"""Scenes for Poncelet's porism: an ellipse, and inside it the conic touching the sides of a pentagon
inscribed in it, beside the conic touching the sides of the same pentagon shrunk a little towards
the centre. A start runs once around the ellipse; from each start the path of five sides closes
for the first pair and misses for the second."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import stack
from examples.math.poncelet import core

CENTRE = np.array([0.4, -0.2])
SEMI = np.array([2.0, 1.2])
TILT = 0.5
# The corners of the pentagon, as angles around the ellipse, bunched to one side, and the next corner
# of each.
CORNERS = np.array([0.1, 0.8, 1.9, 3.0, 5.2])
NEXT = np.array([1, 2, 3, 4, 0])


# --- math -----------------------------------------------------------------------------
def run(angles: np.ndarray, steps: int) -> tuple[core.Conic, core.Conic, core.Point]:
    """The ellipse, the two inner conics, and from the point of the ellipse at each angle the path of
    the given number of sides about each inner conic. Returns the ellipse, the inner conics as maps
    from points, and the vertices of the paths."""
    outer = core.ellipse(CENTRE, SEMI, TILT)                                   # [] Plane <- Point
    corners = stack([on_ellipse(CORNERS, 1.0), on_ellipse(CORNERS, 0.97)])     # [inners, corners] Point
    inner = core.envelope(corners & corners[..., NEXT])                        # [inners] Point <- Plane
    starts = on_ellipse(angles, 1.0)                                           # [starts] Point
    first = core.tangents(inner[:, None], starts[None, :])[..., 0]             # [inners, starts] Plane
    vertices = core.path(outer, inner[:, None], starts[None, :], first, steps)   # [steps + 1, inners, starts] Point

    # --- checks
    # Every vertex lies on the ellipse, and every side touches its inner conic; the path about the
    # first inner conic closes after five sides from every start, the path about the second does not.
    np.testing.assert_allclose((outer(vertices) & vertices).to_array(), 0.0, atol=1e-12)
    sides = (vertices[:-1] & vertices[1:]).normalized()                        # [steps, inners, starts] Plane
    touch = (inner[:, None](sides) & sides).to_array() / np.abs(inner.kernel).max(axis=(-1, -2))[:, None]
    np.testing.assert_allclose(touch, 0.0, atol=1e-10)
    gap = np.abs((vertices[5] - vertices[0]).kernel).max(axis=-1)              # [inners, starts]
    np.testing.assert_allclose(gap[0], 0.0, atol=1e-11)
    assert gap[1].min() > 1e-2
    return outer, inner.inverse(), vertices


def turn(frames: int) -> Iterator[tuple[core.Conic, core.Conic, core.Point]]:
    """A start once around the ellipse: per frame the ellipse, the inner conics and the two paths."""
    outer, inner, vertices = run(np.linspace(0.0, 2 * np.pi, frames, endpoint=False) + 0.05, 5)
    for frame in range(frames):
        yield outer, inner, vertices[:, :, frame]


# --- plumbing -------------------------------------------------------------------------
def on_ellipse(angles: np.ndarray, scale: float) -> core.Point:
    """Points at the given angles on the ellipse, drawn in towards its centre by the scale."""
    # The ellipse about the origin, turned by the tilt and carried to the centre.
    mv = core.mv
    motor = ((mv.xw * CENTRE[0] + mv.yw * CENTRE[1]) * 0.5).exp() * (mv.xy * (-TILT / 2)).exp()
    return motor >> core.point(scale * SEMI * np.stack([np.cos(angles), np.sin(angles)], axis=-1))


if __name__ == "__main__":
    from examples.animation import save_animation, save_figure
    from examples.math.poncelet import render

    scenes = list(turn(120))
    save_figure(render.draw(*scenes[10]), "poncelet_pentagons")
    save_animation(render.animate(scenes), "poncelet_turn", 60)
