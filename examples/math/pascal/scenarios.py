"""Scenes for Pascal's theorem: five points on an ellipse fix the conic, and a sixth point runs once
around it as the second crossing of a line turning about the first. The hexagon of the six changes
shape, convex and crossed, and the three crossings of its opposite sides stay on one line, which
turns with it."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import stack
from examples.math.pascal import core


# --- math -----------------------------------------------------------------------------
def run(five: core.Point, angles: np.ndarray) -> tuple[core.Conic, core.Point, core.Point]:
    """The conic through five points, and for a line through the first at each angle the hexagon of
    the five and the line's second crossing with the conic, with the crossings of its opposite sides."""
    shape = core.conic(five)                                                   # [] Plane <- Point
    sixth = core.second_crossing(shape, five[0], core.headings(angles))        # [angles] Point
    sixth = sixth / (core.mv.w & sixth)                                        # [angles] Point
    hexagon = stack([five[index].broadcast_to(sixth.shape) for index in range(5)] + [sixth], axis=-1)   # [angles, 6] Point
    crossing = core.crossings(hexagon)                                         # [angles, 3] Point

    # --- checks
    # The conic passes through the five points and every sixth; the three crossings lie on one line;
    # Pascal's condition on the sixth point is the conic itself, up to scale.
    np.testing.assert_allclose((shape(five) & five).to_array(), 0.0, atol=1e-11)
    np.testing.assert_allclose((shape(sixth) & sixth).to_array(), 0.0, atol=1e-11)
    # Opposite sides that are nearly parallel meet far out, so the join is measured against the size
    # of the three crossings.
    join = (crossing[..., 0] & crossing[..., 1] & crossing[..., 2]).to_array()   # [angles]
    size = np.prod(np.linalg.norm(crossing.kernel, axis=-1), axis=-1)            # [angles]
    np.testing.assert_allclose(join / size, 0.0, atol=1e-12)
    probes = core.point(np.random.default_rng(0).normal(size=(8, 2)))          # [probes] Point
    ratio = (core.pascal(five)(probes, probes) / (shape(probes) & probes)).to_array()
    np.testing.assert_allclose(ratio, ratio[0], rtol=1e-11)
    return shape, hexagon, crossing


def turn(frames: int) -> Iterator[tuple[core.Conic, core.Point, core.Point]]:
    """The sixth point once around the conic: per frame the conic, the hexagon and its crossings."""
    angles = np.linspace(0.0, np.pi, frames, endpoint=False) + np.pi / (2 * frames)
    shape, hexagon, crossing = run(ellipse_points(np.array([0.4, 1.44, 4.12, 4.77, 6.08])), angles)
    for frame in range(frames):
        yield shape, hexagon[frame], crossing[frame]


# --- plumbing -------------------------------------------------------------------------
def ellipse_points(angles: np.ndarray) -> core.Point:
    """Points at the given angles on an ellipse, off the origin."""
    return core.point(np.stack([1.6 * np.cos(angles) + 0.3, np.sin(angles) - 0.1], axis=-1))


if __name__ == "__main__":
    from examples.animation import save_animation, save_figure
    from examples.math.pascal import render

    scenes = list(turn(120))
    save_figure(render.draw(*scenes[72]), "pascal_hexagon")
    save_animation(render.animate(scenes), "pascal_turn", 60)
