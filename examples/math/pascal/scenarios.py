"""Scenes for Pascal's theorem: five points on an ellipse fix the conic, and a sixth point runs once
around it as the second crossing of a line turning about the first. The hexagon of the six changes
shape, convex and crossed, and the three crossings of its opposite sides stay on one line, which
turns with it."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import stack
from examples.math.pascal import core

# The ellipse the five points lie on: its semi-axes along x and y, and its centre.
SEMI_AXES = np.array([1.6, 1.0])
CENTRE = np.array([0.3, -0.1])
# Where the five points lie on it, by angle.
FIVE = np.array([0.4, 1.44, 4.12, 4.77, 6.08])
# The frames of the turn, the frame shown as a still, and the duration of a frame in milliseconds.
N_FRAMES = 120
STILL = 72
FRAME_DURATION = 60
# Points off the conic at which Pascal's condition is compared with the conic, and their seed.
N_PROBES = 8
SEED = 0


# --- math -----------------------------------------------------------------------------
def run(five: core.Point, angles: np.ndarray) -> tuple[core.Conic, core.Point, core.Plane, core.Point, core.Plane]:
    """The conic through five points, and for a line through the first at each angle the hexagon of
    the five and the line's second crossing with the conic, its sides, the crossings of its opposite
    sides, and the line through them."""
    shape = core.conic(five)                                                   # [] Plane <- Point
    sixth = core.second_crossing(shape, five[0], core.headings(angles))        # [angles] Point
    sixth = sixth / (core.mv.w & sixth)                                        # [angles] Point
    hexagon = stack([five[index].broadcast_to(sixth.shape) for index in range(5)] + [sixth], axis=-1)   # [angles, 6] Point
    sides = core.sides(hexagon)                                                # [angles, 6] Plane
    crossing = core.crossings(sides)                                           # [angles, 3] Point
    return shape, hexagon, sides, crossing, crossing[..., 0] & crossing[..., 2]   # ..., [angles] Plane


def turn(n_frames: int) -> Iterator[tuple[core.Conic, core.Point, core.Plane, core.Point, core.Plane]]:
    """The sixth point once around the conic: per frame the conic, the hexagon, its sides, its crossings
    and the line through them."""
    # Angles of the turning line, half a turn in steps, off the angles where it meets the first point twice.
    angles = np.linspace(0.0, np.pi, n_frames, endpoint=False) + np.pi / (2 * n_frames)
    shape, hexagon, sides, crossing, line = run(ellipse_points(FIVE), angles)
    for frame in range(n_frames):
        yield shape, hexagon[frame], sides[frame], crossing[frame], line[frame]


# --- plumbing -------------------------------------------------------------------------
def ellipse_points(angles: np.ndarray) -> core.Point:
    """Points at the given angles on the ellipse."""
    return core.mv.point(CENTRE + SEMI_AXES * np.stack([np.cos(angles), np.sin(angles)], axis=-1))


def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.math.pascal import render

    scenes = list(turn(N_FRAMES))
    save_figure(render.draw(*scenes[STILL]), "pascal_hexagon")
    save_animation(render.animate(scenes), "pascal_turn", FRAME_DURATION)

    # checks
    # The conic passes through the five points and every sixth; the three crossings lie on one line, the
    # join measured against the size of the crossings, since opposite sides that are nearly parallel meet
    # far out; Pascal's condition on the sixth point is the conic itself, up to scale.
    five = ellipse_points(FIVE)
    shape = scenes[0][0]
    hexagon = stack([scene[1] for scene in scenes])                            # [frames, 6] Point
    crossing = stack([scene[3] for scene in scenes])                           # [frames, 3] Point
    np.testing.assert_allclose((shape(five) & five).to_array(), 0.0, atol=1e-11)
    np.testing.assert_allclose((shape(hexagon) & hexagon).to_array(), 0.0, atol=1e-11)
    join = (crossing[..., 0] & crossing[..., 1] & crossing[..., 2]).to_array()   # [frames]
    size = np.prod(np.linalg.norm(crossing.kernel, axis=-1), axis=-1)            # [frames]
    np.testing.assert_allclose(join / size, 0.0, atol=1e-10)
    probes = core.mv.point(np.random.default_rng(SEED).normal(size=(N_PROBES, 2)))   # [probes] Point
    ratio = (core.pascal(five)(probes, probes) / (shape(probes) & probes)).to_array()
    np.testing.assert_allclose(ratio, ratio[0], rtol=1e-11)


if __name__ == "__main__":
    main()
