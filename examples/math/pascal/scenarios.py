"""Scenes for Pascal's theorem: five points on an ellipsoid fix the quadric, and a sixth point runs once
around it as the second crossing of a line turning about the first. The hexagon of the six changes
shape, convex and crossed, and the three crossings of its opposite sides stay on one line, which
turns with it."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import stack
from examples.math.pascal import core

# The ellipsoid the five points lie on: its semi-axes along x and y, and its centre.
semi_axes = np.array([1.6, 1.0])
centre = np.array([0.3, -0.1])
# Where the five points lie on it, by angle.
five_angles = np.array([0.4, 1.44, 4.12, 4.77, 6.08])
# The next corner of each corner of a hexagon.
next_corner = np.array([1, 2, 3, 4, 5, 0])
# The frames of the turn, the frame shown as a still, and the duration of a frame in milliseconds.
n_frames = 180
still = 3 * n_frames // 5
frame_duration = 40
# Points off the quadric at which Pascal's condition is compared with the quadric, and their seed.
n_probes = 8
seed = 0


# --- math -----------------------------------------------------------------------------
def run(five: core.Point, angles: np.ndarray) -> tuple[core.Quadric, core.Point, core.Line, core.Crossing, core.Line]:
    """The quadric through five points, and for a line through the first at each angle the hexagon of
    the five and the line's second crossing with the quadric, its sides, the crossings of its opposite
    sides, and the line through them."""
    shape = core.quadric(five)                                                 # [] Plane+ <- Point-
    start = five[0]                                                            # [] Point-
    # Points at infinity in the directions of the lines through the start.
    turns = (core.mv.xy * (-angles / 2)).exp()                                 # [angles] Motor+
    headings = turns >> core.mv.x.dual()                                       # [angles] Point-
    # Along the line, `start + heading * t`, the quadric's value is
    # `2 * t * (shape(start) & heading) + t * t * (shape(heading) & heading)`: zero at the start, and at the other root.
    sixth = start * (shape(headings) & headings) - headings * (2 * (shape(start) & headings))   # [angles] Point-
    sixth = sixth.normalized()                                                 # [angles] Point-
    hexagon = stack([five[index].broadcast_to(sixth.shape) for index in range(5)] + [sixth], axis=-1)   # [angles, 6] Point-
    # Each side the join of a corner and the next; opposite sides meet in the three crossings.
    sides = hexagon & hexagon[..., next_corner]                                # [angles, 6] Line-
    crossing = sides[..., :3] ^ sides[..., 3:]                                 # [angles, 3] Crossing+
    return shape, hexagon, sides, crossing, crossing[..., 0] & crossing[..., 2]   # ..., [angles] Line-


def turn(n_frames: int) -> Iterator[tuple[core.Quadric, core.Point, core.Line, core.Crossing, core.Line]]:
    """The sixth point once around the quadric: per frame the quadric, the hexagon, its sides, its crossings
    and the line through them."""
    # Angles of the turning line, half a turn in steps, off the angles where it meets the first point twice.
    angles = np.linspace(0.0, np.pi, n_frames, endpoint=False) + np.pi / (2 * n_frames)
    shape, hexagon, sides, crossing, line = run(ellipsoid_points(five_angles), angles)
    for frame in range(n_frames):
        yield shape, hexagon[frame], sides[frame], crossing[frame], line[frame]


# --- plumbing -------------------------------------------------------------------------
def ellipsoid_points(angles: np.ndarray) -> core.Point:
    """Points at the given angles on the ellipsoid."""
    return core.mv.point(centre + semi_axes * np.stack([np.cos(angles), np.sin(angles)], axis=-1))


def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.math.pascal import render

    scenes = list(turn(n_frames))
    save_figure(render.draw(*scenes[still]), "pascal_hexagon")
    save_animation(render.animate(scenes), "pascal_turn", frame_duration)

    # checks
    # The quadric passes through the five points and every sixth; the three crossings lie on one line, the
    # join measured against the size of the crossings, since opposite sides that are nearly parallel meet
    # far out; Pascal's condition on the sixth point is the quadric itself, up to scale.
    five = ellipsoid_points(five_angles)
    shape = scenes[0][0]
    hexagon = stack([scene[1] for scene in scenes])                            # [frames, 6] Point-
    crossing = stack([scene[3] for scene in scenes])                           # [frames, 3] Crossing+
    np.testing.assert_allclose((shape(five) & five).to_array(), 0.0, atol=1e-11)
    np.testing.assert_allclose((shape(hexagon) & hexagon).to_array(), 0.0, atol=1e-11)
    join = (crossing[..., 0] & crossing[..., 1] & crossing[..., 2]).to_array()   # [frames]
    size = np.prod(np.linalg.norm(crossing.kernel, axis=-1), axis=-1)            # [frames]
    np.testing.assert_allclose(join / size, 0.0, atol=1e-10)
    probes = core.mv.point(np.random.default_rng(seed).normal(size=(n_probes, 2)))   # [probes] Point-
    ratio = (core.pascal(five)(probes, probes) / (shape(probes) & probes)).to_array()
    np.testing.assert_allclose(ratio, ratio[0], rtol=1e-11)


if __name__ == "__main__":
    main()
