"""Scenes for the four-line problem: three lines on the hyperboloid x² + y² = 1 + z², and a steep
fourth line swinging across it, through its waist and out the other side. Outside, the fourth line
crosses the hyperboloid twice and both transversals are real; through the waist it misses the
hyperboloid and the transversals are a complex-conjugate pair."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import stack
from examples.math.klein_quadric import core

# Where the three lines cross the hyperboloid's waist, by angle about z.
WAIST_ANGLES = np.array([0.3, 2.3, 4.4])
# The fourth line: where it crosses the plane z = 0 at rest, the direction it runs in, and how far it
# swings along x either way.
FOURTH_CROSSING = [0.0, 0.4, 0.0]
FOURTH_DIRECTION = [0.3, -0.2, 1.0]
SWING = 1.5
# The lines drawn of each family on the hyperboloid.
N_RULINGS = 36
# The frames of the swing, the frame shown as a still, and the duration of a frame in milliseconds.
N_FRAMES = 96
STILL = 8
FRAME_DURATION = 70


# --- math -----------------------------------------------------------------------------
def scene(three: core.Bivector, fourth: core.Bivector, n_rulings: int) -> tuple[core.Bivector, core.Bivector, core.Bivector, core.Point]:
    """Three lines and a batch of fourth lines: the four lines, both families of lines on the
    quadric of the three, the two transversals of all four and the two points where the fourth line
    crosses the quadric."""
    first, second, third = three[0], three[1], three[2]
    lines = stack([first.broadcast_to(fourth.shape), second.broadcast_to(fourth.shape),
                   third.broadcast_to(fourth.shape), fourth], axis=-1)         # [..., 4] Bivector
    across = core.transversals(lines)                                          # [..., 2] Bivector
    surface = core.ruled_quadric(first, second, third)                         # [] Plane <- Point
    crossing = core.crossings(surface, fourth)                                 # [..., 2] Point
    # The lines meeting the three, through points spread over the first; and the lines meeting
    # three of those, the family of the three.
    sweep = core.through(core.spread(first, n_rulings), second, third)         # [rulings] Bivector
    family = core.through(core.spread(sweep[0], n_rulings), sweep[n_rulings // 3], sweep[2 * n_rulings // 3])   # [rulings] Bivector
    return lines, stack([sweep, family]), across, crossing                     # [..., 4], [families, rulings], [..., 2], [..., 2]


def swing(n_frames: int, n_rulings: int) -> Iterator[tuple[core.Bivector, core.Bivector, core.Bivector, core.Point]]:
    """The fourth line swung once across the hyperboloid and back: per frame the four lines, both
    families of lines on the hyperboloid, the transversals and the crossings."""
    offsets = SWING * np.cos(np.linspace(0.0, 2 * np.pi, n_frames, endpoint=False))
    lines, rulings, across, crossing = scene(waist_lines(WAIST_ANGLES), steep(offsets), n_rulings)
    for frame in range(n_frames):
        yield lines[frame], rulings, across[frame], crossing[frame]


# --- plumbing -------------------------------------------------------------------------
def waist_lines(angles: np.ndarray) -> core.Bivector:
    """Lines of the hyperboloid x² + y² = 1 + z² through the points of its waist at the given angles,
    all leaning the same way."""
    # The line through (1, 0, 0) leaning along (0, 1, 1), the meet of the planes x = 1 and y = z, turned
    # about z by each angle.
    line = (core.mv.x - core.mv.w) ^ (core.mv.y - core.mv.z)
    return (core.mv.xy * (-angles / 2)).exp() >> line


def steep(offsets: np.ndarray) -> core.Bivector:
    """The fourth line, moved along x by each offset: the meet of the two planes through it along x
    and along y."""
    base = (core.mv.xw * (offsets / 2)).exp() >> core.mv.point(FOURTH_CROSSING)
    along = base & core.mv.direction(FOURTH_DIRECTION)
    return (along & core.mv.direction([1.0, 0.0, 0.0])) ^ (along & core.mv.direction([0.0, 1.0, 0.0]))


def main() -> None:
    from examples.animation import save_animation, save_figure
    from examples.math.klein_quadric import render

    scenes = list(swing(N_FRAMES, N_RULINGS))
    save_figure(render.draw(*scenes[STILL], azimuth=render.AZIMUTH), "klein_transversals")
    save_animation(render.animate(scenes), "klein_swing", FRAME_DURATION)

    # checks
    # The transversals are lines and meet all four lines; the crossings lie on the fourth line and on
    # the quadric, each on one of the transversals; both families lie on the quadric.
    three = waist_lines(WAIST_ANGLES)
    lines, rulings, across, crossing = scene(three, steep(SWING * np.cos(np.linspace(0.0, 2 * np.pi, N_FRAMES, endpoint=False))), N_RULINGS)
    surface = core.ruled_quadric(three[0], three[1], three[2])
    np.testing.assert_allclose((across ^ across).kernel, 0.0, atol=1e-12)
    np.testing.assert_allclose((lines[..., :, None] ^ across[..., None, :]).kernel, 0.0, atol=1e-12)
    np.testing.assert_allclose((crossing & lines[..., 3:]).kernel, 0.0, atol=1e-10)
    np.testing.assert_allclose((surface(crossing) & crossing).kernel, 0.0, atol=1e-10)
    off = np.abs((crossing[..., :, None] & across[..., None, :]).kernel).max(axis=-1)   # [frames, 2, 2]
    np.testing.assert_allclose(off.min(axis=-1), 0.0, atol=1e-10)
    ends = core.spread(rulings, 3)                                             # [families, rulings, 3] Point
    np.testing.assert_allclose((surface(ends) & ends).kernel, 0.0, atol=1e-10)


if __name__ == "__main__":
    main()
