"""Scenes for the four-line problem: three lines on the hyperboloid x² + y² = 1 + z², and a steep
fourth line swinging across it, through its waist and out the other side. Outside, the fourth line
crosses the hyperboloid twice and both transversals are real; through the waist it misses the
hyperboloid and the transversals are a complex-conjugate pair."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import stack
from examples.math.klein_quadric import core


# --- math -----------------------------------------------------------------------------
def scene(three: core.Bivector, fourth: core.Bivector, count: int) -> tuple[core.Bivector, core.Bivector, core.Bivector, core.Point]:
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
    sweep = core.through(core.spread(first, count), second, third)             # [count] Bivector
    family = core.through(core.spread(sweep[0], count), sweep[count // 3], sweep[2 * count // 3])   # [count] Bivector
    rulings = stack([sweep, family])                                           # [families, count] Bivector

    # --- checks
    # The transversals are lines and meet all four lines; the crossings lie on the fourth line and
    # on the quadric, each on one of the transversals; both families lie on the quadric.
    np.testing.assert_allclose((across ^ across).kernel, 0.0, atol=1e-12)
    np.testing.assert_allclose((lines[..., :, None] ^ across[..., None, :]).kernel, 0.0, atol=1e-12)
    np.testing.assert_allclose((crossing & fourth[..., None]).kernel, 0.0, atol=1e-10)
    np.testing.assert_allclose((surface(crossing) & crossing).kernel, 0.0, atol=1e-10)
    off = np.abs((crossing[..., :, None] & across[..., None, :]).kernel).max(axis=-1)   # [..., 2, 2]
    np.testing.assert_allclose(off.min(axis=-1), 0.0, atol=1e-10)
    ends = core.spread(rulings, 3)                                             # [families, count, 3] Point
    np.testing.assert_allclose((surface(ends) & ends).kernel, 0.0, atol=1e-10)
    return lines, rulings, across, crossing


def swing(frames: int, count: int) -> Iterator[tuple[core.Bivector, core.Bivector, core.Bivector, core.Point]]:
    """The fourth line swung once across the hyperboloid and back: per frame the four lines, both
    families of lines on the hyperboloid, the transversals and the crossings."""
    offsets = 1.5 * np.cos(np.linspace(0.0, 2 * np.pi, frames, endpoint=False))
    lines, rulings, across, crossing = scene(waist_lines(np.array([0.3, 2.3, 4.4])), steep(offsets), count)
    for frame in range(frames):
        yield lines[frame], rulings, across[frame], crossing[frame]


# --- plumbing -------------------------------------------------------------------------
def waist_lines(angles: np.ndarray) -> core.Bivector:
    """Lines of the hyperboloid x² + y² = 1 + z² through the points of its waist at the given angles,
    all leaning the same way."""
    waist = np.stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)], axis=-1)
    lean = np.stack([-np.sin(angles), np.cos(angles), np.ones_like(angles)], axis=-1)
    return core.point(waist) & core.point(waist + lean)


def steep(offsets: np.ndarray) -> core.Bivector:
    """Steep lines crossing the plane z = 0 at the given offsets along x, beside the x axis."""
    base = np.stack([offsets, np.full_like(offsets, 0.4), np.zeros_like(offsets)], axis=-1)
    return core.point(base) & core.point(base + np.array([0.3, -0.2, 1.0]))


if __name__ == "__main__":
    from examples.animation import save_animation, save_figure
    from examples.math.klein_quadric import render

    scenes = list(swing(96, 36))
    save_figure(render.draw(*scenes[8], azimuth=-60.0), "klein_transversals")
    save_animation(render.animate(scenes), "klein_swing", 70)
