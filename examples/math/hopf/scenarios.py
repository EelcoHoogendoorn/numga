"""Scenes for the Hopf fibration: the fibres over circles of directions, nested tori in space; a
spinor carried around a loop of directions, back onto its own fibre; and fibres added one by one as a
direction spirals over the sphere."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from examples.math.hopf import core

mv = core.mv


# --- math -----------------------------------------------------------------------------
def tori(polars: np.ndarray, per_circle: int, samples: int) -> tuple[core.Vector, core.Vector]:
    """The fibres over directions spaced around circles of latitude, projected into space. Returns the
    directions and the projected fibres."""
    azimuths = np.linspace(0.0, 2 * np.pi, per_circle, endpoint=False)
    directions = sphere(polars[:, None], azimuths[None, :])                   # [circles, per_circle] Vector
    fibres = core.fibre(core.fibre_start(directions), np.linspace(0.0, 2 * np.pi, samples + 1))   # [circles, per_circle, samples + 1] Even
    projected = core.stereographic(fibres)                                     # [circles, per_circle, samples + 1] Vector

    # --- checks
    # Every spinor on a fibre points along the fibre's direction, and any two fibres are linked once.
    np.testing.assert_allclose((core.HOPF(fibres, fibres) - directions[..., None]).kernel, 0.0, atol=1e-9)
    np.testing.assert_allclose(core.linking(projected[0, 0], projected[-1, per_circle // 3]).to_array(), 1.0, atol=1e-2)
    return directions, projected


def lift(polar: float, count: int) -> tuple[core.Vector, core.Vector, core.Vector]:
    """A spinor carried once around the circle of directions at the given polar angle, by the smallest
    rotation from each direction to the next. Returns the loop of directions, the carried spinor's
    projected path, and the projected fibre it starts on."""
    angles = np.linspace(0.0, 2 * np.pi, count + 1)
    loop = sphere(np.full(count + 1, polar), angles)                           # [count + 1] Vector
    start = core.fibre_start(loop[0])                                          # [] Even
    carried = core.transport(loop) * start                                     # [count] Even

    # --- checks
    # The spinor comes back on its own fibre, turned along it by half the solid angle the loop encloses.
    half_solid_angle = np.pi * (1 - np.cos(polar))
    turned = start * (mv.xy * -half_solid_angle).exp()                         # [] Even
    np.testing.assert_allclose((carried[-1] - turned).kernel, 0.0, atol=1e-4)
    path = core.stereographic(carried)                                         # [count] Vector
    return loop, path, core.stereographic(core.fibre(start, angles))


def sweep(frames: int, samples: int) -> Iterator[tuple[core.Vector, core.Vector]]:
    """Directions spiralling from near the south pole to just above the equator, each with its
    projected fibre."""
    for fraction in np.linspace(0.0, 1.0, frames):
        direction = sphere(np.array(np.pi * (0.95 - 0.5 * fraction)), np.array(2 * np.pi * 4 * fraction))   # [] Vector
        yield direction, core.stereographic(core.fibre(core.fibre_start(direction), np.linspace(0.0, 2 * np.pi, samples + 1)))


# --- plumbing -------------------------------------------------------------------------
def sphere(polar: np.ndarray, azimuth: np.ndarray) -> core.Vector:
    """Unit directions at the given polar angles from z and azimuths about it: z turned toward x by
    the polar angle, then about z by the azimuth."""
    polar, azimuth = np.broadcast_arrays(polar, azimuth)
    return (mv.xy * (-azimuth / 2)).exp() * (mv.zx * (-polar / 2)).exp() >> mv.z


if __name__ == "__main__":
    from examples.animation import save_animation, save_figure
    from examples.math.hopf import render

    save_figure(render.draw_tori(*tori(np.pi * np.array([0.9, 0.7, 0.5]), 18, 240)), "hopf_tori")
    save_figure(render.draw_lift(*lift(2.2, 400)), "hopf_lift")
    save_animation(render.animate_sweep(sweep(90, 240)), "hopf_sweep", 60)
