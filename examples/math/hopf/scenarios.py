"""Scenes for the Hopf fibration: the fibres over circles of directions, nested tori in space; a
spinor carried around a loop of directions, back onto its own fibre; and fibres added one by one as a
direction spirals over the sphere."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from examples.math.hopf import core

mv = core.mv

# The polar angles of the circles of directions whose fibres fill the tori, the directions on each circle,
# and the samples of each fibre.
POLARS = np.pi * np.array([0.9, 0.7, 0.5])
N_PER_CIRCLE = 18
N_SAMPLES = 240
# The polar angle of the loop a spinor is carried around, and the steps around it.
LOOP_POLAR = 2.2
N_STEPS = 400
# The spiral of directions: its polar angles at the start and the end, the turns it makes about the pole,
# the frames along it, and the duration of each frame.
SPIRAL_POLARS = np.pi * np.array([0.95, 0.45])
N_TURNS = 4
N_FRAMES = 90
DURATION_MS = 60


# --- math -----------------------------------------------------------------------------
def tori(polars: np.ndarray, n_per_circle: int, n_samples: int) -> tuple[core.Vector, core.Vector]:
    """The fibres over directions spaced around circles of latitude, projected into space. Returns the
    directions and the projected fibres."""
    azimuths = np.linspace(0.0, 2 * np.pi, n_per_circle, endpoint=False)
    directions = sphere(polars[:, None], azimuths[None, :])                   # [circles, per_circle] Vector
    fibres = core.fibre(core.fibre_start(directions), np.linspace(0.0, 2 * np.pi, n_samples + 1))   # [circles, per_circle, samples + 1] Even
    return directions, core.stereographic(fibres)


def lift(polar: float, n_steps: int) -> tuple[core.Vector, core.Even, core.Even]:
    """A spinor carried once around the circle of directions at the given polar angle, by the smallest
    rotation from each direction to the next. Returns the loop of directions, the carried spinor, and the
    fibre it starts on."""
    angles = np.linspace(0.0, 2 * np.pi, n_steps + 1)
    loop = sphere(np.full(n_steps + 1, polar), angles)                         # [steps + 1] Vector
    start = core.fibre_start(loop[0])                                          # [] Even
    carried = core.transport(loop) * start                                     # [steps] Even
    return loop, carried, core.fibre(start, angles)


def sweep(polars: np.ndarray, n_turns: int, n_frames: int, n_samples: int) -> Iterator[tuple[core.Vector, core.Vector]]:
    """Directions spiralling from the first polar angle to the second, each with its projected fibre."""
    for fraction in np.linspace(0.0, 1.0, n_frames):
        direction = sphere(polars[0] + (polars[1] - polars[0]) * fraction, 2 * np.pi * n_turns * fraction)   # [] Vector
        yield direction, core.stereographic(core.fibre(core.fibre_start(direction), np.linspace(0.0, 2 * np.pi, n_samples + 1)))


# --- plumbing -------------------------------------------------------------------------
def sphere(polar: np.ndarray, azimuth: np.ndarray) -> core.Vector:
    """Unit directions at the given polar angles from z and azimuths about it: z turned toward x by
    the polar angle, then about z by the azimuth."""
    polar, azimuth = np.broadcast_arrays(polar, azimuth)
    return (mv.xy * (-azimuth / 2)).exp() * (mv.zx * (-polar / 2)).exp() >> mv.z


def main() -> None:
    from examples.animation import save_animation, save_figure
    from examples.math.hopf import render

    directions, projected = tori(POLARS, N_PER_CIRCLE, N_SAMPLES)
    save_figure(render.draw_tori(directions, projected), "hopf_tori")
    loop, carried, start_fibre = lift(LOOP_POLAR, N_STEPS)
    save_figure(render.draw_lift(loop, core.stereographic(carried), core.stereographic(start_fibre)), "hopf_lift")
    save_animation(render.animate_sweep(sweep(SPIRAL_POLARS, N_TURNS, N_FRAMES, N_SAMPLES)), "hopf_sweep", DURATION_MS)

    # checks
    # Every spinor on a fibre points along the fibre's direction, and any two fibres are linked once.
    fibres = core.fibre(core.fibre_start(directions), np.linspace(0.0, 2 * np.pi, N_SAMPLES + 1))
    np.testing.assert_allclose((core.HOPF(fibres, fibres) - directions[..., None]).kernel, 0.0, atol=1e-9)
    np.testing.assert_allclose(core.linking(projected[0, 0], projected[-1, N_PER_CIRCLE // 3]).to_array(), 1.0, atol=1e-2)
    # The carried spinor comes back on its own fibre, turned along it by half the solid angle the loop encloses.
    half_solid_angle = np.pi * (1 - np.cos(LOOP_POLAR))
    np.testing.assert_allclose((carried[-1] - start_fibre[0] * (mv.xy * -half_solid_angle).exp()).kernel, 0.0, atol=1e-4)


if __name__ == "__main__":
    main()
