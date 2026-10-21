"""Scenes for the rattleback: released tipped from rest it rocks into spin; spun the unfavourable way, it
rocks, stops and spins the other way."""

from __future__ import annotations

from itertools import islice

import numpy as np

from numga import stack
from numga.algebras import PGA3D
from examples import instantiate

core = instantiate("examples.mechanics.rattleback.core", PGA3D)
mv = core.mv
# The shell's semi-axes, the body's mass, gravity, and the turn of the masses' axes against the shell's.
SEMI_AXES = np.array([5, 17 / 16, 1])
MASS = 4
GRAVITY = 9.8
MASS_ANGLE = np.deg2rad(9)
# The two launches: tipped from rest, and spinning the unfavourable way with a little pitching.
TILTS = np.deg2rad([15, 0])
SPINS = np.array([0, -1.5])
PITCH_RATES = np.array([0, 0.1])
# The time step and the steps, the steps between frames drawn, the time the rocking animation runs, and
# the share of the shell's height its flat deck sits at.
TIME_STEP = 0.04
N_STEPS = 250
FRAME_STRIDE = 2
ROCKING_DURATION = 8
DECK_FRACTION = 0.8


# --- math -----------------------------------------------------------------------------
def body() -> tuple[core.Stretch, core.Shape, core.Point, core.Inertia]:
    """The shell's stretch of the unit sphere and its support map, and the six masses and their inertia."""
    mirrors = stack((mv.x, mv.y, mv.z))                                      # [3] Plane
    directions = mirrors.dual()                                              # [3] Direction
    outline = (directions * (mirrors & core.Direction) * SEMI_AXES).sum(axis=0)       # Direction <- Direction
    shape = (directions * (core.Plane & directions) * SEMI_AXES**2).sum(axis=0)       # Direction <- Plane
    # Six equal masses have the second moment of a uniform ellipsoid, their axes turned within the shell.
    mass_axes = (mv.xy * (-MASS_ANGLE / 2)).exp() >> directions             # [3] Direction
    offsets = ((mass_axes * SEMI_AXES)[:, None] * np.array([-1, 1]) * np.sqrt(3 / 5)).reshape(-1)   # [masses] Direction
    mass_points = core.origin + offsets                                      # [masses] Point
    inertia = (mass_points & mass_points.commutator(core.Rate)).mean(axis=-1) * MASS   # Forque <- Rate
    return outline, shape, mass_points, inertia


def launches(dt: float, n_steps: int) -> tuple[core.Stretch, core.Point, core.Motor, core.Spin, core.Scalar]:
    """Both launches from their start over the steps `[steps + 1, cases]`: the shell's stretch, the masses,
    the placements, the body-frame spins and the energies."""
    outline, shape, mass_points, inertia = body()
    up = mv.z.dual()                                                         # [] Direction
    gravity = -up * GRAVITY                                                  # [] Direction
    spin = mv.xy * SPINS - mv.xz * PITCH_RATES                               # [cases] Spin
    tilt = (-mv.xz * TILTS / 2).exp()                                        # [cases] Motor
    # Each launch lifted by its centre's height, so that the shell rests on the table.
    normal = (tilt << up).dual_inverse()                                     # [cases] Plane
    height = (normal & shape(normal)).square_root()                          # [cases] Scalar
    motor = (up * core.origin.inverse() * height / 2).exp() * tilt           # [cases] Motor

    def motion(motor: core.Motor, spin: core.Spin) -> tuple[core.Rate, core.Spin]:
        return core.dynamics(motor, spin, shape, inertia, MASS, gravity, up)

    states = islice(core.evolve(motor, spin, dt, motion), n_steps + 1)
    placements, spins = (stack(list(history)) for history in zip(*states))   # [steps + 1, cases] each
    energies = core.energy(placements, spins, shape, inertia, MASS, gravity, up)   # [steps + 1, cases] Scalar
    return outline, mass_points, placements, spins, energies


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.animation import save_animation, save_figure
    from examples.mechanics.rattleback import render

    duration_ms = round(TIME_STEP * FRAME_STRIDE * 1000)
    outline, mass_points, placements, spins, energies = launches(TIME_STEP, N_STEPS)
    rocking_frames = slice(0, round(ROCKING_DURATION / TIME_STEP) + 1, FRAME_STRIDE)
    reversal_frames = slice(None, None, FRAME_STRIDE)
    save_figure(render.draw_setup(outline, mass_points, DECK_FRACTION), "rattleback_mass_axes")
    save_animation(render.frames(placements[rocking_frames, 0], outline, DECK_FRACTION), "rattleback_rocking",
                   duration_ms)
    save_animation(render.frames(placements[reversal_frames, 1], outline, DECK_FRACTION), "rattleback_reversal",
                   duration_ms)

    # checks
    # The tipped body spins up from rest, the spinning one reverses, and the energy is kept.
    spun = (-(placements >> spins).scalar_product(mv.xy)).to_array()          # [steps + 1, cases]
    assert abs(spun[0, 0]) < 1e-11 and spun[-1, 0] > 1.4
    assert spun[0, 1] < -1.4 and spun[-1, 1] > 1.4
    np.testing.assert_allclose((energies / energies[0]).to_array(), 1, rtol=1e-2)


if __name__ == "__main__":
    main()
