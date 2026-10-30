"""A uniform hollow top, released with equal and opposite spins."""
from __future__ import annotations

from itertools import tee

import numpy as np

from numga import stack
from examples.mechanics.tippe_top import core

# The outer shell about the origin, the cavity carved from it, open at the top, the stem and the ball at its end.
shell_semiaxes = np.array([1.05, 0.95, 1.01])
cavity_centre = core.mv.point([-0.05, -0.04, 0.47])
cavity_semiaxes = np.array([0.72, 0.65, 0.59])
cavity_angle = 0.16
cavity_tilt = 0.07
stem_centre = core.mv.point([0, 0, 0.21])
stem_semiaxes = np.array([0.18, 0.18, 1.02])
tip_centre = core.mv.point([0, 0, 1.23])
tip_radius = 0.20
sample_spacing = 0.025
sample_lower = np.array([-1.125, -1.125, -1.125])
sample_upper = np.array([1.125, 1.125, 1.525])
spin_speed = 13.0
spins = spin_speed * np.array([-1., 1.])
lean = 0.1
gravity = 1.0
dynamic_friction = 0.35
static_friction = 1.2 * dynamic_friction
step = 0.004
# Frames close enough that the spin shows, over the run and, slowed down, over the left top's flip: briefly upright,
# then over, and at rest on its tip.
frame_interval = 0.16
n_steps_per_frame = round(frame_interval / step)
run_time = 80.0
n_states = round(run_time / frame_interval) + 1
flip_start, flip_end = 53.0, 65.0
first_frame = round(flip_start / frame_interval)
n_frames = round((flip_end - flip_start) / frame_interval)
frame_duration_ms = 40


# --- plumbing -------------------------------------------------------------------------
def quadrature_grid() -> core.Point:
    """Midpoints of equal-volume boxes covering the solid."""
    coordinates = [np.arange(low, high, sample_spacing) + sample_spacing / 2
                   for low, high in zip(sample_lower, sample_upper)]
    return core.mv.point(np.stack(np.meshgrid(*coordinates, indexing="ij"), axis=-1).reshape(-1, 3))


# --- math -----------------------------------------------------------------------------
def body() -> core.Body:
    """The four quadrics and the uniform material they enclose."""
    cavity_turn = (core.mv.yz * (-cavity_tilt / 2)).exp() * (core.mv.xy * (-cavity_angle / 2)).exp()
    shell = core.ellipsoid(core.origin, core.axes, shell_semiaxes)
    cavity = core.ellipsoid(cavity_centre, cavity_turn >> core.axes, cavity_semiaxes)
    stem = core.ellipsoid(stem_centre, core.axes, stem_semiaxes)
    tip = core.ellipsoid(tip_centre, core.axes, tip_radius * np.ones(3))
    return core.build(shell, cavity, stem, tip, quadrature_grid(), sample_spacing**3)


def main() -> None:
    from examples.mechanics.tippe_top import render
    from examples.util.animation import save_animation, save_figure

    top = body()
    save_figure(render.draw_setup(top), "tippe_top_construction")
    motor, rate = core.launch(top, spins, lean)
    # One stream of states: the flip is drawn as its states come, and the whole run then gives the history.
    shown, recorded = tee(core.evolve(top, motor, rate, gravity, static_friction, dynamic_friction,
                                      step, n_states, n_steps_per_frame))
    placements = (placement for placement, _ in shown)
    save_animation(render.animate(top, placements, first_frame, n_frames), "tippe_top_opposite_spins", frame_duration_ms)
    vertical = stack([core.mv.z & (placement >> core.up) for placement, _ in recorded])   # [states, spins] Scalar+
    times = np.arange(n_states) * frame_interval
    save_figure(render.history(times, vertical), "tippe_top_inclination")

    # checks
    assert vertical[-1, 0] < -0.8
    assert vertical[-1, 1] > 0.8


if __name__ == "__main__":
    main()
