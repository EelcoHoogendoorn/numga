"""Opposite-spin launches of a uniform hollow top demonstrate chiral spin preference."""
from __future__ import annotations

from itertools import tee

import numpy as np

from numga import stack
from examples.mechanics.tippe_top import core

# Outer shell at the origin, offset tilted cavity breaking through its top, stem, and spherical tip.
shell_semiaxes = np.array([1.05, 0.95, 1.01])
cavity_centre = core.mv.point([-0.05, -0.04, 0.47])                          # [] Point-
cavity_semiaxes = np.array([0.72, 0.65, 0.59])
cavity_angle = 0.16
cavity_tilt = 0.07
stem_centre = core.mv.point([0, 0, 0.21])                                    # [] Point-
stem_semiaxes = np.array([0.18, 0.18, 1.02])
tip_centre = core.mv.point([0, 0, 1.23])                                     # [] Point-
tip_radius = 0.20
sample_spacing = 0.025
sample_lower = np.array([-1.125, -1.125, -1.125])
sample_upper = np.array([1.125, 1.125, 1.525])

cavity_turn = (core.mv.yz * (-cavity_tilt / 2)).exp() * (core.mv.xy * (-cavity_angle / 2)).exp()  # [] Motor+
cavity_axes = cavity_turn >> core.axes                                       # [3] Plane+
shell = core.ellipsoid(core.origin, core.axes, shell_semiaxes)               # [] Plane+ <- Point-
cavity = core.ellipsoid(cavity_centre, cavity_axes, cavity_semiaxes)         # [] Plane+ <- Point-
stem = core.ellipsoid(stem_centre, core.axes, stem_semiaxes)                 # [] Plane+ <- Point-
tip = core.ellipsoid(tip_centre, core.axes, tip_radius * np.ones(3))         # [] Plane+ <- Point-
parts = stack((shell, cavity, stem, tip))                                    # [4] Plane+ <- Point-
contacts = stack((shell, tip))                                               # [2] Plane+ <- Point-

spin_speed = 13.0
spins = spin_speed * np.array([-1, 1])
lean = 0.1
gravity = 1.0
dynamic_friction = 0.35
static_friction = 1.2 * dynamic_friction
step = 0.004
frame_interval = 0.16
n_steps_per_frame = round(frame_interval / step)
run_time = 80.0
n_states = round(run_time / frame_interval) + 1
flip_start, flip_end = 53.0, 65.0
first_frame = round(flip_start / frame_interval)
n_frames = round((flip_end - flip_start) / frame_interval)
frame_duration_ms = 40


# --- plumbing -------------------------------------------------------------------------
def quadrature_grid(spacing: float) -> core.Point:
    """Midpoints of equal-volume boxes `spacing` apart cover the solid's bounding box (`[...] Point-`)."""
    coordinates = [np.arange(low, high, spacing) + spacing / 2
                   for low, high in zip(sample_lower, sample_upper)]
    return core.mv.point(np.stack(np.meshgrid(*coordinates, indexing="ij"), axis=-1).reshape(-1, 3))


def main() -> None:
    from examples.mechanics.tippe_top import render
    from examples.util.animation import save_animation, save_figure

    mass, inertia = core.mass_distribution(parts, quadrature_grid(sample_spacing), sample_spacing**3)  # [] Point-, Forque- <- Rate+
    save_figure(render.setup_figure(parts, mass), "tippe_top_construction")
    motor, rate = core.release_states(contacts, mass, spins, lean)           # [2] Motor+, Rate+
    shown_stream, recorded_stream = tee(core.motion(
        contacts, mass, inertia, motor, rate, gravity,
        static_friction, dynamic_friction, step, n_states, n_steps_per_frame,
    ))
    save_animation(render.frames(parts, mass, first_frame, n_frames, shown_stream),
                   "tippe_top_opposite_spins", frame_duration_ms)
    placements = stack([placement for placement, _ in recorded_stream])      # [n_states, 2] Motor+
    stem_directions = placements >> core.up                                  # [n_states, 2] Point-
    vertical = core.mv.z & stem_directions                                   # [n_states, 2] Scalar+
    times = np.arange(n_states) * frame_interval
    save_figure(render.inclination_figure(times, vertical), "tippe_top_inclination")

    # checks
    assert (core.mv.w & mass) > 0
    assert ((core.mv.xy & inertia(core.mv.yz)) - (core.mv.yz & inertia(core.mv.xy))).abs() < 1e-12
    assert ((placements[-1].norm() - 1).abs() < 1e-12).all()
    assert vertical[-1, 0] < -0.8
    assert vertical[-1, 1] > 0.8


if __name__ == "__main__":
    main()
