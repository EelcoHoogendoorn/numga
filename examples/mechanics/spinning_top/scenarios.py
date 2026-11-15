"""Four variations of a three-part ellipsoid top spin and precess on flat ground."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace

import numpy as np

from numga import stack
from examples.mechanics.spinning_top import core

mv = core.mv
gravity = 9.81
static_friction = 0.4
# Semi-axes, relative heights, and masses of the disc, stem, and rounded tip.
disc_semi_axes = np.array([0.25, 0.25, 0.04])
stem_semi_axes = np.array([0.02, 0.02, 0.15])
stem_above_disc = 0.16
stem_mass = 0.05
tip_height = 0.14
tip_mass = 0.15
# Initial release tilt and vertical clearance above the ground.
tilt = 0.15
clearance = 0.002
# Simulation horizon, integration time step, substeps per rendered frame, pixel size, and frame duration.
runtime = 5.0
dt = 4e-3
n_steps_per_frame = 12
n_pixels = 260
frame_duration_ms = 40
view, view_centre, view_extent = (30, -60), core.origin + core.up * 0.15, 0.55


@dataclass(frozen=True)
class Setup:
    """Parameters specify one spinning top and the ground it meets."""
    spin_rate: float
    tip_radius: float
    disc_height: float
    disc_mass: float
    ground_curvature: float
    dynamic_friction: float
    drag: float


base_setup = Setup(18.0, 0.05, 0.10, 1.0, 0.0, 0.3, 0.13)
variations = {
    "base": base_setup,
    "sharp_tip": replace(base_setup, tip_radius=0.02),
    "heavy_disc": replace(base_setup, disc_mass=2.0),
    "slippery": replace(base_setup, dynamic_friction=0.1),
}

# Shared four-top batch arrays for the side-by-side notebook comparison.
part_heights = np.array([base_setup.disc_height, base_setup.disc_height + stem_above_disc, tip_height])
tip_radii = np.array([setup.tip_radius for setup in variations.values()])
disc_masses = np.array([setup.disc_mass for setup in variations.values()])
dynamic_frictions = np.array([setup.dynamic_friction for setup in variations.values()])
ground_curvature = base_setup.ground_curvature
spin_rate = base_setup.spin_rate
drag = base_setup.drag
n_tops = len(variations)
n_frames = round(runtime / dt) // n_steps_per_frame


# --- math -----------------------------------------------------------------------------
def launch_top(setup: Setup, n_launch_frames: int, step_dt: float, substeps: int
               ) -> tuple[core.Quadric, core.Quadric, Iterator[tuple[core.Motor, core.Rate]]]:
    """Each `setup` assembles the three ellipsoid parts, the ground quadric, and the streamed `(motor, rate)` trajectory."""
    heights = np.array([setup.disc_height, setup.disc_height + stem_above_disc, tip_height])
    semi_axes = np.stack([disc_semi_axes, stem_semi_axes, np.array([setup.tip_radius, setup.tip_radius, tip_height])])
    masses = np.array([setup.disc_mass, stem_mass, tip_mass])
    parts, inertia, centre_height = core.top_body(heights, semi_axes, masses)  # [3] Plane+ <- Point-, [] Forque- <- Rate+
    ground = core.bowl(setup.ground_curvature)                                 # [] Plane+ <- Point-
    # Lifting by `centre_height * cos(tilt) + clearance` places the tilted tip clear of the ground.
    lift = (mv.zw * ((centre_height * np.cos(tilt) + clearance) / 2)).exp()    # [] Rotor+
    lean_turn = (mv.zx * (tilt / 2)).exp()                                     # [] Rotor+
    start_motor = (lift * lean_turn).normalized()                              # [] Rotor+
    start_rate = mv.xy * setup.spin_rate                                       # [] Rate+
    states = core.spin(start_motor, start_rate, parts, inertia, np.asarray(masses.sum()), ground,
                       gravity, setup.drag, static_friction, np.asarray(setup.dynamic_friction),
                       step_dt, n_launch_frames, substeps)
    return parts, ground, states


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    """Streams and saves each variation's animation to `plots/` and verifies upright precession."""
    from examples.mechanics.spinning_top import render
    from examples.util.animation import save_animation

    runs: dict[str, list[tuple[core.Motor, core.Rate]]] = {}
    for name, setup in variations.items():
        parts, ground, state_stream = launch_top(setup, n_frames, dt, n_steps_per_frame)
        recorded: list[tuple[core.Motor, core.Rate]] = []

        def frames() -> Iterator[np.ndarray]:
            for motor, rate in state_stream:
                recorded.append((motor, rate))
                yield render.frame(motor, parts, ground, view, view_centre, view_extent, n_pixels)

        save_animation(frames(), f"spinning_top_{name}", frame_duration_ms)
        runs[name] = recorded

    # checks
    # Every variation stays upright below 45 degrees and spins in the same sense while air drag slows it.
    for states in runs.values():
        motors = stack([motor for motor, _ in states])                         # [n_frames] Rotor+
        rates = stack([rate for _, rate in states])                            # [n_frames] Rate+
        leans = core.lean_angle(motors)                                        # [n_frames] Scalar+
        spins = -(mv.xy | rates)                                               # [n_frames] Scalar+
        assert (leans < np.pi / 4).all()
        assert (spins > 0).all() and (spins[-1] < spins[0]).all()


if __name__ == "__main__":
    main()
