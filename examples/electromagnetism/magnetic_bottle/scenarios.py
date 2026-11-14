"""Scenario parameters launch relativistic particles across a fan of pitch angles in the magnetic bottle,
separating escaping trajectories inside the loss cone from trapped trajectories that conserve their
turn-averaged magnetic moment."""

from __future__ import annotations

import numpy as np

from examples.electromagnetism.magnetic_bottle import core
from numga import stack

mv, time, x_axis, z_axis = core.mv, core.time, core.x_axis, core.z_axis

coil_radius = 1.0
coil_heights = np.array([-2.0, 2.0])
n_pieces = 48
coil_current = 1.78

launch_speed = 0.3
n_particles = 10
launch_pitches = np.linspace(15, 80, n_particles)
launch_offsets = np.linspace(0, 0.6, n_particles)

proper_step = 0.1
n_steps = 1000
n_steps_per_frame = 8
frame_duration_ms = 40
n_turn = 31

view_half_length = 3.0
view_half_width = 1.2
n_grid = 120
n_axis = 401


# --- math -----------------------------------------------------------------------------
def launch() -> tuple[core.Antivector, core.Antivector]:
    """Initial midplane events and unit four-velocities (`[n_particles] Antivector-`) fan out in radial
    offset and pitch angle."""
    tilts = (mv.zy * (-np.radians(launch_pitches) / 2)).exp()                 # [n_particles] Rotor+
    directions = tilts >> z_axis                                              # [n_particles] Antivector-
    velocities = (time + launch_speed * directions) / np.sqrt(1 - launch_speed**2)   # [n_particles] Antivector-
    return x_axis * launch_offsets, velocities


def section(
    currents: core.Antivector, midpoints: core.Antivector, n_grid: int,
) -> tuple[core.Antivector, core.Scalar]:
    """Meridional grid points (`[n_grid, 2 * n_grid] Antivector-`) and magnetic field strength
    (`[n_grid, 2 * n_grid] Scalar+`) span the viewing plane through the bottle axis."""
    axial_points = z_axis * np.linspace(-view_half_length, view_half_length, 2 * n_grid)   # [2 * n_grid] Antivector-
    radial_offsets = x_axis * np.linspace(-view_half_width, view_half_width, n_grid)[:, None]   # [n_grid, 1] Antivector-
    points = radial_offsets + axial_points                                    # [n_grid, 2 * n_grid] Antivector-
    return points, core.field(points, currents, midpoints).norm()


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.electromagnetism.magnetic_bottle import render
    from examples.util.animation import save_animation

    currents, midpoints = core.coils(coil_radius, coil_heights, coil_current, n_pieces)
    axis_samples = z_axis * np.linspace(coil_heights[0], coil_heights[-1], n_axis)   # [n_axis] Antivector-
    axis_strengths = core.field(axis_samples, currents, midpoints).norm()     # [n_axis] Scalar+
    loss_cone = (axis_strengths.min() / axis_strengths.max()).square_root().arcsin()   # [] Scalar+

    points, strength = section(currents, midpoints, n_grid)
    recorded_states: list[tuple[core.Antivector, core.Antivector]] = []
    states = (
        recorded_states.append(state) or state
        for state in core.evolve(*launch(), currents, midpoints, proper_step, n_steps)
    )
    images = render.animate_flights(
        launch_pitches, points, strength, coil_radius, coil_heights, n_steps_per_frame, states,
    )
    save_animation(images, "magnetic_bottle", frame_duration_ms)

    events, velocities = (stack(items) for items in zip(*recorded_states))    # [n_steps, n_particles] Antivector-, Antivector-
    magnetic = core.field(events, currents, midpoints)                        # [n_steps, n_particles] Field+
    moments = core.moment(velocities, magnetic)                               # [n_steps, n_particles] Scalar+
    cumulative_moments = moments.cumsum(0)                                    # [n_steps, n_particles] Scalar+
    means = (cumulative_moments[n_turn:] - cumulative_moments[:-n_turn]) / n_turn   # [n_steps - n_turn, n_particles] Scalar+
    moment_drift = (means / means[0] - 1).abs().max(axis=0)                   # [n_particles] Scalar+
    escape_mask = (mv.z & events[-1]).abs() > coil_heights[-1]

    # checks
    lorentz_factor = 1 / np.sqrt(1 - launch_speed**2)
    assert ((velocities.commutator(magnetic).dual() - (magnetic & velocities)).norm() < 1e-8).all()
    assert (((mv.t & velocities) - lorentz_factor).abs() < 1e-8).all()
    assert np.array_equal(escape_mask, np.radians(launch_pitches) < loss_cone)
    assert (moment_drift[~escape_mask] < 0.2).all()
    assert (moment_drift[escape_mask] > 1.0).all()


if __name__ == "__main__":
    main()
