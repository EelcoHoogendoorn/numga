"""Scenes for the magnetic bottle: particles launched from its centre at a range of angles to the field,
the steep ones trapped and bouncing between the coils, the shallow ones escaping through the ends."""

from __future__ import annotations

from itertools import islice

import numpy as np

from numga import stack
from examples.electromagnetism.magnetic_bottle import core

mv, time, axis = core.mv, core.time, core.axis
# The coils' radius, their heights along the axis, the straight pieces each is made of, and the current
# in them, which makes the field at the centre two units.
coil_radius = 1.0
coil_heights = np.array([-2.0, 2.0])
n_pieces = 48
coil_current = 1.78
# The particles' speed as a fraction of the speed of light, their angles to the axis in degrees, and
# their distances from the axis at launch, the steepest furthest out.
launch_speed = 0.3
launch_pitches = np.linspace(15, 80, 10)
launch_offsets = np.linspace(0, 0.6, 10)
# The step in the particles' own time, the steps, and the steps between the frames drawn.
proper_step = 0.1
n_steps = 1000
n_steps_per_frame = 12
duration_ms = 60
# The half widths of the view along the axis and across it, and the points across it the field's
# strength is drawn at.
view_half_widths = (3.0, 1.2)
n_grid = 120


# --- math -----------------------------------------------------------------------------
def launch() -> tuple[core.Antivector, core.Antivector]:
    """The particles in the bottle's middle plane, each at its distance from the axis `[particles]`,
    moving at its angle to the axis, tilted from it sideways."""
    direction = (mv.zy * (-np.radians(launch_pitches) / 2)).exp() >> axis           # [particles] Antivector
    return core.across * launch_offsets, (time + launch_speed * direction) / np.sqrt(1 - launch_speed**2)


def flights(pieces: core.Antivector, midpoints: core.Antivector) -> tuple[core.Antivector, core.Antivector]:
    """The particles' events and velocities at every step `[steps, particles]`."""
    history = islice(core.evolve(*launch(), pieces, midpoints, coil_current, proper_step), n_steps)
    events, velocities = zip(*history)
    return stack(list(events)), stack(list(velocities))


def section(pieces: core.Antivector, midpoints: core.Antivector) -> tuple[core.Antivector, core.Scalar]:
    """Points of the plane through the axis, on a grid over the view `[across, along]`, and the field's
    strength at each."""
    along = axis * np.linspace(-view_half_widths[0], view_half_widths[0], 2 * n_grid)                 # [along] Antivector
    points = core.across * np.linspace(-view_half_widths[1], view_half_widths[1], n_grid)[:, None] + along   # [across, along] Antivector
    return points, core.field(points, pieces, midpoints, coil_current).scalar_norm_squared().square_root()


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_animation
    from examples.electromagnetism.magnetic_bottle import render

    pieces, midpoints = core.coils(coil_radius, coil_heights, n_pieces)
    events, velocities = flights(pieces, midpoints)                          # [steps, particles] Antivector
    points, strength = section(pieces, midpoints)
    images = render.animate_flights(events, launch_pitches, points, strength, coil_radius, coil_heights, n_steps_per_frame)
    save_animation(images, "magnetic_bottle", duration_ms)

    # checks
    # The Lorentz force is the dual of the field's turn of the velocity, and the field keeps the speed.
    magnetic = core.field(events, pieces, midpoints, coil_current)                 # [steps, particles] Bivector
    np.testing.assert_allclose((velocities.commutator(magnetic).dual() - (velocities & magnetic)).kernel, 0, atol=1e-10)
    np.testing.assert_allclose((mv.t & velocities).to_array(), 1 / np.sqrt(1 - launch_speed**2), atol=1e-11)
    # The particles inside the loss cone escape through the coils, those outside it stay between them;
    # the trapped ones keep their magnetic moment through their bounces.
    on_axis = core.field(axis * np.linspace(coil_heights[0], coil_heights[-1], 401), pieces, midpoints, coil_current)
    strengths = on_axis.scalar_norm_squared().square_root().to_array()
    loss_cone = np.degrees(np.arcsin(np.sqrt(strengths.min() / strengths.max())))
    heights = (mv.z & events[-1]).to_array()                                  # [particles]
    np.testing.assert_array_equal(np.abs(heights) > coil_heights[-1], launch_pitches < loss_cone)
    # The moment is read with the field at the particle, which swings within a turn off the axis; the
    # invariant is its mean over a turn, the steps the velocity takes to turn once in the central field.
    moments = core.moment(velocities, magnetic).to_array()[:, launch_pitches > loss_cone]   # [steps, trapped]
    n_turn = int(2 * np.pi / (strengths.min() * proper_step))
    means = np.apply_along_axis(np.convolve, 0, moments, np.ones(n_turn) / n_turn, "valid")
    np.testing.assert_allclose(means / means[0], 1, rtol=0.2)


if __name__ == "__main__":
    main()
