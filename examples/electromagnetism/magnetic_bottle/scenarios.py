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
RADIUS = 1.0
HEIGHTS = np.array([-2.0, 2.0])
N_PIECES = 48
CURRENT = 1.78
# The particles' speed as a fraction of the speed of light, their angles to the axis in degrees, and
# their distances from the axis at launch, the steepest furthest out.
SPEED = 0.3
PITCHES = np.linspace(15, 80, 10)
OFFSETS = np.linspace(0, 0.6, 10)
# The step in the particles' own time, the steps, and the steps between the frames drawn.
PROPER_STEP = 0.1
N_STEPS = 1000
N_STEPS_PER_FRAME = 12
DURATION_MS = 60
# The half widths of the view along the axis and across it, and the points across it the field's
# strength is drawn at.
VIEW = (3.0, 1.2)
N_GRID = 120


# --- math -----------------------------------------------------------------------------
def launch() -> tuple[core.Antivector, core.Antivector]:
    """The particles in the bottle's middle plane, each at its distance from the axis `[particles]`,
    moving at its angle to the axis, tilted from it sideways."""
    direction = (mv.zy * (-np.radians(PITCHES) / 2)).exp() >> axis           # [particles] Antivector
    return core.across * OFFSETS, (time + SPEED * direction) / np.sqrt(1 - SPEED**2)


def flights(pieces: core.Antivector, midpoints: core.Antivector) -> tuple[core.Antivector, core.Antivector]:
    """The particles' events and velocities at every step `[steps, particles]`."""
    history = islice(core.evolve(*launch(), pieces, midpoints, CURRENT, PROPER_STEP), N_STEPS)
    events, velocities = zip(*history)
    return stack(list(events)), stack(list(velocities))


def section(pieces: core.Antivector, midpoints: core.Antivector) -> tuple[core.Antivector, core.Scalar]:
    """Points of the plane through the axis, on a grid over the view `[across, along]`, and the field's
    strength at each."""
    along = axis * np.linspace(-VIEW[0], VIEW[0], 2 * N_GRID)                 # [along] Antivector
    points = core.across * np.linspace(-VIEW[1], VIEW[1], N_GRID)[:, None] + along   # [across, along] Antivector
    return points, core.field(points, pieces, midpoints, CURRENT).scalar_norm_squared().square_root()


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.animation import save_animation
    from examples.electromagnetism.magnetic_bottle import render

    pieces, midpoints = core.coils(RADIUS, HEIGHTS, N_PIECES)
    events, velocities = flights(pieces, midpoints)                          # [steps, particles] Antivector
    points, strength = section(pieces, midpoints)
    images = render.animate_flights(events, PITCHES, points, strength, RADIUS, HEIGHTS, N_STEPS_PER_FRAME)
    save_animation(images, "magnetic_bottle", DURATION_MS)

    # checks
    # The Lorentz force is the dual of the field's turn of the velocity, and the field keeps the speed.
    magnetic = core.field(events, pieces, midpoints, CURRENT)                 # [steps, particles] Bivector
    np.testing.assert_allclose((velocities.commutator(magnetic).dual() - (velocities & magnetic)).norm().to_array(),
                               0, atol=1e-10)
    np.testing.assert_allclose((mv.t & velocities).to_array(), 1 / np.sqrt(1 - SPEED**2), atol=1e-11)
    # The particles inside the loss cone escape through the coils, those outside it stay between them;
    # the trapped ones keep their magnetic moment through their bounces.
    on_axis = core.field(axis * np.linspace(HEIGHTS[0], HEIGHTS[-1], 401), pieces, midpoints, CURRENT)
    strengths = on_axis.scalar_norm_squared().square_root().to_array()
    loss_cone = np.degrees(np.arcsin(np.sqrt(strengths.min() / strengths.max())))
    heights = (mv.z & events[-1]).to_array()                                  # [particles]
    np.testing.assert_array_equal(np.abs(heights) > HEIGHTS[-1], PITCHES < loss_cone)
    # The moment is read with the field at the particle, which swings within a turn off the axis; the
    # invariant is its mean over a turn, the steps the velocity takes to turn once in the central field.
    moments = core.moment(velocities, magnetic).to_array()[:, PITCHES > loss_cone]   # [steps, trapped]
    n_turn = int(2 * np.pi / (strengths.min() * PROPER_STEP))
    means = np.apply_along_axis(np.convolve, 0, moments, np.ones(n_turn) / n_turn, "valid")
    np.testing.assert_allclose(means / means[0], 1, rtol=0.2)


if __name__ == "__main__":
    main()
