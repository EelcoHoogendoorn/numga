"""The magnetic bottle: the Lorentz force is the dual of the field's turn of the velocity, the field
keeps the speed, particles inside the loss cone escape and the others stay, keeping their magnetic moment,
and the scene draws."""

from __future__ import annotations

from itertools import islice

import numpy as np

from numga import stack
from examples.electromagnetism.magnetic_bottle import core, render, scenarios

mv = core.mv


def bottle() -> tuple[core.Antivector, core.Antivector, core.Antivector, core.Antivector]:
    pieces, midpoints = core.coils(scenarios.RADIUS, scenarios.HEIGHTS, scenarios.N_PIECES)
    return pieces, midpoints, *scenarios.flights(pieces, midpoints)


def test_field_turns_velocity_by_the_lorentz_force_at_constant_speed():
    pieces, midpoints = core.coils(scenarios.RADIUS, scenarios.HEIGHTS, scenarios.N_PIECES)
    history = islice(core.evolve(*scenarios.launch(), pieces, midpoints, scenarios.CURRENT, scenarios.PROPER_STEP), 50)
    event, velocity = list(history)[-1]
    magnetic = core.field(event, pieces, midpoints, scenarios.CURRENT)
    mismatch = velocity.commutator(magnetic).dual() - (velocity & magnetic)
    np.testing.assert_allclose(mismatch.norm().to_array(), 0, atol=1e-10)
    np.testing.assert_allclose((mv.t & velocity).to_array(), 1 / np.sqrt(1 - scenarios.SPEED**2), atol=1e-11)


def test_loss_cone_and_magnetic_moment():
    """Escape through the coils exactly inside the loss cone of the field on the axis; the trapped
    particles keep their magnetic moment, averaged over a turn, through their bounces."""
    pieces, midpoints, events, velocities = bottle()
    on_axis = core.field(core.axis * np.linspace(scenarios.HEIGHTS[0], scenarios.HEIGHTS[-1], 401), pieces, midpoints, scenarios.CURRENT)
    strengths = on_axis.scalar_norm_squared().square_root().to_array()
    loss_cone = np.degrees(np.arcsin(np.sqrt(strengths.min() / strengths.max())))
    escaped = np.abs((mv.z & events[-1]).to_array()) > scenarios.HEIGHTS[-1]
    np.testing.assert_array_equal(escaped, scenarios.PITCHES < loss_cone)
    assert escaped.any() and not escaped.all()
    magnetic = core.field(events, pieces, midpoints, scenarios.CURRENT)
    moments = core.moment(velocities, magnetic).to_array()                    # [steps, particles]
    n_turn = int(2 * np.pi / (strengths.min() * scenarios.PROPER_STEP))
    means = np.apply_along_axis(np.convolve, 0, moments, np.ones(n_turn) / n_turn, "valid")
    np.testing.assert_allclose(means[:, ~escaped] / means[0, ~escaped], 1, rtol=0.2)
    assert (np.abs(means[:, escaped] / means[0, escaped] - 1).max(axis=0) > 1).all()


def test_scene_draws():
    pieces, midpoints = core.coils(scenarios.RADIUS, scenarios.HEIGHTS, scenarios.N_PIECES)
    history = islice(core.evolve(*scenarios.launch(), pieces, midpoints, scenarios.CURRENT, scenarios.PROPER_STEP), 24)
    events = stack([event for event, _ in history])
    points, strength = scenarios.section(pieces, midpoints)
    images = render.animate_flights(events, scenarios.PITCHES, points, strength, scenarios.RADIUS,
                                    scenarios.HEIGHTS, scenarios.N_STEPS_PER_FRAME)
    assert len(images) == 2
