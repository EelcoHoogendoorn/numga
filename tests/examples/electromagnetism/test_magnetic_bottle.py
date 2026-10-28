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
    pieces, midpoints = core.coils(scenarios.coil_radius, scenarios.coil_heights, scenarios.n_pieces)
    return pieces, midpoints, *scenarios.flights(pieces, midpoints)


def test_field_turns_velocity_by_the_lorentz_force_at_constant_speed():
    pieces, midpoints = core.coils(scenarios.coil_radius, scenarios.coil_heights, scenarios.n_pieces)
    history = islice(core.evolve(*scenarios.launch(), pieces, midpoints, scenarios.coil_current, scenarios.proper_step), 50)
    event, velocity = list(history)[-1]
    magnetic = core.field(event, pieces, midpoints, scenarios.coil_current)
    mismatch = velocity.commutator(magnetic).dual() - (velocity & magnetic)
    np.testing.assert_allclose(mismatch.kernel, 0, atol=1e-10)
    np.testing.assert_allclose((mv.t & velocity).to_array(), 1 / np.sqrt(1 - scenarios.launch_speed**2), atol=1e-11)


def test_loss_cone_and_magnetic_moment():
    """Escape through the coils exactly inside the loss cone of the field on the axis; the trapped
    particles keep their magnetic moment, averaged over a turn, through their bounces."""
    pieces, midpoints, events, velocities = bottle()
    field_on_axis = core.field(core.axis * np.linspace(scenarios.coil_heights[0], scenarios.coil_heights[-1], scenarios.n_axis), pieces, midpoints, scenarios.coil_current)
    strengths = field_on_axis.scalar_norm_squared().square_root().to_array()
    loss_cone = np.degrees(np.arcsin(np.sqrt(strengths.min() / strengths.max())))
    escaped = np.abs((mv.z & events[-1]).to_array()) > scenarios.coil_heights[-1]
    np.testing.assert_array_equal(escaped, scenarios.launch_pitches < loss_cone)
    assert escaped.any() and not escaped.all()
    magnetic = core.field(events, pieces, midpoints, scenarios.coil_current)
    moments = core.moment(velocities, magnetic).to_array()                    # [steps, particles]
    n_turn = int(2 * np.pi / (strengths.min() * scenarios.proper_step))
    means = np.apply_along_axis(np.convolve, 0, moments, np.ones(n_turn) / n_turn, "valid")
    np.testing.assert_allclose(means[:, ~escaped] / means[0, ~escaped], 1, rtol=0.2)
    assert (np.abs(means[:, escaped] / means[0, escaped] - 1).max(axis=0) > 1).all()


def test_scene_draws():
    pieces, midpoints = core.coils(scenarios.coil_radius, scenarios.coil_heights, scenarios.n_pieces)
    history = islice(core.evolve(*scenarios.launch(), pieces, midpoints, scenarios.coil_current, scenarios.proper_step), 24)
    events = stack([event for event, _ in history])
    points, strength = scenarios.section(pieces, midpoints)
    images = render.animate_flights(events, scenarios.launch_pitches, points, strength, scenarios.coil_radius,
                                    scenarios.coil_heights, scenarios.n_steps_per_frame)
    assert len(list(images)) == 24 // scenarios.n_steps_per_frame
