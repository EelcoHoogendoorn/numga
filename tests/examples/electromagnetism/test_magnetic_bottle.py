"""The magnetic bottle: the Lorentz force is the dual of the field's turn of the velocity, and the field keeps the
speed."""

from __future__ import annotations

from itertools import islice

import numpy as np

from examples.electromagnetism.magnetic_bottle import core, render, scenarios

mv = core.mv


def test_field_turns_velocity_by_the_lorentz_force_at_constant_speed():
    pieces, midpoints = core.coils(scenarios.coil_radius, scenarios.coil_heights, scenarios.n_pieces)
    history = islice(core.evolve(*scenarios.launch(), pieces, midpoints, scenarios.coil_current, scenarios.proper_step), 50)
    event, velocity = list(history)[-1]
    magnetic = core.field(event, pieces, midpoints, scenarios.coil_current)
    mismatch = velocity.commutator(magnetic).dual() - (velocity & magnetic)
    np.testing.assert_allclose(mismatch.kernel, 0, atol=1e-10)
    np.testing.assert_allclose((mv.t & velocity).to_array(), 1 / np.sqrt(1 - scenarios.launch_speed**2), atol=1e-11)


def test_one_step_draws():
    pieces, midpoints = core.coils(scenarios.coil_radius, scenarios.coil_heights, 8)
    events, _ = scenarios.flights(pieces, midpoints, n_steps=1)
    points, strength = scenarios.section(pieces, midpoints, n_grid=4)
    next(render.animate_flights(events, scenarios.launch_pitches, points, strength, scenarios.coil_radius,
                                scenarios.coil_heights, 1))
