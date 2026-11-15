"""Coverage tests check Lorentz-force duality, speed conservation, magnetic moment positivity, and GPU
frame rendering on minimal coil and step counts."""

from __future__ import annotations

import numpy as np

from examples.electromagnetism.magnetic_bottle import core, render, scenarios

mv = core.mv


def test_field_turns_velocity_by_the_lorentz_force_at_constant_speed() -> None:
    currents, midpoints = core.coils(scenarios.coil_radius, scenarios.coil_heights, scenarios.coil_current, 8)
    states = list(core.evolve(*scenarios.launch(), currents, midpoints, scenarios.proper_step, 2))
    event, velocity = states[-1]
    magnetic = core.field(event, currents, midpoints)
    mismatch = velocity.commutator(magnetic).dual() - (magnetic & velocity)
    lorentz_factor = 1 / np.sqrt(1 - scenarios.launch_speed**2)
    assert (mismatch.norm() < 1e-8).all()
    assert (((mv.t & velocity) - lorentz_factor).abs() < 1e-8).all()
    assert (core.moment(velocity, magnetic) > 0).all()


def test_one_step_draws() -> None:
    currents, midpoints = core.coils(scenarios.coil_radius, scenarios.coil_heights, scenarios.coil_current, 8)
    states = core.evolve(*scenarios.launch(), currents, midpoints, scenarios.proper_step, 1)
    points, strength = scenarios.section(currents, midpoints, 4)
    frame = next(render.animate_flights(
        scenarios.launch_pitches, points, strength, scenarios.coil_radius, scenarios.coil_heights, 1, states,
    ))
    assert frame.shape[1:] == (render.n_pixels, 3)
