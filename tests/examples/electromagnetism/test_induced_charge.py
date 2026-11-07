"""Small coverage of charge relaxation and its picture."""
import numpy as np
from examples.electromagnetism.induced_charge import core, render, scenarios


def test_charge_relaxation():
    topology, vertices, fractions, memberships, boundary = scenarios.scene(9, 2)
    states = core.relaxation(topology, vertices, fractions, boundary, scenarios.permittivity,
                             scenarios.conductivities, scenarios.body_charges, scenarios.frame_step, 1)
    next(states)
    state = next(states)
    charge = state[1]
    assert np.all((charge.sites.sum() - scenarios.body_charges.sum()).abs() < .01)
    next(render.animate(topology, vertices, fractions.sum(axis=0), [state]))
