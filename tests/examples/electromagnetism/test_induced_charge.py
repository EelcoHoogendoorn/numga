"""Small coverage of charge relaxation and its picture."""
import numpy as np
from examples.electromagnetism.induced_charge import core, render, scenarios
from examples.util.animation import save_animation


def test_charge_relaxation():
    topology, vertices, fractions, memberships, boundary = scenarios.scene(9, 2)
    states = list(core.relaxation(topology, vertices, fractions, boundary, scenarios.permittivity,
                                 scenarios.conductivities, scenarios.body_charges, scenarios.frame_step, 1))
    save_animation(render.animate(topology, vertices, fractions.sum(axis=0), states),
                   "induced_charge_test", scenarios.frame_duration_ms)
    charge = states[-1][1]
    assert np.all((charge.sites.sum() - scenarios.body_charges.sum()).abs() < .01)
