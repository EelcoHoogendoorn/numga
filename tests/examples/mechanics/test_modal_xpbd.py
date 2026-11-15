"""Coverage tests for modal extended position-based dynamics across sparse and differentiated
couplings."""

import matplotlib.pyplot as plt
import numpy as np

from numga.algebras import PGA2D
from examples import instantiate
from examples.mechanics.modal_xpbd import render, scenarios

core = instantiate("examples.mechanics.modal_xpbd.core", PGA2D)


def test_chain_swings_and_beam_compresses_and_draws():
    shape = core.girder(1, scenarios.length, scenarios.height, scenarios.stiffness, scenarios.density, 1)
    assert (shape.compliance > 0).all()
    plt.close(render.mode_shapes(shape.rest, shape.rest + shape.modes[:1] * 0.03, shape.edges))

    scenarios.cantilever(core.mv, shape, scenarios.damping, np.ones((1, 1)))
    bodies, hinges = scenarios.hinged_chain(core.mv, shape, scenarios.damping, np.ones((1, 1)))
    initial_points = core.world_points(bodies, shape)
    plt.close(render.chain_figure(initial_points, shape.edges))

    gravity = core.mv.direction([0, -scenarios.gravity_strength])           # [] Antivector-
    chain_states = list(scenarios.swing_states(core, shape, bodies, hinges, gravity, scenarios.dt, 1, 2))
    list(render.swinging_chain(shape.edges, initial_points, chain_states[:1]))
    _, _, _, final_gap = core.coupling(chain_states[-1][1], hinges)
    assert (final_gap.dual_inverse().norm() < 1e-4).all()

    beam_bodies, splices = scenarios.clamped_beam(core.mv, shape, 1, scenarios.beam_damping)
    beam_states = list(scenarios.compression_states(core, shape, beam_bodies, splices, np.zeros(1), scenarios.beam_dt, 1))
    list(render.beam_animation(shape.edges, beam_states))
    plt.close(render.buckling_figure(np.zeros(1), beam_states[0][1][None], scenarios.critical_displacement(scenarios.height, scenarios.length)))


def test_chain_swings_with_derived_couplings_and_draws():
    from examples.mechanics.modal_xpbd import derived

    shape = derived.core.girder(1, scenarios.length, scenarios.height, scenarios.stiffness, scenarios.density, 1)
    bodies, hinges = scenarios.hinged_chain(derived.mv, shape, scenarios.damping, np.ones((1, 1)))
    gravity = derived.mv.direction([0, -scenarios.gravity_strength])        # [] Antivector-
    states = list(derived.swing_states(bodies, hinges, gravity, scenarios.dt, 1, 2))
    frames = (derived.core.world_points(state, shape) for state in states)
    list(render.swinging_chain(shape.edges, derived.core.world_points(bodies, shape), frames))
    _, _, _, final_gap = derived.core.coupling(states[-1], hinges)
    assert (final_gap.dual_inverse().norm() < 1e-4).all()
