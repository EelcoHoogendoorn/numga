"""Coverage tests for modal extended position-based dynamics across sparse and differentiated
couplings."""

import matplotlib.pyplot as plt
import numpy as np

from numga import dim
from numga.algebras import PGA2D
from examples import instantiate
from examples.mechanics.modal_xpbd import render, scenarios

core = instantiate("examples.mechanics.modal_xpbd.core", PGA2D, units=True)


def test_chain_swings_and_beam_compresses_and_draws():
    mv = core.mv
    force_unit = dim.mass * dim.length / dim.time**2
    density_unit = dim.mass / dim.length**2
    length = mv.scalar(scenarios.length, unit=dim.length)
    height = mv.scalar(scenarios.height, unit=dim.length)
    shape = core.girder(1, length, height, mv.scalar(scenarios.stiffness, unit=force_unit), mv.scalar(scenarios.density, unit=density_unit), 1)
    assert shape.compliance.unit == dim.time**2
    assert shape.modes.unit == 1 / dim.mass.root(2)
    assert (shape.compliance > 0).all()
    mode_scale = mv.scalar(0.03, unit=dim.length * dim.mass.root(2))
    plt.close(render.mode_shapes(shape.rest, shape.rest + shape.modes[:1] * mode_scale, shape.edges))

    scenarios.cantilever(mv, shape, scenarios.damping, np.ones((1, 1)))
    bodies, hinges = scenarios.hinged_chain(mv, shape, scenarios.damping, np.ones((1, 1)))
    initial_points = core.world_points(bodies, shape)
    plt.close(render.chain_figure(initial_points, shape.edges))

    gravity = mv.direction([0, -scenarios.gravity_strength], unit=dim.length / dim.time**2)  # [] Antivector-, length/time^2
    dt = mv.scalar(scenarios.dt, unit=dim.time)
    chain_states = list(scenarios.swing_states(core, shape, bodies, hinges, gravity, dt, 1, 2))
    list(render.swinging_chain(shape.edges, initial_points, chain_states[:1]))
    _, _, _, final_gap = core.coupling(chain_states[-1][1], hinges)
    assert final_gap.dual_inverse().norm().unit == dim.length
    assert (final_gap.dual_inverse().norm() < mv.scalar(1e-4, unit=dim.length)).all()

    beam_bodies, splices = scenarios.clamped_beam(mv, shape, 1, scenarios.beam_damping)
    zero_displacements = mv.scalar(np.zeros((1, 1)), unit=dim.length)
    beam_states = list(scenarios.compression_states(core, shape, beam_bodies, splices, zero_displacements, mv.scalar(scenarios.beam_dt, unit=dim.time), 1))
    list(render.beam_animation(shape.edges, beam_states))
    plt.close(render.buckling_figure(zero_displacements, beam_states[0][1][None], scenarios.critical_displacement(height, length)))


def test_chain_swings_with_derived_couplings_and_draws():
    from examples.mechanics.modal_xpbd import derived

    mv = derived.mv
    force_unit = dim.mass * dim.length / dim.time**2
    density_unit = dim.mass / dim.length**2
    shape = derived.core.girder(
        1,
        mv.scalar(scenarios.length, unit=dim.length),
        mv.scalar(scenarios.height, unit=dim.length),
        mv.scalar(scenarios.stiffness, unit=force_unit),
        mv.scalar(scenarios.density, unit=density_unit),
        1,
    )
    bodies, hinges = scenarios.hinged_chain(mv, shape, scenarios.damping, np.ones((1, 1)))
    gravity = mv.direction([0, -scenarios.gravity_strength], unit=dim.length / dim.time**2)  # [] Antivector-, length/time^2
    dt = mv.scalar(scenarios.dt, unit=dim.time)
    states = list(derived.swing_states(bodies, hinges, gravity, dt, 1, 2))
    frames = (derived.core.world_points(state, shape) for state in states)
    list(render.swinging_chain(shape.edges, derived.core.world_points(bodies, shape), frames))
    _, _, _, final_gap = derived.core.coupling(states[-1], hinges)
    assert final_gap.dual_inverse().norm().unit == dim.length
    assert (final_gap.dual_inverse().norm() < mv.scalar(1e-4, unit=dim.length)).all()
