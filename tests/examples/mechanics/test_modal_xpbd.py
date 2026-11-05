"""A chain of one girder of one cell swings a step, by the sparse couplings and by the derived ones, and a beam of one
such girder is compressed a step, and a frame of each draws."""

import numpy as np

from numga.algebras import PGA2D
from examples import instantiate
from examples.mechanics.modal_xpbd import render, scenarios

core = instantiate("examples.mechanics.modal_xpbd.core", PGA2D)


def test_a_chain_swings_and_a_beam_is_compressed_a_step_and_both_draw():
    shape = core.girder(1, scenarios.LENGTH, scenarios.HEIGHT, scenarios.STIFFNESS, scenarios.DENSITY, 1)
    bodies, hinges = scenarios.hinged_chain(core, shape, scenarios.DAMPING, np.ones((1, 1)))
    gravity = core.mv.direction([0, -scenarios.GRAVITY])                    # [] Displacement-
    geometry = scenarios.swing(core, shape, bodies, hinges, gravity, scenarios.INTERVAL, 1, 1)
    list(render.swinging_chain(geometry, shape.edges, core.points(bodies, shape)))
    bodies, splices = scenarios.clamped_beam(core, shape, 1, scenarios.BEAM_DAMPING)
    compressed = scenarios.compress(core, shape, bodies, splices, np.zeros(1), scenarios.BEAM_INTERVAL, 1)
    list(render.beam((points for points, _ in compressed), shape.edges))


def test_a_chain_swings_a_step_by_the_derived_couplings_and_draws():
    from examples.mechanics.modal_xpbd import derived

    shape = derived.core.girder(1, scenarios.LENGTH, scenarios.HEIGHT, scenarios.STIFFNESS, scenarios.DENSITY, 1)
    bodies, hinges = scenarios.hinged_chain(derived.core, shape, scenarios.DAMPING, np.ones((1, 1)))
    gravity = derived.mv.direction([0, -scenarios.GRAVITY])                 # [] Displacement-
    frames = (derived.core.points(moment, shape) for moment in derived.swing(bodies, hinges, gravity,
                                                                             scenarios.INTERVAL, 1, 2))
    list(render.swinging_chain(frames, shape.edges, derived.core.points(bodies, shape)))
