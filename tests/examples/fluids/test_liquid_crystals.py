"""Nematic liquid crystals: a defect pair and a disordered film on the smallest square film, one step of their turning
and of their active motion, their energy, windings and defects read off, and a frame of each drawn."""

import numpy as np

from examples.fluids.liquid_crystals import core, render, scenarios

mv = core.mv


def test_one_step_runs_and_draws():
    topology, vertices = scenarios.film(3, scenarios.half_width)
    centres = mv.vector([[-scenarios.separation / 2, 0], [scenarios.separation / 2, 0]]).dual()   # [2] Point-
    pair = core.defect_pair(vertices, centres, scenarios.core_radius, mv.xy * 0)
    wavevectors = mv.vector([[scenarios.wave_number, 0]])                     # [waves] Plane+
    amplitudes = core.alignment(mv.x.dual()[None], 1.0)                       # [waves] Point- <- Point-
    disordered = core.disordered_alignment(vertices, wavevectors, np.zeros(1), amplitudes, scenarios.softness)
    areas, gradients, cells = core.triangles(topology, vertices)
    stiffness = core.elasticity(topology, areas, gradients)
    core.free_energy(stiffness, cells, scenarios.elastic, scenarios.bulk, pair)
    core.windings(topology, pair, mv.x.dual())
    turned = list(core.turn_relaxation(topology, stiffness, cells, scenarios.elastic, scenarios.bulk, pair[None],
                                       scenarios.rotation_step, 2, 1))
    driven = list(core.evolution(topology, areas, gradients, cells, stiffness, scenarios.elastic, scenarios.bulk,
                                 scenarios.activities[:1], disordered[None], scenarios.time_step, 2, 1))
    next(render.animate(topology, vertices, render.turn_camera,
                        core.with_defects(topology, vertices, turned[-1:], mv.x.dual())))
    next(render.animate(topology, vertices, render.dense_camera,
                        core.with_defects(topology, vertices, driven[-1:], mv.x.dual())))
