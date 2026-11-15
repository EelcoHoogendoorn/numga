"""One-step defect-pair rotation and disordered active flow run on a tiny square film and render through both cameras."""

import numpy as np

from examples.fluids.liquid_crystals import core, render, scenarios

mv = core.mv


def test_one_step_runs_and_draws():
    topology, vertices = scenarios.film(3, scenarios.half_width)
    centres = mv.vector([[-scenarios.separation / 2 + 0.03, 0.07],
                         [scenarios.separation / 2 + 0.03, 0.07]]).dual()     # [2] Point-
    pair_order = core.defect_pair(vertices, centres, scenarios.core_radius, mv.xy * 0)
    wavevectors = mv.vector([[scenarios.wave_number, 0]])                     # [1] Plane+
    amplitudes = core.alignment(mv.x.dual()[None], 1.0)                       # [1] Point- <- Point-
    disordered_order = core.disordered_alignment(vertices, wavevectors, np.zeros(1), amplitudes, scenarios.softness)
    areas, gradients, dual_areas = core.triangles(topology, vertices)
    stiffness = core.elasticity(topology, areas, gradients)
    energy = core.free_energy(stiffness, dual_areas, scenarios.elastic, scenarios.bulk, pair_order)
    core.windings(topology, pair_order, mv.x.dual())
    turned_states = list(core.turn_relaxation(topology, stiffness, dual_areas, scenarios.elastic, scenarios.bulk,
                                              pair_order[None], scenarios.rotation_step, 2, 1))
    driven_states = list(core.evolution(topology, areas, gradients, dual_areas, stiffness, scenarios.elastic,
                                        scenarios.bulk, scenarios.activities[:1], disordered_order[None],
                                        scenarios.time_step, 2, 1))
    turn_frame = next(render.animate(topology, vertices, render.turn_camera,
                                     core.with_defects(topology, vertices, turned_states[-1:], mv.x.dual())))
    dense_frame = next(render.animate(topology, vertices, render.dense_camera,
                                      core.with_defects(topology, vertices, driven_states[-1:], mv.x.dual())))
    assert energy.isfinite().all()
    assert turn_frame.shape[2] == 3 and dense_frame.shape[2] == 3
