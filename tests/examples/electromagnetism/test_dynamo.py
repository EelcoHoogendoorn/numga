"""The dynamo at a small ball, a once subdivided icosahedron in three shells, with a handful of flow modes, stepped once, its
first flow mode drawn."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.electromagnetism.dynamo import core, render, scenarios

mv = core.mv


def test_the_dynamo_steps_and_draws():
    topology, _, vertices = scenarios.ball(n_levels=1, n_shells=3)
    ball = core.Ball(topology, vertices)
    fluid_topology, fluid_vertices, embedding = scenarios.outer_core(topology, vertices)
    flow = core.Flow(core.Ball(fluid_topology, fluid_vertices), embedding, 4, 2)
    stirred = flow.reading(ball.prisms.norm() * scenarios.helical_flow(topology, vertices).dual())
    radii = ball.centres.antinorm()
    conductivity = mv.scalar(np.where((radii < 1).reshape(-1), scenarios.reynolds,
                                      scenarios.reynolds * scenarios.mantle_conductance)[:, None]).field()
    potential = scenarios.seed(topology, vertices, mv.xy * 1e-2 + mv.yz * 3e-3)
    states = core.dynamo(ball, flow, conductivity, scenarios.viscosity, scenarios.pull, stirred, potential,
                         scenarios.dt, 1)
    list(scenarios.frames(ball, flow, states))
    plt.close(render.draw_modes(ball, scenarios.mantle_radius, flow.velocity(flow.modes[:1])))
