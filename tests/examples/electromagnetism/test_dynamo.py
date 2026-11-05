"""A dynamo in a planet's core: the flow's modes are orthonormal in kinetic energy and carrying itself the flow makes
no energy, the field's force on a current has the power of the current paired with the motional field, the stirred
flow grows a field without sources, and the pictures draw."""

from __future__ import annotations

from itertools import islice

import matplotlib.pyplot as plt
import numpy as np

from numga.sparse import spdiag
from examples.electromagnetism.dynamo import core, render, scenarios

mv = core.mv
unit = mv.scalar([1.0])


def ball_and_flow():
    topology, _, vertices = scenarios.ball()
    ball = core.Ball(topology, vertices)
    fluid_topology, fluid_vertices, embedding = scenarios.outer_core(topology, vertices)
    flow = core.Flow(core.Ball(fluid_topology, fluid_vertices), embedding, scenarios.n_scalar_modes, scenarios.n_modes)
    return topology, vertices, ball, flow


def test_the_modes_are_orthonormal_and_carrying_itself_the_flow_makes_no_energy():
    topology, vertices, ball, flow = ball_and_flow()
    momenta = ball.prisms.norm() * flow.velocity(flow.modes).dual()            # [modes] Plane+[tets]
    gram = (flow.modes[:, None] & flow.velocity.adjugate()(momenta)[None, :]).sites.sum()   # [modes, modes] Scalar+
    np.testing.assert_allclose(gram.to_array(), np.eye(scenarios.n_modes), atol=1e-10)
    amplitudes = mv.scalar(np.random.default_rng(0).normal(size=(scenarios.n_modes, 1))).field()
    potential = scenarios.seed(topology, vertices, mv.xy)
    _, rate = core.advected(ball, flow, 0 * flow.velocity(flow.circulation(amplitudes)), amplitudes, potential)
    scale = ((rate * rate).sites.sum() * (amplitudes * amplitudes).sites.sum()).square_root().to_array()
    np.testing.assert_allclose((amplitudes * rate).sites.sum().to_array(), 0.0, atol=1e-12 * scale)


def test_the_force_on_a_current_has_the_power_of_its_motional_field():
    topology, vertices, ball, flow = ball_and_flow()
    rng = np.random.default_rng(1)
    velocity = flow.velocity(flow.circulation(mv.scalar(rng.normal(size=(scenarios.n_modes, 1))).field()))   # Point-[tets]
    potential = mv.scalar(rng.normal(size=(topology.n_elements[1], 1))).field()   # Scalar+[edges]
    conductivity = mv.scalar(np.ones((topology.n_elements[3], 1))).field()     # Scalar+[tets]
    currents = core.ohm(ball, conductivity)(ball.electric(mv.scalar(rng.normal(size=(topology.n_elements[1], 1))).field()))
    motional = spdiag((core.Point & ball.curl(potential)) ^ mv.t)              # (Field+ <- Point-)[tets <- tets]
    force_power = (motional.adjugate()(currents) & velocity).sites.sum().to_array()
    motional_power = (currents & motional(velocity)).sites.sum().to_array()
    np.testing.assert_allclose(force_power, motional_power, rtol=1e-10)


def test_the_stirred_flow_grows_a_field_without_sources_and_the_pictures_draw():
    topology, vertices, ball, flow = ball_and_flow()
    radii = ball.centres.antinorm()
    conductivity = mv.scalar(np.where((radii < 1).reshape(-1), scenarios.reynolds,
                                      scenarios.reynolds * scenarios.mantle_conductance)[:, None]).field()
    stirred = flow.reading(ball.prisms.norm() * scenarios.helical_flow(topology, vertices).dual())
    potential = scenarios.seed(topology, vertices, mv.xy * 1e-2 + mv.yz * 3e-3)
    states = list(islice(core.dynamo(ball, flow, conductivity, scenarios.viscosity, scenarios.pull, stirred, potential,
                                      scenarios.dt, 600), 0, None, 300))
    magnetic = [((state * ball.stiffness(state)).sites.sum() & unit).to_array() for _, state in states]
    assert magnetic[2] > 2 * magnetic[1]
    fluxes = topology.d[1] * states[-1][1]
    np.testing.assert_allclose((topology.d[2] * fluxes).to_array(), 0.0, atol=1e-12 * np.abs(fluxes.to_array()).max())
    frames = list(scenarios.frames(ball, flow, states[1:]))
    assert len(list(render.animate_lines(ball, frames))) == 2
    assert len(list(render.animate(ball, scenarios.mantle_radius, frames))) == 2
    figure = render.draw_modes(ball, scenarios.mantle_radius, flow.velocity(flow.modes[:2]))
    assert isinstance(figure, plt.Figure)
    plt.close(figure)
