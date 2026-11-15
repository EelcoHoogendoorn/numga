"""Potential flow past a Joukowski wing."""

import matplotlib.pyplot as plt

from examples.fluids.wing import core, render, scenarios
from numga.topology import Topology


def test_wing_flow_has_vanishing_derivative_and_kutta_joukowski_lift():
    centre, trailing_edge = scenarios.wing_geometry(scenarios.highest_attack)
    stream = scenarios.stream
    swirl = core.kutta_circulation(centre, trailing_edge, stream)

    # Kutta stagnation condition at the cylinder trailing edge, and free-stream recovery far away.
    trailing_velocity, _, _ = core.cylinder_flow(centre, trailing_edge, swirl, stream, trailing_edge)
    assert (trailing_velocity.norm() < 1e-12).all()
    _, far_velocity, _, _ = core.wing_flow(
        centre, trailing_edge, swirl, stream,
        core.cylinder_rings(centre, trailing_edge, 2, 8, 1e4)[-1], trailing_edge,
    )
    assert ((far_velocity - stream).norm() < 1e-3).all()

    # Vanishing geometric derivative past the cylinder, constant surface stream function, and pressure lift.
    n_test_angles = 180
    positions = core.cylinder_rings(centre, trailing_edge, 2, n_test_angles, scenarios.outer_distance)
    _, gradient, _ = core.cylinder_flow(centre, trailing_edge, swirl, stream, positions)
    assert (core.geometric_derivative(gradient).norm() < 1e-12).all()

    wing_points, wing_velocity, _, wing_flux = core.wing_flow(
        centre, trailing_edge, swirl, stream, positions, trailing_edge,
    )
    assert ((wing_flux[0] - wing_flux[0, :1]).norm() < 1e-12).all()

    ring = Topology.loops([n_test_angles])
    edge_average = ring.incidence[1, 0].mean(axis=0)
    surface_points = wing_points[0].field()
    edge_steps = ring.d[0] * surface_points
    edge_pressures = -0.5 * scenarios.density * (edge_average * wing_velocity[0].squared().field())
    tails, heads = ring.incidence[1, 0] * surface_points
    winding = (tails & heads).batch().sum(axis=-1)
    pressure_force = (winding / winding.antinorm()) * (edge_pressures * edge_steps).batch().sum(axis=-1)
    assert ((pressure_force - core.kutta_lift(swirl, stream, scenarios.density)).norm() < 5e-3).all()


def test_render_draws_wing_flow():
    centre, trailing_edge = scenarios.wing_geometry(scenarios.highest_attack)
    stream = scenarios.stream
    swirl = core.kutta_circulation(centre, trailing_edge, stream)
    positions = core.cylinder_rings(centre, trailing_edge, 2, 24, scenarios.outer_distance)
    wing_points, wing_velocity, _, wing_flux = core.wing_flow(
        centre, trailing_edge, swirl, stream, positions, trailing_edge,
    )
    lift = core.kutta_lift(swirl, stream, scenarios.density)
    plt.close(render.figure(wing_points, wing_velocity, wing_flux, lift, scenarios.speed))
