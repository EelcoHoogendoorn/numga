"""A triangulated cylinder has zero developability energy, and one step of the bunny flow decreases energy and renders."""

from __future__ import annotations

from itertools import islice

import matplotlib.pyplot as plt
import numpy as np

from examples.surfaces.developability import core, render, scenarios
from numga.topology import Topology

mv = core.mv


def open_cylinder(n_rings: int, n_sectors: int) -> tuple[Topology, core.Point]:
    """Sweeping an `n_sectors`-gon loop along an `n_rings`-vertex chain builds an open unit cylinder (`Point-[V]`)."""
    angles = np.linspace(0, 2 * np.pi, n_sectors, endpoint=False)
    ring_rotors = (mv.xy * (-angles / 2)).exp()                                 # [n_sectors] Rotor+
    circle_points = ring_rotors >> mv.x.dual()                                  # [n_sectors] Point-
    axis_step = mv.z.dual()                                                     # [] Point-
    vertices = (circle_points[None, :] + axis_step * np.linspace(-1, 1, n_rings)[:, None]).reshape(-1).field()  # Point-[V]
    return Topology.loops([n_sectors]).product(Topology.chains([n_rings])), vertices


def test_cylinder_has_zero_developability_energy_and_gradient():
    """Every vertex normal neighbourhood on a cylinder lies on one great circle."""
    topology, vertices = open_cylinder(6, 24)
    edges = topology.boundary[2, 1] * (topology.d[0] * vertices)                # [3] Point-[F]
    angles = core.corner_angles(topology, edges)                                # [3] Scalar+[F]
    normals, spreads = core.normal_spread(topology, edges, angles)              # Patch-[F], (Point- <- Point-)[V]
    eigenvalues, directions = spreads.batch().eigh()                            # [V, 3] Scalar+, [V, 3] Point-
    assert (eigenvalues[:, 0].abs() < 1e-10).all()
    cylinder_gradient = core.energy_gradient(topology, edges, angles, normals, directions[:, 0].field())  # Plane+[V]
    assert (cylinder_gradient.norm() < 1e-10).all()


def test_bunny_flows_one_step_and_renders():
    """One stiffness-scaled flow step reduces the bunny's developability energy and renders static and animated frames."""
    topology, state_stream = scenarios.bunny_flow(n_kept=2, n_step_stride=1)
    states = list(state_stream)
    first_energy = states[0][1].batch().sum(axis=0)                             # [] Scalar+
    second_energy = states[1][1].batch().sum(axis=0)                            # [] Scalar+
    assert second_energy < first_energy
    figure = render.surface_figure(topology, states[0][0])
    plt.close(figure)
    frames = list(render.flow_animation(topology, islice(states, 1)))
    assert len(frames) == 1
