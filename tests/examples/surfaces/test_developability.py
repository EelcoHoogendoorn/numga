"""Developability: a cylinder has no energy, and the bunny's flow runs one step and draws."""

from __future__ import annotations

import numpy as np

from examples.surfaces.developability import core, render, scenarios

mv = core.mv


def cylinder(rings: int, sectors: int) -> tuple[core.Topology, core.Point]:
    """An open cylinder of unit radius around the z axis, `rings` circles of `sectors` points."""
    angles = np.linspace(0, 2 * np.pi, sectors, endpoint=False)
    around = (mv.xy * (-angles / 2)).exp() >> mv.x.dual()           # [sectors] Point-
    height = mv.z.dual()                                             # [] Point-
    vertices = (around[None, :] + height * np.linspace(-1, 1, rings)[:, None]).reshape(-1).field()   # Point-[vertices]
    # A loop of sectors swept along a chain of rings, its vertices ring after ring.
    return core.Topology.loops([sectors]).product(core.Topology.chains([rings])), vertices


def test_a_cylinder_is_developable():
    """Around every vertex of a cylinder the face normals lie on one great circle."""
    topology, vertices = cylinder(6, 24)
    edges = topology.boundary[2, 1] * (topology.d[0] * vertices)
    tails, heads = topology.corners[2, 1]
    after, before = edges[tails], edges[heads]
    angles = (-(after.dual_inverse() & before) / (after.norm() * before.norm())).arccos()
    _, spreads = core.spread(topology, edges, angles)
    values, _ = spreads.batch().eigh()
    np.testing.assert_allclose(values[:, 0].to_array(), 0.0, atol=1e-13)


def test_the_bunny_flows_one_step_and_draws():
    topology, states = scenarios.flowed(n_steps=1)
    assert len(list(render.animate_flow(topology, states))) == len(states)
