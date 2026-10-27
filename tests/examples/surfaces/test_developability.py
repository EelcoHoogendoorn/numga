"""Developability: a cylinder has no energy, the gradient is the energy's derivative, and the bunny's flow takes
away nearly all its energy."""

from __future__ import annotations

from itertools import islice

import numpy as np

from examples.surfaces.developability import core, render, scenarios

mv = core.mv


def cylinder(rings: int, sectors: int) -> tuple[core.Mesh, core.Point]:
    """An open cylinder of unit radius around the z axis, `rings` circles of `sectors` points."""
    angles = np.linspace(0, 2 * np.pi, sectors, endpoint=False)
    around = (mv.xy * (-angles / 2)).exp() >> mv.x.dual()           # [sectors] Point-
    height = mv.z.dual()                                             # [] Point-
    vertices = (around[None, :] + height * np.linspace(-1, 1, rings)[:, None]).reshape(-1).field()   # Point-[vertices]
    ring, sector = np.meshgrid(np.arange(rings - 1), np.arange(sectors), indexing="ij")
    here, along = ring * sectors + sector, ring * sectors + (sector + 1) % sectors
    faces = np.concatenate([np.stack([here, along, along + sectors], axis=-1).reshape(-1, 3),
                            np.stack([here, along + sectors, here + sectors], axis=-1).reshape(-1, 3)])
    return core.Mesh(rings * sectors, faces), vertices


def energy(mesh: core.Mesh, vertices: core.Point) -> float:
    edges = (mesh.d0 * vertices).batch()[mesh.face_edges.T] * mesh.face_edge_orientation.T
    after, before = edges[[1, 2, 0]], edges[[2, 0, 1]]
    angles = (-(after.dual_inverse() & before) / (after.norm() * before.norm())).arccos()
    _, spreads = core.spread(mesh, edges, angles)
    values, _ = spreads.batch().eigh()
    return values[:, 0].sum(axis=0).to_array()


def test_a_cylinder_is_developable():
    """Around every vertex of a cylinder the face normals lie on one great circle."""
    mesh, vertices = cylinder(6, 24)
    edges = (mesh.d0 * vertices).batch()[mesh.face_edges.T] * mesh.face_edge_orientation.T
    after, before = edges[[1, 2, 0]], edges[[2, 0, 1]]
    angles = (-(after.dual_inverse() & before) / (after.norm() * before.norm())).arccos()
    _, spreads = core.spread(mesh, edges, angles)
    values, _ = spreads.batch().eigh()
    np.testing.assert_allclose(values[:, 0].to_array(), 0.0, atol=1e-13)


def test_the_gradient_is_the_energys_derivative():
    mesh, vertices = scenarios.bunny()
    edges = (mesh.d0 * vertices).batch()[mesh.face_edges.T] * mesh.face_edge_orientation.T
    after, before = edges[[1, 2, 0]], edges[[2, 0, 1]]
    angles = (-(after.dual_inverse() & before) / (after.norm() * before.norm())).arccos()
    normals, spreads = core.spread(mesh, edges, angles)
    _, directions = spreads.batch().eigh()
    gradient = core.gradient(mesh, edges, angles, normals, directions[:, 0]).batch()     # [vertices] Vector+
    points = render.coordinates(vertices)                                         # [V, 3]
    epsilon = 1e-6
    rng = np.random.default_rng(0)
    for vertex, axis in zip(rng.integers(len(points), size=6), rng.integers(3, size=6)):
        nudge = np.zeros_like(points)
        nudge[vertex, axis] = epsilon
        ahead, behind = (energy(mesh, mv.antivector(points + sign * nudge).field()) for sign in (1, -1))
        # The gradient paired with a unit step along the axis; the tolerance tests the central difference.
        along = (gradient[vertex] & mv.antivector(np.eye(3)[axis])).to_array()
        np.testing.assert_allclose((ahead - behind) / (2 * epsilon), along, rtol=1e-6, atol=1e-8)


def test_the_bunny_flow_takes_away_nearly_all_its_energy():
    mesh, vertices = scenarios.bunny()
    states = list(islice(core.flow(mesh, vertices, scenarios.STEP), 0, 101, 25))
    energies = np.array([values.sum(axis=0).to_array() for _, values in states])
    assert np.all(np.diff(energies) < 0)
    assert energies[-1] < 0.2 * energies[0]
    assert len(list(render.animate_flow(mesh, states[:1]))) == 1
