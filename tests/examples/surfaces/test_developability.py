"""Developability: a cylinder has no energy, the gradient is the energy's derivative, and the bunny's flow takes
away nearly all its energy."""

from __future__ import annotations

from itertools import islice

import numpy as np

from examples.surfaces.developability import core, render, scenarios

mv = core.mv


def cylinder(rings: int, sectors: int) -> core.Mesh:
    """An open cylinder of unit radius around the z axis, `rings` circles of `sectors` points."""
    angles = np.linspace(0, 2 * np.pi, sectors, endpoint=False)
    around = (mv.xy * (-angles / 2)).exp() >> mv.x                               # [sectors] Vector
    vertices = (around[None, :] + mv.z * np.linspace(-1, 1, rings)[:, None]).reshape(-1).field()   # Vector[V]
    ring, sector = np.meshgrid(np.arange(rings - 1), np.arange(sectors), indexing="ij")
    here, along = ring * sectors + sector, ring * sectors + (sector + 1) % sectors
    faces = np.concatenate([np.stack([here, along, along + sectors], axis=-1).reshape(-1, 3),
                            np.stack([here, along + sectors, here + sectors], axis=-1).reshape(-1, 3)])
    return core.Mesh(vertices, faces)


def energy(mesh: core.Mesh) -> float:
    _, spreads = core.spread(mesh)
    values, _ = spreads.batch().eigh()
    return values[:, 0].sum(axis=0).to_array()


def test_a_cylinder_is_developable():
    """Around every vertex of a cylinder the face normals lie on one great circle."""
    _, spreads = core.spread(cylinder(6, 24))
    values, _ = spreads.batch().eigh()
    np.testing.assert_allclose(values[:, 0].to_array(), 0.0, atol=1e-13)


def test_the_gradient_is_the_energys_derivative():
    mesh = scenarios.bunny()
    normals, spreads = core.spread(mesh)
    _, vectors = spreads.batch().eigh()
    gradient = render.coordinates(core.gradient(mesh, normals, vectors[:, 0]))   # [V, 3]
    points = render.coordinates(mesh.vertices)                                    # [V, 3]
    epsilon = 1e-6
    rng = np.random.default_rng(0)
    for vertex, axis in zip(rng.integers(len(points), size=6), rng.integers(3, size=6)):
        nudge = np.zeros_like(points)
        nudge[vertex, axis] = epsilon
        ahead, behind = (energy(mesh.copy(mv.vector(points + sign * nudge).field())) for sign in (1, -1))
        # The tolerance tests the central difference.
        np.testing.assert_allclose((ahead - behind) / (2 * epsilon), gradient[vertex, axis], rtol=1e-6, atol=1e-8)


def test_the_bunny_flow_takes_away_nearly_all_its_energy():
    mesh = scenarios.bunny()
    states = list(islice(core.flow(mesh, scenarios.STEP), 0, 101, 25))
    energies = np.array([values.sum(axis=0).to_array() for _, values in states])
    assert np.all(np.diff(energies) < 0)
    assert energies[-1] < 0.2 * energies[0]
    assert len(render.animate_flow(mesh, states[:1])) == 1
