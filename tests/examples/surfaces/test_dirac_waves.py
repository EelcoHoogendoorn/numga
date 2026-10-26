"""Spinor waves on the sphere: the leapfrog keeps the energy, the pulse gathers at the south pole, the
frequencies are the whole numbers, and the scenes draw."""

from __future__ import annotations

import numpy as np

from numga import stack
from examples.surfaces.dirac_waves import core, render, scenarios

mv = core.mv


def test_the_pulse_keeps_its_energy_and_gathers_at_the_south_pole():
    sphere, points, states = scenarios.pulse()
    edges = ((sphere.d0 * points).batch()[sphere.face_edges.T] * sphere.face_edge_orientation.T).field()
    triangle_areas = edges[1].commutator(edges[0]).norm() / 2
    thirds = core.as_ga_sparse(sphere.faces, core.as_scalar(np.ones_like(sphere.faces) / 3))
    face_areas, vertex_areas = core.spdiag(triangle_areas), core.spdiag(~thirds * triangle_areas)
    energies = stack([(vertex_areas * vertices.scalar_norm_squared()).sites.sum()
                      + (face_areas * before.reverse().scalar_product(after)).sites.sum()
                      for (vertices, after), (_, before) in zip(states[1:], states)]).to_array()
    np.testing.assert_allclose(energies, energies[0], rtol=1e-12)
    density = (vertex_areas * states[int(round(np.pi / scenarios.INTERVAL))][0].scalar_norm_squared()).to_array()
    assert density[(points & mv.z) < -0.9].sum() > 0.5 * density.sum()
    vertices, faces = states[0]
    assert len(list(render.animate_pulse(sphere, points, vertices[None], faces[None]))) == 1


def test_the_standing_waves_ring_at_whole_numbers():
    sphere, points, squared, vertices, faces = scenarios.modes()
    measured = squared.to_array()
    np.testing.assert_allclose(measured[:scenarios.N_CONSTANT], 0.0, atol=1e-9)
    levels = np.arange(1, scenarios.N_LEVELS + 1)
    np.testing.assert_allclose(np.sqrt(measured[scenarios.N_CONSTANT:]), np.repeat(levels, 8 * levels), rtol=0.03)
    assert len(list(render.animate_modes(sphere, points, vertices[:, :1], faces[:, :1]))) == 1
