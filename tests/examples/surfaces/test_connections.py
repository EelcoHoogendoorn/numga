"""Connections on curved surfaces: a loop's holonomy grows by the circulation of the edges' extra turns,
the scenes of parallel transport, trivial connections and stripe patterns pass their checks, and the
figures and animations draw."""

from __future__ import annotations

from itertools import islice

import matplotlib.pyplot as plt
import numpy as np
from numga import stack

from examples.surfaces.connections import core, render, scenarios


def test_extra_turns_add_to_every_face_holonomy():
    """Transported around a face by the adjusted rotors, a tangent vector turns by the face's curvature
    plus the circulation of the extra turns around it."""
    sphere, vertices = core.Mesh.icosphere(2)
    unit_normals = core.vertex_normals(sphere, vertices)                          # Vector-[vertices]
    angles = core.as_scalar(np.random.default_rng(0).normal(size=len(sphere.edges)) * 0.3).field()   # Scalar+[edges]
    forward = core.adjusted_transport(sphere, unit_normals, angles)[:, 0]         # [edges] Rotor+
    both = stack([forward, forward.reverse()], axis=-1)                           # [edges, 2] Rotor+
    # Around each face, counter-clockwise from its first corner: the edges facing the third, first and
    # second corners, each taken forward where it runs that way and backward where it does not.
    around = [both[sphere.face_edges[:, corner], (sphere.face_edge_orientation[:, corner] < 0).astype(int)]
              for corner in (2, 0, 1)]
    loop = around[2] * around[1] * around[0]                                      # [faces] Rotor+
    normals = unit_normals.batch()                                                # [vertices] Vector-
    turned = ((loop * loop).log() | normals[sphere.faces[:, 0]].dual()).to_array()   # [faces]
    expected = (core.loop_turn(normals[sphere.faces]) + sphere.circulation([])(angles).batch()).to_array()
    np.testing.assert_allclose(np.angle(np.exp(1j * (turned - expected))), 0.0, atol=1e-12)


def test_transport_turns_by_the_solid_angle_and_stays_tangent():
    """Around a latitude circle the carried vector turns by the cap's solid angle; on the torus, heat
    diffusion and the harmonic modes stay tangent, and the modes' energies come in equal pairs."""
    sphere, sphere_vertices, rings, starts = scenarios.holonomy()
    carried = stack(tuple(core.transported(rings, starts)))
    np.testing.assert_allclose((carried & rings).to_array(), 0.0, atol=1e-12)
    np.testing.assert_allclose(carried[-1].dot(starts).to_array(), np.cos(2 * np.pi * (1 - np.cos(scenarios.COLATITUDES))), atol=2e-3)
    frames = list(render.animate_holonomy(sphere, sphere_vertices, rings, islice(scenarios.carried_around(rings, starts, 60), 2)))
    torus, vertices, diffused, transported, modes, energies, _ = scenarios.torus_fields()
    normals = core.vertex_normals(torus, vertices)                                # Vector-[vertices]
    np.testing.assert_allclose((diffused & normals).to_array(), 0.0, atol=1e-12)
    np.testing.assert_allclose((modes & normals).to_array(), 0.0, atol=1e-11)
    np.testing.assert_allclose(energies.to_array()[0::2], energies.to_array()[1::2], rtol=1e-10)
    plt.close(render.draw_torus_fields(torus, vertices, [transported, modes[0]]))
    assert len(frames) == 2 and len(list(render.animate_diffusion(torus, vertices, diffused[:1]))) == 1


def test_designed_fields_are_parallel_and_their_indices_add_up():
    """Every design's circulation is met and its parallel fields carry no energy; indices one short of
    the Euler characteristic leave one whole turn spread over the faces."""
    sphere, sphere_vertices, sphere_scenes, residuals, energies = scenarios.field_spheres()
    torus, torus_vertices, torus_scenes, torus_residuals, torus_energies = scenarios.field_tori()
    np.testing.assert_allclose(np.concatenate([residual.to_array() for residual in residuals + torus_residuals]), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.concatenate([pair.to_array() for pair in energies + torus_energies]), 0.0, atol=1e-12)
    angles, circulation, target = core.trivial_connection(sphere, core.vertex_normals(sphere, sphere_vertices), scenarios.face_indices(sphere, np.array([0]), [1]), [])
    np.testing.assert_allclose((circulation(angles) - target).to_array(), 2 * np.pi / len(sphere.faces), rtol=1e-9)
    plt.close(render.draw_fields(sphere, sphere_vertices, sphere_scenes))
    plt.close(render.draw_fields(torus, torus_vertices, torus_scenes))
    field, singular, indices = sphere_scenes[0]
    turned = scenarios.turning(sphere, sphere_vertices, field)[:1]
    assert len(list(render.animate_turning(sphere, sphere_vertices, turned, singular, indices))) == 1


def test_stripes_follow_their_targets_and_crosses_land_on_their_lines():
    """Across most edges the stripe phase advances by the edge's target, the stripe fields come in
    equal pairs, and each line of a cross lands along or against one of its neighbour's lines."""
    sphere, vertices, patterns, energies = scenarios.stripe_spheres()
    for (lines, stripes, advance), pair in zip(patterns, energies):
        np.testing.assert_allclose(pair.to_array()[:, 1], pair.to_array()[:, 0], rtol=1e-6)
        assert np.median(scenarios.stripe_mismatch(sphere, vertices, lines, stripes, advance), axis=0).max() < 0.25
    _, _, crosses, cross_angles = scenarios.crosses()
    normals = core.vertex_normals(sphere, vertices)
    for (lines, _, _), angles in zip(crosses, cross_angles):
        forward = core.adjusted_transport(sphere, normals, angles)[:, 0]
        match = lines[sphere.edges[:, 1]][:, :, None].dot((forward[:, None] >> lines[sphere.edges[:, 0]])[:, None, :]).to_array()
        np.testing.assert_allclose(np.abs(match) * (1 - np.abs(match)), 0, atol=1e-8)
    bands = [tuple(family[..., :1] for family in pattern) for pattern in patterns]
    plt.close(render.draw_bands(sphere, vertices, bands))
    plt.close(render.draw_lines(sphere, vertices, crosses))
    lines, stripes, advance = (family[..., :1] for family in patterns[0])
    assert len(list(render.animate_flow(sphere, vertices, lines, scenarios.sliding(sphere, vertices, stripes)[:1], advance))) == 1
