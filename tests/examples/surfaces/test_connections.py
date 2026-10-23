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
    sphere = core.Mesh.icosphere(2)
    angles = core.as_scalar(np.random.default_rng(0).normal(size=len(sphere.edges)) * 0.3).field()   # Scalar[E]
    forward = core.adjusted_transport(sphere, angles)[:, 0]                       # [E] Rotor: tail to head
    both = stack([forward, forward.reverse()], axis=-1)                           # [E, 2] Rotor: either way
    # Around each face, counter-clockwise from its first corner: the edges facing the third, first and
    # second corners, each taken forward where it runs that way and backward where it does not.
    around = [both[sphere.face_edges[:, corner], (sphere.face_edge_orientation[:, corner] < 0).astype(int)]
              for corner in (2, 0, 1)]
    loop = around[2] * around[1] * around[0]                                      # [F] Rotor
    normals = sphere.vertex_normals.batch()                                       # [V] Vector
    turned = core.turn(loop, normals[sphere.faces[:, 0]]).to_array()              # [F]
    expected = (core.loop_turn(normals[sphere.faces]) + core.loops(sphere, [])(angles).batch()).to_array()
    np.testing.assert_allclose(np.angle(np.exp(1j * (turned - expected))), 0.0, atol=1e-12)


def test_transport_turns_by_the_solid_angle_and_stays_tangent():
    """Around a latitude circle the carried vector turns by the cap's solid angle; on the torus, heat
    diffusion and the harmonic modes stay tangent, and the modes' energies come in equal pairs."""
    sphere, rings, starts = scenarios.holonomy()
    carried = stack(tuple(core.transported(rings, starts)))
    np.testing.assert_allclose((carried | rings).to_array(), 0.0, atol=1e-12)
    np.testing.assert_allclose((carried[-1] | starts).to_array(), np.cos(2 * np.pi * (1 - np.cos(scenarios.COLATITUDES))), atol=2e-3)
    frames = list(render.animate_holonomy(sphere, rings, islice(scenarios.carried_around(rings, starts, 60), 2)))
    torus, diffused, transported, modes, energies, _ = scenarios.torus_fields()
    np.testing.assert_allclose((diffused | torus.vertex_normals).to_array(), 0.0, atol=1e-12)
    np.testing.assert_allclose((modes | torus.vertex_normals).to_array(), 0.0, atol=1e-11)
    np.testing.assert_allclose(energies.to_array()[0::2], energies.to_array()[1::2], rtol=1e-10)
    plt.close(render.draw_torus_fields(torus, [transported, modes[0]]))
    assert len(frames) == 2 and len(list(render.animate_diffusion(torus, diffused[:1]))) == 1


def test_designed_fields_are_parallel_and_their_indices_add_up():
    """Every design's circulation is met and its parallel fields carry no energy; indices one short of
    the Euler characteristic leave one whole turn spread over the faces."""
    sphere, sphere_scenes, residuals, energies = scenarios.field_spheres()
    torus, torus_scenes, torus_residuals, torus_energies = scenarios.field_tori()
    np.testing.assert_allclose(np.concatenate([residual.to_array() for residual in residuals + torus_residuals]), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.concatenate([pair.to_array() for pair in energies + torus_energies]), 0.0, atol=1e-12)
    angles, circulation, target = core.trivial_connection(sphere, scenarios.face_indices(sphere, np.array([0]), [1]), [])
    np.testing.assert_allclose((circulation(angles) - target).to_array(), 2 * np.pi / len(sphere.faces), rtol=1e-9)
    plt.close(render.draw_fields(sphere, sphere_scenes, render.sphere_limits, render.sphere_needle))
    plt.close(render.draw_fields(torus, torus_scenes, render.torus_limits, render.torus_needle))
    field, singular, indices = sphere_scenes[0]
    assert len(list(render.animate_turning(sphere, scenarios.turning(sphere, field)[:1], singular, indices))) == 1


def test_stripes_follow_their_targets_and_crosses_land_on_their_lines():
    """Across most edges the stripe phase advances by the edge's target, the stripe fields come in
    equal pairs, and each line of a cross lands along or against one of its neighbour's lines."""
    sphere, patterns, energies = scenarios.stripe_spheres()
    for (lines, stripes, advance), pair in zip(patterns, energies):
        np.testing.assert_allclose(pair.to_array()[:, 1], pair.to_array()[:, 0], rtol=1e-6)
        assert np.median(scenarios.stripe_mismatch(sphere, lines, stripes, advance), axis=0).max() < 0.25
    _, crosses, cross_angles = scenarios.crosses()
    for (lines, _, _), angles in zip(crosses, cross_angles):
        forward = core.adjusted_transport(sphere, angles)[:, 0]
        match = (lines[sphere.edges[:, 1]][:, :, None] | (forward[:, None] >> lines[sphere.edges[:, 0]])[:, None, :]).to_array()
        np.testing.assert_allclose(np.abs(match) * (1 - np.abs(match)), 0, atol=1e-8)
    views = np.broadcast_to(render.view, (len(patterns), 2))
    bands = [tuple(family[..., :1] for family in pattern) for pattern in patterns]
    plt.close(render.draw_patterns(sphere, bands, render.draw_bands, render.sphere_reach, views))
    plt.close(render.draw_patterns(sphere, crosses, render.draw_lines, render.sphere_reach, views))
    lines, stripes, advance = (family[..., :1] for family in patterns[0])
    assert len(list(render.animate_flow(sphere, lines, scenarios.sliding(sphere, stripes)[:1], advance, render.sphere_reach))) == 1
