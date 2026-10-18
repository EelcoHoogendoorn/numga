"""Connections on curved surfaces: a loop's holonomy grows by the circulation of the edges' extra turns,
the scenes of parallel transport, trivial connections and stripe patterns pass their checks, and the
figures and animations draw."""

from __future__ import annotations

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


def test_the_transport_scenes_pass_their_checks_and_draw():
    sphere, rings, carried = scenarios.holonomy()
    plt.close(render.draw_holonomy(sphere, rings, carried))
    torus, diffused, transported, modes = scenarios.torus_fields()
    plt.close(render.draw_torus_fields(torus, [transported, modes[0]]))
    assert len(render.animate_diffusion(torus, diffused[:1])) == 1


def test_the_field_scenes_pass_their_checks_and_draw():
    sphere, sphere_scenes = scenarios.field_spheres()
    plt.close(render.draw_fields(sphere, sphere_scenes, render.SPHERE_LIMITS, render.SPHERE_NEEDLE))
    torus, torus_scenes = scenarios.field_tori()
    plt.close(render.draw_fields(torus, torus_scenes, render.TORUS_LIMITS, render.TORUS_NEEDLE))
    field, singular, indices = sphere_scenes[0]
    assert len(render.animate_turning(sphere, scenarios.turning(sphere, field)[:1], singular, indices)) == 1


def test_the_stripe_scenes_pass_their_checks_and_draw():
    sphere, patterns = scenarios.stripe_spheres()
    views = np.broadcast_to(render.VIEW, (len(patterns), 2))
    bands = [tuple(family[..., :1] for family in pattern) for pattern in patterns]
    plt.close(render.draw_patterns(sphere, bands, render.draw_bands, render.SPHERE_REACH, views))
    sphere, crosses = scenarios.crosses()
    plt.close(render.draw_patterns(sphere, crosses, render.draw_lines, render.SPHERE_REACH, views))
    lines, stripes, advance = (family[..., :1] for family in patterns[0])
    assert len(render.animate_flow(sphere, lines, scenarios.sliding(sphere, stripes)[:1], advance, render.SPHERE_REACH)) == 1
