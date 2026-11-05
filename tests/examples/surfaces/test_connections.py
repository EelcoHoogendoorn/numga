"""Connections on curved surfaces: the transport, the designed fields, the stripes and the crosses run on the
icosahedron and a coarse torus, and one picture of the transport, the fields and the stripes draws."""

from __future__ import annotations

from itertools import islice

import matplotlib.pyplot as plt

from examples.surfaces.connections import render, scenarios


def test_the_scenes_run_and_draw():
    sphere_topology, sphere_vertices, rings, starts = scenarios.holonomy(sphere_levels=0, n_ring_steps=2)
    assert len(list(render.animate_holonomy(sphere_topology, sphere_vertices, rings,
                                            islice(scenarios.carried_around(rings, starts, 1), 1)))) == 1
    scenarios.torus_fields(n_major=6, n_minor=3, n_heat_steps=1, n_modes=2)

    topology, vertices, scenes, _, _ = scenarios.field_spheres(sphere_levels=0)
    plt.close(render.draw_fields(topology, vertices, scenes[:1]))
    field, _, _ = scenes[0]
    scenarios.turning(topology, vertices, field)

    topology, vertices, patterns, _ = scenarios.stripe_spheres(sphere_levels=0)
    plt.close(render.draw_bands(topology, vertices, [tuple(family[:1] for family in patterns[0])]))
    scenarios.crosses(sphere_levels=0)
