"""Coverage tests for surface connections, parallel transport, trivial connections, and stripes."""

from __future__ import annotations

from itertools import islice

import matplotlib.pyplot as plt
import pytest

from numga import stack
from examples.surfaces.connections import core, scenarios

tolerance = 1e-9


def test_connections_math_and_rendering():
    sphere_topology, sphere_vertices, rings, starts = scenarios.holonomy_scene(0, 4)
    carried_tangents = stack(tuple(core.parallel_transport(rings, starts)))       # [5, 3] Point-
    assert ((carried_tangents & rings).antinorm() < tolerance).all()
    assert ((carried_tangents.norm() - 1).abs() < tolerance).all()

    (torus_topology, torus_vertices, torus_patches, torus_areas,
     _, torus_source, _, laplacian) = scenarios.torus_setup(6, 3)
    diffused_field, = core.heat_diffusion(laplacian, torus_areas, torus_source, scenarios.step_time, 1)
    assert ((diffused_field & torus_patches).batch().antinorm() < tolerance).all()
    mode_energies, modes = core.low_modes(laplacian, torus_patches, torus_areas, scenarios.normal_stiffness, 2)
    assert ((mode_energies[0] - mode_energies[1]).abs() < tolerance).all()
    assert ((modes & torus_patches).batch().antinorm() < tolerance).all()

    topology, vertices, scenes, residuals, energies = scenarios.sphere_field_scenes(0)
    assert (residuals[1].batch().abs() < tolerance).all()
    assert (energies[1].abs() < tolerance).all()
    field, _, _ = scenes[1]
    sphere_patches = core.vertex_patches(topology, vertices)
    turned_frame, = islice(core.turned_fields(sphere_patches, field, 4), 1)
    assert ((turned_frame & sphere_patches).batch().antinorm() < tolerance).all()

    stripe_topology, stripe_vertices, patterns, stripe_energies = scenarios.sphere_stripe_scenes(0)
    assert ((stripe_energies[1][:, 1] - stripe_energies[1][:, 0]).abs() < 1e-5).all()
    _, _, cross_patterns, cross_angles = scenarios.cross_stripe_scenes(0)
    stripe_patches = core.vertex_patches(stripe_topology, stripe_vertices)
    cross_lines, cross_stripes, _ = cross_patterns[0]
    assert ((cross_lines & stripe_patches).batch().antinorm() < tolerance).all()
    assert ((cross_stripes & stripe_patches).batch().antinorm() < tolerance).all()
    assert len(cross_angles) == 2

    pytest.importorskip("wgpu")
    from examples.surfaces.connections import render

    holonomy_states = islice(zip(rings, core.parallel_transport(rings, starts)), 1)
    assert len(list(render.holonomy_animation(sphere_topology, sphere_vertices, rings, holonomy_states))) == 1
    assert len(list(render.diffusion_animation(torus_topology, torus_vertices, [diffused_field]))) == 1
    assert len(list(render.turning_animation(topology, vertices, [turned_frame]))) == 1
    plt.close(render.torus_fields_figure(torus_topology, torus_vertices, [modes[0]]))
    plt.close(render.fields_figure(topology, vertices, scenes[:1]))
    single_band = [tuple(family[:1] for family in patterns[0])]
    plt.close(render.bands_figure(stripe_topology, stripe_vertices, single_band))
    plt.close(render.lines_figure(stripe_topology, stripe_vertices, cross_patterns[:1], scenarios.cross_scale))
    band_lines, band_stripes, band_advance = single_band[0]
    assert len(list(render.sliding_animation(stripe_topology, stripe_vertices, band_lines, [band_stripes], band_advance))) == 1
