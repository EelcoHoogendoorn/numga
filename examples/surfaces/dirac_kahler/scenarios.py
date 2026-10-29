"""A disk and a perforated annulus carry mixed-grade Dirac–Kähler fields, a Hodge decomposition, and scattered waves."""

from __future__ import annotations

from itertools import islice

import numpy as np

from numga import concatenate, stack
from numga.algebras import VGA2D as ga
from examples import instantiate

mesh_module = instantiate("examples.util.mesh", ga)
Mesh, as_scalar, mv = mesh_module.Mesh, mesh_module.as_scalar, mesh_module.mv
Even = ga.gatype.even()                                                      # Even+
Point = ga.gatype.antivector()                                               # Antivector-

horizontal, vertical = mv.x.dual(), mv.y.dual()                              # [] Point-, [] Point-
n_disk_divisions = 9
disk_radius = 1.0
inner_radius = 0.32
outer_radius = 1.0
n_annulus_rings = 15
circulation_strength = 0.65
pulse_centre = -0.58 * horizontal + 0.18 * vertical                          # [] Point-
pulse_width = 0.15
dt = 0.005
n_steps = 320
stride = 8
frame_duration_ms = 60


# --- plumbing -------------------------------------------------------------------------
def disk(radius: float, n_divisions: int) -> tuple[Mesh, Point]:
    """The disk is triangulated into nearly equilateral triangles, with its first vertex at the origin."""
    spacing = radius / n_divisions
    indices = np.arange(-n_divisions, n_divisions + 1)
    columns, rows = np.meshgrid(indices, indices, indexing="ij")
    layers = np.maximum.reduce((abs(columns), abs(rows), abs(columns + rows)))
    inside = (layers > 0) & (layers <= n_divisions)
    lattice = np.stack((columns[inside], rows[inside]), axis=-1)             # [nodes, axes]
    unit = mv.x.dual()                                                       # [] Point-
    axes = stack((unit, (mv.xy * (-np.pi / 6)).exp() >> unit))               # [axes] Point-
    positions = (axes * lattice).sum(axis=-1)                                # [nodes] Point-
    positions = spacing * layers[inside] * positions.normalized()            # [nodes] Point-
    vertices = concatenate((unit[None] * 0, positions)).field()              # Point-[vertices]
    return Mesh.triangulate(vertices), vertices


def annulus(hole_radius: float, rim_radius: float,
            n_rings: int) -> tuple[Mesh, Point, np.ndarray, np.ndarray]:
    """The perforated disk has staggered circular rings of vertices from `hole_radius` to `rim_radius`."""
    radii = np.linspace(hole_radius, rim_radius, n_rings)
    spacing = (rim_radius - hole_radius) / (n_rings - 1)
    edge_length = 2 * spacing / np.sqrt(3)
    n_fewest_per_ring = 18
    populations = np.maximum(n_fewest_per_ring, np.rint(2 * np.pi * radii / edge_length).astype(int))
    ring = np.repeat(np.arange(len(radii)), populations)
    offsets = np.repeat(np.cumsum(populations) - populations, populations)
    angles = 2 * np.pi * (np.arange(populations.sum()) - offsets + (ring % 2) / 2) / populations[ring]
    directions = (mv.xy * (-angles / 2)).exp() >> mv.x.dual()                # [vertices] Point-
    vertices = (directions * radii[ring]).field()                            # Point-[vertices]
    raw_mesh = Mesh.triangulate(vertices)
    # Every Delaunay triangle spanning the central hole has all three corners on the innermost ring.
    keep = ~np.all(ring[raw_mesh.faces] == 0, axis=-1)
    mesh = Mesh(len(vertices.batch()), raw_mesh.faces[keep])
    inner_ring = np.flatnonzero(ring == 0)
    interior = np.flatnonzero((ring > 0) & (ring < n_rings - 1))
    return mesh, vertices, inner_ring, interior


def main() -> None:
    from examples.surfaces.dirac_kahler.core import (
        Operators, harmonic_circulation, hodge_decomposition, propagate,
    )
    from examples.surfaces.dirac_kahler import render
    from examples.util.animation import save_animation, save_figure

    # 1. Mixed-grade Whitney reconstruction and the single-product Dirac-Kahler derivative on a disk.
    disk_mesh, disk_vertices = disk(disk_radius, n_disk_divisions)
    disk_ops = Operators(disk_mesh, disk_vertices)
    x, y = mv.x & disk_vertices, mv.y & disk_vertices                        # Scalar+[vertices], Scalar+[vertices]
    scalar_potential = 0.5 * (x.squared() - y.squared()) + 0.35 * x          # Scalar+[vertices]
    vertex_1form = (0.6 * x - 0.8 * y) * mv.x + (0.8 * x + 0.6 * y) * mv.y   # Plane+[vertices]
    stream_bivector = (x * y - 0.35 * y) * mv.xy                             # Bivector+[vertices]
    vertex_multivector = scalar_potential + vertex_1form + stream_bivector   # Full+[vertices]
    scalar_gradient = disk_ops.gradient(scalar_potential)                    # Plane+[faces]
    face_dirac = disk_ops.dirac(vertex_multivector)                          # Full+[faces]

    edge_midpoints = disk_mesh.incidence[1, 0].mean(axis=0) * disk_vertices  # Point-[edges]
    mx, my = mv.x & edge_midpoints, mv.y & edge_midpoints                    # Scalar+[edges], Scalar+[edges]
    midpoint_1form = (0.6 * mx - 0.8 * my) * mv.x + (0.8 * mx + 0.6 * my) * mv.y  # Plane+[edges]
    edge_cochain = midpoint_1form & (disk_mesh.d[0] * disk_vertices)         # Scalar+[edges]
    face_cochain = disk_ops.d1(edge_cochain)                                 # Scalar+[faces]
    whitney_multivector = (
        disk_ops.whitney0(scalar_potential)
        + disk_ops.whitney1(edge_cochain)
        + disk_ops.whitney2(face_cochain)
    )                                                                        # Full+[faces]
    save_figure(
        render.draw_dirac_grades(disk_mesh, disk_vertices, vertex_multivector,
                                 scalar_gradient, face_dirac, whitney_multivector),
        "dirac_kahler_grades",
    )

    # 2. Three-way Hodge-Helmholtz decomposition of a velocity on a perforated disk.
    mesh, vertices, inner_ring, interior = annulus(inner_radius, outer_radius, n_annulus_rings)
    operators = Operators(mesh, vertices)
    squared_radii = vertices.norm_squared()                                  # Scalar+[vertices]
    bubble = (squared_radii - inner_radius**2) * (outer_radius**2 - squared_radii)  # Scalar+[vertices]
    abscissa, ordinate = mv.x & vertices, mv.y & vertices                    # Scalar+[vertices], Scalar+[vertices]
    true_potential = bubble * (-2.2 * abscissa + 0.9 * ordinate)             # Scalar+[vertices]
    true_stream = bubble * (1.6 * abscissa + 1.9 * ordinate)                 # Scalar+[vertices]
    harmonic_stream, true_harmonic = harmonic_circulation(operators, inner_ring, interior)
    total_flow = (operators.potential_velocity(true_potential) + operators.stream_velocity(true_stream)
                  + circulation_strength * true_harmonic)                    # Point-[faces]
    potential, stream, potential_flow, stream_flow, harmonic_flow = hodge_decomposition(
        operators, total_flow, interior,
    )
    save_figure(
        render.draw_hodge_decomposition(
            mesh, vertices, total_flow, potential, stream, potential_flow, harmonic_stream,
        ),
        "dirac_kahler_hodge",
    )

    # 3. Coupled acoustic-vortical Dirac-Kahler wave dynamics around the central obstacle.
    envelope = (-(vertices - pulse_centre).norm_squared() / (2 * pulse_width**2)).exp()  # Scalar+[vertices]
    initial_even = envelope.cast(Even)                                       # Even+[vertices]
    initial_face_field = operators.even_gradient(initial_even * 0.0)         # Plane+[faces]

    energy_trace = []
    for even_state, face_field in propagate(operators, initial_even, initial_face_field, dt, n_steps):
        even_energy = (operators.vertex_areas * even_state.scalar_norm_squared()).sites.sum()
        field_energy = (operators.areas * face_field.scalar_norm_squared()).sites.sum()
        energy_trace.append((even_energy + field_energy) / 2)
    energies = stack(energy_trace)                                           # [steps + 1] Scalar+

    sampled_frames = islice(propagate(operators, initial_even, initial_face_field, dt, n_steps), 0, None, stride)
    save_animation(
        render.animate_dirac_waves(mesh, vertices, sampled_frames),
        "dirac_kahler_waves",
        frame_duration_ms,
    )

    # checks
    # Grade-by-grade identity of the single geometric product dirac(vertex_multivector).
    u_corners = disk_mesh.incidence[2, 0] * vertex_1form
    omega_corners = disk_mesh.incidence[2, 0] * stream_bivector
    div_u = (disk_ops.corner_gradients | u_corners).sum(axis=0)
    grad_phi_plus_rot_omega = disk_ops.gradient(scalar_potential) + (disk_ops.corner_gradients | omega_corners).sum(axis=0)
    curl_u = (disk_ops.corner_gradients ^ u_corners).sum(axis=0)
    np.testing.assert_allclose((face_dirac.select[0] - div_u).kernel, 0.0, atol=1e-14)
    np.testing.assert_allclose((face_dirac.select[1] - grad_phi_plus_rot_omega).kernel, 0.0, atol=1e-14)
    np.testing.assert_allclose(
        (face_dirac.select[1] - disk_ops.even_gradient(scalar_potential + stream_bivector)).kernel,
        0.0, atol=1e-14,
    )
    np.testing.assert_allclose((face_dirac.select[2] - curl_u).kernel, 0.0, atol=1e-14)
    np.testing.assert_allclose((whitney_multivector.select[2] - face_dirac.select[2]).kernel, 0.0, atol=1e-13)

    # Exact recovery and pairwise energy orthogonality of the 3-way Hodge-Helmholtz decomposition.
    np.testing.assert_allclose((potential - true_potential).kernel, 0.0, atol=1e-12)
    np.testing.assert_allclose((stream - true_stream).kernel, 0.0, atol=1e-12)
    np.testing.assert_allclose((harmonic_flow - circulation_strength * true_harmonic).kernel, 0.0, atol=1e-12)
    np.testing.assert_allclose((operators.isotropic_resistance(potential_flow) & stream_flow).sites.sum().kernel, 0.0, atol=1e-12)
    np.testing.assert_allclose((operators.isotropic_resistance(potential_flow) & harmonic_flow).sites.sum().kernel, 0.0, atol=1e-12)
    np.testing.assert_allclose((operators.isotropic_resistance(stream_flow) & harmonic_flow).sites.sum().kernel, 0.0, atol=1e-12)

    # Skew-adjoint power balance and RK4 wave energy conservation.
    np.testing.assert_allclose((energies / energies[0]).kernel, 1.0, rtol=1e-4)


if __name__ == "__main__":
    main()
