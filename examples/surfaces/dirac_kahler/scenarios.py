"""Scene setup and verification for Dirac–Kähler derivatives, Hodge decomposition, and wave scattering."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator

import numpy as np

from numga import NumpyContext, stack
from numga.algebras import VGA2D as ga
from examples.util import shapes
from numga.topology import Topology

mv = NumpyContext(ga).multivector
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
dt = 0.00533
n_steps = 300
stride = 5
frame_duration_ms = 40


# --- plumbing -------------------------------------------------------------------------
def perforated_annulus(
    hole_radius: float, rim_radius: float, n_rings: int,
) -> tuple[Topology, Point, np.ndarray, np.ndarray]:
    """Staggered circular vertex rings from `hole_radius` to `rim_radius` triangulate a perforated annulus (`Point-[V]`)."""
    radii = np.linspace(hole_radius, rim_radius, n_rings)
    spacing = (rim_radius - hole_radius) / (n_rings - 1)
    edge_length = 2 * spacing / np.sqrt(3)
    n_fewest_per_ring = 18
    populations = np.maximum(n_fewest_per_ring, np.rint(2 * np.pi * radii / edge_length).astype(int))
    ring = np.repeat(np.arange(len(radii)), populations)
    offsets = np.repeat(np.cumsum(populations) - populations, populations)
    angles = 2 * np.pi * (np.arange(populations.sum()) - offsets + (ring % 2) / 2) / populations[ring]
    ring_turns = (mv.xy * (-angles / 2)).exp()                               # [...] Rotor+
    directions = ring_turns >> mv.x.dual()                                   # [...] Point-
    vertices = (directions * radii[ring]).field()                            # Point-[V]
    # Build the Delaunay triangulation from coordinate pairings across x and y, then drop triangles spanning the hole.
    raw_topology, _ = shapes.delaunay((stack((mv.x, mv.y)) & vertices.batch()[..., None]).to_array())
    triangles = raw_topology.elements[2]
    keep = ~np.all(ring[triangles] == 0, axis=-1)
    topology = raw_topology.subset(keep, -1)
    inner_ring = np.flatnonzero(ring == 0)
    interior = np.flatnonzero((ring > 0) & (ring < n_rings - 1))
    return topology, vertices, inner_ring, interior


def save_states(
    states: Iterable, animate: Callable[..., Iterable[np.ndarray]], *arguments,
    name: str, duration_ms: int,
) -> list:
    """Streams `states` once through `animate(*arguments, states)` into `save_animation` and returns the collected states."""
    from examples.util.animation import save_animation

    recorded = []

    def passing() -> Iterator:
        for state in states:
            recorded.append(state)
            yield state

    save_animation(animate(*arguments, passing()), name, duration_ms)
    return recorded


def main() -> None:
    from examples.surfaces.dirac_kahler.core import (
        dirac_derivatives,
        harmonic_circulation,
        hodge_decomposition,
        medium_operators,
        simplex_geometry,
        wave_divergence,
        wave_states,
        whitney_maps,
    )
    from examples.surfaces.dirac_kahler import render
    from examples.util.animation import save_figure

    # 1. Mixed-grade Whitney reconstruction and the single-product Dirac-Kahler derivative on a triangulated disc.
    disk_topology, disk_coordinates = shapes.hexagonal_disk(disk_radius, n_disk_divisions)
    disk_vertices = mv.vector(disk_coordinates).dual().field()               # Point-[V]
    _, disk_patches, _, disk_gradients = simplex_geometry(disk_topology, disk_vertices)
    exterior_curl, whitney0, whitney1, whitney2 = whitney_maps(disk_topology, disk_patches, disk_gradients)
    disk_gradient, dirac, disk_even_gradient = dirac_derivatives(disk_topology, disk_gradients)

    x, y = mv.x & disk_vertices, mv.y & disk_vertices                        # Scalar+[V], Scalar+[V]
    scalar_potential = 0.5 * (x.squared() - y.squared()) + 0.35 * x          # Scalar+[V]
    vertex_1form = (0.6 * x - 0.8 * y) * mv.x + (0.8 * x + 0.6 * y) * mv.y   # Plane+[V]
    stream_bivector = (x * y - 0.35 * y) * mv.xy                             # Bivector+[V]
    vertex_multivector = scalar_potential + vertex_1form + stream_bivector   # Full+[V]
    scalar_gradient = disk_gradient(scalar_potential)                        # Plane+[F]
    face_dirac = dirac(vertex_multivector)                                   # Full+[F]

    edge_midpoints = disk_topology.incidence[1, 0].mean(axis=0) * disk_vertices  # Point-[E]
    midpoint_x, midpoint_y = mv.x & edge_midpoints, mv.y & edge_midpoints    # Scalar+[E], Scalar+[E]
    midpoint_1form = (0.6 * midpoint_x - 0.8 * midpoint_y) * mv.x + (0.8 * midpoint_x + 0.6 * midpoint_y) * mv.y  # Plane+[E]
    edge_cochain = midpoint_1form & (disk_topology.d[0] * disk_vertices)     # Scalar+[E]
    face_cochain = exterior_curl(edge_cochain)                               # Scalar+[F]
    whitney_multivector = (
        whitney0(scalar_potential)
        + whitney1(edge_cochain)
        + whitney2(face_cochain)
    )                                                                        # Full+[F]
    save_figure(
        render.grades_figure(
            disk_topology, disk_vertices, vertex_multivector,
            scalar_gradient, face_dirac, whitney_multivector,
        ),
        "dirac_kahler_grades",
    )

    # 2. Three-way Hodge-Helmholtz decomposition of a velocity on a perforated annulus.
    topology, vertices, inner_ring, interior = perforated_annulus(inner_radius, outer_radius, n_annulus_rings)
    _, patches, areas, gradients = simplex_geometry(topology, vertices)
    annulus_gradient, _, even_gradient = dirac_derivatives(topology, gradients)
    (
        isotropic_medium, isotropic_resistance, potential_velocity,
        stream_velocity, even_velocity, stream_laplacian, even_laplacian,
    ) = medium_operators(topology, patches, areas, annulus_gradient, even_gradient)

    squared_radii = vertices.norm_squared()                                  # Scalar+[V]
    bubble = (squared_radii - inner_radius**2) * (outer_radius**2 - squared_radii)  # Scalar+[V]
    abscissa, ordinate = mv.x & vertices, mv.y & vertices                    # Scalar+[V], Scalar+[V]
    true_potential = bubble * (-2.2 * abscissa + 0.9 * ordinate)             # Scalar+[V]
    true_stream = bubble * (1.6 * abscissa + 1.9 * ordinate)                 # Scalar+[V]
    _, true_harmonic = harmonic_circulation(topology, stream_velocity, stream_laplacian, inner_ring, interior)
    total_flow = (
        potential_velocity(true_potential)
        + stream_velocity(true_stream)
        + circulation_strength * true_harmonic
    )                                                                        # Point-[F]
    potential, stream, potential_flow, stream_flow, harmonic_flow = hodge_decomposition(
        topology, even_gradient, even_velocity, even_laplacian, total_flow, interior,
    )
    save_figure(
        render.hodge_figure(
            topology, vertices, total_flow, potential, potential_flow, stream_flow, harmonic_flow,
        ),
        "dirac_kahler_hodge",
    )

    # 3. Coupled acoustic-vortical Dirac-Kahler wave dynamics around the central obstacle.
    dual_areas, even_divergence = wave_divergence(topology, areas, even_gradient, isotropic_medium)
    envelope = (-(vertices - pulse_centre).norm_squared() / (2 * pulse_width**2)).exp()  # Scalar+[V]
    initial_even = envelope.cast(Even)                                       # Even+[V]
    initial_face_field = even_gradient(initial_even * 0.0)                   # Plane+[F]
    recorded_waves = save_states(
        wave_states(even_gradient, even_divergence, initial_even, initial_face_field, dt, n_steps, stride),
        render.waves_animation,
        topology,
        vertices,
        name="dirac_kahler_waves",
        duration_ms=frame_duration_ms,
    )

    # checks
    # Grade-by-grade identity of the single geometric product dirac(vertex_multivector).
    corner_1form = disk_topology.incidence[2, 0] * vertex_1form              # [3] Plane+[F]
    corner_stream = disk_topology.incidence[2, 0] * stream_bivector          # [3] Bivector+[F]
    div_1form = (disk_gradients | corner_1form).sum(axis=0)                  # Scalar+[F]
    grad_plus_rot = scalar_gradient + (disk_gradients | corner_stream).sum(axis=0)  # Plane+[F]
    curl_1form = (disk_gradients ^ corner_1form).sum(axis=0)                 # Bivector+[F]
    assert ((face_dirac.select[0] - div_1form).abs() < 1e-12).all()
    assert ((face_dirac.select[1] - grad_plus_rot).scalar_norm_squared() < 1e-24).all()
    assert ((face_dirac.select[1] - disk_even_gradient(scalar_potential + stream_bivector)).scalar_norm_squared() < 1e-24).all()
    assert ((face_dirac.select[2] - curl_1form).scalar_norm_squared() < 1e-24).all()
    assert ((whitney_multivector.select[2] - face_dirac.select[2]).scalar_norm_squared() < 1e-24).all()

    # Exact recovery and pairwise energy orthogonality of the 3-way Hodge-Helmholtz decomposition.
    assert ((potential - true_potential).abs() < 1e-11).all()
    assert ((stream - true_stream).abs() < 1e-11).all()
    assert ((harmonic_flow - circulation_strength * true_harmonic).scalar_norm_squared() < 1e-22).all()
    assert (isotropic_resistance(potential_flow) & stream_flow).sites.sum().abs() < 1e-11
    assert (isotropic_resistance(potential_flow) & harmonic_flow).sites.sum().abs() < 1e-11
    assert (isotropic_resistance(stream_flow) & harmonic_flow).sites.sum().abs() < 1e-11

    # Skew-adjoint wave energy conservation and boundary conversion on the recorded states.
    energies = stack([
        ((dual_areas * even_state.scalar_norm_squared()).sites.sum()
         + (areas * face_field.scalar_norm_squared()).sites.sum()) / 2
        for even_state, face_field in recorded_waves
    ])                                                                       # [...] Scalar+
    final_vortical_energy = (dual_areas * recorded_waves[-1][0].select[2].scalar_norm_squared()).sites.sum() / 2  # [] Scalar+
    assert ((energies / energies[0] - 1.0).abs() < 1e-4).all()
    assert final_vortical_energy > 1e-4


if __name__ == "__main__":
    main()
