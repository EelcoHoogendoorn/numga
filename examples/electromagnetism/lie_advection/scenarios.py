"""Moving-triangle and swirling-disk parameters configure the magnetic Lie advection figures, animation, and checks."""

from __future__ import annotations

from collections.abc import Iterator
from itertools import islice

import numpy as np

from numga import stack
from examples.electromagnetism.lie_advection.core import Scalar, Upright, mv
from examples.util import shapes

horizontal, vertical = mv.x.dual(), mv.y.dual()                              # [] Point-, Point-
ribbon_step = 0.55
triangle_corners = stack((horizontal * 0.0 + vertical * 0.0,
                          horizontal * 1.15 + vertical * 0.1,
                          horizontal * 0.34 + vertical * 0.92)).field()      # Point-[3]
corner_potentials = mv.vector([[0.45, -0.65, 0.0],
                               [1.35, 0.25, 0.0],
                               [-0.55, 1.15, 0.0]]).field()                  # Plane+[3]
corner_velocities = stack((0.48 * horizontal + 0.26 * vertical,
                           0.62 * horizontal + 0.34 * vertical,
                           0.38 * horizontal + 0.44 * vertical)).field()     # Point-[3]

n_disk_divisions = 16
disk_radius = 1.0
swirl_rate = 5.2
ellipticity = 0.2
n_lobes = 2
lobe_centres = stack((-0.36 * horizontal - 0.06 * vertical,
                      0.36 * horizontal + 0.06 * vertical))                  # [n_lobes] Point-
lobe_strengths = np.array([1.0, -0.65])                                      # [n_lobes]
lobe_radius = 0.28
n_cases = 2
resistivities = np.array([0.0, 0.015])                                       # [n_cases]
dt = 0.0027
n_steps = 356
n_steps_per_frame = 4
n_frames = n_steps // n_steps_per_frame + 1
frame_duration_ms = 40


def main() -> None:
    """The Cartan ribbon figure, swirling-disk animation, and conservation comparison save to disk before running checks."""
    from examples.electromagnetism.lie_advection import render
    from examples.electromagnetism.lie_advection.core import (
        advection_map, cartan_decomposition, cochain_trajectory,
        curl_laplacian_map, curl_stiffness_map, flux_density_map,
        initial_cochain, swept_edge_patches, time, triangle,
        vortex_streamfunction, whitney_curl,
    )
    from examples.util.animation import save_animation, save_figure

    # 1. Edge world sheets and Cartan's decomposition on a moving triangle.
    endpoint_differences, swept_fluxes, _, face_field = cartan_decomposition(
        triangle_corners, corner_potentials, corner_velocities,
    )
    save_figure(
        render.cartan_ribbon_figure(triangle_corners, corner_potentials, corner_velocities,
                                    ribbon_step, endpoint_differences, swept_fluxes),
        "lie_advection_cartan",
    )

    # 2. Side-by-side ideal and resistive magnetic flux advection on a swirling disk.
    topology, coordinates = shapes.hexagonal_disk(disk_radius, n_disk_divisions)
    vertices = mv(Upright, coordinates).dual().field()                       # Point-[V]
    stream = vortex_streamfunction(vertices, swirl_rate, ellipticity)        # Scalar+[V]
    start_cochain = initial_cochain(topology, vertices, lobe_centres, lobe_strengths, lobe_radius)  # Scalar+[E]
    curl = whitney_curl(topology, vertices)                                  # (Magnetic+ <- Scalar+)[F <- E]
    swept_patches = swept_edge_patches(topology, vertices, stream)           # Patch-[E]
    advection = advection_map(topology, vertices, curl, swept_patches)       # (Scalar+ <- Scalar+)[E <- E]
    curl_stiffness = curl_stiffness_map(topology, vertices, curl)            # (Antiscalar- <- Scalar+)[E <- E]
    curl_laplacian = curl_laplacian_map(topology, curl_stiffness)            # (Scalar+ <- Scalar+)[E <- E]
    flux_density = flux_density_map(topology, vertices)                      # (Scalar+ <- Scalar+)[F <- E]

    rate = advection - curl_laplacian * resistivities[:, None]               # [n_cases] (Scalar+ <- Scalar+)[E <- E]
    batched_start = stack((start_cochain, start_cochain))                    # [n_cases] Scalar+[E]

    sampled_cochains = []

    def streamed_cochains() -> Iterator[Scalar]:
        for state in islice(cochain_trajectory(batched_start, rate, dt, n_steps), 0, None, n_steps_per_frame):
            sampled_cochains.append(state)
            yield state

    save_animation(
        render.advection_frames(topology, vertices, flux_density, streamed_cochains()),
        "lie_advection_flow",
        frame_duration_ms,
    )

    # 3. Energy dissipation and conserved total flux from the streamed cochains.
    cochain_states = stack(sampled_cochains)                                 # [n_frames, n_cases] Scalar+[E]
    d1 = topology.d[1] * Scalar                                              # (Scalar+ <- Scalar+)[F <- E]
    energies = (cochain_states & curl_stiffness(cochain_states)).sites.sum() / 2  # [n_frames, n_cases] Scalar+
    total_fluxes = d1(cochain_states).sites.sum()                            # [n_frames, n_cases] Scalar+
    final_flux_density = flux_density(cochain_states[-1])                    # [n_cases] Scalar+[F]
    frame_times = np.arange(n_frames) * (n_steps_per_frame * dt)
    save_figure(
        render.advection_comparison_figure(topology, vertices, final_flux_density,
                                           frame_times, energies, total_fluxes),
        "lie_advection_snapshots",
    )

    # checks
    # Associativity of the regressive product equates world-sheet flux to the interior-product line integral,
    # and endpoint differences cancel around the closed triangle boundary.
    triangle_sides = triangle.boundary[2, 1] * (triangle.d[0] * triangle_corners)
    edge_motions = triangle.incidence[2, 1] * (triangle.incidence[1, 0].mean(axis=0) * (time + corner_velocities))
    interior_line_integrals = (face_field & edge_motions) & triangle_sides
    assert ((swept_fluxes - interior_line_integrals).abs() < 1e-12).all()
    assert (endpoint_differences.sum(axis=0).abs() < 1e-12).all()

    # Across every edge, the swept spatial patch area equals minus the streamfunction difference.
    d0 = topology.d[0] * Scalar
    assert (((mv.xy & swept_patches) + d0(stream)).abs() < 1e-12).all()

    # Ideal advection produces zero instantaneous magnetic power and conserves flux and energy,
    # while interior resistance dissipates magnetic energy monotonically.
    ideal_initial_rate = advection(start_cochain)
    assert ((ideal_initial_rate & curl_stiffness(start_cochain)).sites.sum().abs() < 1e-12).all()
    assert ((total_fluxes - total_fluxes[0]).abs() < 1e-12).all()
    assert (((energies[:, 0] / energies[0, 0]) - 1).abs() < 1e-4).all()
    assert ((energies[1:, 1] - energies[:-1, 1]) < 0).all()


if __name__ == "__main__":
    main()
