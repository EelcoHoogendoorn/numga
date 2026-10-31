"""Moving triangle ribbons and disk meshes for discrete Lie advection of magnetic flux."""

from __future__ import annotations

from itertools import islice

import numpy as np

from numga import concatenate, stack
from numga.algebras import STA2D as ga
from numga.sparse import spdiag
from examples import instantiate

mesh_module = instantiate("examples.util.mesh", ga)
Mesh, as_scalar, mv = mesh_module.Mesh, mesh_module.as_scalar, mesh_module.mv
Scalar = ga.gatype.scalar()                                                  # Scalar+
Bivector = ga.gatype.bivector()                                              # Bivector+
Magnetic = Bivector.derive.within("x y")                                     # Bivector+
Point = ga.gatype.antivector()                                               # Antivector-

horizontal, vertical = mv.x.dual(), mv.y.dual()                              # [] Point-, Point-
ribbon_step = 0.55
corner_forms = mv.vector([[0.45, -0.65, 0.0], [1.35, 0.25, 0.0], [-0.55, 1.15, 0.0]])  # [corners] Plane+
corner_velocities = stack((0.48 * horizontal + 0.26 * vertical,
                           0.62 * horizontal + 0.34 * vertical,
                           0.38 * horizontal + 0.44 * vertical))             # [corners] Point-

n_disk_divisions = 16
disk_radius = 1.0
swirl_rate = 5.2
ellipticity = 0.2
lobe_centres = stack((-0.36 * horizontal - 0.06 * vertical,
                      0.36 * horizontal + 0.06 * vertical))                  # [lobes] Point-
lobe_strengths = np.array([1.0, -0.65])                                      # [lobes]
lobe_radius = 0.28
resistivities = np.array([0.0, 0.015])                                       # [cases]
dt = 0.0027
n_steps = 356
n_steps_per_frame = 4
frame_duration_ms = 40


# --- plumbing -------------------------------------------------------------------------
def single_triangle() -> Point:
    """A scalene triangle's corners."""
    corners = stack((horizontal * 0.0 + vertical * 0.0,
                     horizontal * 1.15 + vertical * 0.1,
                     horizontal * 0.34 + vertical * 0.92))                   # [corners] Point-
    return corners


def disk(radius: float, n_divisions: int) -> tuple[Mesh, Point]:
    """A circular disk of nearly equilateral triangles with its first vertex at the origin."""
    spacing = radius / n_divisions
    indices = np.arange(-n_divisions, n_divisions + 1)
    columns, rows = np.meshgrid(indices, indices, indexing="ij")
    layers = np.maximum.reduce((abs(columns), abs(rows), abs(columns + rows)))
    inside = (layers > 0) & (layers <= n_divisions)
    lattice = np.stack((columns[inside], rows[inside]), axis=-1)             # [nodes, axes]
    unit = mv.x.dual()                                                       # [] Point-
    axes = stack((unit, (mv.xy * (-np.pi / 6)).exp() >> unit))               # [axes] Point-
    positions = (axes * lattice).sum(axis=-1)                                # [nodes] Point-
    positions = spacing * layers[inside] * positions / positions.squared().square_root()  # [nodes] Point-
    vertices = concatenate((unit[None] * 0, positions)).field()              # Point-[vertices]
    return Mesh.triangulate(vertices), vertices


def main() -> None:
    from examples.electromagnetism.lie_advection.core import (
        Operators, advect, cartan_decomposition, edge_world_sheets,
        initial_cochain, time, triangle, vortex_streamfunction,
    )
    from examples.electromagnetism.lie_advection import render
    from examples.util.animation import save_animation, save_figure

    # 1. Edge world sheets and Cartan's formula on a moving triangle.
    corners = single_triangle()
    sides = (triangle.boundary[2, 1] * (triangle.d[0] * corners.field())).batch()[:, 0]  # [corners] Point-
    patch = (sides[0] & sides[1]) / 2                                        # [] Patch-
    gradients = (time & sides) / (2 * (time & patch))                        # [corners] Plane+
    endpoint_differences, swept_fluxes, _, face_bivector = cartan_decomposition(
        corners, corner_forms, corner_velocities,
    )
    save_figure(
        render.draw_cartan_ribbon(corners, corner_forms, corner_velocities,
                                  ribbon_step, endpoint_differences, swept_fluxes),
        "lie_advection_cartan",
    )

    # 2. Flux winding and resistive diffusion on a disk.
    mesh, vertices = disk(disk_radius, n_disk_divisions)
    operators = Operators(mesh, vertices)
    stream = vortex_streamfunction(vertices, swirl_rate, ellipticity)        # Scalar+[vertices]
    face_velocity, fixed_sheets, comoving_sheets = edge_world_sheets(operators, stream)  # Point-[faces], Patch-[edges], Patch-[edges]
    start_cochain = initial_cochain(operators, lobe_centres, lobe_strengths, lobe_radius)  # Scalar+[edges]
    batched_start = stack((start_cochain, start_cochain))                    # [cases] Scalar+[edges]

    # A fixed edge's reading changes by the field on its world sheet; ideal Ohm's law zeroes the field on the
    # comoving sheet, leaving the patch of space between the two, which the magnetic plane alone reads.
    mesh_swept_patches = fixed_sheets - comoving_sheets                      # Patch-[edges]
    advection = spdiag(Magnetic & mesh_swept_patches)(operators.edge_curl)   # (Scalar+ <- Scalar+)[edges <- edges]
    rate = advection - operators.curl_laplacian * resistivities[:, None]     # [cases] (Scalar+ <- Scalar+)[edges <- edges]

    # Retain scalar diagnostics and the final state; the mesh fields stream through one at a time.
    energy_trace, flux_trace = [], []
    for cochain in advect(batched_start, rate, dt, n_steps):
        energy_trace.append((cochain & operators.curl_stiffness(cochain)).sites.sum() / 2)
        flux_trace.append(operators.d1(cochain).sites.sum())
    energies = stack(energy_trace)                                           # [steps + 1, cases] Scalar+
    total_fluxes = stack(flux_trace)                                         # [steps + 1, cases] Scalar+
    final_flux_density = operators.flux_density(cochain)                     # [cases] Scalar+[faces]
    times = np.arange(n_steps + 1) * dt
    save_figure(
        render.draw_advection_comparison(mesh, vertices, final_flux_density,
                                         times, energies, total_fluxes),
        "lie_advection_snapshots",
    )

    # 3. Animated magnetic flux winding side by side.
    sampled_states = islice(advect(batched_start, rate, dt, n_steps), 0, None, n_steps_per_frame)
    frames = (operators.flux_density(state) for state in sampled_states)
    save_animation(
        render.animate_advection(mesh, vertices, frames),
        "lie_advection_flow",
        frame_duration_ms,
    )

    # checks
    # Associativity of the regressive product equates the world-sheet flux to the interior-product line integral.
    corner_motions = time + corner_velocities
    edge_motions = (triangle.incidence[2, 1] * (triangle.incidence[1, 0].mean(axis=0) * corner_motions.field())).batch()[:, 0]
    interior_line_integrals = (face_bivector & edge_motions) & sides
    np.testing.assert_allclose((swept_fluxes - interior_line_integrals).kernel, 0.0, atol=1e-13)
    np.testing.assert_allclose(endpoint_differences.sum(axis=0).kernel, 0.0, atol=1e-15)

    # Cartan's decomposition matches the exact material derivative of the edge line integrals.
    plus_corners = corners + 0.5 * corner_motions
    minus_corners = corners - 0.5 * corner_motions
    plus_sides = (triangle.boundary[2, 1] * (triangle.d[0] * plus_corners.field())).batch()[:, 0]
    minus_sides = (triangle.boundary[2, 1] * (triangle.d[0] * minus_corners.field())).batch()[:, 0]
    plus_midpoints = (triangle.incidence[2, 1] * (triangle.incidence[1, 0].mean(axis=0) * plus_corners.field())).batch()[:, 0]
    minus_midpoints = (triangle.incidence[2, 1] * (triangle.incidence[1, 0].mean(axis=0) * minus_corners.field())).batch()[:, 0]
    plus_hats = 1 + (gradients[:, None] & (plus_midpoints[None, :] - corners[:, None]))
    minus_hats = 1 + (gradients[:, None] & (minus_midpoints[None, :] - corners[:, None]))
    plus_forms = (plus_hats * corner_forms[:, None]).sum(axis=0)
    minus_forms = (minus_hats * corner_forms[:, None]).sum(axis=0)
    material_rates = (plus_forms & plus_sides) - (minus_forms & minus_sides)
    np.testing.assert_allclose(((endpoint_differences + swept_fluxes) - material_rates).kernel, 0.0, atol=1e-13)

    # Across every mesh edge, the swept patch area equals minus the streamfunction difference.
    d0 = mesh.d[0] * Scalar
    np.testing.assert_allclose(((mv.xy & mesh_swept_patches) + d0(stream)).kernel, 0.0, atol=1e-12)
    np.testing.assert_allclose((operators.gradient(stream) & face_velocity).kernel, 0.0, atol=1e-12)

    # Ideal advection has zero instantaneous magnetic energy production and conserves energy and flux over RK4.
    ideal_initial_rate = advection(start_cochain)
    np.testing.assert_allclose(
        (ideal_initial_rate & operators.curl_stiffness(start_cochain)).sites.sum().kernel,
        0.0, atol=1e-12,
    )
    np.testing.assert_allclose((total_fluxes[:, 0] - total_fluxes[0, 0]).kernel, 0.0, atol=1e-13)
    np.testing.assert_allclose((total_fluxes[:, 1] - total_fluxes[0, 1]).kernel, 0.0, atol=1e-14)
    np.testing.assert_allclose((energies[:, 0] / energies[0, 0]).kernel, 1.0, rtol=1e-4)
    assert np.all(np.diff(energies[:, 1].to_array()) < 0)


if __name__ == "__main__":
    main()
