"""The scenes relax a dome onto the soap film of a four-arch wire, recover the conjugates of that film and of Enneper's surface, and turn a slit catenoid through Bonnet's family into a helicoid."""

from __future__ import annotations

import numpy as np

from numga import Algebra, stack
from examples.util import shapes
from examples.surfaces.minimal_surfaces.core import (
    Point, Topology, bonnet_family, conjugate_minimal_surface, ga, mv,
    plateau_states,
)

# Coordinate planes of x and y across the planar disk domain, whose duals give horizontal displacements.
Upright = ga.gatype.vector().derive.within(Algebra("x+y+"))                  # Vector+

disk_radius = 1.0
n_disk_divisions = 10
arch_height = 0.35
dome_height = 0.55
relaxation_step_size = 0.14
n_relaxation_steps = 40

enneper_radius = 0.8
n_enneper_divisions = 10

catenoid_height = 1.15
catenoid_azimuth_span = 0.85 * np.pi
n_catenoid_heights = 22
n_catenoid_azimuths = 44
n_bonnet_frames = 36
frame_duration_ms = 40


# --- plumbing -------------------------------------------------------------------------
def four_arch_dome(
    radius: float, n_divisions: int, rim_arch: float, interior_dome: float,
) -> tuple[Topology, Point, Point]:
    """A solid-harmonic rim arch and parabolic interior dome lift a triangulated disk (`Point-[V]`) onto an initial surface (`Point-[V]`) spanning a four-arch wire."""
    topology, flat_coordinates = shapes.triangular_disk(radius, n_divisions)
    flat_vertices = mv(Upright, flat_coordinates).dual().field()             # Point-[V]
    vertical_direction = mv.z.dual()                                         # [] Point-
    # Multiplying each planar point by the unit x-direction forms a planar spinor whose fourth power carries the four-arch solid harmonic in its scalar part.
    spinors = -flat_vertices * mv.x.dual()                                   # Even+[V]
    fourth_power = spinors.squared().squared()                               # Even+[V]
    envelope = 1 - flat_vertices.norm_squared() / radius**2                  # Scalar+[V]
    initial_vertices = flat_vertices + (rim_arch * fourth_power.select[0] + interior_dome * envelope) * vertical_direction  # Point-[V]
    return topology, flat_vertices, initial_vertices


def enneper_disk(
    radius: float, n_divisions: int,
) -> tuple[Topology, Point, Point, Point]:
    """Weierstrass spinor powers map a triangulated disk (`Point-[V]`) onto Enneper's minimal surface (`Point-[V]`) and its closed-form harmonic conjugate (`Point-[V]`)."""
    topology, flat_coordinates = shapes.triangular_disk(radius, n_divisions)
    flat_vertices = mv(Upright, flat_coordinates).dual().field()             # Point-[V]
    spinors = -flat_vertices * mv.x.dual()                                   # Even+[V]
    spinors_squared = spinors.squared()                                      # Even+[V]
    spinors_cubed = spinors_squared * spinors                                # Even+[V]
    enneper = (spinors.reverse() - spinors_cubed / 3) * mv.x.dual() + spinors_squared.select[0] * mv.z.dual()  # Point-[V]
    exact_conjugate = -(mv.xy * (spinors.reverse() + spinors_cubed / 3)) * mv.x.dual() + (mv.xy * spinors_squared).select[0] * mv.z.dual()  # Point-[V]
    return topology, flat_vertices, enneper, exact_conjugate


def slit_catenoid(
    height: float, azimuth_span: float, n_heights: int, n_azimuths: int,
) -> tuple[Topology, Point, Point, np.ndarray, np.ndarray]:
    """Turning a hyperbolic cosine profile around the vertical axis builds a triangulated slit catenoid (`Point-[V]`), its closed-form helicoid conjugate (`Point-[V]`), and ring (`[n_heights, n_azimuths]`) and meridian (`[n_azimuths, n_heights]`) vertex indices."""
    heights = np.linspace(-height, height, n_heights)
    azimuths = np.linspace(-azimuth_span, azimuth_span, n_azimuths)
    radial_directions = (mv.xy * (-azimuths / 2)).exp() >> mv.x.dual()       # [n_azimuths] Point-
    tangent_directions = (mv.xy * (-np.pi / 4)).exp() >> radial_directions   # [n_azimuths] Point-
    vertical_direction = mv.z.dual()                                         # [] Point-

    catenoid = (np.cosh(heights)[:, None] * radial_directions[None, :] + heights[:, None] * vertical_direction).reshape(-1).field()  # Point-[V]
    helicoid = (np.sinh(heights)[:, None] * tangent_directions[None, :] - azimuths[None, :] * vertical_direction).reshape(-1).field()  # Point-[V]

    grid = np.arange(n_heights * n_azimuths).reshape(n_heights, n_azimuths)
    # Split each parameter quad into two counter-clockwise triangles seen from outside.
    first_triangles = np.stack((grid[:-1, :-1], grid[1:, 1:], grid[1:, :-1]), axis=-1).reshape(-1, 3)
    second_triangles = np.stack((grid[:-1, :-1], grid[:-1, 1:], grid[1:, 1:]), axis=-1).reshape(-1, 3)
    topology = Topology(n_heights * n_azimuths, np.concatenate((first_triangles, second_triangles)))
    return topology, catenoid, helicoid, grid, grid.T


def main() -> None:
    """Runs the Plateau relaxation, harmonic conjugate recovery, and Bonnet family animations and checks."""
    import matplotlib.pyplot as plt
    from examples.surfaces.minimal_surfaces import render
    from examples.util.animation import save_animation, save_figure

    # Relax the initial dome onto the minimal soap film spanning the four-arch wire.
    plateau_topology, flat_vertices, initial_vertices = four_arch_dome(
        disk_radius, n_disk_divisions, arch_height, dome_height,
    )
    relaxation_states = list(plateau_states(
        plateau_topology, initial_vertices, relaxation_step_size, n_relaxation_steps,
    ))
    save_animation(
        render.plateau_animation(plateau_topology, flat_vertices, relaxation_states),
        "minimal_surfaces_plateau",
        frame_duration_ms,
        forward_then_back=True,
    )
    _, _, initial_curvature = relaxation_states[0]
    minimal_vertices, _, final_curvature = relaxation_states[-1]

    # Recover harmonic conjugates of the relaxed soap film and Enneper's surface from boundary side sums.
    plateau_conjugate, plateau_flux = conjugate_minimal_surface(plateau_topology, minimal_vertices)
    enneper_topology, enneper_flat, enneper_vertices, exact_enneper_conjugate = enneper_disk(
        enneper_radius, n_enneper_divisions,
    )
    enneper_conjugate, enneper_flux = conjugate_minimal_surface(enneper_topology, enneper_vertices)
    figure = render.conjugates_figure(
        plateau_topology, flat_vertices, minimal_vertices, plateau_conjugate,
        enneper_topology, enneper_flat, enneper_vertices, enneper_conjugate,
    )
    save_figure(figure, "minimal_surfaces_conjugates")
    plt.close(figure)

    # Sweep Bonnet's isometric family from the slit catenoid to its recovered helicoid conjugate.
    catenoid_topology, catenoid_vertices, exact_helicoid, catenoid_rings, catenoid_meridians = slit_catenoid(
        catenoid_height, catenoid_azimuth_span, n_catenoid_heights, n_catenoid_azimuths,
    )
    helicoid_vertices, catenoid_flux = conjugate_minimal_surface(catenoid_topology, catenoid_vertices)
    bonnet_angles = np.linspace(0.0, np.pi / 2, n_bonnet_frames)
    bonnet_states = list(bonnet_family(catenoid_topology, catenoid_vertices, helicoid_vertices, bonnet_angles))
    save_animation(
        render.bonnet_animation(catenoid_topology, catenoid_rings, catenoid_meridians, bonnet_states),
        "minimal_surfaces_bonnet",
        frame_duration_ms,
        forward_then_back=True,
    )

    # checks
    relaxation_areas = stack([area for _, area, _ in relaxation_states])     # [n_relaxation_steps + 1] Scalar+
    assert (relaxation_areas[1:] < relaxation_areas[:-1]).all()
    assert (final_curvature.norm() < 0.1 * initial_curvature.norm().batch().max()).all()

    # Facing sides cancel across shared interior edges, leaving boundary sums alone.
    for topology, flux in (
        (plateau_topology, plateau_flux),
        (enneper_topology, enneper_flux),
        (catenoid_topology, catenoid_flux),
    ):
        assert ((topology.reading(topology.interior[0], 0) * flux).norm() < 1e-10).all()

    # Recovered conjugates match the closed-form conjugates up to global translation.
    centred_enneper = enneper_conjugate - enneper_conjugate.sites.mean()     # Point-[V]
    centred_exact_enneper = exact_enneper_conjugate - exact_enneper_conjugate.sites.mean()  # Point-[V]
    assert ((centred_enneper - centred_exact_enneper).norm() < 2e-2).all()

    centred_helicoid = helicoid_vertices - helicoid_vertices.sites.mean()    # Point-[V]
    centred_exact_helicoid = exact_helicoid - exact_helicoid.sites.mean()    # Point-[V]
    assert ((centred_helicoid - centred_exact_helicoid).norm() < 2e-2).all()

    # Bonnet's family preserves edge lengths and vertex Gauss normals between catenoid and helicoid.
    catenoid_frame, catenoid_normals = bonnet_states[0]
    helicoid_frame, helicoid_normals = bonnet_states[-1]
    catenoid_lengths = (catenoid_topology.d[0] * catenoid_frame).norm()      # Scalar+[E]
    helicoid_lengths = (catenoid_topology.d[0] * helicoid_frame).norm()      # Scalar+[E]
    assert (((helicoid_lengths - catenoid_lengths).abs() / catenoid_lengths) < 1e-2).all()
    assert ((helicoid_normals - catenoid_normals).norm() < 5e-2).all()


if __name__ == "__main__":
    main()
