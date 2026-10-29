"""The scenes relax a dome onto the soap film of a four-arch wire, recover the conjugates of that film and of Enneper's
surface, and turn a slit catenoid through Bonnet's family into a helicoid."""

from __future__ import annotations

import numpy as np

from numga import stack
from numga.sparse import spdiag
from examples.surfaces.minimal_surfaces.core import (
    Mesh, Point, bonnet_family, conjugate_minimal_surface, mv,
    relax_plateau,
)

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
) -> tuple[Mesh, Point, Point]:
    """A triangulated disk and an initial domed surface spanning a four-arch boundary wire."""
    mesh, flat_vertices = Mesh.triangular_disk(radius, n_divisions)
    vertical = mv.z.dual()                                                   # [] Point-
    # Multiplying each planar point by the unit x-direction gives a planar spinor whose
    # fourth power carries the four-arch solid harmonic in its scalar part.
    planar = -flat_vertices * mv.x.dual()                                    # Even+[vertices]
    fourth_power = planar.squared().squared()                                # Even+[vertices]
    envelope = 1 - flat_vertices.norm_squared() / radius**2                  # Scalar+[vertices]
    initial_vertices = flat_vertices + (rim_arch * fourth_power.select[0] + interior_dome * envelope) * vertical  # Point-[vertices]
    return mesh, flat_vertices, initial_vertices


def enneper_disk(
    radius: float, n_divisions: int,
) -> tuple[Mesh, Point, Point, Point]:
    """A triangulated disk, Enneper's minimal surface, and its exact analytical harmonic conjugate."""
    mesh, flat_vertices = Mesh.triangular_disk(radius, n_divisions)
    planar = -flat_vertices * mv.x.dual()                                    # Even+[vertices]
    planar_squared = planar.squared()                                        # Even+[vertices]
    planar_cubed = planar_squared * planar                                   # Even+[vertices]
    enneper = (planar.reverse() - planar_cubed / 3) * mv.x.dual() + planar_squared.select[0] * mv.z.dual()  # Point-[vertices]
    exact_conjugate = -(mv.xy * (planar.reverse() + planar_cubed / 3)) * mv.x.dual() + (mv.xy * planar_squared).select[0] * mv.z.dual()  # Point-[vertices]
    return mesh, flat_vertices, enneper, exact_conjugate


def slit_catenoid(
    height: float, azimuth_span: float, n_heights: int, n_azimuths: int,
) -> tuple[Mesh, Point, Point, np.ndarray, np.ndarray]:
    """A slit catenoid mesh built by turning a hyperbolic cosine profile around the z-axis, its exact analytical
    helicoid conjugate, and its parameter lines as vertex indices, the rings `[heights, azimuths]` and the meridians
    `[azimuths, heights]`."""
    heights = np.linspace(-height, height, n_heights)
    azimuths = np.linspace(-azimuth_span, azimuth_span, n_azimuths)
    radial = (mv.xy * (-azimuths / 2)).exp() >> mv.x.dual()                  # [azimuths] Point-
    tangent = (mv.xy * (-np.pi / 4)).exp() >> radial                         # [azimuths] Point-
    vertical = mv.z.dual()                                                   # [] Point-

    catenoid = (np.cosh(heights)[:, None] * radial[None, :] + heights[:, None] * vertical).reshape(-1).field()  # Point-[vertices]
    helicoid = (np.sinh(heights)[:, None] * tangent[None, :] - azimuths[None, :] * vertical).reshape(-1).field()  # Point-[vertices]

    grid = np.arange(n_heights * n_azimuths).reshape(n_heights, n_azimuths)
    # Each quad's two triangles, counter-clockwise seen from outside.
    first = np.stack((grid[:-1, :-1], grid[1:, 1:], grid[1:, :-1]), axis=-1).reshape(-1, 3)
    second = np.stack((grid[:-1, :-1], grid[:-1, 1:], grid[1:, 1:]), axis=-1).reshape(-1, 3)
    mesh = Mesh(n_heights * n_azimuths, np.concatenate((first, second)))
    return mesh, catenoid, helicoid, grid, grid.T


def main() -> None:
    from examples.surfaces.minimal_surfaces import render
    from examples.util.animation import save_animation, save_figure

    # The dome relaxes onto the soap film of the four-arch wire.
    plateau_mesh, flat_vertices, initial_vertices = four_arch_dome(
        disk_radius, n_disk_divisions, arch_height, dome_height,
    )
    relaxation_states = list(relax_plateau(
        plateau_mesh, initial_vertices, relaxation_step_size, n_relaxation_steps,
    ))
    save_animation(
        render.animate_plateau(plateau_mesh, flat_vertices, relaxation_states),
        "minimal_surfaces_plateau",
        frame_duration_ms,
        forward_then_back=True,
    )
    minimal_vertices, _, _ = relaxation_states[-1]

    # The conjugates of the film and of Enneper's surface, from the sums of the sides facing each vertex.
    plateau_conjugate, plateau_flux = conjugate_minimal_surface(plateau_mesh, minimal_vertices)
    enneper_mesh, enneper_flat, enneper_vertices, exact_enneper_conjugate = enneper_disk(
        enneper_radius, n_enneper_divisions,
    )
    enneper_conjugate, enneper_flux = conjugate_minimal_surface(enneper_mesh, enneper_vertices)
    save_figure(
        render.draw_conjugates(
            plateau_mesh, flat_vertices, minimal_vertices, plateau_conjugate,
            enneper_mesh, enneper_flat, enneper_vertices, enneper_conjugate,
        ),
        "minimal_surfaces_conjugates",
    )

    # Bonnet's family turns the slit catenoid into the helicoid.
    catenoid_mesh, catenoid_vertices, exact_helicoid, catenoid_rings, catenoid_meridians = slit_catenoid(
        catenoid_height, catenoid_azimuth_span, n_catenoid_heights, n_catenoid_azimuths,
    )
    helicoid_vertices, catenoid_flux = conjugate_minimal_surface(catenoid_mesh, catenoid_vertices)
    bonnet_angles = np.linspace(0.0, np.pi / 2, n_bonnet_frames)
    save_animation(
        render.animate_bonnet(
            catenoid_mesh, catenoid_rings, catenoid_meridians,
            bonnet_family(catenoid_mesh, catenoid_vertices, helicoid_vertices, bonnet_angles),
        ),
        "minimal_surfaces_bonnet",
        frame_duration_ms,
        forward_then_back=True,
    )

    # checks
    # The quarter-turned sides are twice the area-weighted corner gradients, and summed at the vertices they are the
    # area gradient and the cotangent Laplacian's pull alike.
    corners = plateau_mesh.incidence[2, 0]                                   # [corners] Scalar+[faces <- vertices]
    sides = plateau_mesh.boundary[2, 1] * (plateau_mesh.d[0] * initial_vertices)  # [corners] Point-[faces]
    patches = (sides[0] & sides[1]) / 2                                      # Patch-[faces]
    areas = patches.norm()                                                   # Scalar+[faces]
    normals = patches / areas                                                # Patch-[faces]
    corner_gradients = (sides | patches.inverse()) / 2                       # [corners] Plane+[faces]
    turned_sides = sides.commutator(normals.dual())                          # [corners] Point-[faces]
    np.testing.assert_allclose((turned_sides - 2 * areas * corner_gradients.dual()).kernel, 0.0, atol=1e-13)

    local_laplacian = (corner_gradients[:, None] | corner_gradients[None, :]) * areas  # [corners, corners] Scalar+[faces]
    laplacian = (~corners[:, None] * spdiag(local_laplacian) * corners[None, :]).sum(axis=0).sum(axis=0)  # Scalar+[vertices <- vertices]
    area_pull = (~corners * turned_sides).sum(axis=0) / 2                    # Point-[vertices]
    area_gradient = (~corners * (areas * corner_gradients)).sum(axis=0)      # Plane+[vertices]
    np.testing.assert_allclose((area_pull - area_gradient.dual()).kernel, 0.0, atol=1e-13)
    np.testing.assert_allclose((area_pull - laplacian * initial_vertices).kernel, 0.0, atol=1e-12)

    relaxation_areas = stack([area for _, area, _ in relaxation_states])     # [steps + 1] Scalar+
    assert np.all(relaxation_areas[1:] < relaxation_areas[:-1])

    # The sums of the sides facing each vertex cancel in the interior.
    for mesh, flux in ((plateau_mesh, plateau_flux), (enneper_mesh, enneper_flux), (catenoid_mesh, catenoid_flux)):
        np.testing.assert_allclose((mesh.reading(mesh.interior(0)) * flux).kernel, 0.0, atol=1e-14)

    # The recovered conjugates match the exact ones up to a translation.
    centred_enneper = (enneper_conjugate.batch() - enneper_conjugate.batch().mean(axis=0)).field()
    centred_exact_enneper = (exact_enneper_conjugate.batch() - exact_enneper_conjugate.batch().mean(axis=0)).field()
    np.testing.assert_allclose(centred_enneper.kernel, centred_exact_enneper.kernel, atol=2e-2)

    centred_helicoid = (helicoid_vertices.batch() - helicoid_vertices.batch().mean(axis=0)).field()
    centred_exact_helicoid = (exact_helicoid.batch() - exact_helicoid.batch().mean(axis=0)).field()
    np.testing.assert_allclose(centred_helicoid.kernel, centred_exact_helicoid.kernel, atol=2e-2)


if __name__ == "__main__":
    main()
