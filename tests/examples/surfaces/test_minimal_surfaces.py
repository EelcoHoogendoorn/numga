"""Discrete minimal surfaces, harmonic conjugates, and Bonnet's isometric family."""

import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from numga.sparse import spdiag
from examples.surfaces.minimal_surfaces import core, render, scenarios


def test_minimal_surfaces_math_does_not_import_plotting():
    probe = (
        "import sys, examples.surfaces.minimal_surfaces.core, examples.surfaces.minimal_surfaces.scenarios; "
        "print([m for m in sys.modules if m.split('.')[0] in ('matplotlib', 'PIL')])"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]", out.stdout


def test_turned_sides_area_gradient_and_plateau_relaxation():
    mesh, _, initial_vertices = scenarios.four_arch_dome(1.0, 6, 0.35, 0.55)
    sides = mesh.boundary[2, 1] * (mesh.d[0] * initial_vertices)
    patches = (sides[0] & sides[1]) / 2
    areas = patches.norm()
    normals = patches / areas
    corner_gradients = (sides | patches.inverse()) / 2
    turned_sides = sides.commutator(normals.dual())

    # Quarter-turned opposite sides equal twice the area-weighted outward corner hat gradients.
    np.testing.assert_allclose((turned_sides - 2 * areas * corner_gradients.dual()).kernel, 0.0, atol=1e-13)

    # Vertex sum of turned sides matches both the first variation of total area and the cotangent Laplacian pull.
    corners = mesh.incidence[2, 0]
    local_laplacian = (corner_gradients[:, None] | corner_gradients[None, :]) * areas
    laplacian = (~corners[:, None] * spdiag(local_laplacian) * corners[None, :]).sum(axis=0).sum(axis=0)
    area_pull = (~corners * turned_sides).sum(axis=0) / 2
    area_gradient = (~corners * (areas * corner_gradients)).sum(axis=0)
    np.testing.assert_allclose((area_pull - area_gradient.dual()).kernel, 0.0, atol=1e-13)
    np.testing.assert_allclose((area_pull - laplacian * initial_vertices).kernel, 0.0, atol=1e-12)

    # Finite-difference check against the exact triangle-area sum, at the first interior vertex.
    displacement = 1e-7 * (0.4 * core.mv.x.dual() - 0.7 * core.mv.z.dual())
    step = ~mesh.reading(mesh.interior(0)[:1]) * (displacement * np.ones(1)).field()
    plus_sides = mesh.boundary[2, 1] * (mesh.d[0] * (initial_vertices + step))
    minus_sides = mesh.boundary[2, 1] * (mesh.d[0] * (initial_vertices - step))
    plus_area = ((plus_sides[0] & plus_sides[1]).norm() / 2).sites.sum()
    minus_area = ((minus_sides[0] & minus_sides[1]).norm() / 2).sites.sum()
    # The area gradient, a form on displacements, reads the step.
    directional_derivative = (area_gradient & step).sites.sum()
    np.testing.assert_allclose(((plus_area - minus_area) / 2 - directional_derivative).kernel, 0.0, atol=1e-12)

    # Damped Plateau relaxation keeps the boundary pinned and strictly decreases total surface area.
    states = list(core.relax_plateau(mesh, initial_vertices, 0.25, 16))
    relaxation_areas = stack([area for _, area, _ in states])
    assert np.all(relaxation_areas[1:] < relaxation_areas[:-1])

    exterior = mesh.reading(mesh.exterior(0))
    for vertices, _, _ in states:
        np.testing.assert_allclose((exterior * (vertices - initial_vertices)).kernel, 0.0, atol=1e-14)
    initial_mean_square_curvature = states[0][2].squared().sites.mean()
    final_mean_square_curvature = states[-1][2].squared().sites.mean()
    assert final_mean_square_curvature < 1e-2 * initial_mean_square_curvature


def test_conjugate_minimal_surface_interior_cancellation_and_enneper():
    mesh, _, enneper, exact_conjugate = scenarios.enneper_disk(0.8, 10)
    conjugate, boundary_flux = core.conjugate_minimal_surface(mesh, enneper)

    # The sides facing each vertex sum to zero at every interior vertex.
    np.testing.assert_allclose((mesh.reading(mesh.interior(0)) * boundary_flux).kernel, 0.0, atol=1e-14)

    # Recovered conjugate matches the exact analytical Weierstrass-Enneper conjugate up to translation.
    centred_conjugate = (conjugate.batch() - conjugate.batch().mean(axis=0)).field()
    centred_exact = (exact_conjugate.batch() - exact_conjugate.batch().mean(axis=0)).field()
    np.testing.assert_allclose(centred_conjugate.kernel, centred_exact.kernel, atol=2e-2)

    # Applying the conjugate construction a second time turns differentials by pi, negating the interior surface.
    second_conjugate, second_flux = core.conjugate_minimal_surface(mesh, exact_conjugate)
    np.testing.assert_allclose((mesh.reading(mesh.interior(0)) * second_flux).kernel, 0.0, atol=1e-14)
    centred_second = (second_conjugate.batch() - second_conjugate.batch().mean(axis=0)).field()
    centred_enneper = (enneper.batch() - enneper.batch().mean(axis=0)).field()
    np.testing.assert_allclose((centred_second + centred_enneper).kernel, 0.0, atol=2e-2)


def test_bonnet_isometric_family_catenoid_to_helicoid():
    mesh, catenoid, exact_helicoid, _, _ = scenarios.slit_catenoid(1.15, 0.85 * np.pi, 22, 44)
    helicoid, boundary_flux = core.conjugate_minimal_surface(mesh, catenoid)

    interior = mesh.reading(mesh.interior(0))
    np.testing.assert_allclose((interior * boundary_flux).kernel, 0.0, atol=1e-14)
    centred_helicoid = (helicoid.batch() - helicoid.batch().mean(axis=0)).field()
    centred_exact = (exact_helicoid.batch() - exact_helicoid.batch().mean(axis=0)).field()
    np.testing.assert_allclose(centred_helicoid.kernel, centred_exact.kernel, atol=2e-2)

    # Along Bonnet's circle between the catenoid and its exact helicoid conjugate, the normal at each interior vertex,
    # of the patches around it, keeps its direction.
    angles = np.linspace(0.0, np.pi / 2, 5)
    normals = [interior * normal for _, normal in core.bonnet_family(mesh, catenoid, exact_helicoid, angles)]
    for normal in normals[1:]:
        np.testing.assert_allclose((normal - normals[0]).kernel, 0.0, atol=5e-4)


def test_render_smoke():
    plateau_mesh, flat_vertices, initial_vertices = scenarios.four_arch_dome(1.0, 4, 0.35, 0.55)
    states = list(core.relax_plateau(plateau_mesh, initial_vertices, 0.3, 2))
    plateau_frames = list(render.animate_plateau(plateau_mesh, flat_vertices, states[:2]))
    assert len(plateau_frames) == 2
    assert plateau_frames[0].dtype == np.uint8

    minimal_vertices, _, _ = states[-1]
    plateau_conjugate, _ = core.conjugate_minimal_surface(plateau_mesh, minimal_vertices)
    enneper_mesh, enneper_flat, enneper_vertices, _ = scenarios.enneper_disk(0.8, 4)
    enneper_conjugate, _ = core.conjugate_minimal_surface(enneper_mesh, enneper_vertices)
    fig = render.draw_conjugates(
        plateau_mesh, flat_vertices, minimal_vertices, plateau_conjugate,
        enneper_mesh, enneper_flat, enneper_vertices, enneper_conjugate,
    )
    plt.close(fig)

    catenoid_mesh, catenoid, exact_helicoid, rings, meridians = scenarios.slit_catenoid(1.0, 0.7 * np.pi, 8, 12)
    bonnet_frames = list(render.animate_bonnet(
        catenoid_mesh, rings, meridians, core.bonnet_family(catenoid_mesh, catenoid, exact_helicoid, np.array([0.0, np.pi / 4])),
    ))
    assert len(bonnet_frames) == 2
    assert bonnet_frames[0].dtype == np.uint8
