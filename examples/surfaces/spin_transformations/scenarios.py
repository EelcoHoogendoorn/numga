"""Scenes for spin transformations: a sphere given a dipole change in mean curvature, Spot the cow
rounded by conformal curvature flow, and the Dirac spheres."""

from __future__ import annotations

import numpy as np

from examples.surfaces.spin_transformations import core

# How often the sphere's icosahedron is split, and how strongly its mean curvature changes along y.
N_SUBDIVISIONS = 3
DIPOLE = 0.5
# The share of each face's curvature departure each step of Spot's flow removes, the number of steps,
# and the duration of a frame.
RATE = 0.2
N_STEPS = 15
DURATION_MS = 160
# The finer sphere the Dirac spheres are spun from, the eigenvalues shown, and the states shown of each.
N_DIRAC_SUBDIVISIONS = 4
EIGENVALUES = (1, 2, 3, -2, -3)
N_STATES = 3


# --- math -----------------------------------------------------------------------------
def dipole() -> tuple[core.Mesh, core.Point, core.Scalar, core.Point]:
    """The sphere and its vertices, the change in mean curvature on each face, growing along y, and the
    vertices after the spin transformation that makes it."""
    mesh, vertices = core.Mesh.icosphere(N_SUBDIVISIONS)
    centroids = mesh.incidence[2, 0].mean(axis=0) * vertices
    # Each centroid paired with the direction y: its height along y.
    curvature_change = (centroids & core.context.multivector.y) * DIPOLE        # Scalar+[faces]
    return mesh, vertices, curvature_change, core.spin_transform_deform(mesh, vertices, curvature_change)


def rounding() -> tuple[core.Mesh, list[core.Point]]:
    """Spot the cow under conformal curvature flow: her mesh, and its vertices at every step."""
    mesh, vertices = core.spot()
    return mesh, list(core.conformal_smooth(mesh, vertices, N_STEPS, RATE))     # [steps + 1] Point-[vertices]


def dirac() -> tuple[core.Mesh, core.Point, dict[int, core.Point], np.ndarray]:
    """The sphere and its vertices; a few of the Dirac spheres for each of the eigenvalues, the vertices
    of the surfaces spun by the Dirac operator's eigenfields; and the energies of their fields."""
    mesh, vertices = core.Mesh.icosphere(N_DIRAC_SUBDIVISIONS)
    energies, spun = core.dirac_spheres(mesh, vertices, np.array(EIGENVALUES, dtype=float), N_STATES)   # [eigenvalues, states]
    return mesh, vertices, {eigenvalue: spheres for eigenvalue, spheres in zip(EIGENVALUES, spun)}, energies.to_array().ravel()


# --- plumbing -------------------------------------------------------------------------
def roundness(mesh: core.Mesh, vertices: core.Point) -> float:
    """The spread of the vertices' distances from the surface's centre, its area-weighted centroid, over
    their mean: nothing on a sphere, however its vertices crowd."""
    centroids = mesh.incidence[2, 0].mean(axis=0) * vertices                    # Point-[faces]
    sides = mesh.boundary[2, 1] * (mesh.d[0] * vertices)                        # [3] Point-[faces]
    areas = sides[1].commutator(sides[0]).norm() / 2                            # Scalar+[faces]
    middle = (centroids * areas).sites.sum() / areas.sites.sum()
    distances = (vertices - middle).norm().to_array()
    return distances.std() / distances.mean()


def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.surfaces.spin_transformations import render

    mesh, vertices, curvature_change, deformed = dipole()
    save_figure(render.draw_dipole(mesh, vertices, curvature_change, deformed), "spin_transformation")
    sphere, sphere_vertices, gallery, energies = dirac()
    save_figure(render.draw_dirac(sphere, gallery, sphere_vertices), "dirac_spheres")
    spot, frames = rounding()
    save_animation(render.animate_rounding(spot, frames), "spin_rounding", DURATION_MS)

    # checks
    cosines = core.corner_cosines
    # No change in curvature leaves the sphere as it is, centred.
    unchanged = core.spin_transform_deform(mesh, vertices, curvature_change * 0.0) - (vertices - vertices.sites.mean())
    np.testing.assert_allclose(unchanged.kernel, 0.0, atol=1e-11)
    # The deformation keeps every corner's angle, up to the mesh's resolution, and mean curvature rises
    # where it was asked to rise.
    np.testing.assert_allclose((cosines(mesh, deformed) - cosines(mesh, vertices)).to_array(), 0.0, atol=0.05)
    risen = core.mean_curvature(mesh, core.recentred(deformed)) - core.mean_curvature(mesh, vertices)
    assert np.corrcoef(risen.to_array().ravel(), curvature_change.to_array().ravel())[0, 1] > 0.5
    # Spot rounds: the spread of the vertices' distances from her centre falls tenfold, while the angles
    # hold up to the mesh's resolution, in the median.
    assert roundness(spot, frames[-1]) < 0.1 * roundness(spot, frames[0])
    assert np.median(np.abs((cosines(spot, frames[-1]) - cosines(spot, frames[0])).to_array())) < 0.05
    # The eigenvalues are integers: each asks for a field of zero energy, met up to the mesh's resolution
    # against the spectrum's unit spacing; and every Dirac sphere keeps the angles, up to the mesh's
    # resolution away from where its field vanishes.
    assert np.all(np.abs(energies) < 0.05)
    for spheres in gallery.values():
        for spun in spheres:
            assert np.median(np.abs((cosines(sphere, spun) - cosines(sphere, sphere_vertices)).to_array())) < 0.1


if __name__ == "__main__":
    main()
