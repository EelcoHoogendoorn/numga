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
def dipole() -> tuple[core.Mesh, core.Scalar, core.Mesh]:
    """The sphere, the change in mean curvature on each face, growing along y, and the sphere after
    the spin transformation that makes it."""
    mesh = core.icosphere(N_SUBDIVISIONS)
    centroids = core.as_ga_sparse(mesh.faces, core.as_scalar(np.ones_like(mesh.faces) / 3)) * mesh.vertices
    # Each centroid paired with the direction y: its height along y.
    curvature_change = (centroids & core.context.multivector.y) * DIPOLE        # Scalar[F]
    return mesh, curvature_change, core.spin_transform_deform(mesh, curvature_change)


def rounding() -> list[core.Mesh]:
    """Spot the cow under conformal curvature flow: the mesh at every step."""
    return list(core.conformal_smooth(core.spot(), N_STEPS, RATE))              # [steps + 1] Mesh


def dirac() -> tuple[core.Mesh, dict[int, list[core.Mesh]], np.ndarray]:
    """The sphere; a few of the Dirac spheres for each of the eigenvalues, the surfaces spun by the
    Dirac operator's eigenfields; and the energies of their fields."""
    mesh = core.icosphere(N_DIRAC_SUBDIVISIONS)
    spun = {eigenvalue: core.dirac_spheres(mesh, eigenvalue, N_STATES) for eigenvalue in EIGENVALUES}
    energies = np.concatenate([energy.to_array().ravel() for energy, _ in spun.values()])
    return mesh, {eigenvalue: spheres for eigenvalue, (_, spheres) in spun.items()}, energies


# --- plumbing -------------------------------------------------------------------------
def roundness(mesh: core.Mesh) -> float:
    """The spread of the vertices' distances from the surface's centre, its area-weighted centroid, over
    their mean: nothing on a sphere, however its vertices crowd."""
    centroids = core.as_ga_sparse(mesh.faces, core.as_scalar(np.ones_like(mesh.faces) / 3)) * mesh.vertices   # Vector[F]
    middle = (centroids * mesh.triangle_areas).sites.sum() / mesh.triangle_areas.sites.sum()
    distances = (mesh.vertices - middle).norm().to_array()
    return distances.std() / distances.mean()


def main() -> None:
    from examples.animation import save_animation, save_figure
    from examples.surfaces.spin_transformations import render

    mesh, curvature_change, deformed = dipole()
    save_figure(render.draw_dipole(mesh, curvature_change, deformed), "spin_transformation")
    sphere, gallery, energies = dirac()
    save_figure(render.draw_dirac(gallery, sphere), "dirac_spheres")
    frames = rounding()
    save_animation(render.animate_rounding(frames), "spin_rounding", DURATION_MS)

    # checks
    # No change in curvature leaves the sphere as it is, centred.
    unchanged = core.spin_transform_deform(mesh, curvature_change * 0.0).vertices - (mesh.vertices - mesh.vertices.sites.mean())
    np.testing.assert_allclose(unchanged.kernel, 0.0, atol=1e-11)
    # The deformation keeps every corner's angle, up to the mesh's resolution, and mean curvature rises
    # where it was asked to rise.
    np.testing.assert_allclose((deformed.corner_cosines() - mesh.corner_cosines()).to_array(), 0.0, atol=0.05)
    risen = core.mean_curvature(core._recenter(deformed)) - core.mean_curvature(mesh)
    assert np.corrcoef(risen.to_array().ravel(), curvature_change.to_array().ravel())[0, 1] > 0.5
    # Spot rounds: the spread of the vertices' distances from her centre falls tenfold, while the angles
    # hold up to the mesh's resolution, in the median.
    assert roundness(frames[-1]) < 0.1 * roundness(frames[0])
    assert np.median(np.abs((frames[-1].corner_cosines() - frames[0].corner_cosines()).to_array())) < 0.05
    # The eigenvalues are integers: each asks for a field of zero energy, met up to the mesh's resolution
    # against the spectrum's unit spacing; and every Dirac sphere keeps the angles, up to the mesh's
    # resolution away from where its field vanishes.
    assert np.all(np.abs(energies) < 0.05)
    for spheres in gallery.values():
        for spun in spheres:
            assert np.median(np.abs((spun.corner_cosines() - sphere.corner_cosines()).to_array())) < 0.1


if __name__ == "__main__":
    main()
