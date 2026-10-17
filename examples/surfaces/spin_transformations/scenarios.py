"""Scenes for spin transformations: a sphere given a dipole change in mean curvature, Spot the cow
rounded by conformal curvature flow, and the Dirac spheres."""

from __future__ import annotations

import numpy as np

from examples.surfaces.spin_transformations import core

# How often the sphere's icosahedron is split, and how strongly its mean curvature changes along y.
LEVELS = 3
DIPOLE = 0.5
# The share of each face's curvature departure each step of Spot's flow removes, and the number of steps.
RATE = 0.1
STEPS = 30
# The finer sphere the Dirac spheres are spun from, the eigenvalues shown, and the states shown of each.
DIRAC_LEVELS = 4
EIGENVALUES = (1, 2, 3, -2, -3)
STATES = 3


# --- math -----------------------------------------------------------------------------
def dipole():
    """The sphere, the change in mean curvature on each face, growing along y, and the sphere after
    the spin transformation that makes it."""
    mesh = core.icosphere(LEVELS)
    centroids = core.as_ga_sparse(mesh.faces, core.as_scalar(np.ones_like(mesh.faces) / 3)) * mesh.vertices
    rho = (centroids | core.context.multivector.y) * DIPOLE                    # Scalar[F]
    deformed = core.spin_transform_deform(mesh, rho)

    # --- checks
    # No change in curvature leaves the sphere as it is, centred.
    unchanged = core.spin_transform_deform(mesh, rho * 0.0).vertices - (mesh.vertices - mesh.vertices.sites.mean())
    np.testing.assert_allclose(unchanged.kernel, 0.0, atol=1e-11)
    # The deformation keeps every corner's angle, up to the mesh's resolution.
    np.testing.assert_allclose((deformed.corner_cosines() - mesh.corner_cosines()).to_array(), 0.0, atol=0.05)
    # Mean curvature rises where it was asked to rise.
    risen = core.mean_curvature(core._recenter(deformed)) - core.mean_curvature(mesh)
    assert np.corrcoef(risen.to_array().ravel(), rho.to_array().ravel())[0, 1] > 0.5
    return mesh, rho, deformed


def rounding():
    """Spot the cow under conformal curvature flow: the mesh at every step."""
    frames = list(core.conformal_smooth(core.spot(), STEPS, RATE))            # [STEPS + 1] Mesh

    # --- checks
    # The surface rounds: the spread of the vertices' distances from the centre of the surface falls
    # tenfold.
    assert roundness(frames[-1]) < 0.1 * roundness(frames[0])
    # The angles hold up to the mesh's resolution, in the median.
    assert np.median(np.abs((frames[-1].corner_cosines() - frames[0].corner_cosines()).to_array())) < 0.05
    return frames


def dirac():
    """The sphere, and a few of the Dirac spheres for each of the eigenvalues: the surfaces spun by the
    Dirac operator's eigenfields."""
    mesh = core.icosphere(DIRAC_LEVELS)
    spun = {eigenvalue: core.dirac_spheres(mesh, eigenvalue, STATES) for eigenvalue in EIGENVALUES}
    gallery = {eigenvalue: spheres for eigenvalue, (_, spheres) in spun.items()}

    # --- checks
    # The eigenvalues are integers: each asks for a field of zero energy, met up to the mesh's resolution
    # against the spectrum's unit spacing.
    energies = np.concatenate([energy.to_array().ravel() for energy, _ in spun.values()])
    assert np.all(np.abs(energies) < 0.05)
    # Every Dirac sphere keeps the angles, up to the mesh's resolution away from where its field vanishes.
    for spheres in gallery.values():
        for sphere in spheres:
            assert np.median(np.abs((sphere.corner_cosines() - mesh.corner_cosines()).to_array())) < 0.1
    return mesh, gallery


# --- plumbing -------------------------------------------------------------------------
def roundness(mesh: core.Mesh) -> float:
    """The spread of the vertices' distances from the surface's centre, its area-weighted centroid, over
    their mean: nothing on a sphere, however its vertices crowd."""
    centroids = core.as_ga_sparse(mesh.faces, core.as_scalar(np.ones_like(mesh.faces) / 3)) * mesh.vertices   # Vector[F]
    middle = (centroids * mesh.triangle_areas).sites.sum() / mesh.triangle_areas.sites.sum()
    distances = (mesh.vertices - middle).norm().to_array()
    return distances.std() / distances.mean()


if __name__ == "__main__":
    from examples.animation import save_animation, save_figure
    from examples.surfaces.spin_transformations import render

    save_figure(render.draw_dipole(*dipole()), "spin_transformation")
    mesh, gallery = dirac()
    save_figure(render.draw_dirac(gallery, mesh), "dirac_spheres")
    save_animation(render.animate_rounding(rounding()), "spin_rounding", 120)
