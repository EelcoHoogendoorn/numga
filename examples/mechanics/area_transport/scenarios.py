"""A shear-compression sweep flattens the unit cube into a sheet and verifies area, flux, and projective plane transport."""

from __future__ import annotations

import numpy as np

from examples.mechanics.area_transport import core

side = 1.0
n_frames = 36
duration_ms = 40
max_shear = 0.70
# Ease the height from the unit cube down to a flat sheet while shearing horizontal layers along x.
thicknesses = (1 + np.cos(np.linspace(0, np.pi, n_frames))) / 2
shears = max_shear * (1 - thicknesses)
sample_thickness = 0.35
sample_shear = 0.45
translation = 0.5


# --- math -----------------------------------------------------------------------------
def scene(thicknesses: np.ndarray, shears: np.ndarray) -> tuple[core.Point, core.Point, core.Patch, core.Scalar]:
    """The shear-compression sweep carries the unit cube's corners, face centres, face patches, and volume factor across every frame."""
    crush = core.deformation(thicknesses, shears)                               # [...] Point- <- Point-
    areas, _, volumes = core.transport(crush)                                   # [...] Patch- <- Patch-, Flux+ <- Flux+, Scalar+
    corners, centres, patches = core.cube(side)                                 # [8] Point-, [6] Point-, [6] Patch-
    return crush[..., None](corners), crush[..., None](centres), areas[..., None](patches), volumes


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_animation
    from examples.mechanics.area_transport import render

    crush = core.deformation(thicknesses, shears)                               # [n_frames] Point- <- Point-
    areas, flux_pullback, volumes = core.transport(crush)                       # [n_frames] Patch- <- Patch-, Flux+ <- Flux+, Scalar+
    rest_corners, rest_centres, rest_patches = core.cube(side)                  # [8] Point-, [6] Point-, [6] Patch-
    vertices = crush[:, None](rest_corners)                                     # [n_frames, 8] Point-
    centres = crush[:, None](rest_centres)                                      # [n_frames, 6] Point-
    patches = areas[:, None](rest_patches)                                      # [n_frames, 6] Patch-
    horizontal_flux = flux_pullback(core.mv.xy)                                 # [n_frames] Flux+
    null_patch = (core.mv.z - shears[-1] * core.mv.x).dual() & core.mv.y.dual() # [] Patch-
    save_animation(render.animate(zip(vertices, centres, patches, volumes)), "area_transport", duration_ms,
                   forward_then_back=True)

    placement = core.projective_placement(sample_thickness, sample_shear, translation)  # [] ProjectivePoint- <- ProjectivePoint-
    plane_pullback = placement.adjugate()                                       # [] ProjectivePlane+ <- ProjectivePlane+
    corner_points, face_planes = core.projective_cube(side)                     # [8] ProjectivePoint-, [6] ProjectivePlane+
    image_planes = plane_pullback.solve(face_planes)                            # [6] ProjectivePlane+

    # checks
    # Pulling horizontal flux back gives the same flux through every undeformed face as through the deformed face.
    assert (((horizontal_flux[:, None] & rest_patches[None, :]) - (core.mv.xy & patches)).abs() < 1e-12).all()
    # The deformation and its area map scale every face pyramid's volume by the determinant, including at zero volume.
    assert (((centres & patches) - volumes[:, None] * (rest_centres & rest_patches)).norm() < 1e-12).all()
    # At the flat sheet, the pulled-back horizontal flux vanishes on the collapsed patch that the area map crushes.
    assert (horizontal_flux[-1] & null_patch).abs() < 1e-12
    assert areas[-1](null_patch).norm() < 1e-12
    # Projective plane pullback preserves every face-corner incidence across shear, compression, and translation.
    rest_incidence = face_planes[:, None] & corner_points[None, :]              # [6, 8] Scalar+
    image_incidence = image_planes[:, None] & placement(corner_points)[None, :] # [6, 8] Scalar+
    assert ((image_incidence - rest_incidence).abs() < 1e-12).all()


def pinned() -> None:
    """The gallery build saves the looping animation of the sheared cube collapsing into a flat sheet."""
    from examples.util.animation import pin_animation
    from examples.mechanics.area_transport import render

    vertices, centres, patches, volumes = scene(thicknesses, shears)
    pin_animation(render.animate(zip(vertices, centres, patches, volumes)), "area_transport", duration_ms,
                  forward_then_back=True)


if __name__ == "__main__":
    main()
