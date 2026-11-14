"""Unit tests cover area transport, flux pullback at flat collapse, projective plane transport, and frame rendering."""

import numpy as np

from numga import stack
from examples.mechanics.area_transport import core, render, scenarios


def test_transported_faces_and_volume_agree_with_deformed_edges():
    thickness = np.array([1.0, 0.3, 0.0, -0.4])
    shear = np.array([0.0, 0.5, 0.7, 0.2])
    turn = (core.mv.xy * 0.31 + core.mv.yz * 0.17).exp()                        # [] Rotor+
    crush = turn >> core.deformation(thickness, shear)                          # [4] Point- <- Point-
    areas, _, volume_factor = core.transport(crush)                             # [4] Patch- <- Patch-, Flux+ <- Flux+, Scalar+
    corners, _, rest_patches = core.cube(1.0)                                   # [8] Point-, [6] Point-, [6] Patch-
    vertices = crush[:, None](corners)                                          # [4, 8] Point-
    image_patches = areas[:, None](rest_patches)                                # [4, 6] Patch-

    edges = vertices[:, [4, 2, 1]] - vertices[:, 0, None]                       # [4, 3] Point-
    x_edge, y_edge, z_edge = edges[:, 0], edges[:, 1], edges[:, 2]              # [4] Point-, Point-, Point-
    edge_patches = stack([y_edge & z_edge, z_edge & x_edge, x_edge & y_edge], axis=-1)  # [4, 3] Patch-
    rest_edges = corners[[4, 2, 1]] - corners[0]                                # [3] Point-
    volume_ratio = (x_edge & y_edge & z_edge) / (rest_edges[0] & rest_edges[1] & rest_edges[2])  # [4] Scalar+

    assert ((edge_patches - image_patches[:, :3]).norm() < 1e-12).all()
    assert ((volume_ratio - volume_factor).abs() < 1e-12).all()
    assert ((volume_factor - thickness).abs() < 1e-12).all()


def test_adjugate_identifies_collapsed_direction():
    thickness = np.array([1.0, 0.2, 0.0])
    shear = np.array([0.0, 0.4, 0.7])
    areas, flux_pullback, _ = core.transport(core.deformation(thickness, shear))  # [3] Patch- <- Patch-, Flux+ <- Flux+, Scalar+
    horizontal_flux = flux_pullback[-1](core.mv.xy)                             # [] Flux+
    assert (horizontal_flux - (core.mv.xy - shear[-1] * core.mv.yz)).norm() < 1e-12
    # A patch spanning the collapsed direction pairs to zero with the pulled-back flux and vanishes under the area map.
    null_patch = (core.mv.z - shear[-1] * core.mv.x).dual() & core.mv.y.dual()  # [] Patch-
    assert (horizontal_flux & null_patch).abs() < 1e-12
    assert areas[-1](null_patch).norm() < 1e-12


def test_projective_plane_transport_preserves_incidence():
    placement = core.projective_placement(0.35, 0.45, 0.5)                      # [] ProjectivePoint- <- ProjectivePoint-
    corner_points, face_planes = core.projective_cube(1.0)                      # [8] ProjectivePoint-, [6] ProjectivePlane+
    image_planes = placement.adjugate().solve(face_planes)                      # [6] ProjectivePlane+
    rest_incidence = face_planes[:, None] & corner_points[None, :]              # [6, 8] Scalar+
    image_incidence = image_planes[:, None] & placement(corner_points)[None, :] # [6, 8] Scalar+
    assert ((image_incidence - rest_incidence).abs() < 1e-12).all()


def test_render_frames_cover_full_and_collapsed_cube():
    vertices, centres, patches, volumes = scenarios.scene(np.array([1.0, 0.0]), np.array([0.0, 0.7]))
    frames = list(render.animate(zip(vertices, centres, patches, volumes)))
    assert len(frames) == 2
    assert frames[0].ndim == 3 and frames[0].shape[-1] == 3
