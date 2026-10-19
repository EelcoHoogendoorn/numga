"""Area transport through invertible, collapsed and orientation-reversing deformations."""

import numpy as np

from numga import stack
from examples.mechanics.area_transport import core


def test_transported_faces_and_volume_agree_with_the_deformed_edges():
    thickness = np.array([1.0, 0.3, 0.0, -0.4])
    shear = np.array([0.0, 0.5, 0.7, 0.2])
    turn = (core.mv.xy * 0.31 + core.mv.yz * 0.17).exp()
    deformation = turn >> core.deformation(thickness, shear)
    areas, _, cofactor, volume_factor = core.transport(deformation)
    corners, _, rest_patches = core.cube(1.0)
    vertices = deformation[:, None](corners)
    carried_patches = areas[:, None](rest_patches)

    edges = vertices[:, [4, 2, 1]] - vertices[:, 0, None]
    first, second, third = (edges[:, i] for i in range(3))
    patches = stack([second ^ third, third ^ first, first ^ second], axis=-1)
    volume = (first ^ second ^ third) / core.mv.xyz
    normals = cofactor[:, None](rest_patches.dual())

    np.testing.assert_allclose((patches - carried_patches[:, :3]).kernel, 0, atol=1e-12)
    np.testing.assert_allclose((volume - volume_factor).kernel, 0, atol=1e-12)
    np.testing.assert_allclose((normals - carried_patches.dual()).kernel, 0, atol=1e-12)
    np.testing.assert_allclose(volume_factor.kernel[..., 0], thickness, atol=1e-12)


def test_the_adjugate_identifies_the_lost_direction_at_collapse():
    thickness = np.array([1.0, 0.2, 0.0])
    shear = np.array([0.0, 0.4, 0.7])
    areas, adjugate, _, _ = core.transport(core.deformation(thickness, shear))
    lost = adjugate[-1](core.mv.xy.dual())
    np.testing.assert_allclose((lost - (core.mv.xy - shear[-1] * core.mv.yz).dual()).kernel, 0, atol=1e-12)
    # A patch through the lost direction: the adjugate's image pairs to nothing with it, and the area map
    # crushes it.
    crushed = (core.mv.z - shear[-1] * core.mv.x) ^ core.mv.y
    np.testing.assert_allclose((lost & crushed).kernel, 0, atol=1e-12)
    np.testing.assert_allclose(areas[-1](crushed).kernel, 0, atol=1e-12)
