"""Scenes for projective cameras cast shadows from a point light and a directional sun and image a subject from a stereo camera pair."""

from __future__ import annotations

import numpy as np

from numga import stack
from examples.geometry.projection.core import CUBE_EDGES, circle, cube, mv, shadows, stereo

# The lit body: a cube of this side, lifted this far above the ground; the point light, the sun's direction,
# and the circle of light positions a corner's shadow is swept over, its centre and its count.
BODY_SIDE, BODY_HEIGHT = 1.0, 1.5
POINT_LIGHT = [1, -1, 4]
SUN = [-1, 0.6, -2.5]
LIGHT_PATH_CENTRE = np.array([0, -1, 4])
N_LIGHT_PATH = 24
# The stereo pair: each rig shifted this far either side along x and turned this far toward the other, and
# the subject, a cube of this side this far in front of them.
RIG_SHIFT, RIG_TURN = 0.3, 0.06
SUBJECT_SIDE, SUBJECT_DEPTH = 1.6, 5.0


# --- math -----------------------------------------------------------------------------
def projection():
    """The shadow scene of a lit cube over the ground and the stereo views of a second cube from two camera poses."""
    origin = mv.w.dual()
    body = (mv.zw * (BODY_HEIGHT / 2)).exp() >> cube(BODY_SIDE)               # [corners] Point-
    ground = mv.z
    point_light, sun = mv.point(POINT_LIGHT), mv.direction(SUN)
    path_shift = (stack((mv.xw, mv.yw, mv.zw)) * LIGHT_PATH_CENTRE).sum(axis=0) / 2
    light_path = path_shift.exp() >> circle(N_LIGHT_PATH)                     # [lights] Point-

    # Two copies of the rig, shifted apart along x and turned about y so they converge.
    screen = mv.z - mv.w
    rig_1 = (mv.xw * -RIG_SHIFT).exp() * (mv.xz * +RIG_TURN).exp()
    rig_2 = (mv.xw * +RIG_SHIFT).exp() * (mv.xz * -RIG_TURN).exp()
    subject = (mv.zw * (SUBJECT_DEPTH / 2)).exp() >> cube(SUBJECT_SIDE)       # [corners] Point-

    cast = shadows(body, ground, point_light, sun, body[7], light_path)
    views = stereo(subject, origin, screen, rig_1, rig_2)
    return body, CUBE_EDGES, cast, point_light, sun, views, rig_1, rig_2


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_figure
    from examples.geometry.projection import render

    scene = projection()
    save_figure(render.draw_projection(*scene), "projection")

    # checks
    # Every shadow lies on the ground; corresponding images annihilate the correspondence form, the
    # epipole lies in its kernel, and each epipolar line passes through its image.
    _, _, (point_shadow, sun_shadow, shadow_trail), _, _, (image_1, image_2, epipole_2, lines_2, correspondence), _, _ = scene
    for shadow in (point_shadow, sun_shadow, shadow_trail):
        assert np.abs((mv.z & shadow).to_array()).max() < 1e-12
    assert np.abs(correspondence(image_1, image_2).dual().to_array()).max() < 1e-12
    assert np.abs(correspondence.bind({1: epipole_2}).kernel).max() < 1e-12
    assert (lines_2 & image_2).norm().to_array().max() < 1e-12


if __name__ == "__main__":
    main()
