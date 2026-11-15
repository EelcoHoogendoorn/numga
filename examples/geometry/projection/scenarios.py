"""Scene parameters and figure runner for shadow casting and stereo camera projection."""

from __future__ import annotations

import numpy as np

from numga import stack
from examples.geometry.projection.core import circle, cube, mv, shadows, stereo

cube_edges = np.array([
    [0, 1], [1, 3], [3, 2], [2, 0],
    [4, 5], [5, 7], [7, 6], [6, 4],
    [0, 4], [1, 5], [2, 6], [3, 7],
])                                                                              # [12, 2]
body_side, body_height = 1.0, 1.5
point_light_coords = [1, -1, 4]
sun_direction = [-1, 0.6, -2.5]
light_path_centre = np.array([0, -1, 4])
n_light_path = 24

rig_shift, rig_turn = 0.3, 0.06
subject_side, subject_depth = 1.6, 5.0


# --- math -----------------------------------------------------------------------------
def projection_scene():
    """A lifted cube casts shadows onto the ground and a second cube is imaged from two stereo camera poses."""
    origin = mv.w.dual()                                                        # [] Point-
    body_lift = (mv.zw * (body_height / 2)).exp()                               # [] Motor+
    body = body_lift >> cube(body_side)                                         # [8] Point-
    ground = mv.z                                                               # [] Plane+
    point_light, sun = mv.point(point_light_coords), mv.direction(sun_direction)  # [] Point-, [] Point-
    path_shift = ((stack((mv.xw, mv.yw, mv.zw)) * light_path_centre).sum(axis=0) / 2).exp()  # [] Motor+
    light_path = path_shift >> circle(n_light_path)                             # [n_light_path] Point-

    # Shift two copies of the camera apart along x and turn them inward toward the subject.
    screen = mv.z - mv.w                                                        # [] Plane+
    rig_1 = (mv.xw * -rig_shift).exp() * (mv.xz * +rig_turn).exp()              # [] Motor+
    rig_2 = (mv.xw * +rig_shift).exp() * (mv.xz * -rig_turn).exp()              # [] Motor+
    subject_lift = (mv.zw * (subject_depth / 2)).exp()                          # [] Motor+
    subject = subject_lift >> cube(subject_side)                                # [8] Point-

    cast = shadows(body, ground, point_light, sun, body[7], light_path)
    views = stereo(subject, origin, screen, rig_1, rig_2)
    return body, cube_edges, cast, point_light, sun, views, rig_1, rig_2


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_figure
    from examples.geometry.projection import render

    scene = projection_scene()
    save_figure(render.projection_figure(*scene), "projection")

    # checks
    # Every shadow lies on the ground; corresponding images annihilate the correspondence form, the
    # epipole lies in its kernel, and each epipolar line passes through its image.
    _, _, (point_shadow, sun_shadow, shadow_trail), _, _, (image_1, image_2, epipole_2, lines_2, correspondence), _, _ = scene
    for shadow in (point_shadow, sun_shadow, shadow_trail):
        assert ((mv.z & shadow).abs() < 1e-12).all()
    assert (correspondence(image_1, image_2).dual().abs() < 1e-12).all()
    assert (correspondence(image_1, epipole_2).dual().abs() < 1e-12).all()
    assert ((lines_2 & image_2).norm() < 1e-12).all()


if __name__ == "__main__":
    main()
