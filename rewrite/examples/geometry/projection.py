"""Projective cameras, shadows and epipolar geometry in PGA3D = R_{3,0,1}.

The usual pipeline builds a 4x4 projection matrix and applies it at the very end.
Here the camera is an expression with a hole, (centre ∨ point) ∧ screen, and the
matrix is what you get by choosing which hole to leave open. Every question below is
the same expression with a different slot open.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples import PLOT_DIR
from examples.geometry.projection_plumbing import (
    ga,
    Line,
    Point,
    Pseudoscalar,
    direction,
    circle,
    cube,
    mv,
    origin,
    point,
    render_shadow_scene,
    render_stereo_scene,
)

# Map types (GATypes) read output <= inputs:
Camera = ga.gatype((Point, Point))                           # image point <= world point
LineCamera = ga.gatype((Line, Line))                         # image line <= world line
ShadowTrail = ga.gatype((Point, Point))                      # shadow <= light position
Correspondence = ga.gatype((Pseudoscalar, Point, Point))     # ray incidence <= (image 1, image 2)


def main(plot_path: str = str(PLOT_DIR / "projection.png")) -> plt.Figure:
    """Step through the projective camera as a late-bound extensor and draw the results."""

    # -----------------------------------------------------------------------
    # 1. Shadows: leave the point open
    # -----------------------------------------------------------------------
    body = (mv.zw * 0.75).exp() >> cube(1.0)
    ground = mv.z
    point_light = point(np.array([1.0, -1.0, 4.0]))
    sun = direction(np.array([-1.0, 0.6, -2.5]))

    # Join the point with the light into a ray, then meet the ray with the ground. With the
    # point slot open this is a linear map Point -> Point: the shadow matrix. Nothing in the
    # expression cares whether the light is a finite point or a direction at infinity, so
    # the sun's orthographic shadow is the same line of code.
    spotlight: Camera = point_light.regressive(Point).wedge(ground)
    sunlight: Camera = sun.regressive(Point).wedge(ground)
    point_shadow = spotlight(body)
    sun_shadow = sunlight(body)

    # Bind a different slot and you ask a different question. Fix one corner and leave
    # the light open: the corner's shadow is now a linear map of the light position, so a
    # whole path of light positions binds in one call. A fixed pipeline cannot express this
    # without rebuilding its matrix per light.
    corner = body[7]
    shadow_of_corner: ShadowTrail = Point.regressive(corner).wedge(ground)
    light_path = (mv.yw * -0.5 + mv.zw * 2.0).exp() >> circle(24)
    shadow_trail = shadow_of_corner(light_path)
    np.testing.assert_allclose(shadow_of_corner(point_light).kernel, point_shadow[7].kernel, atol=1e-14)

    # -----------------------------------------------------------------------
    # 2. Cameras: one rig, moved as a map
    # -----------------------------------------------------------------------
    # A canonical rig: pinhole at the origin, screen z = 1. The same join-then-meet with a
    # line in the open slot is the camera for lines.
    screen = mv.z - mv.w
    camera: Camera = origin.regressive(Point).wedge(screen)
    line_camera: LineCamera = origin.regressive(Line).wedge(screen)

    # Two copies of the rig, each translated 0.6 along x and turned 0.12 rad about the y
    # axis so they converge. The camera moves as a map: pull world points back into the rig
    # frame, push image points forward. Because the map is an extensor and not a matrix
    # baked at the end, the line camera moves the same way and stays consistent with the
    # point camera, image of a join = join of the images.
    rig_1 = (mv.xw * -0.3).exp() * (mv.xz * +0.06).exp()
    rig_2 = (mv.xw * +0.3).exp() * (mv.xz * -0.06).exp()
    camera_1 = rig_1 >> camera(rig_1 << Point)
    camera_2 = rig_2 >> camera(rig_2 << Point)
    line_camera_2 = rig_2 >> line_camera(rig_2 << Line)
    centre_1 = rig_1 >> origin
    centre_2 = rig_2 >> origin

    subject = (mv.zw * 2.5).exp() >> cube(1.6)
    image_1 = camera_1(subject)
    image_2 = camera_2(subject)

    # -----------------------------------------------------------------------
    # 3. Epipolar geometry: leave the point slots of two rays open
    # -----------------------------------------------------------------------
    # The epipole is the image of the other centre. The epipolar line of an image point is
    # the image of its ray, which the line camera gives directly.
    epipole_2 = camera_2(centre_1)
    epipolar_lines_2 = line_camera_2(centre_1.regressive(image_1))

    # Two image points correspond iff their rays meet, and two lines meet iff their wedge
    # vanishes. Leave both point slots open and that sentence is a bilinear form whose
    # kernel is the fundamental matrix. There is no matrix to slap on at the end here; the
    # form exists only because the rays were never evaluated.
    correspondence: Correspondence = centre_1.regressive(Point).wedge(centre_2.regressive(Point))
    np.testing.assert_allclose(correspondence(image_1, image_2).kernel, 0.0, atol=1e-12)
    np.testing.assert_allclose(correspondence.bind({1: epipole_2}).kernel, 0.0, atol=1e-12)
    np.testing.assert_allclose(epipolar_lines_2.regressive(image_2).kernel, 0.0, atol=1e-12)

    # -----------------------------------------------------------------------
    # 4. Draw
    # -----------------------------------------------------------------------
    fig = plt.figure(figsize=(16, 5), dpi=120)
    render_shadow_scene(fig.add_subplot(1, 3, 1, projection="3d"), body, point_light, sun, point_shadow, sun_shadow, shadow_trail)
    render_stereo_scene(fig.add_subplot(1, 3, 2), fig.add_subplot(1, 3, 3), rig_1, rig_2, image_1, image_2, epipolar_lines_2)
    plt.tight_layout()
    if plot_path:
        plt.savefig(plot_path, bbox_inches="tight")
        print(f"Figure saved to {plot_path}")
    return fig


if __name__ == "__main__":
    main()
    plt.show()
