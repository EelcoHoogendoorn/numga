"""Scenes of the thin-lens example: one lens and two lenses, and an animated optical train."""

from __future__ import annotations

import numpy as np

from examples.optics.thin_lens.core import (
    ELEMENTS, Line, LineMap, Motor, Point, mv, origin, point, thin_lens, trace,
)


def lenses():
    """Image an object point through one lens, and focus parallel rays through two.

    Returns, for each system, its legs (rays, start plane, stop plane) and its lens planes.
    """
    # Each lens is a plane (a vertical line) and a focal length; nothing else is stored.
    plane_1, focal_1 = mv.x - mv.w * 1.5, 1.0
    plane_2, focal_2 = mv.x - mv.w * 2.1, 0.5
    lens_1, lens_2 = thin_lens(plane_1, focal_1), thin_lens(plane_2, focal_2)

    # 1. Rays from an object point through a pupil of points on the lens plane. Every
    #    transformed ray passes through one point: the image, the meet of any two of them.
    obj: Point = point(-2.0, 0.5)
    pupil: Point = plane_1 ^ (mv.y - mv.w * np.linspace(-0.8, 0.8, 9))
    rays: Line = obj & pupil
    out: Line = lens_1(rays)
    image: Point = out[0] ^ out[1]

    # 2. Two lenses. The system is a composition of maps. Parallel rays are joins with the
    #    ideal point along x; where they meet after the system is the back focal point.
    system: LineMap = lens_2(lens_1)
    parallel: Line = mv.yw & pupil
    focused: Line = system(parallel)
    focus: Point = focused[0] ^ focused[1]

    # Draw each bundle between planes: from a vertical through its start, to half a unit past its end.
    beyond: Motor = (mv.wx * -0.25).exp()
    one_lens = [(rays, obj & mv.wx, plane_1), (out, plane_1, beyond >> (image & mv.wx))]
    two_lenses = [
        (parallel, (mv.wx * 0.5).exp() >> plane_1, plane_1),
        (lens_1(parallel), plane_1, plane_2),
        (focused, plane_2, beyond >> (focus & mv.wx)),
    ]

    # --- checks -------------------------------------------------------------
    # Every transformed ray passes through the image. A point's incidence with the lens plane
    # over its incidence with the line at infinity (its weight) is its signed distance, which
    # gives the thin lens equation and, for two lenses, Gullstrand's back focal distance.
    np.testing.assert_allclose((out ^ image).kernel, 0.0, atol=1e-11)
    d_obj = -(plane_1 & obj) / (mv.w & obj)
    d_img = (plane_1 & image) / (mv.w & image)
    np.testing.assert_allclose((1 / d_obj + 1 / d_img).to_array(), 1 / focal_1, atol=1e-12)
    gap = (plane_1 - plane_2) & origin
    f_eff = 1 / (1 / focal_1 + 1 / focal_2 - gap / (focal_1 * focal_2))
    back_focal = f_eff * (focal_1 - gap) / focal_1
    np.testing.assert_allclose(((plane_2 & focus) / (mv.w & focus)).to_array(), back_focal.to_array(), atol=1e-12)
    np.testing.assert_allclose((mv.y ^ focus).kernel, 0.0, atol=1e-12)

    return one_lens, [plane_1], two_lenses, [plane_1, plane_2]


def train(frames: int):
    """An optical train of a lens, a prism, a lens and a mirror, with the first lens and the mirror moving.

    A motor per element places it. A lens still images when tilted, because the ideal thin
    lens is a collineation and concurrent rays stay concurrent. The first lens slides and
    tilts and the mirror rocks; the image is the meet of two output rays.

    Yields per frame the subject, the element planes, the bundle before and after each
    element, the composed train, and the image.
    """
    subject: Point = point(-1.0, 0.5)
    fan: Line = subject & point(1.0, np.linspace(-0.6, 0.6, 7))
    for t in np.linspace(0.0, 2 * np.pi, frames, endpoint=False):
        motors = (
            # The first lens slides and tilts.
            (mv.wx * (-(1.0 + 0.3 * np.sin(t)) / 2)).exp() * (mv.xy * (0.3 * np.sin(2 * t) / 2)).exp(),
            # The prism and the second lens are fixed.
            (mv.wx * (-1.9 / 2)).exp(),
            (mv.wx * (-2.2 / 2)).exp(),
            # The mirror rocks about its pivot.
            (mv.wx * (-3.2 / 2)).exp() * (mv.xy * ((np.pi / 4 + 0.1 * np.cos(t)) / 2)).exp(),
        )
        planes, legs, composed = trace(fan, motors, ELEMENTS)
        back: Line = composed(fan)
        yield subject, planes, legs, composed, back[0] ^ back[-1]


if __name__ == "__main__":
    from examples.animation import save_animation, save_figure
    from examples.optics.thin_lens import render

    save_figure(render.draw_lenses(*lenses()), "thin_lens")
    save_animation(render.animate_train(train(72)), "thin_lens", 60)
