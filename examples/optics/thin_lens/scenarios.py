"""Scenes of the thin-lens example: one lens and two lenses, and an animated optical train."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from examples.optics.thin_lens.core import (
    elements, Crossing, Line, LineMap, Motor, Plane, Point, mv, origin, thin_lens, trace,
)

# The two lenses: where each stands along x, and its focal length.
FIRST_LENS, FIRST_FOCAL = 1.5, 1.0
SECOND_LENS, SECOND_FOCAL = 2.1, 0.5
# The object imaged by the first lens, and the half-height of the pupil its rays pass through, in rays.
OBJECT = [-2.0, 0.5]
PUPIL = 0.8
N_RAYS = 9
# Where the parallel rays start, ahead of the first lens, and how far past the focus each bundle is drawn.
LEAD = 1.0
PAST = 0.5
# The train: its subject, the point the fan of rays aims at, the fan's half-height in rays; where each
# element stands along x; how far the first lens slides and tilts and the mirror rocks; and the mirror's
# tilt at rest.
SUBJECT = [-1.0, 0.5]
AIM = [1.0, 0.0]
FAN = 0.6
N_FAN_RAYS = 7
STATIONS = np.array([1.0, 1.9, 2.2, 3.2])
SLIDE, LENS_TILT, ROCK = 0.3, 0.3, 0.1
MIRROR_TILT = np.pi / 4
# The frames of the train's loop, and the duration of a frame in milliseconds.
N_FRAMES = 72
FRAME_DURATION = 60


# --- math -----------------------------------------------------------------------------
def lenses():
    """Image an object point through one lens, and focus parallel rays through two.

    Returns, for each system, its legs (rays, start plane, stop plane) and its lens planes, and the image
    and the focus.
    """
    # Each lens is a plane (a vertical line) and a focal length; nothing else is stored.
    first_plane, second_plane = mv.x - mv.w * FIRST_LENS, mv.x - mv.w * SECOND_LENS   # [] Plane each
    first_lens, second_lens = thin_lens(first_plane, FIRST_FOCAL), thin_lens(second_plane, SECOND_FOCAL)

    # 1. Rays from an object point through a pupil of points on the lens plane. Every transformed ray
    #    passes through one point: the image, the meet of any two of them. The pupil starts where the
    #    lens plane meets the optical axis, the join of the origin and the ideal point along x, and
    #    steps along the ideal point along y.
    subject = mv.point(OBJECT)                                                  # Point
    axis = origin & mv.direction([1, 0])                                        # Line
    pupil = (first_plane ^ axis) + mv.direction([0, 1]) * np.linspace(-PUPIL, PUPIL, N_RAYS)  # Point
    rays = subject & pupil                                                      # Line
    out = first_lens(rays)                                                      # Line
    image = out[0] ^ out[1]                                                     # Crossing

    # 2. Two lenses. The system is a composition of maps. Parallel rays are joins with the ideal point
    #    along x; where they meet after the system is the back focal point.
    system = second_lens(first_lens)                                            # LineMap
    parallel = mv.direction([1, 0]) & pupil                                     # Line
    focused = system(parallel)                                                  # Line
    focus = focused[0] ^ focused[1]                                             # Crossing

    # Draw each bundle between planes: from a vertical through its start, to a little past its end.
    beyond = (mv.wx * (-PAST / 2)).exp()                                        # Motor
    vertical = mv.direction([0, 1])                                             # Point
    one_lens = [(rays, subject & vertical, first_plane), (out, first_plane, beyond >> (image & vertical))]
    two_lenses = [
        (parallel, (mv.wx * (LEAD / 2)).exp() >> first_plane, first_plane),
        (first_lens(parallel), first_plane, second_plane),
        (focused, second_plane, beyond >> (focus & vertical)),
    ]
    return one_lens, [first_plane], two_lenses, [first_plane, second_plane], image, focus


def train(n_frames: int) -> Iterator[tuple]:
    """An optical train of a lens, a prism, a lens and a mirror, with the first lens and the mirror moving.

    A motor per element places it. A lens still images when tilted, because the ideal thin
    lens is a collineation and concurrent rays stay concurrent. The first lens slides and
    tilts and the mirror rocks; the image is the meet of two output rays.

    Yields per frame the subject, the element planes, the bundle before and after each
    element, the composed train, and the image.
    """
    subject = mv.point(SUBJECT)                                                 # Point
    fan = subject & (mv.point(AIM) + mv.direction([0, 1]) * np.linspace(-FAN, FAN, N_FAN_RAYS))  # Line
    for t in np.linspace(0.0, 2 * np.pi, n_frames, endpoint=False):
        motors = (
            # The first lens slides and tilts.
            (mv.wx * (-(STATIONS[0] + SLIDE * np.sin(t)) / 2)).exp() * (mv.xy * (LENS_TILT * np.sin(2 * t) / 2)).exp(),
            # The prism and the second lens are fixed.
            (mv.wx * (-STATIONS[1] / 2)).exp(),
            (mv.wx * (-STATIONS[2] / 2)).exp(),
            # The mirror rocks about its pivot.
            (mv.wx * (-STATIONS[3] / 2)).exp() * (mv.xy * ((MIRROR_TILT + ROCK * np.cos(t)) / 2)).exp(),
        )
        planes, legs, composed = trace(fan, motors, elements)
        back = composed(fan)                                                    # Line
        yield subject, planes, legs, composed, back[0] ^ back[-1]


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.optics.thin_lens import render

    one_lens, one_planes, two_lenses, two_planes, image, focus = lenses()
    save_figure(render.draw_lenses(one_lens, one_planes, two_lenses, two_planes), "thin_lens")
    save_animation(render.animate_train(train(N_FRAMES)), "thin_lens", FRAME_DURATION)

    # checks
    # Every transformed ray passes through the image. A point's incidence with the lens plane over its
    # incidence with the line at infinity, its weight, is its signed distance, which gives the thin lens
    # equation and, for two lenses, Gullstrand's back focal distance. Through the train every output ray
    # passes through the image, wherever the elements stand.
    first_plane, second_plane = two_planes
    out = one_lens[1][0]
    subject = mv.point(OBJECT)
    np.testing.assert_allclose((out ^ image).kernel, 0.0, atol=1e-11)
    object_distance = -(first_plane & subject) / (mv.w & subject)
    image_distance = (first_plane & image) / (mv.w & image)
    np.testing.assert_allclose((1 / object_distance + 1 / image_distance).to_array(), 1 / FIRST_FOCAL, atol=1e-12)
    gap = (first_plane - second_plane) & origin
    effective = 1 / (1 / FIRST_FOCAL + 1 / SECOND_FOCAL - gap / (FIRST_FOCAL * SECOND_FOCAL))
    back_focal = effective * (FIRST_FOCAL - gap) / FIRST_FOCAL
    np.testing.assert_allclose(((second_plane & focus) / (mv.w & focus)).to_array(), back_focal.to_array(), atol=1e-12)
    np.testing.assert_allclose((mv.y ^ focus).kernel, 0.0, atol=1e-12)
    for _, _, legs, composed, picture in train(N_FRAMES):
        np.testing.assert_allclose((composed(legs[0]) ^ picture).kernel, 0.0, atol=1e-10)


if __name__ == "__main__":
    main()
