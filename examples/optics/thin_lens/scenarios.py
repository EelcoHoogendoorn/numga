"""Scenes of the thin-lens example: one lens and two lenses, and an animated optical train."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import stack

from examples.optics.thin_lens.core import (
    elements, Crossing, Line, LineMap, Plane, Point, mv, origin, thin_lens, trace,
)

# The two lenses: where each stands along x, and its focal length.
first_station, first_focal = 1.5, 1.0
second_station, second_focal = 2.1, 0.5
# The subject imaged by the first lens, and the half-height of the pupil its rays pass through, in rays.
subject = mv.point([-2.0, 0.5])                                                 # [] Point-
pupil_height = 0.8
n_rays = 9
# The train: its subject, the point the fan of rays aims at, the fan's half-height in rays; where each
# element stands along x; how far the first lens slides and tilts and the mirror rocks; and the mirror's
# tilt at rest.
train_subject = mv.point([-1.0, 0.5])                                           # [] Point-
aim = mv.point([1.0, 0.0])                                                      # [] Point-
fan_height = 0.6
n_fan_rays = 7
stations = np.array([1.0, 1.9, 2.2, 3.2])
slide, lens_tilt, rock = 0.3, 0.3, 0.1
mirror_tilt = np.pi / 4
# The frames of the train's loop, and the duration of a frame in milliseconds.
n_frames = 72
frame_duration = 60


# --- math -----------------------------------------------------------------------------
def lenses() -> tuple[Line, Crossing, Line, Crossing, Plane]:
    """One lens imaging the subject, and two lenses focusing parallel rays.

    The result is each system's bundle before and after each lens [legs, rays], with the image and the
    focus, and the planes of the two lenses.
    """
    # Each lens is a plane and a focal length; nothing else is stored.
    first_plane, second_plane = mv.x - mv.w * first_station, mv.x - mv.w * second_station   # [] Plane+, Plane+
    first_lens = thin_lens(first_plane, first_focal)                            # [] Line- <- Line-
    second_lens = thin_lens(second_plane, second_focal)                         # [] Line- <- Line-

    # 1. Rays from the subject through a pupil of points on the lens plane. Every refracted ray
    #    passes through one point: the image, the meet of any two of them. The pupil starts where the
    #    lens plane meets the optical axis, the join of the origin and the ideal point along x, and
    #    steps along the ideal point along y.
    axis = origin & mv.direction([1, 0])                                        # [] Line-
    pupil = (first_plane ^ axis) + mv.direction([0, 1]) * np.linspace(-pupil_height, pupil_height, n_rays)   # [rays] Point-
    rays = subject & pupil                                                      # [rays] Line-
    refracted = first_lens(rays)                                                # [rays] Line-
    image = refracted[0] ^ refracted[1]                                         # [] Crossing+

    # 2. Two lenses. The system is a composition of maps. Parallel rays are joins with the ideal point
    #    along x; where they meet after the system is the back focal point.
    system = second_lens(first_lens)                                            # [] Line- <- Line-
    parallel = mv.direction([1, 0]) & pupil                                    # [rays] Line-
    between_lenses = first_lens(parallel)                                       # [rays] Line-
    focused = system(parallel)                                                  # [rays] Line-
    focus = focused[0] ^ focused[1]                                             # [] Crossing+

    imaging = stack([rays, refracted])                                          # [legs, rays] Line-
    focusing = stack([parallel, between_lenses, focused])                       # [legs, rays] Line-
    return imaging, image, focusing, focus, stack([first_plane, second_plane])


def train(n_frames: int) -> Iterator[tuple[Point, Plane, Line, LineMap, Crossing]]:
    """An optical train of a lens, a prism, a lens and a mirror, with the first lens and the mirror moving.

    A motor per element places it. A lens still images when tilted, because the ideal thin
    lens is a collineation and concurrent rays stay concurrent. The first lens slides and
    tilts and the mirror rocks; the image is the meet of two output rays.

    Yields per frame the subject, the element planes, the bundle before and after each
    element, the composed train, and the image.
    """
    fan = train_subject & (aim + mv.direction([0, 1]) * np.linspace(-fan_height, fan_height, n_fan_rays))   # [rays] Line-
    # Each element's motor onto its station.
    stops = (mv.wx * (-stations / 2)).exp()                                     # [elements] Motor+
    for phase in np.linspace(0.0, 2 * np.pi, n_frames, endpoint=False):
        # The first lens slides and tilts, and the mirror rocks about its pivot; the prism and the second lens are fixed.
        lens_slide = (mv.wx * (-slide * np.sin(phase) / 2)).exp()               # [] Motor+
        lens_turn = (mv.xy * (lens_tilt * np.sin(2 * phase) / 2)).exp()         # [] Motor+
        mirror_turn = (mv.xy * ((mirror_tilt + rock * np.cos(phase)) / 2)).exp()   # [] Motor+
        motors = (lens_slide * stops[0] * lens_turn, stops[1], stops[2], stops[3] * mirror_turn)
        planes, legs, composed = trace(fan, motors, elements)
        outgoing = composed(fan)                                                # [rays] Line-
        yield train_subject, planes, legs, composed, outgoing[0] ^ outgoing[-1]


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.optics.thin_lens import render

    imaging, image, focusing, focus, planes = lenses()
    save_figure(render.draw_lenses(subject, imaging, image, focusing, focus, planes), "thin_lens")
    save_animation(render.animate_train(train(n_frames)), "thin_lens", frame_duration)

    # checks
    # Every refracted ray passes through the image. A point's incidence with the lens plane over its
    # incidence with the plane at infinity, its weight, is its signed distance, which gives the thin lens
    # equation and, for two lenses, Gullstrand's back focal distance. Through the train every output ray
    # passes through the image, wherever the elements stand.
    first_plane, second_plane = planes[0], planes[1]
    np.testing.assert_allclose((imaging[1] ^ image).kernel, 0.0, atol=1e-11)
    object_distance = -(first_plane & subject) / (mv.w & subject)
    image_distance = (first_plane & image) / (mv.w & image)
    np.testing.assert_allclose((1 / object_distance + 1 / image_distance).to_array(), 1 / first_focal, atol=1e-12)
    gap = (first_plane - second_plane) & origin
    effective = 1 / (1 / first_focal + 1 / second_focal - gap / (first_focal * second_focal))
    back_focal = effective * (first_focal - gap) / first_focal
    np.testing.assert_allclose(((second_plane & focus) / (mv.w & focus)).to_array(), back_focal.to_array(), atol=1e-12)
    np.testing.assert_allclose((mv.y ^ focus).kernel, 0.0, atol=1e-12)
    for _, _, legs, composed, picture in train(n_frames):
        np.testing.assert_allclose((composed(legs[0]) ^ picture).kernel, 0.0, atol=1e-10)


if __name__ == "__main__":
    main()
