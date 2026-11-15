"""Single-lens imaging, two-lens focusing, and a four-element moving optical train in two-dimensional projective space."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator

import numpy as np

from examples.optics.thin_lens.core import (
    Crossing,
    Line,
    LineMap,
    Plane,
    Point,
    mv,
    optical_train_trace,
    origin,
    thin_lens_map,
    train_elements,
)
from numga import stack

# Axial stations and focal lengths of the two static lenses, plus object position and pupil ray count.
first_station, first_focal = 1.5, 1.0
second_station, second_focal = 2.1, 0.5
subject = mv.point([-2.0, 0.5])                                                 # [] Point-
pupil_height = 0.8
n_rays = 9
# Moving-train object, aim point, fan ray count, axial element stations, motion amplitudes, and loop frame count.
train_subject = mv.point([-1.0, 0.5])                                           # [] Point-
aim = mv.point([1.0, 0.0])                                                      # [] Point-
fan_height = 0.6
n_fan_rays = 7
stations = np.array([1.0, 1.9, 2.2, 3.2])
slide_amplitude, lens_tilt_amplitude, rock_amplitude = 0.3, 0.3, 0.1
mirror_tilt = np.pi / 4
n_frames = 108
frame_duration_ms = 40


# --- math -----------------------------------------------------------------------------
def static_lens_systems() -> tuple[Line, Crossing, Line, Crossing, Plane]:
    """Single-lens point imaging and two-lens parallel focusing produce the two ray-stage bundles (`[2, n_rays] Line-`, `[3, n_rays] Line-`), their image and back-focus crossings (`[] Crossing+`), and the two lens planes (`[2] Plane+`)."""
    first_plane, second_plane = mv.x - mv.w * first_station, mv.x - mv.w * second_station  # [] Plane+, Plane+
    first_lens = thin_lens_map(first_plane, first_focal)                        # [] Line- <- Line-
    second_lens = thin_lens_map(second_plane, second_focal)                     # [] Line- <- Line-

    # Join the subject with pupil points along the first lens plane; any two refracted rays meet at the image.
    axis = origin & mv.direction([1, 0])                                        # [] Line-
    pupil = (first_plane ^ axis) + mv.direction([0, 1]) * np.linspace(-pupil_height, pupil_height, n_rays)  # [n_rays] Point-
    rays = subject & pupil                                                      # [n_rays] Line-
    refracted = first_lens(rays)                                                # [n_rays] Line-
    image = refracted[0] ^ refracted[1]                                         # [] Crossing+

    # Composing both lens maps focuses horizontal rays (joins with the ideal x-direction) onto the back focal point.
    system = second_lens(first_lens)                                            # [] Line- <- Line-
    parallel = mv.direction([1, 0]) & pupil                                     # [n_rays] Line-
    between_lenses = first_lens(parallel)                                       # [n_rays] Line-
    focused = system(parallel)                                                  # [n_rays] Line-
    focus = focused[0] ^ focused[1]                                             # [] Crossing+

    imaging = stack([rays, refracted])                                          # [2, n_rays] Line-
    focusing = stack([parallel, between_lenses, focused])                       # [3, n_rays] Line-
    return imaging, image, focusing, focus, stack([first_plane, second_plane])


def moving_train_states(n_frames: int) -> Iterator[tuple[Point, Plane, Line, LineMap, Crossing]]:
    """Yields `(train_subject, planes, legs, composed, picture)` across `n_frames` phases as the first lens slides and tilts and the mirror rocks."""
    fan = train_subject & (aim + mv.direction([0, 1]) * np.linspace(-fan_height, fan_height, n_fan_rays))  # [n_fan_rays] Line-
    stops = (mv.wx * (-stations / 2)).exp()                                     # [4] Rotor+
    for phase in np.linspace(0.0, 2 * np.pi, n_frames, endpoint=False):
        lens_slide = (mv.wx * (-slide_amplitude * np.sin(phase) / 2)).exp()     # [] Rotor+
        lens_turn = (mv.xy * (lens_tilt_amplitude * np.sin(2 * phase) / 2)).exp()  # [] Rotor+
        mirror_turn = (mv.xy * ((mirror_tilt + rock_amplitude * np.cos(phase)) / 2)).exp()  # [] Rotor+
        motors = (lens_slide * stops[0] * lens_turn, stops[1], stops[2], stops[3] * mirror_turn)
        planes, legs, composed = optical_train_trace(fan, motors, train_elements)
        outgoing = composed(fan)                                                # [n_fan_rays] Line-
        yield train_subject, planes, legs, composed, outgoing[0] ^ outgoing[-1]


# --- plumbing -------------------------------------------------------------------------
def save_states(
    states: Iterable, animate: Callable[..., Iterable[np.ndarray]], *arguments, name: str, duration_ms: int,
) -> list:
    """Streams `states` once through `animate(*arguments, states)` into `save_animation` and returns the collected states."""
    from examples.util.animation import save_animation

    kept = []

    def passing() -> Iterator:
        for state in states:
            kept.append(state)
            yield state

    save_animation(animate(*arguments, passing()), name, duration_ms)
    return kept


def main() -> None:
    """Saves the static ray diagram and moving-train animation once, then checks thin-lens, Gullstrand, and train concurrency."""
    from examples.optics.thin_lens import render
    from examples.util.animation import save_figure

    imaging, image, focusing, focus, planes = static_lens_systems()
    save_figure(render.lenses_figure(subject, imaging, image, focusing, focus, planes), "thin_lens")
    recorded = save_states(moving_train_states(n_frames), render.train_animation, name="thin_lens", duration_ms=frame_duration_ms)

    # checks
    # Every refracted ray passes through the image crossing, signed object and image distances satisfy the
    # thin-lens equation, two-lens parallel focusing matches Gullstrand's back focal distance on the optical axis,
    # and the moving train keeps all outgoing rays concurrent at the image in every recorded frame.
    first_plane, second_plane = planes[0], planes[1]                            # [] Plane+, Plane+
    axis = origin & mv.direction([1, 0])                                        # [] Line-
    assert bool(((focus & axis).abs() < 1e-12).all())
    assert bool(((image & imaging[1]).abs() < 1e-11).all())

    object_distance = -(first_plane & subject) / (mv.w & subject)               # [] Scalar+
    image_distance = (first_plane & image) / (mv.w & image)                     # [] Scalar+
    assert bool(((1 / object_distance + 1 / image_distance - 1 / first_focal).abs() < 1e-12).all())

    gap = (first_plane - second_plane) & origin                                 # [] Scalar+
    effective = 1 / (1 / first_focal + 1 / second_focal - gap / (first_focal * second_focal))  # [] Scalar+
    back_focal = effective * (first_focal - gap) / first_focal                  # [] Scalar+
    assert bool((((second_plane & focus) / (mv.w & focus) - back_focal).abs() < 1e-12).all())

    for _, _, legs, composed, picture in recorded:
        assert bool(((picture & composed(legs[0])).abs() < 1e-10).all())


if __name__ == "__main__":
    main()
