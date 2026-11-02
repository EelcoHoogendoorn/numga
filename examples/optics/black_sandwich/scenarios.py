"""Rotate two identical half-wave plates between fixed crossed polarizers."""

from __future__ import annotations

import numpy as np

from examples.optics.black_sandwich.core import (
    Direction, Field, OpticalMap, field_along, mv, polarizer, through_pair, time,
)

# The duration of a frame, and the gallery's sandwich: its dots per inch.
duration_ms = 40
gallery_dpi = 720 / 9.2


# --- math
def rotating_plate() -> tuple[Field, Direction, Direction, Direction, OpticalMap]:
    """The field through both stacks, the plates' fast axes, the crossed polarizers' axes, and one
    stack's map, over a swing of the plates from the entrance axis to 90 degrees."""
    n_frames = 108
    # The plates' swing out; the animation plays it forward and back.
    cycle = np.linspace(0, np.pi, n_frames // 2 + 1)
    plate_angles = (1 - np.cos(cycle)) * np.pi / 4
    input_axis = mv.x.dual()                                                  # [] Direction-
    output_axis = mv.y.dual()                                                 # [] Direction-

    plate_turns = (mv.xy * (-plate_angles / 2)).exp()                         # [frames] Rotor+
    plate_axes = plate_turns >> input_axis                                    # [frames] Direction-
    entrance = polarizer(input_axis, input_axis)                              # [] Field+ <- Field+
    # The half-wave plate keeps the field along its fast axis and reverses the field across it.
    plate = 2 * polarizer(plate_axes, plate_axes) - Field                     # [frames] Field+ <- Field+
    exit = polarizer(output_axis, output_axis)                                # [] Field+ <- Field+
    sandwich = exit(plate(entrance))                                          # [frames] Field+ <- Field+
    incident = field_along(input_axis) * np.ones(len(cycle))                  # [frames] Field+
    fields = through_pair(entrance, plate, exit, incident)                    # [frames, stages] Field+
    return fields, plate_axes, input_axis, output_axis, sandwich


# --- plumbing
def main() -> None:
    from examples.util.animation import save_animation
    from examples.optics.black_sandwich import render

    fields, plate_axes, input_axis, output_axis, sandwich = rotating_plate()
    save_animation(render.frames(fields, plate_axes, input_axis, output_axis), "black_sandwich", duration_ms,
                   forward_then_back=True)

    # checks
    # One stack passes the whole field at some plate angle, yet two in succession pass none.
    passed = (fields[:, 3] & (output_axis & time)).to_array()                 # [frames]
    np.testing.assert_allclose(passed.max(), 1, atol=1e-12)
    np.testing.assert_allclose(sandwich(sandwich).kernel, 0, atol=1e-12)
    np.testing.assert_allclose(fields[:, 4:].kernel, 0, atol=1e-12)


def pinned() -> None:
    """The pictures the repository tracks: the plates turning between crossed polarizers, for the gallery."""
    from examples.util.animation import pin_animation
    from examples.optics.black_sandwich import render

    fields, plate_axes, input_axis, output_axis, sandwich = rotating_plate()
    pin_animation(render.sized_frames(fields, plate_axes, input_axis, output_axis, gallery_dpi), "sandwich",
                  duration_ms, forward_then_back=True)


if __name__ == "__main__":
    main()
