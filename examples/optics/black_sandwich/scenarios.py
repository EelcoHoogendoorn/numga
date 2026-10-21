"""Rotate two identical half-wave plates between fixed crossed polarizers."""

from __future__ import annotations

import numpy as np

from examples.optics.black_sandwich.core import (
    Direction, Field, OpticalMap, half_wave, mv, polarizer, through_pair, time,
)


# --- math
def rotating_plate() -> tuple[Field, Direction, Direction, Direction, OpticalMap]:
    """The field through both stacks, the plates' fast axes, the crossed polarizers' axes, and one
    stack's map, over a swing of the plates from the entrance axis to 90 degrees and back."""
    n_frames = 48
    cycle = np.linspace(0, 2 * np.pi, n_frames, endpoint=False)
    plate_angles = (1 - np.cos(cycle)) * np.pi / 4
    input_axis = mv.x.dual()                                                  # Direction
    output_axis = mv.y.dual()                                                 # Direction

    plate_axes = (mv.xy * (-plate_angles / 2)).exp() >> input_axis            # [frames] Direction
    entrance = polarizer(input_axis)                                          # [] Field <- Field
    plate = half_wave(plate_axes)                                             # [frames] Field <- Field
    exit = polarizer(output_axis)                                             # [] Field <- Field
    sandwich = exit(plate(entrance))                                          # [frames] Field <- Field
    # The field along the entrance axis, which pairs to one with the plane that axis sweeps in time.
    incident = (input_axis & time).dual_inverse() * np.ones(n_frames)         # [frames] Field
    fields = through_pair(entrance, plate, exit, incident)                    # [frames, stages] Field
    return fields, plate_axes, input_axis, output_axis, sandwich


# --- plumbing
def main() -> None:
    from examples.animation import save_animation
    from examples.optics.black_sandwich import render

    duration_ms = 90
    fields, plate_axes, input_axis, output_axis, sandwich = rotating_plate()
    save_animation(render.frames(fields, plate_axes, input_axis, output_axis), "black_sandwich", duration_ms)

    # checks
    # One stack passes the whole field at some plate angle, yet two in succession pass none.
    passed = (fields[:, 3] & (output_axis & time)).to_array()                 # [frames]
    np.testing.assert_allclose(passed.max(), 1, atol=1e-12)
    np.testing.assert_allclose(sandwich(sandwich).kernel, 0, atol=1e-12)
    np.testing.assert_allclose(fields[:, 4:].kernel, 0, atol=1e-12)


if __name__ == "__main__":
    main()
