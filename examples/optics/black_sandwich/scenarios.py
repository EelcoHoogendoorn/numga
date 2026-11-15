"""Rotating half-wave plate states between fixed crossed polarizers across two identical stacks."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import stack
from examples.optics.black_sandwich.core import (
    Direction, Field, OpticalMap, field_along, mv, polarizer, stack_stage_fields, time,
)

# Frame duration in milliseconds, swing frame count, and gallery rendering DPI.
frame_duration_ms = 40
n_frames = 108
gallery_dpi = 720 / 9.2

# Crossed entrance (`x`) and exit (`y`) polarizer axes.
input_axis = mv.x.dual()                                                       # [] Direction-
output_axis = mv.y.dual()                                                      # [] Direction-


# --- math -----------------------------------------------------------------------------
def plate_swing_states(n_half_frames: int) -> Iterator[tuple[Field, Direction, OpticalMap]]:
    """Yields `(stage_fields, plate_axis, sandwich)` (`[7] Field+`, `[] Direction-`, `[] Field+ <- Field+`)
    across `n_half_frames` angles as both half-wave plates swing smoothly from `0` to `90` degrees."""
    cycle = np.linspace(0, np.pi, n_half_frames)
    plate_angles = (1 - np.cos(cycle)) * np.pi / 4
    entrance = polarizer(input_axis, input_axis)                               # [] Field+ <- Field+
    exit_polarizer = polarizer(output_axis, output_axis)                       # [] Field+ <- Field+
    incident_field = field_along(input_axis)                                   # [] Field+

    for plate_angle in plate_angles:
        plate_rotor = (mv.xy * (-plate_angle / 2)).exp()                       # [] Rotor+
        plate_axis = plate_rotor >> input_axis                                 # [] Direction-
        # A half-wave plate preserves the fast-axis component and reverses the orthogonal transverse component.
        plate = 2 * polarizer(plate_axis, plate_axis) - Field                  # [] Field+ <- Field+
        sandwich = exit_polarizer(plate(entrance))                             # [] Field+ <- Field+
        stage_fields = stack_stage_fields(entrance, plate, exit_polarizer, incident_field)  # [7] Field+
        yield stage_fields, plate_axis, sandwich


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_animation
    from examples.optics.black_sandwich import render

    recorded: list[tuple[Field, Direction, OpticalMap]] = []

    def recording() -> Iterator[tuple[Field, Direction, OpticalMap]]:
        for state in plate_swing_states(n_frames // 2 + 1):
            recorded.append(state)
            yield state

    save_animation(render.sandwich_animation(input_axis, output_axis, recording()), "black_sandwich",
                   frame_duration_ms)

    # checks
    # One stack transmits the full amplitude at 45 degrees, whereas two identical stacks extinguish every angle.
    fields = stack([state[0] for state in recorded])                           # [n_half_frames, 7] Field+
    sandwiches = stack([state[2] for state in recorded])                       # [n_half_frames] Field+ <- Field+
    basis_fields = stack((mv.xy, mv.xz, mv.yz, mv.xt, mv.yt, mv.zt))           # [6] Field+
    transmitted = fields[:, 3] & (output_axis & time)                          # [n_half_frames] Scalar+
    assert ((transmitted.max() - 1).abs() < 1e-12).all()
    assert ((sandwiches(sandwiches)(basis_fields[:, None])[..., None] & basis_fields.dual()).abs() < 1e-12).all()
    assert ((fields[:, 4:, None] & basis_fields.dual()).abs() < 1e-12).all()


def pinned() -> None:
    """Regenerates the gallery GIF `plots/sandwich.gif` of the turning half-wave plates between crossed polarizers."""
    from examples.util.animation import pin_animation
    from examples.optics.black_sandwich import render

    pin_animation(render.sized_sandwich_animation(input_axis, output_axis, gallery_dpi,
                                                  plate_swing_states(n_frames // 2 + 1)),
                  "sandwich", frame_duration_ms)


if __name__ == "__main__":
    main()
