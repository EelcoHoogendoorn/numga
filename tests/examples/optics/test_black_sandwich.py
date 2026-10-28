"""The two optical stacks extinguish every input, in any orientation of the apparatus."""

import numpy as np

from examples.optics.black_sandwich import core
from examples.optics.black_sandwich.scenarios import rotating_plate


round_off = 1e-12


def test_half_wave_transmission_and_nilpotence():
    fields, plate_axes, input_axis, output_axis, sandwich = rotating_plate()
    plate = 2 * core.polarizer(plate_axes, plate_axes) - core.Field
    entrance_sweep, exit_sweep = input_axis & core.time, output_axis & core.time
    # One stack passes the field turned by twice the plate's angle, read along the exit axis.
    passed = fields[:, 3] & exit_sweep
    expected = 2 * (core.mv.x & plate_axes) * (core.mv.y & plate_axes)
    dyad = core.field_along(output_axis) * (core.Field & entrance_sweep)
    strength = plate(core.field_along(input_axis)) & exit_sweep

    # checks
    np.testing.assert_allclose((plate(plate) - core.Field).kernel, 0, atol=round_off, rtol=0)
    np.testing.assert_allclose((passed - expected).kernel, 0, atol=round_off, rtol=0)
    np.testing.assert_allclose((sandwich - strength * dyad).kernel, 0, atol=round_off, rtol=0)
    np.testing.assert_allclose(sandwich(sandwich).kernel, 0, atol=round_off, rtol=0)
    np.testing.assert_allclose(fields[:, 4:].kernel, 0, atol=round_off, rtol=0)
    np.testing.assert_allclose(passed.kernel.max(), 1, atol=round_off, rtol=0)


def test_moving_the_components_moves_the_whole_map():
    fields, plate_axes, input_axis, output_axis, sandwich = rotating_plate()
    mirror = 0.6 * core.mv.x + 0.8 * core.mv.y
    moved_input, moved_plate_axes, moved_output = mirror >> input_axis, mirror >> plate_axes, mirror >> output_axis
    moved_entrance = core.polarizer(moved_input, moved_input)
    moved_plate = 2 * core.polarizer(moved_plate_axes, moved_plate_axes) - core.Field
    moved_exit = core.polarizer(moved_output, moved_output)
    rebuilt = moved_exit(moved_plate(moved_entrance))
    moved = mirror >> sandwich(mirror << core.Field)
    incident = core.field_along(moved_input) * np.ones(len(plate_axes))
    moved_fields = core.through_pair(moved_entrance, moved_plate, moved_exit, incident)

    # checks
    np.testing.assert_allclose((rebuilt - moved).kernel, 0, atol=round_off, rtol=0)
    np.testing.assert_allclose((moved_fields - (mirror >> fields)).kernel, 0, atol=round_off, rtol=0)
    np.testing.assert_allclose(rebuilt(rebuilt).kernel, 0, atol=round_off, rtol=0)
