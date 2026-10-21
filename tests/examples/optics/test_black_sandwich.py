"""The two optical stacks extinguish every input, in any orientation of the apparatus."""

import numpy as np

from examples.optics.black_sandwich import core
from examples.optics.black_sandwich.scenarios import rotating_plate


ROUND_OFF = 1e-12


def test_half_wave_transmission_and_nilpotence():
    fields, plate_axes, input_axis, output_axis, sandwich = rotating_plate()
    plate = core.half_wave(plate_axes)
    entrance_sweep, exit_sweep = input_axis & core.time, output_axis & core.time
    # One stack passes the field turned by twice the plate's angle, read along the exit axis.
    passed = fields[:, 3] & exit_sweep
    expected = 2 * (core.mv.x & plate_axes) * (core.mv.y & plate_axes)
    dyad = exit_sweep.dual_inverse() * (core.Field & entrance_sweep)
    strength = plate(entrance_sweep.dual_inverse()) & exit_sweep

    # checks
    np.testing.assert_allclose((plate(plate) - core.Field).kernel, 0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose((passed - expected).kernel, 0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose((sandwich - strength * dyad).kernel, 0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose(sandwich(sandwich).kernel, 0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose(fields[:, 4:].kernel, 0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose(passed.kernel.max(), 1, atol=ROUND_OFF, rtol=0)


def test_moving_the_components_moves_the_whole_map():
    fields, plate_axes, input_axis, output_axis, sandwich = rotating_plate()
    mirror = 0.6 * core.mv.x + 0.8 * core.mv.y
    moved_entrance = core.polarizer(mirror >> input_axis)
    moved_plate = core.half_wave(mirror >> plate_axes)
    moved_exit = core.polarizer(mirror >> output_axis)
    rebuilt = moved_exit(moved_plate(moved_entrance))
    moved = mirror >> sandwich(mirror << core.Field)
    incident = ((mirror >> input_axis) & core.time).dual_inverse() * np.ones(len(plate_axes))
    moved_fields = core.through_pair(moved_entrance, moved_plate, moved_exit, incident)

    # checks
    np.testing.assert_allclose((rebuilt - moved).kernel, 0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose((moved_fields - (mirror >> fields)).kernel, 0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose(rebuilt(rebuilt).kernel, 0, atol=ROUND_OFF, rtol=0)
