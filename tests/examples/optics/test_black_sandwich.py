"""Tests for half-wave transmission, nilpotence, and reflection equivariance of black polarization sandwiches."""

import numpy as np

from numga import stack
from examples.optics.black_sandwich import core, render, scenarios


round_off = 1e-12


def test_half_wave_transmission_and_nilpotence():
    states = list(scenarios.plate_swing_states(5))
    fields = stack([state[0] for state in states])                             # [5, 7] Field+
    plate_axes = stack([state[1] for state in states])                         # [5] Direction-
    sandwiches = stack([state[2] for state in states])                         # [5] Field+ <- Field+

    input_axis, output_axis = scenarios.input_axis, scenarios.output_axis
    plates = 2 * core.polarizer(plate_axes, plate_axes) - core.Field           # [5] Field+ <- Field+
    entrance_sweep, exit_sweep = input_axis & core.time, output_axis & core.time
    passed = fields[:, 3] & exit_sweep                                         # [5] Scalar+
    expected = 2 * (core.mv.x & plate_axes) * (core.mv.y & plate_axes)         # [5] Scalar+
    dyad = core.field_along(output_axis) * (core.Field & entrance_sweep)       # [] Field+ <- Field+
    strength = plates(core.field_along(input_axis)) & exit_sweep               # [5] Scalar+

    np.testing.assert_allclose((plates(plates) - core.Field).kernel, 0, atol=round_off)
    assert ((passed - expected).abs() < round_off).all()
    np.testing.assert_allclose((sandwiches - strength * dyad).kernel, 0, atol=round_off)
    np.testing.assert_allclose(sandwiches(sandwiches).kernel, 0, atol=round_off)
    np.testing.assert_allclose(fields[:, 4:].kernel, 0, atol=round_off)
    assert ((passed.max() - 1).abs() < round_off).all()

    frame = next(render.sandwich_animation(input_axis, output_axis, states[:1]))
    assert frame.ndim == 3 and frame.shape[2] == 3


def test_reflecting_components_reflects_optical_map():
    states = list(scenarios.plate_swing_states(5))
    fields = stack([state[0] for state in states])                             # [5, 7] Field+
    plate_axes = stack([state[1] for state in states])                         # [5] Direction-
    sandwiches = stack([state[2] for state in states])                         # [5] Field+ <- Field+

    mirror = 0.6 * core.mv.x + 0.8 * core.mv.y                                 # [] Vector+
    moved_input = mirror >> scenarios.input_axis                               # [] Direction-
    moved_plate_axes = mirror >> plate_axes                                    # [5] Direction-
    moved_output = mirror >> scenarios.output_axis                             # [] Direction-
    moved_entrance = core.polarizer(moved_input, moved_input)                  # [] Field+ <- Field+
    moved_plates = 2 * core.polarizer(moved_plate_axes, moved_plate_axes) - core.Field  # [5] Field+ <- Field+
    moved_exit = core.polarizer(moved_output, moved_output)                    # [] Field+ <- Field+
    rebuilt = moved_exit(moved_plates(moved_entrance))                         # [5] Field+ <- Field+
    moved = mirror >> sandwiches(mirror << core.Field)                         # [5] Field+ <- Field+
    incident = core.field_along(moved_input)                                   # [] Field+
    moved_fields = core.stack_stage_fields(moved_entrance, moved_plates, moved_exit, incident)  # [5, 7] Field+

    np.testing.assert_allclose((rebuilt - moved).kernel, 0, atol=round_off)
    np.testing.assert_allclose((moved_fields - (mirror >> fields)).kernel, 0, atol=round_off)
    np.testing.assert_allclose(rebuilt(rebuilt).kernel, 0, atol=round_off)
