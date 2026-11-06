"""Two spins: the exchange, the steering, the levels in a field and the return to the singlet, computed and drawn
at a few samples."""

from __future__ import annotations

import numpy as np

from examples.quantum.two_spins import core, render, scenarios


def test_the_scenes_draw_at_a_few_samples():
    scenarios.singlet()
    angles, first, second, bells, _ = scenarios.swap(2)
    render.draw_state(first[0], second[0], render.plane_reading)
    render.draw_exchange(angles, bells, first, render.plane_reading)
    planes, probability, steered, _ = scenarios.steering(1, 2, 2)
    next(render.animate_steering(planes, probability, steered, render.plane_reading))
    fields = np.linspace(0.0, scenarios.MAX_FIELD, 2)
    four_first, four_second = core.spin_planes(scenarios.four)
    render.draw_levels(fields, scenarios.field_sweep(fields, scenarios.EXCHANGE_RATE),
                       core.field_plane.dot(four_first + four_second))
    times, rates = np.linspace(0.0, np.pi, 2), scenarios.EXCHANGE_RATES[:1]
    states, back_in_singlet = scenarios.singlet_return(times, rates, scenarios.DIFFERENCE)
    render.draw_qubit(*core.qubit(states), times, back_in_singlet, rates)
