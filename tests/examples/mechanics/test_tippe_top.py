"""Coarse quadrature samples verify one contact step and exercise setup, animation, and inclination rendering."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from examples.mechanics.tippe_top import core, render, scenarios

coarse_spacing = 0.5


def test_launched_top_steps_and_renders() -> None:
    samples = scenarios.quadrature_grid(coarse_spacing)                      # [...] Point-
    mass, inertia = core.mass_distribution(scenarios.parts, samples, coarse_spacing**3)  # [] Point-, Forque- <- Rate+
    assert (core.mv.w & mass) > 0

    motor, rate = core.release_states(scenarios.contacts, mass, scenarios.spins, scenarios.lean)  # [2] Motor+, Rate+
    states = list(core.motion(
        scenarios.contacts, mass, inertia, motor, rate, scenarios.gravity,
        scenarios.static_friction, scenarios.dynamic_friction, scenarios.step, 2, 1,
    ))
    stepped_motor, _ = states[-1]                                            # [2] Motor+
    assert ((stepped_motor.norm() - 1).abs() < 1e-12).all()

    plt.close(render.setup_figure(scenarios.parts, mass))
    frame = next(render.frames(scenarios.parts, mass, render.n_smoothing, 1, [states[-1]] * render.n_window))
    assert frame.shape == (render.n_pixels, 2 * render.n_pixels + 6, 3)

    placements = stack([placement for placement, _ in states])               # [2, 2] Motor+
    stem_directions = placements >> core.up                                  # [2, 2] Point-
    vertical = core.mv.z & stem_directions                                   # [2, 2] Scalar+
    plt.close(render.inclination_figure(np.array([0.0, scenarios.step]), vertical))
