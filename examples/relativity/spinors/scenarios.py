"""Scenario parameters and figure generation cover double-cover spinor rotation, Weyl currents, and Majorana phase sweeps."""

from __future__ import annotations

import numpy as np

from numga import stack
from examples.relativity.spinors import core

n_rotation_frames = 192
rotation_angles = np.linspace(0, 4 * np.pi, n_rotation_frames, endpoint=False)
frame_duration_ms = 40
rapidity = 0.7
n_phase_samples = 129
phase_angles = np.linspace(0, 2 * np.pi, n_phase_samples)


def main() -> None:
    """Renders the rotation animation and static chirality and Majorana figures to the shared plots directory and checks invariants."""
    from examples.relativity.spinors import render
    from examples.util.animation import save_animation, save_figure

    rotation_history = list(core.rotation_states(rotation_angles))
    left_current, right_current, dirac_current = core.chiral_currents(rapidity)
    fixed_density, negated_density = core.majorana_densities(phase_angles)

    save_animation(render.rotation_animation(rotation_history), "spinor_rotation", frame_duration_ms)
    save_figure(render.chirality_figure(left_current, right_current, dirac_current), "spinor_chirality")
    save_figure(render.majorana_figure(phase_angles, fixed_density, negated_density), "spinor_majorana")

    # checks
    mv = core.mv
    spacetime_axes = stack((mv.t, mv.x, mv.y, mv.z))                            # [4] Vector+
    initial_spin, initial_sum, initial_difference = rotation_history[0]
    half_turn_spin, half_turn_sum, half_turn_difference = rotation_history[n_rotation_frames // 2]
    assert (spacetime_axes & (half_turn_spin - initial_spin)).abs().max() < 1e-9
    assert (initial_sum - 1).abs() < 1e-9 and initial_difference.abs() < 1e-9
    assert half_turn_sum.abs() < 1e-9 and (half_turn_difference - 1).abs() < 1e-9

    assert left_current.squared().abs() < 1e-9
    assert right_current.squared().abs() < 1e-9
    assert dirac_current.squared() < 0
    assert (spacetime_axes & (left_current + right_current - dirac_current)).abs().max() < 1e-9

    assert (fixed_density + negated_density - 1).abs().max() < 1e-9
    assert (fixed_density[0] - 1).abs() < 1e-9
    assert (negated_density[(n_phase_samples - 1) // 4] - 1).abs() < 1e-9
    assert (fixed_density[(n_phase_samples - 1) // 2] - 1).abs() < 1e-9


if __name__ == "__main__":
    main()
