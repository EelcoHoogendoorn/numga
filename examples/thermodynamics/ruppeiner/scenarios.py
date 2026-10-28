"""Stable states of a van der Waals fluid approaching its critical point, and their fluctuations."""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from examples.thermodynamics.ruppeiner import core

# The fluid: the strength of its molecules' attraction, the volume they exclude, and its heat capacity at fixed volume.
attraction, exclusion, capacity = 9 / 8, 1 / 3, 1.5


# --- math -----------------------------------------------------------------------------
def metric_of(states: core.State) -> core.Metric:
    return core.metric(states, attraction, exclusion, capacity)


def curvature_of(states: core.State) -> core.Scalar:
    return core.thermodynamic_curvature(states, attraction, exclusion, capacity)


def pressure_of(states: core.State) -> core.Scalar:
    return core.pressure(states, attraction, exclusion)


def approach(temperatures: np.ndarray) -> Iterator[core.State]:
    """States at the critical volume cooling through the temperatures toward the critical one."""
    for frame in range(len(temperatures)):
        yield core.mv.t.dual() * temperatures[frame] + core.mv.v.dual()


if __name__ == "__main__":
    from examples.util.animation import save_animation, save_figure
    from examples.thermodynamics.ruppeiner import render

    n_frames, frame_duration_ms, n_samples = 48, 80, 100
    warm = core.mv.t.dual() * 1.8 + core.mv.v.dual()                                  # [] State-
    temperatures = np.linspace(1.8, 1.02, n_frames)
    # Distances above the critical temperature, along the critical volume.
    offsets = np.geomspace(0.002, 0.8, n_samples)
    above_critical = core.mv.t.dual() * (1 + offsets) + core.mv.v.dual()              # [samples] State-
    save_figure(render.draw_landscape(metric_of, curvature_of, pressure_of, warm), "ruppeiner_landscape")
    save_animation(render.fluctuations(metric_of, curvature_of, pressure_of, approach(temperatures)),
                   "ruppeiner_fluctuations", frame_duration_ms)
    save_figure(render.scaling(above_critical, curvature_of), "ruppeiner_scaling")
