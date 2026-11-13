"""Cools a van der Waals fluid along its critical isochore toward the critical point and tracks its fluctuation metric
and scalar curvature."""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from examples.thermodynamics.ruppeiner import core

attraction, exclusion, capacity = 9 / 8, 1 / 3, 1.5


# --- math -----------------------------------------------------------------------------
def metric_of(states: core.State) -> core.Metric:
    return core.metric(states, attraction, exclusion, capacity)


def curvature_of(states: core.State) -> core.Scalar:
    return core.thermodynamic_curvature(states, attraction, exclusion, capacity)


def pressure_of(states: core.State) -> core.Scalar:
    return core.pressure(states, attraction, exclusion)


def cooling_path(temperatures: np.ndarray) -> Iterator[core.State]:
    """Generates states along the critical volume cooling steadily through the given temperatures."""
    for frame in range(len(temperatures)):
        yield core.mv.t.dual() * temperatures[frame] + core.mv.v.dual()


def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.thermodynamics.ruppeiner import render

    n_frames, frame_duration_ms, n_samples = 96, 40, 100
    warm_state = core.mv.t.dual() * 1.8 + core.mv.v.dual()                            # [] State-
    temperatures = np.linspace(1.8, 1.02, n_frames)
    offsets = np.geomspace(0.002, 0.8, n_samples)
    supercritical_states = core.mv.t.dual() * (1 + offsets) + core.mv.v.dual()        # [n_samples] State-
    save_figure(render.draw_landscape(metric_of, curvature_of, pressure_of, warm_state), "ruppeiner_landscape")
    save_animation(render.fluctuations(metric_of, curvature_of, pressure_of, cooling_path(temperatures)),
                   "ruppeiner_fluctuations", frame_duration_ms)
    save_figure(render.draw_scaling(supercritical_states, curvature_of), "ruppeiner_scaling")

    # checks
    path_states = core.mv.t.dual() * temperatures + core.mv.v.dual()                  # [n_frames] State-
    assert (((metric_of(path_states)(core.mv.t.dual()) & core.mv.t.dual())
             - capacity / temperatures ** 2).abs() < 1e-12).all()
    assert ((metric_of(path_states)(core.mv.t.dual()) & core.mv.v.dual()).abs() < 1e-12).all()
    assert (curvature_of(supercritical_states[0]) * (3 * offsets[0] ** 2) - 1).abs() < 0.01


if __name__ == "__main__":
    main()
