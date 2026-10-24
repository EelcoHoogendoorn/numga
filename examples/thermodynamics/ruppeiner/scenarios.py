"""Stable states of a van der Waals fluid approaching its critical point, and their fluctuations."""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from examples.thermodynamics.ruppeiner import core

ATTRACTION, EXCLUSION, CAPACITY = 9 / 8, 1 / 3, 1.5


# --- math -----------------------------------------------------------------------------
def metric_of(states: core.State) -> core.Metric:
    return core.metric(states, ATTRACTION, EXCLUSION, CAPACITY)


def curvature_of(states: core.State) -> core.Scalar:
    return core.thermodynamic_curvature(states, ATTRACTION, EXCLUSION, CAPACITY)


def pressure_of(states: core.State) -> core.Scalar:
    return core.pressure(states, ATTRACTION, EXCLUSION)


def approach(n_frames: int) -> Iterator[core.State]:
    """States at the critical volume cooling toward the critical temperature."""
    for warmth in np.linspace(1.8, 1.02, n_frames):
        yield core.mv.t.dual() * warmth + core.mv.v.dual()


if __name__ == "__main__":
    from examples.util.animation import save_animation, save_figure
    from examples.thermodynamics.ruppeiner import render

    save_figure(render.draw_landscape(metric_of, curvature_of, pressure_of, next(approach(2))), "ruppeiner_landscape")
    save_animation(render.fluctuations(metric_of, curvature_of, pressure_of, approach(48)), "ruppeiner_fluctuations", 80)
    offsets = np.geomspace(0.002, 0.8, 100)
    save_figure(render.scaling(core.mv.t.dual() * (1 + offsets) + core.mv.v.dual(), curvature_of), "ruppeiner_scaling")
