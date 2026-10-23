"""The cone of a rational Bézier curve: a quadratic curve whose middle weight sweeps it from an ellipse
through a parabola to a hyperbola and back, with its complement and the cone of its conic."""

from __future__ import annotations

import numpy as np

from numga import stack
from examples.geometry.bezier import core

# Samples along each curve.
N_SAMPLES = 400
# The control triangle of the quadratic curve.
QUADRATIC_POINTS = np.array([[-1.0, 0.0], [0.0, 1.5], [1.0, 0.0]])
# The middle weight swept from an ellipse through a parabola to a hyperbola and back, the frames of the
# sweep, and the duration of each.
SWEEP = (0.3, 2.0)
N_FRAMES = 48
DURATION_MS = 80


# --- math -----------------------------------------------------------------------------
def sweep() -> tuple[core.Point, core.Point, core.Point, core.Conic]:
    """As the middle weight sweeps out and back: the weighted control points of the curve and of its
    complement, the curve, its complement, and the form of its cone."""
    phase = np.linspace(0, 2 * np.pi, N_FRAMES, endpoint=False)
    middle = np.exp(np.log(SWEEP).mean() - np.log(SWEEP[1] / SWEEP[0]) / 2 * np.cos(phase))   # [frames]
    weights = np.stack([np.ones_like(middle), middle, np.ones_like(middle)], axis=-1)   # [frames, 3]
    controls = core.point(QUADRATIC_POINTS)                                    # [controls] Point
    parameters = np.linspace(0.0, 1.0, N_SAMPLES + 1)                            # [samples + 1]
    weighted = stack([controls * weights, controls * core.complementary(weights)], axis=1)   # [frames, 2, controls] Point
    curves = core.blend(controls, weights, parameters)                         # [frames, samples + 1] Point
    complements = core.blend(controls, core.complementary(weights), parameters)   # [frames, samples + 1] Point
    return weighted, curves, complements, core.conic(controls, weights)        # [frames] Plane <- Point


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.animation import save_animation
    from examples.geometry.bezier import render

    weighted, curves, complements, shapes = sweep()
    save_animation(render.animate_rays(weighted, curves, complements, shapes), "bezier_cone", DURATION_MS)

    # checks
    # Both arcs of the curve lie on the cone of its conic, at every middle weight.
    assert np.abs((shapes[:, None](curves) & curves).to_array()).max() < 1e-10
    assert np.abs((shapes[:, None](complements) & complements).to_array()).max() < 1e-10


if __name__ == "__main__":
    main()
