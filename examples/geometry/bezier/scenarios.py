"""Scenes for rational quadratic Bézier curves sweep the middle weight from an ellipse through a parabola to a hyperbola and back, together with the complementary arc and the cone of their conic."""

from __future__ import annotations

import numpy as np

from numga import stack
from examples.geometry.bezier import core

# Samples along each curve.
n_samples = 400
# The control triangle of the quadratic curve.
quadratic_points = np.array([[-1.0, 0.0], [0.0, 1.5], [1.0, 0.0]])
# The middle weight swept from an ellipse through a parabola to a hyperbola and back, the frames of the
# sweep, and the duration of each.
weight_sweep = (0.3, 2.0)
n_frames = 96
frame_duration_ms = 40


# --- math -----------------------------------------------------------------------------
def conic_sweep(n_frames: int, n_samples: int) -> tuple[core.Point, core.Point, core.Point, core.Conic]:
    """As the middle weight sweeps out and back over `n_frames` frames, the quadratic curve and its complementary arc (`[n_frames, n_samples + 1] Point-`) and the cone of their conic (`[n_frames] Plane+ <- Point-`) pass from an ellipse through a parabola to a hyperbola."""
    phases = np.linspace(0, 2 * np.pi, n_frames, endpoint=False)
    middle_weights = np.exp(np.log(weight_sweep).mean() - np.log(weight_sweep[1] / weight_sweep[0]) / 2 * np.cos(phases))  # [n_frames]
    weights = np.stack([np.ones_like(middle_weights), middle_weights, np.ones_like(middle_weights)], axis=-1)              # [n_frames, 3]
    # Points (x, y) at unit weight in the plane z = 1 are the duals of x * mv.x + y * mv.y + mv.z.
    controls = (core.mv.x * quadratic_points[:, 0] + core.mv.y * quadratic_points[:, 1] + core.mv.z).dual()                # [3] Point-
    parameters = np.linspace(0.0, 1.0, n_samples + 1)                                                                      # [n_samples + 1]
    weighted_controls = stack([controls * weights, controls * core.complementary_weights(weights)], axis=1)                # [n_frames, 2, 3] Point-
    curves = core.bezier_curve(controls, weights, parameters)                                                              # [n_frames, n_samples + 1] Point-
    complements = core.bezier_curve(controls, core.complementary_weights(weights), parameters)                             # [n_frames, n_samples + 1] Point-
    return weighted_controls, curves, complements, core.curve_conic(controls, weights)                                     # [n_frames] Plane+ <- Point-


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_animation
    from examples.geometry.bezier import render

    weighted_controls, curves, complements, conics = conic_sweep(n_frames, n_samples)
    save_animation(render.animate_rays(weighted_controls, curves, complements, conics), "bezier_cone", frame_duration_ms)

    # checks
    # Both arcs lie on the cone of their conic at every middle weight, and the conic polarity map is symmetric.
    assert ((conics[:, None](curves) & curves).abs() < 1e-10).all()
    assert ((conics[:, None](complements) & complements).abs() < 1e-10).all()
    assert (((conics[:, None](curves) & complements) - (conics[:, None](complements) & curves)).abs() < 1e-12).all()


if __name__ == "__main__":
    main()
