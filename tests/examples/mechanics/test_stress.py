"""Coverage and Hooke's-law tests for the stressed elastic cube."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.mechanics.stress import core, render, scenarios


def test_isotropic_stress_matches_lame_law():
    """Pairing `core.cauchy_stress` with the basis planes reproduces Lamé's isotropic relation."""
    lame, shear_modulus = 1.8, 1.1
    matrix = np.array([[0.02, 0.015, -0.005], [0.015, -0.01, 0.008], [-0.005, 0.008, 0.025]])
    strain = (core.axes[:, None] * matrix * (core.axes & core.Point)[None, :]).sum(axis=(0, 1))   # [] Plane+ <- Point-
    stress = core.cauchy_stress(strain, lame, shear_modulus)                   # [] Point- <- Plane+
    entries = core.axes[:, None] & stress(core.axes)[None, :]                  # [3, 3] Scalar+
    expected = lame * np.trace(matrix) * np.eye(3) + 2 * shear_modulus * matrix
    assert ((entries - expected).abs() < 1e-12).all()


def test_scenes_render():
    """Both the static multi-frame comparison figure and a single rotating animation frame render."""
    views, stresses = scenarios.three_frames()
    figure = render.frames_figure(views, stresses)
    plt.close(figure)
    frames = list(render.turning_frames(stresses, scenarios.turning_stream(1)))
    assert len(frames) == 1
