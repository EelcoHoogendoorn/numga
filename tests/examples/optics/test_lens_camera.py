"""Three-still exposure and single-frame zoom rendering for the projective lens camera."""

from __future__ import annotations

import matplotlib.pyplot as plt

from examples.optics.lens_camera import core, render, scenarios


def test_stills_and_single_zoom_frame():
    """The three still exposures verify cone-vertex incidence, render their comparison figure, and stream one zoom frame."""
    exposures = scenarios.still_exposures()
    collineation, _, _, cones, _, _ = exposures[-1]
    assert ((cones(collineation(scenarios.scene)) & collineation(scenarios.scene)).abs() < 1e-9).all()
    plt.close(render.stills_figure(exposures))
    next(render.camera_animation(scenarios.scene, scenarios.zoom_states(1)))
