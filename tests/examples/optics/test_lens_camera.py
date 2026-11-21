"""Three-still exposure and single-frame zoom rendering for the projective lens camera."""

from __future__ import annotations

import matplotlib.pyplot as plt

from numga import dim
from examples.optics.lens_camera import core, render, scenarios


def test_stills_and_single_zoom_frame():
    """The three still exposures verify cone-vertex incidence, render their comparison figure, and stream one zoom frame."""
    exposures = scenarios.still_exposures()
    collineation, _, _, cones, _, _ = exposures[-1]
    incidence = cones(collineation(scenarios.scene)) & collineation(scenarios.scene)
    assert incidence.unit == dim.length**4
    assert (incidence.abs() < core.mv.scalar(1e-9, unit=dim.length**4)).all()
    plt.close(render.stills_figure(exposures))
    next(render.camera_animation(scenarios.scene, scenarios.zoom_states(1)))
