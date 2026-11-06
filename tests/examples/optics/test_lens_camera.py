"""The lens camera's stills, and one frame of its zoom drawn."""

from __future__ import annotations

from examples.optics.lens_camera import render, scenarios


def test_the_stills_and_one_frame_of_the_zoom():
    scenarios.stills()
    next(render.animate_camera(scenarios.motion(1), scenarios.scene))
