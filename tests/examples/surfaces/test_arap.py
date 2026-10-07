"""As rigid as possible: turning the whole problem turns its answer, and the bending scene passes its
checks and draws."""

from __future__ import annotations

import numpy as np

from examples.surfaces.arap import core, render, scenarios


def test_turning_the_bar_and_its_handles_turns_the_answer():
    """The same bend of the bar turned as a whole: as rigid as possible gives the turned answer."""
    bar = core.bar(4, 4.0)
    along = bar.vertices | core.mv.x
    handles = (along < -1.999) | (along > 1.999)
    pose = bar.vertices + (along > 1.999) * core.mv.z * 0.8
    turn = ((core.mv.x ^ core.mv.y) * -0.3).exp() * ((core.mv.y ^ core.mv.z) * 0.2).exp()
    (shape, _, _), = core.deform(bar, handles, [pose], 1e3, 4)
    (turned, _, _), = core.deform(bar.copy(vertices=turn >> bar.vertices), handles, [turn >> pose], 1e3, 4)
    np.testing.assert_allclose(turned.kernel, (turn >> shape).kernel, atol=1e-9)


def test_the_bending_scene_passes_its_checks_and_draws():
    bar, rigid, laplacian = scenarios.bending()
    images = render.animate(bar, rigid[-1:], np.array([scenarios.LENGTH, 2.0, 2.0]) / scenarios.DIVISIONS)
    assert len(images) == 1
