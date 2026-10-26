"""As rigid as possible: turning the whole problem turns its answer, and the bending scene passes its
checks and draws."""

from __future__ import annotations

import numpy as np

from examples.surfaces.arap import core, render, scenarios


def test_turning_the_bar_and_its_handles_turns_the_answer():
    """The same bend of the bar turned as a whole: as rigid as possible gives the turned answer."""
    bar, at_rest = core.bar(4, 4.0)
    along = core.mv.x & at_rest
    handles = (along < -1.999) | (along > 1.999)
    pose = at_rest + core.as_scalar(along > 1.999).field() * core.mv.z.dual() * 0.8
    turn = ((core.mv.x ^ core.mv.y) * -0.3).exp() * ((core.mv.y ^ core.mv.z) * 0.2).exp()
    (shape, _, _), = core.deform(bar, at_rest, handles, [pose], 1e3, 4)
    (turned, _, _), = core.deform(bar, turn >> at_rest, handles, [turn >> pose], 1e3, 4)
    np.testing.assert_allclose(turned.kernel, (turn >> shape).kernel, atol=1e-9)


def test_the_bending_scene_passes_its_checks_and_draws():
    bar, at_rest, rigid, laplacian, handles, target, last_energies = scenarios.bending()
    # At the last frame the energy falls with every iteration, and the handles are where they were moved to.
    energies = last_energies.to_array()
    assert np.all(np.diff(energies) <= 1e-9 * energies[0])
    assert (rigid[-1] - target).norm().batch()[handles].to_array().max() < 1e-3
    # The bar keeps its angles: its corners' cosines change a fraction as much as by Laplacian editing.
    change = [np.abs((core.corner_cosines(bar, shape) - core.corner_cosines(bar, at_rest)).to_array()).mean()
              for shape in (rigid[-1], laplacian[-1])]
    assert change[0] < 0.5 * change[1]
    images = list(render.animate(bar, at_rest, rigid[-1:], np.array([scenarios.LENGTH, 2.0, 2.0]) / scenarios.N_DIVISIONS))
    assert len(images) == 1
