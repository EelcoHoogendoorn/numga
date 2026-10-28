"""Penrose tilings: each face is a tile as often as its area, thick against thin in the golden ratio,
the tiles of a patch follow it, sliding the cut within itself selects the same faces, and the scenes
draw."""

from __future__ import annotations

from itertools import islice

import matplotlib.pyplot as plt
import numpy as np

from examples.math.penrose import core, render, scenarios

mv, axes = core.mv, core.axes
golden = (1 + np.sqrt(5)) / 2


def test_faces_are_tiles_as_often_as_their_area_in_the_golden_ratio():
    """The shadow of each dual cube square to the cut is as large as the face laid flat onto it."""
    areas, volumes = core.shadows(scenarios.along)
    np.testing.assert_allclose(volumes.to_array(), areas.to_array(), atol=1e-12)
    thick = areas > areas.mean()                                              # [faces]
    np.testing.assert_allclose(areas.to_array()[thick] / areas.to_array()[~thick][:, None], golden, rtol=1e-12)


def test_thick_tiles_outnumber_thin_ones_by_the_golden_ratio():
    """In a patch of a few hundred tiles, to within its edge."""
    centres, _ = scenarios.faces(scenarios.along)
    areas, _ = core.shadows(scenarios.along)
    margins = core.margins(scenarios.shift, centres, core.reciprocal(scenarios.cut))
    tiles = np.all(margins > 0, axis=-1)                                      # [points, faces]
    thick = areas > areas.mean()                                              # [faces]
    np.testing.assert_allclose(tiles[:, thick].sum() / tiles[:, ~thick].sum(), golden, rtol=5e-2)


def test_sliding_the_cut_within_itself_selects_the_same_faces():
    """The reciprocal frames read nothing of the cut, so a shift within it only slides the tiling."""
    frames = core.reciprocal(scenarios.cut)                                   # [faces, 3] Plane+
    centres = core.lattice(2)[:, None] + axes[core.sides_index].sum(axis=-1) / 2   # [points, faces] Antivector-
    shift = scenarios.shift                                                   # [] Antivector-
    shift_within_cut = shift + scenarios.along(0.7 * axes[2] - 1.3 * axes[4])  # [] Antivector-
    difference = core.margins(shift_within_cut, centres, frames) - core.margins(shift, centres, frames)
    np.testing.assert_allclose(difference.to_array(), 0, atol=1e-11)


def test_scenes_draw():
    frames = core.reciprocal(scenarios.cut)
    centres, corners = scenarios.faces(scenarios.along)
    areas, _ = core.shadows(scenarios.along)
    reading = scenarios.view(scenarios.along)
    margins = core.margins(scenarios.shift, centres, frames)
    figure = render.draw_tiling(reading, corners, areas, margins, scenarios.extent)
    assert isinstance(figure, plt.Figure)
    plt.close(figure)
    flips = islice(scenarios.flips(scenarios.phasons(scenarios.across), centres, frames), 2)
    assert len(list(render.animate_flips(reading, corners, areas, flips, scenarios.extent))) == 2
