"""Penrose tilings: each face is a tile as often as its area, thick against thin in the golden ratio,
the tiles of a patch follow it, sliding the cut within itself selects the same faces, and the scenes
draw."""

from __future__ import annotations

from itertools import islice

import matplotlib.pyplot as plt
import numpy as np

from examples.math.penrose import core, render, scenarios

mv, axes = core.mv, core.axes
GOLDEN = (1 + np.sqrt(5)) / 2


def test_faces_are_tiles_as_often_as_their_area_in_the_golden_ratio():
    """The shadow of each dual cube square to the cut is as large as the face laid flat onto it."""
    along, _, _ = scenarios.cut()
    areas, volumes = core.shadows(along)
    np.testing.assert_allclose(volumes.to_array(), areas.to_array(), atol=1e-12)
    thick = areas.to_array() > areas.to_array().mean()                        # [faces]
    np.testing.assert_allclose(areas.to_array()[thick] / areas.to_array()[~thick][:, None], GOLDEN, rtol=1e-12)


def test_thick_tiles_outnumber_thin_ones_by_the_golden_ratio():
    """In a patch of a few hundred tiles, to within its edge."""
    along, _, plane = scenarios.cut()
    centres, _ = scenarios.faces(along)
    areas, _ = core.shadows(along)
    margins = core.margins(scenarios.shift(), centres, core.reciprocal(plane))
    tiles = np.all(margins.to_array() > 0, axis=-1)                           # [points, faces]
    thick = areas.to_array() > areas.to_array().mean()                        # [faces]
    np.testing.assert_allclose(tiles[:, thick].sum() / tiles[:, ~thick].sum(), GOLDEN, rtol=5e-2)


def test_sliding_the_cut_within_itself_selects_the_same_faces():
    """The reciprocal frames read nothing of the cut, so a shift within it only slides the tiling."""
    along, _, plane = scenarios.cut()
    frames = core.reciprocal(plane)                                           # [faces, 3] Plane
    centres = core.lattice(2)[:, None] + axes[core.SIDES].sum(axis=-1) / 2    # [points, faces] Antivector
    shift = scenarios.shift()
    slid = shift + along(0.7 * axes[2] - 1.3 * axes[4])
    difference = core.margins(slid, centres, frames) - core.margins(shift, centres, frames)
    np.testing.assert_allclose(difference.to_array(), 0, atol=1e-11)


def test_scenes_draw():
    along, across, plane = scenarios.cut()
    frames = core.reciprocal(plane)
    centres, corners = scenarios.faces(along)
    areas, _ = core.shadows(along)
    reading = scenarios.view(along)
    margins = core.margins(scenarios.shift(), centres, frames)
    figure = render.draw_tiling(reading, corners, areas, margins, scenarios.EXTENT)
    assert isinstance(figure, plt.Figure)
    plt.close(figure)
    flips = islice(scenarios.flips(scenarios.phasons(across), centres, frames), 2)
    assert len(list(render.animate_flips(reading, corners, areas, flips, scenarios.EXTENT))) == 2
