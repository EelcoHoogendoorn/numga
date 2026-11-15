"""Tests for Penrose tilings check shadow measures, golden-ratio frequencies, cut-shift invariance, and rendering."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.math.penrose import core, render, scenarios

axes = core.axes
golden = (1 + np.sqrt(5)) / 2


def test_dual_cube_shadows_match_tile_areas_in_the_golden_ratio():
    """Dual-cube perpendicular shadow volumes equal projected tile areas, with thick over thin in the golden ratio."""
    areas, volumes = core.shadow_measures(scenarios.cut_projection)
    assert ((volumes - areas).abs() < 1e-12).all()
    thick = areas > areas.mean()                                              # [10]
    assert ((areas[thick] / areas[~thick][:, None] - golden).abs() < 1e-12).all()


def test_patch_tile_counts_follow_the_golden_ratio():
    """Selected thick and thin rhombi in the lattice patch match the golden ratio within finite-patch boundary spread."""
    centres, _ = core.lattice_faces(scenarios.cut_projection, scenarios.lattice_radius)
    areas, _ = core.shadow_measures(scenarios.cut_projection)
    margins = core.cube_margins(scenarios.origin_shift, centres, core.reciprocal_frames(scenarios.cut))
    selected = np.all(margins > 0, axis=-1)                                   # [..., 10]
    thick = areas > areas.mean()                                              # [10]
    assert abs(selected[:, thick].sum() / selected[:, ~thick].sum() - golden) < 5e-2 * golden


def test_sliding_the_cut_within_itself_selects_the_same_faces():
    """Shifting the cut along a direction inside the cut leaves every dual-cube margin unchanged."""
    frames = core.reciprocal_frames(scenarios.cut)                            # [10, 3] Plane+
    centres, _ = core.lattice_faces(scenarios.cut_projection, 1)              # [243, 10] Point-
    shifted_along_cut = scenarios.origin_shift + scenarios.cut_projection(0.7 * axes[2] - 1.3 * axes[4])  # [] Point-
    difference = core.cube_margins(shifted_along_cut, centres, frames) - core.cube_margins(scenarios.origin_shift, centres, frames)
    assert (difference.abs() < 1e-11).all()


def test_tiling_figure_and_one_phason_frame_render():
    """Static tiling figure and one streamed phason frame render on a minimal radius-1 lattice."""
    frames = core.reciprocal_frames(scenarios.cut)                            # [10, 3] Plane+
    centres, corners = core.lattice_faces(scenarios.cut_projection, 1)        # [243, 10] Point-, [243, 10, 4] Point-
    areas, _ = core.shadow_measures(scenarios.cut_projection)                 # [10] Scalar+
    reading = core.reading_planes(scenarios.cut_projection)                   # [2] Plane+
    figure = render.tiling_figure(
        reading, corners, areas, scenarios.view_extent, core.cube_margins(scenarios.origin_shift, centres, frames),
    )
    plt.close(figure)
    loop_shifts = core.phason_loop(scenarios.phason_projection, scenarios.origin_shift, scenarios.phason_radius, 1)
    states = core.margin_stream(loop_shifts, centres, frames)
    frame = next(render.phason_frames(reading, corners, areas, scenarios.view_extent, states))
    assert frame.ndim == 3 and frame.shape[2] == 3
