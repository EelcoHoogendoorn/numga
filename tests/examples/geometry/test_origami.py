"""Kabuto folds commute with rigid motions of the sheet, bisector creases fold points and lines onto each other,
folds keep turning layers on the exit side of overlapping stationary paper, and the render entry points draw."""

from itertools import islice

import matplotlib.pyplot as plt
import numpy as np

from numga import stack, where
from numga.topology import Topology

from examples.geometry.origami import core, scenarios


def test_full_folding_sequence_follows_a_moved_sheet():
    state, creases = scenarios.kabuto_recipe()
    points, counts, offsets, sheet = state
    frame = (core.mv.xw * 0.4 + core.mv.zw * 0.6).exp() * (core.mv.xz * 0.45).exp()
    displaced = (frame >> points, counts, frame >> offsets, frame >> sheet)
    for crease, selected, end in zip(creases, scenarios.selections, scenarios.ends):
        split_points, split_counts, split_offsets, turning = core.split_polygons(
            points, counts, offsets, crease, selected, scenarios.crease_tolerance,
        )
        displaced_crease = frame >> crease
        d_points, d_counts, d_offsets, d_turning = core.split_polygons(
            displaced[0], displaced[1], displaced[2], displaced_crease, selected, scenarios.crease_tolerance,
        )
        points, counts, offsets, sheet = core.folded_state(
            split_points, split_counts, split_offsets, sheet, crease ^ sheet, turning, end,
        )
        displaced = core.folded_state(
            d_points, d_counts, d_offsets, displaced[3], displaced_crease ^ displaced[3], d_turning, end,
        )
        assert (((displaced[0] - (frame >> points)).dual().scalar_norm_squared()) < 1e-20).all()
        assert (((displaced[2] - (frame >> offsets)).dual().scalar_norm_squared()) < 1e-18).all()


def test_point_and_line_coincidence_crease_constructions():
    state, _ = scenarios.kabuto_recipe()
    first, second, third, _ = state[0].batch()
    sheet = state[3]
    crease = core.point_bisector(first, third)
    motion = ((crease ^ sheet).normalized() * (np.pi / 2)).exp()
    first_line, second_line = first & second, first & third
    bisector = core.line_bisector(first_line, second_line, first)
    turn = ((bisector ^ sheet).normalized() * (np.pi / 2)).exp()
    aligned_line = turn >> first_line.normalized()
    target_line = second_line.normalized()

    # checks
    assert (((motion >> first) & third).antinorm() < 1e-10).all()
    assert (((aligned_line + target_line).dual().scalar_norm_squared()) < 1e-20).all()


def test_folds_do_not_pull_buried_layers_through_stationary_paper():
    state, creases = scenarios.kabuto_recipe()
    points, counts, offsets, sheet = state
    for crease, selected, peak, end in zip(creases, scenarios.selections, scenarios.peaks, scenarios.ends):
        split_points, split_counts, split_offsets, turning = core.split_polygons(
            points, counts, offsets, crease, selected, scenarios.crease_tolerance,
        )
        faces = core.polygon_faces(split_counts)
        batch_points = split_points.batch()
        heights = (sheet & split_offsets).batch()
        for turning_face in np.flatnonzero(turning):
            for stationary_face in np.flatnonzero(~turning):
                first, second = batch_points[faces[turning_face]], batch_points[faces[stationary_face]]
                # Separating planes perpendicular to the sheet along every polygon edge distinguish
                # overlapping face interiors from faces that only share a boundary crease.
                first_tails, first_heads = (Topology.loops([len(first)]).incidence[1, 0] * first.field()).batch()
                second_tails, second_heads = (Topology.loops([len(second)]).incidence[1, 0] * second.field()).batch()
                edges = stack([*(first_tails & first_heads), *(second_tails & second_heads)])
                edge_planes = (edges | sheet).normalized()
                first_proj = edge_planes[:, None] & first[None, :]
                second_proj = edge_planes[:, None] & second[None, :]
                overlap = (
                    where(first_proj.max(axis=1) < second_proj.max(axis=1), first_proj.max(axis=1), second_proj.max(axis=1))
                    - where(first_proj.min(axis=1) > second_proj.min(axis=1), first_proj.min(axis=1), second_proj.min(axis=1))
                )
                if (overlap > 1e-8).all():
                    assert bool((heights[turning_face] - heights[stationary_face]) * np.sign(peak) >= -1e-10)
        points, counts, offsets, sheet = core.folded_state(
            split_points, split_counts, split_offsets, sheet, crease ^ sheet, turning, end,
        )


def test_origami_render_entry_points():
    from examples.geometry.origami import render

    state, creases = scenarios.kabuto_recipe()
    figure = render.paper_figure(state)
    plt.close(figure)
    states = core.folding_states(
        state, creases, scenarios.selections, scenarios.peaks, scenarios.ends, 1, scenarios.crease_tolerance,
    )
    frames = list(render.folding_animation(islice(states, 2)))
    assert len(frames) == 2
