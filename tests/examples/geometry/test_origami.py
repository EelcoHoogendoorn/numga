"""Kabuto folds follow a moved sheet, creases fold points and lines onto each other, folds never pull buried layers
through stationary paper, and the first step of the folding draws."""

from itertools import islice

import numpy as np

from examples.geometry.origami import core, scenarios


def test_full_folding_sequence_follows_a_moved_sheet():
    paper, creases = scenarios.kabuto()
    frame = (core.mv.xw * 0.4 + core.mv.zw * 0.6).exp() * (core.mv.xz * 0.45).exp()
    displaced_paper = core.Paper(frame >> paper.points, paper.counts, frame >> paper.offsets, frame >> paper.sheet)
    for crease, selected, end in zip(creases, scenarios.SELECTIONS, scenarios.ENDS):
        paper, turning, _ = paper.split(crease, selected, scenarios.CREASE_TOLERANCE)
        displaced_crease = frame >> crease
        displaced_paper, displaced_turning, _ = displaced_paper.split(
            displaced_crease, selected, scenarios.CREASE_TOLERANCE
        )
        paper = paper.folded(crease ^ paper.sheet, turning, end)
        displaced_paper = displaced_paper.folded(displaced_crease ^ displaced_paper.sheet, displaced_turning, end)
        np.testing.assert_allclose((displaced_paper.points - (frame >> paper.points)).kernel, 0, atol=1e-11)
        np.testing.assert_allclose((displaced_paper.offsets - (frame >> paper.offsets)).kernel, 0, atol=1e-10)


def test_point_and_line_coincidence_crease_constructions():
    paper, _ = scenarios.kabuto()
    first, second, third, _ = paper.points
    crease = core.point_bisector(first, third)
    motion = ((crease ^ paper.sheet).normalized() * (np.pi / 2)).exp()
    first_line, second_line = first & second, first & third
    bisector = core.line_bisector(first_line, second_line, first)
    turn = ((bisector ^ paper.sheet).normalized() * (np.pi / 2)).exp()
    aligned_line = turn >> first_line.normalized()
    target_line = second_line.normalized()

    # checks: folding aligns the given points or the unoriented lines.
    np.testing.assert_allclose(((motion >> first) - third).kernel, 0, atol=1e-12)
    np.testing.assert_allclose((aligned_line + target_line).kernel, 0, atol=1e-12)


def test_folds_do_not_pull_buried_layers_through_stationary_paper():
    from examples.geometry.origami import render

    paper, creases = scenarios.kabuto()
    for crease, selected, peak, end in zip(creases, scenarios.SELECTIONS, scenarios.PEAKS, scenarios.ENDS):
        split_paper, turning, _ = paper.split(crease, selected, scenarios.CREASE_TOLERANCE)
        faces = split_paper.faces()
        points = render.coordinates(split_paper.points)[:, :2]
        heights = (split_paper.sheet & split_paper.offsets).to_array()
        for turning_face in np.flatnonzero(turning):
            for stationary_face in np.flatnonzero(~turning):
                first, second = points[faces[turning_face]], points[faces[stationary_face]]
                # The separating-axis test distinguishes overlapping interiors from
                # polygons that merely share a crease. Every face remains convex.
                edges = np.concatenate([np.roll(first, -1, axis=0) - first,
                                        np.roll(second, -1, axis=0) - second])
                axes = np.column_stack([-edges[:, 1], edges[:, 0]])
                axes /= np.linalg.norm(axes, axis=1, keepdims=True)
                first_projection, second_projection = first @ axes.T, second @ axes.T
                overlap = (np.minimum(first_projection.max(axis=0), second_projection.max(axis=0))
                           - np.maximum(first_projection.min(axis=0), second_projection.min(axis=0)))
                if np.all(overlap > 1e-8):
                    # A positive turn lifts out of the top; a negative turn exits below.
                    assert (heights[turning_face] - heights[stationary_face]) * np.sign(peak) >= -1e-10
        paper = split_paper.folded(crease ^ paper.sheet, turning, end)



def test_the_first_step_of_the_folding_draws():
    from examples.geometry.origami import render

    paper, creases = scenarios.kabuto()
    states = core.folding(paper, creases, scenarios.SELECTIONS, scenarios.PEAKS, scenarios.ENDS, 1,
                          scenarios.CREASE_TOLERANCE)
    list(render.animate(islice(states, 2)))
