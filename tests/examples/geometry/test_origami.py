"""Kabuto folds preserve paper area, facet lengths and the prescribed creases."""

import numpy as np
import pytest
from numga import stack

from examples.geometry.origami import core, scenarios


def area(paper: core.Paper):
    normals = stack([(paper.points[face].dual() ^ paper.points[np.roll(face, -1)].dual()).sum(axis=0)
                     for face in paper.faces()])
    return normals.norm().sum(axis=0) / 2


def test_kabuto_preserves_area_and_facet_lengths_through_every_fold():
    paper, creases = scenarios.kabuto()
    initial_area = area(paper)
    area_error, length_error, hinge_error = [], [], []
    for crease, selected, end in zip(creases, scenarios.SELECTIONS, scenarios.ENDS):
        split_paper, turning, _ = paper.split(crease, selected, scenarios.CREASE_TOLERANCE)
        hinge = crease ^ paper.sheet
        next_corners = np.concatenate([np.roll(face, -1) for face in split_paper.faces()])
        lengths = (split_paper.points & split_paper.points[next_corners]).scalar_norm_squared()
        on_crease = (crease & split_paper.points).abs() < scenarios.CREASE_TOLERANCE
        for angle in (0.0, np.pi / 2, end):
            folded_paper = split_paper.folded(hinge, turning, angle)
            folded_lengths = (folded_paper.points & folded_paper.points[next_corners]).scalar_norm_squared()
            area_error.append(np.max(np.abs((area(folded_paper) - initial_area).kernel)))
            length_error.append(np.max(np.abs((folded_lengths - lengths).kernel)))
            hinge_error.append(np.max(np.abs((folded_paper.points[on_crease] - split_paper.points[on_crease]).kernel)))
        paper = split_paper.folded(hinge, turning, end)

    # checks: every split retains all the paper; the motors neither stretch it nor move a crease.
    assert max(area_error) < 1e-10
    assert max(length_error) < 1e-10
    assert max(hinge_error) < 1e-11
    # The two horn tips are the line-bisector construction's characteristic landmarks.
    tips = core.mv.point([[-1, 1 - np.sqrt(2), 0], [np.sqrt(2) - 1, 1, 0]])
    squared_distance = (paper.points[:, None] - tips).dual().scalar_norm_squared()
    np.testing.assert_allclose(squared_distance.kernel.min(axis=0), 0, atol=1e-20)
    np.testing.assert_allclose((paper.sheet & paper.points).kernel, 0, atol=1e-11)


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


def test_folds_keep_shared_material_points_joined():
    paper, creases = scenarios.kabuto()
    placements = core.mv.rotor()[None]
    for crease, selected, peak, end in zip(creases, scenarios.SELECTIONS, scenarios.PEAKS, scenarios.ENDS):
        split_paper, turning, parents = paper.split(crease, selected, scenarios.CREASE_TOLERANCE)
        placements = placements[parents]
        face_index = np.repeat(np.arange(len(split_paper.counts)), split_paper.counts)
        # Undo each facet's accumulated motor to identify points on the initial sheet.
        # Equal material points on adjacent facets must coincide throughout a fold.
        material_points = placements[face_index].inverse() >> split_paper.points
        shared = (material_points[:, None] & material_points[None, :]).scalar_norm_squared() < 1e-18
        hinge = (crease ^ paper.sheet).normalized()
        for angle in (peak / 2, peak, (peak + end) / 2, end):
            folded_paper = split_paper.folded(hinge, turning, angle)
            gaps = (folded_paper.points[:, None] & folded_paper.points[None, :]).scalar_norm_squared()
            np.testing.assert_allclose(gaps.kernel[shared], 0, atol=1e-22)

        placements = (hinge * (end / 2 * turning)).exp() * placements
        paper = split_paper.folded(hinge, turning, end)


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


@pytest.mark.parametrize("elevation,azimuth", [(35, 135), (-35, 135), (45, 45), (-45, 225)])
def test_paper_shading_has_no_colour_jump_at_fold_boundaries(monkeypatch, elevation, azimuth):
    from examples.geometry.origami import render

    monkeypatch.setattr(render, "n_pixels", 120)
    monkeypatch.setattr(render, "elevation", np.radians(elevation))
    monkeypatch.setattr(render, "azimuth", np.radians(azimuth + 90))
    paper, creases = scenarios.kabuto()
    # Resolve a visible displacement beyond the infinitesimal layer gaps.
    angle_step = 1e-3
    for crease, selected, peak, end in zip(creases, scenarios.SELECTIONS, scenarios.PEAKS, scenarios.ENDS):
        split_paper, turning, _ = paper.split(crease, selected, scenarios.CREASE_TOLERANCE)
        hinge = crease ^ paper.sheet
        initial_fold = split_paper.folded(hinge, turning, np.copysign(angle_step, peak))
        view = render.viewport(split_paper.points)
        before = render.picture(split_paper, view).astype(float)
        after = render.picture(initial_fold, view).astype(float)
        changed = np.max(np.abs(after - before), axis=-1) > 5

        # An infinitesimal motion may change edge coverage, but cannot recolour a flap.
        assert changed.mean() < 0.02
        displacement = (initial_fold.offsets - split_paper.offsets).dual().norm()
        assert np.max(displacement.kernel) < angle_step * 100

        final_step = split_paper.folded(hinge, turning, end - np.copysign(angle_step, peak)
                                        if end else np.copysign(angle_step, peak))
        paper = split_paper.folded(hinge, turning, end)
        before = render.picture(final_step, view).astype(float)
        after = render.picture(paper, view).astype(float)
        changed = np.max(np.abs(after - before), axis=-1) > 5
        assert changed.mean() < 0.02


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

