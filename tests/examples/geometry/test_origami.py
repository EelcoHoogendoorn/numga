"""Kabuto folds preserve paper area, facet lengths and the prescribed creases."""

import numpy as np
import pytest
from numga import stack

from examples.geometry.origami import core, scenarios


def area(paper: core.Paper) -> core.Scalar:
    normals = stack([(paper.points[face].dual() ^ paper.points[np.roll(face, -1)].dual()).sum(axis=0)
                     for face in paper.faces()])
    return normals.norm().sum(axis=0) / 2


def test_kabuto_preserves_area_and_facet_lengths_through_every_fold():
    paper, planes = scenarios.kabuto()
    initial_area = area(paper)
    area_error, length_error, hinge_error = [], [], []
    for plane, selected, end in zip(planes, scenarios.SELECTIONS, scenarios.ENDS):
        split, turning, _ = paper.split(plane, selected, scenarios.CREASE_TOLERANCE)
        hinge = plane ^ paper.sheet
        following = np.concatenate([np.roll(face, -1) for face in split.faces()])
        lengths = (split.points & split.points[following]).scalar_norm_squared()
        on_crease = (plane & split.points).abs() < scenarios.CREASE_TOLERANCE
        for angle in (0.0, np.pi / 2, end):
            folded = split.folded(hinge, turning, angle)
            folded_lengths = (folded.points & folded.points[following]).scalar_norm_squared()
            area_error.append(np.max(np.abs((area(folded) - initial_area).kernel)))
            length_error.append(np.max(np.abs((folded_lengths - lengths).kernel)))
            hinge_error.append(np.max(np.abs((folded.points[on_crease] - split.points[on_crease]).kernel)))
        paper = split.folded(hinge, turning, end)

    # checks: every split retains all the paper; the motors neither stretch it nor move a crease.
    assert max(area_error) < 1e-10
    assert max(length_error) < 1e-10
    assert max(hinge_error) < 1e-11
    # The two horn tips are the line-bisector construction's characteristic landmarks.
    tips = (core.mv("x y z", [[-1, 1 - np.sqrt(2), 0], [np.sqrt(2) - 1, 1, 0]]) + core.mv.w).dual()
    squared_distance = (paper.points[:, None] - tips).dual().scalar_norm_squared()
    np.testing.assert_allclose(squared_distance.kernel.min(axis=0), 0, atol=1e-20)
    np.testing.assert_allclose((paper.sheet & paper.points).kernel, 0, atol=1e-11)


def test_full_folding_sequence_follows_a_moved_sheet():
    paper, planes = scenarios.kabuto()
    frame = (core.mv.xw * 0.4 + core.mv.zw * 0.6).exp() * (core.mv.xz * 0.45).exp()
    moved = core.Paper(frame >> paper.points, paper.counts, frame >> paper.offsets, frame >> paper.sheet)
    for plane, selected, end in zip(planes, scenarios.SELECTIONS, scenarios.ENDS):
        paper, turning, _ = paper.split(plane, selected, scenarios.CREASE_TOLERANCE)
        moved_plane = frame >> plane
        moved, moved_turning, _ = moved.split(moved_plane, selected, scenarios.CREASE_TOLERANCE)
        paper = paper.folded(plane ^ paper.sheet, turning, end)
        moved = moved.folded(moved_plane ^ moved.sheet, moved_turning, end)
        np.testing.assert_allclose((moved.points - (frame >> paper.points)).kernel, 0, atol=1e-11)
        np.testing.assert_allclose((moved.offsets - (frame >> paper.offsets)).kernel, 0, atol=1e-10)


def test_folds_keep_shared_material_points_joined():
    paper, planes = scenarios.kabuto()
    placements = core.mv.rotor()[None]
    for plane, selected, peak, end in zip(planes, scenarios.SELECTIONS, scenarios.PEAKS, scenarios.ENDS):
        split, turning, parents = paper.split(plane, selected, scenarios.CREASE_TOLERANCE)
        placements = placements[parents]
        face_index = np.repeat(np.arange(len(split.counts)), split.counts)
        # Undo each facet's accumulated motor to identify points on the initial sheet.
        # Equal material points on adjacent facets must coincide throughout a fold.
        material = placements[face_index].inverse() >> split.points
        shared = (material[:, None] & material[None, :]).scalar_norm_squared() < 1e-18
        hinge = (plane ^ paper.sheet).normalized()
        for angle in (peak / 2, peak, (peak + end) / 2, end):
            folded = split.folded(hinge, turning, angle)
            gaps = (folded.points[:, None] & folded.points[None, :]).scalar_norm_squared()
            np.testing.assert_allclose(gaps.kernel[shared], 0, atol=1e-22)

        placements = (hinge * (end / 2 * turning)).exp() * placements
        paper = split.folded(hinge, turning, end)


def test_point_and_line_coincidence_crease_constructions():
    paper, _ = scenarios.kabuto()
    first, second, third, _ = paper.points
    plane = core.point_bisector(first, third)
    motion = ((plane ^ paper.sheet).normalized() * (np.pi / 2)).exp()
    first_line, second_line = first & second, first & third
    bisector = core.line_bisector(first_line, second_line, paper.sheet)
    turn = ((bisector ^ paper.sheet).normalized() * (np.pi / 2)).exp()
    matched = turn >> first_line.normalized()
    target = second_line.normalized()

    # checks: folding aligns the given points or the unoriented lines.
    np.testing.assert_allclose(((motion >> first) - third).kernel, 0, atol=1e-12)
    np.testing.assert_allclose((matched + target).kernel, 0, atol=1e-12)


def test_depth_test_hides_covered_creases_and_resolves_intersecting_planes():
    from examples.geometry.origami import render

    square = np.array([[8, 8], [56, 8], [56, 56], [8, 56]])
    triangle = np.array([[16, 16], [48, 16], [32, 48]])
    xy = np.concatenate([square, triangle])
    screen = np.column_stack([xy, xy @ [0.2, 0.1]])
    # The square lies a little above the triangle.
    screen[:4, 2] += 1e-3
    faces = (np.arange(4), np.arange(4, 7))
    colours = np.array([[0.9, 0.3, 0.1], [0.1, 0.4, 0.9]])
    image = render.rasterize(screen, faces, colours, 64)
    reversed_image = render.rasterize(screen, faces[::-1], colours[::-1], 64)

    # checks: a buried triangular crease leaves no mark, irrespective of drawing order.
    np.testing.assert_allclose(image[16:48, 16:48] - colours[0], 0, atol=1e-12)
    np.testing.assert_allclose(image - reversed_image, 0, atol=1e-12)

    xy = np.concatenate([square, square])
    screen = np.column_stack([xy, xy @ [0.2, 0.1]])
    screen[4:, 2] += (square[:, 0] - 32) * 0.1
    faces = (np.arange(4), np.arange(4, 8))
    image = render.rasterize(screen, faces, colours, 64)
    # The nearest plane changes across the image; one polygon-level order cannot draw this.
    np.testing.assert_allclose(image[20:40, 16:28] - colours[0], 0, atol=1e-12)
    np.testing.assert_allclose(image[20:40, 36:48] - colours[1], 0, atol=1e-12)


@pytest.mark.parametrize("elevation,azimuth", [(35, 135), (-35, 135), (45, 45), (-45, 225)])
def test_paper_shading_has_no_colour_jump_at_fold_boundaries(monkeypatch, elevation, azimuth):
    from examples.geometry.origami import render

    monkeypatch.setattr(render, "PIXELS", 120)
    monkeypatch.setattr(render, "SUPERSAMPLE", 1)
    monkeypatch.setattr(render, "ELEVATION", elevation)
    monkeypatch.setattr(render, "AZIMUTH", azimuth)
    paper, planes = scenarios.kabuto()
    # Resolve a visible displacement beyond the infinitesimal layer gaps.
    angle_step = 1e-3
    for plane, selected, peak, end in zip(planes, scenarios.SELECTIONS, scenarios.PEAKS, scenarios.ENDS):
        split, turning, _ = paper.split(plane, selected, scenarios.CREASE_TOLERANCE)
        hinge = plane ^ paper.sheet
        starting = split.folded(hinge, turning, np.copysign(angle_step, peak))
        centre, scale = render.viewport(render.coordinates(split.points))
        before = render.picture(split, centre, scale).astype(float)
        after = render.picture(starting, centre, scale).astype(float)
        changed = np.max(np.abs(after - before), axis=-1) > 5

        # An infinitesimal motion may change edge coverage, but cannot recolour a flap.
        assert changed.mean() < 0.02
        displacement = (starting.offsets - split.offsets).dual().norm()
        assert np.max(displacement.kernel) < angle_step * 100

        approaching = split.folded(hinge, turning, end - np.copysign(angle_step, peak)
                                   if end else np.copysign(angle_step, peak))
        paper = split.folded(hinge, turning, end)
        before = render.picture(approaching, centre, scale).astype(float)
        after = render.picture(paper, centre, scale).astype(float)
        changed = np.max(np.abs(after - before), axis=-1) > 5
        assert changed.mean() < 0.02


def test_folds_do_not_pull_buried_layers_through_stationary_paper():
    from examples.geometry.origami import render

    paper, planes = scenarios.kabuto()
    for plane, selected, peak, end in zip(planes, scenarios.SELECTIONS, scenarios.PEAKS, scenarios.ENDS):
        split, turning, _ = paper.split(plane, selected, scenarios.CREASE_TOLERANCE)
        faces = split.faces()
        points = render.coordinates(split.points)[:, :2]
        heights = (split.sheet & split.offsets).kernel[:, 0]
        for moving_face in np.flatnonzero(turning):
            for fixed_face in np.flatnonzero(~turning):
                first, second = points[faces[moving_face]], points[faces[fixed_face]]
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
                    assert (heights[moving_face] - heights[fixed_face]) * np.sign(peak) >= -1e-10
        paper = split.folded(plane ^ paper.sheet, turning, end)

