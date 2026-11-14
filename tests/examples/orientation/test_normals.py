"""Reflection preserves a plane's chosen side and reverses triangle winding consistently."""

import matplotlib.pyplot as plt

from examples.orientation.normals import render, scenarios
from examples.orientation.normals.core import mv


def test_mirrored_normals_keep_their_geometric_meanings():
    vertices, sides, windings, mirror = scenarios.normals()
    witness = vertices[0].mean(axis=0) + mv.z.dual()
    reflected_witness = mirror >> witness

    # Both normals initially point upward, and only the winding reverses in the vertical mirror.
    initial_diff = sides[0] - windings[0] * mv.antipseudoscalar([1])
    side_diff = sides[1] - sides[0]
    winding_sum = windings[1] + windings[0]
    assert initial_diff.scalar_norm_squared() + initial_diff.dual().scalar_norm_squared() < 1e-12
    assert side_diff.scalar_norm_squared() + side_diff.dual().scalar_norm_squared() < 1e-12
    assert winding_sum.scalar_norm_squared() + winding_sum.dual().scalar_norm_squared() < 1e-12

    # The positive half-space remains positive, while its pairing with winding reverses sign.
    assert ((sides[0] & witness) - 1).abs() < 1e-6
    assert ((sides[1] & reflected_witness) - 1).abs() < 1e-6
    assert ((windings[0] & witness) - mv.antipseudoscalar([1])).scalar_norm_squared() < 1e-12
    assert ((windings[1] & reflected_witness) + mv.antipseudoscalar([1])).scalar_norm_squared() < 1e-12

    # Reflecting a constructed plane agrees with joining the reflected vertices in order.
    rebuilt_winding = (vertices[1, 0] & vertices[1, 1] & vertices[1, 2]).normalized()
    rebuilt_diff = rebuilt_winding - windings[1]
    assert rebuilt_diff.scalar_norm_squared() + rebuilt_diff.dual().scalar_norm_squared() < 1e-12

    figure = render.normals_figure(vertices, sides, windings, mirror)
    figure.canvas.draw()
    plt.close(figure)


def test_a_line_lying_in_the_mirror_is_its_own_mirror_image():
    ends, lines, mirror = scenarios.line_in_mirror()
    line = ends[0] & ends[1]

    # Both points lie in the mirror, so `>>` preserves them and their joined line while the plain sandwich reverses the line.
    ends_diff = (mirror >> ends) - ends
    line_diff = lines[0] - line
    reflected_diff = lines[1] - line
    sandwich_sum = lines[2] + line
    assert ((ends_diff.scalar_norm_squared() + ends_diff.dual().scalar_norm_squared()) < 1e-12).all()
    assert line_diff.scalar_norm_squared() + line_diff.dual().scalar_norm_squared() < 1e-12
    assert reflected_diff.scalar_norm_squared() + reflected_diff.dual().scalar_norm_squared() < 1e-12
    assert sandwich_sum.scalar_norm_squared() + sandwich_sum.dual().scalar_norm_squared() < 1e-12

    figure = render.line_in_mirror_figure(ends, lines, mirror)
    figure.canvas.draw()
    plt.close(figure)


def test_planes_crossing_in_the_mirror_swap_while_points_on_their_line_stay():
    plane_pairs, point_pairs, meets, joins, mirror = scenarios.pairs_in_mirror()

    # Reflection swaps the crossing planes and fixes the points, so the meet negates while the join stays unchanged.
    swapped_planes_diff = plane_pairs[1] - plane_pairs[0, ::-1]
    fixed_points_diff = point_pairs[1] - point_pairs[0]
    meets_sum = meets[1] + meets[0]
    joins_diff = joins[1] - joins[0]
    assert ((swapped_planes_diff.scalar_norm_squared() + swapped_planes_diff.dual().scalar_norm_squared()) < 1e-12).all()
    assert ((fixed_points_diff.scalar_norm_squared() + fixed_points_diff.dual().scalar_norm_squared()) < 1e-12).all()
    assert meets_sum.scalar_norm_squared() + meets_sum.dual().scalar_norm_squared() < 1e-12
    assert joins_diff.scalar_norm_squared() + joins_diff.dual().scalar_norm_squared() < 1e-12

    figure = render.pairs_in_mirror_figure(plane_pairs, point_pairs, meets, joins, mirror)
    figure.canvas.draw()
    plt.close(figure)
