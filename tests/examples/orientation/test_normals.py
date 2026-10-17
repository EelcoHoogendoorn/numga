"""Reflection preserves a plane's chosen side and the construction of triangle winding."""

import matplotlib.pyplot as plt
import numpy as np

from examples.orientation.normals import render, scenarios
from examples.orientation.normals.core import Plane, mv


def test_mirrored_normals_keep_their_geometric_meanings():
    vertices, sides, windings, mirror = scenarios.normals()
    witness = vertices[0].mean(axis=0) + mv.z.dual()
    reflected_witness = mirror >> witness

    # Both normals initially point up; only winding reverses in the vertical mirror.
    np.testing.assert_allclose(sides[0].cast(Plane).kernel, windings[0].kernel, atol=1e-12)
    np.testing.assert_allclose(sides[1].kernel, sides[0].kernel, atol=1e-12)
    np.testing.assert_allclose(windings[1].kernel, -windings[0].kernel, atol=1e-12)

    # The positive half-space remains positive, while its pairing with winding reverses.
    np.testing.assert_allclose((sides[0] & witness).kernel, 1, atol=1e-12)
    np.testing.assert_allclose((sides[1] & reflected_witness).kernel, 1, atol=1e-12)
    np.testing.assert_allclose((windings[0] & witness).kernel, 1, atol=1e-12)
    np.testing.assert_allclose((windings[1] & reflected_witness).kernel, -1, atol=1e-12)

    # Reflecting a constructed plane agrees with constructing it from reflected vertices.
    rebuilt = (vertices[1, 0] & vertices[1, 1] & vertices[1, 2]).normalized()
    np.testing.assert_allclose(rebuilt.kernel, windings[1].kernel, atol=1e-12)

    figure = render.draw_normals(vertices, sides, windings, mirror)
    figure.canvas.draw()
    plt.close(figure)


def test_a_line_lying_in_the_mirror_is_its_own_mirror_image():
    ends, lines, mirror = scenarios.line_in_mirror()
    line = ends[0] & ends[1]

    # Both points lie in the mirror, so they and the line through them are their own mirror images;
    # the plain sandwich keeps the points and reverses the line.
    np.testing.assert_allclose((mirror >> ends).kernel, ends.kernel, atol=1e-12)
    np.testing.assert_allclose(lines[0].kernel, line.kernel, atol=1e-12)
    np.testing.assert_allclose(lines[1].kernel, line.kernel, atol=1e-12)
    np.testing.assert_allclose(lines[2].kernel, -line.kernel, atol=1e-12)

    figure = render.draw_line_in_mirror(ends, lines, mirror)
    figure.canvas.draw()
    plt.close(figure)


def test_planes_crossing_in_the_mirror_swap_while_points_on_their_line_stay():
    planes, ends, meets, joins, mirror = scenarios.pairs_in_mirror()

    # The mirror swaps the two planes and leaves the points; so the meet of the planes is negated,
    # by the antisymmetry of the outer product, while the join of the points is unchanged.
    np.testing.assert_allclose(planes[1].kernel, planes[0, ::-1].kernel, atol=1e-12)
    np.testing.assert_allclose(ends[1].kernel, ends[0].kernel, atol=1e-12)
    np.testing.assert_allclose(meets[1].kernel, -meets[0].kernel, atol=1e-12)
    np.testing.assert_allclose(joins[1].kernel, joins[0].kernel, atol=1e-12)

    figure = render.draw_pairs_in_mirror(planes, ends, meets, joins, mirror)
    figure.canvas.draw()
    plt.close(figure)
