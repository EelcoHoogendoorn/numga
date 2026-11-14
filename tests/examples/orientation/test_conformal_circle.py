"""Tests for intrinsic point-join and extrinsic surface-meet circles and their conformal motions."""

import matplotlib.pyplot as plt
import numpy as np

from numga import stack

from examples.orientation.conformal_circle import core, render, scenarios
from examples.orientation.conformal_circle.core import mv


def test_circle_constructions_commute_with_reflection():
    points, curves, surfaces, planes, spheres = scenarios.reflected_circle_scene()
    point_join, surface_meet = core.circle_constructions(points[0], planes[0], spheres[0])
    mirror = planes[0]                                                          # [] Sphere+
    rebuilt_join, rebuilt_meet = core.circle_constructions(points[1], planes[1], spheres[1])

    basis_v = mv.basis()                                                        # [5] Sphere+
    trivectors = basis_v[:, None, None] ^ basis_v[None, :, None] ^ basis_v[None, None, :]  # [5, 5, 5] Trivector+
    antibivectors = (basis_v[:, None] ^ basis_v[None, :]).dual()                # [5, 5] Antibivector-

    # Reflecting in the circle's own plane fixes the point join while reversing the surface meet.
    assert ((trivectors & ((mirror >> point_join) - point_join)).abs() < 1e-12).all()
    assert (((mirror >> surface_meet) + surface_meet) & antibivectors).abs().max() < 1e-12
    assert ((trivectors & (rebuilt_join - (mirror >> point_join))).abs() < 1e-12).all()
    assert ((rebuilt_meet - (mirror >> surface_meet)) & antibivectors).abs().max() < 1e-12
    assert ((planes[0] & curves[0]).abs() < 1e-12).all()
    assert ((spheres[0] & curves[0]).abs() < 1e-12).all()

    figure = render.circle_figure(points, curves, surfaces, planes, spheres)
    figure.canvas.draw()
    plt.close(figure)


def test_circulation_and_glide_preserve_circle_with_different_motion():
    points = scenarios.circle_points                                            # [3] Point-
    point_join, _ = core.circle_constructions(points, mv.z, mv.w)
    horizon = scenarios.horizon                                                 # [] Sphere+
    phases = np.array([0.0, 0.4, np.pi / 2])
    axis = (point_join.dual() | horizon).normalized()                           # [] Circle+
    grid = core.disk_samples(mv.x.dual(), scenarios.origin, scenarios.infinity,
                             axis, np.linspace(0, 1, 3), np.linspace(0, 2 * np.pi, 8, endpoint=False))
    pairs = stack(list(core.circle_motion_states(point_join, horizon, mv.x, grid, phases, 0.8)))  # [3, 2, 3, 8] Point-
    rotated_rim, glided_rim = pairs[:, 0, -1], pairs[:, 1, -1]                  # [3, 8] Point-, Point-

    # Both motions keep rim points on the circle.
    assert ((point_join & rotated_rim).antinorm().abs() < 1e-12).all()
    assert ((point_join & glided_rim).antinorm().abs() < 1e-12).all()

    # Squared chord lengths (`2 * (P | Q)` on horizon-normalized points) stay rigid under rotation and distort under glide.
    def chord_squares(rim: core.Point) -> core.Scalar:
        unit_rim = rim / (-horizon & rim)                                       # [..., 8] Point-
        return 2 * (unit_rim[..., :-1] | unit_rim[..., 1:])                     # [..., 7] Scalar+

    initial_chords = chord_squares(grid[-1])                                    # [7] Scalar+
    assert ((chord_squares(rotated_rim) - initial_chords).abs() < 1e-12).all()
    assert bool((chord_squares(glided_rim[-1]) - initial_chords).abs().max() > 0.2)

    frame = next(render.circle_motions_animation(scenarios.fixed_points, [pairs[0]]))
    assert frame.ndim == 3 and frame.shape[2] == 3
