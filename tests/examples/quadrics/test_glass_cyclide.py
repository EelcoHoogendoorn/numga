"""Coverage tests verify conformal quadric surfaces, Dupin cyclide inversion, tangent normals, and GPU ray tracing."""

from __future__ import annotations

import numpy as np
import pytest

from examples.quadrics.glass_cyclide import core, scenarios

mv = core.mv
tolerance = 1e-10


def test_inverted_torus_lies_on_the_cyclide():
    n_around, n_across = 16, 8
    points = scenarios.inversion >> core.torus_points(scenarios.torus_radius, scenarios.torus_tube, n_around, n_across)  # [n_across, n_around] Point-
    inversion = scenarios.inversion
    cyclide = inversion >> core.torus(scenarios.torus_radius, scenarios.torus_tube)(inversion << core.Point)             # [] Sphere+ <- Point-
    residuals = (cyclide(points) & points) / (core.horizon & points).squared()                                           # [n_across, n_around] Scalar+
    assert (residuals.abs() < tolerance).all()


def test_tangent_sphere_gives_the_outward_normal():
    centre = core.translator(mv.x.dual() - mv.z.dual() * 0.5) >> core.origin                    # [] Point-
    solid = core.ball(core.sphere(centre, 0.8))                                                 # [] Sphere+ <- Point-
    radial = (mv.x.dual() * 0.6 + mv.y.dual() * 0.48 + mv.z.dual() * 0.64) * 0.8               # [] Direction-
    point = core.translator(radial) >> centre                                                   # [] Point-
    direction = core.outward_normal(solid(point), point)                                        # [] Direction-
    # Collinear directions have vanishing regressive join, and stepping outward leaves the solid.
    assert ((direction & radial).scalar_norm_squared() < tolerance**2).all()
    beyond = core.translator(direction * 0.01) >> point                                         # [] Point-
    assert (solid(beyond) & beyond > 0).all()
    # Pairing is negative inside the ball, positive outside, and negated by negating the quadric.
    inside = centre                                                                             # [] Point-
    outside = core.translator(mv.x.dual() * 2) >> centre                                        # [] Point-
    assert (solid(inside) & inside < 0).all()
    assert (solid(outside) & outside > 0).all()
    assert ((-solid)(inside) & inside > 0).all()


def test_one_frame_traces():
    pytest.importorskip("wgpu")
    import matplotlib.pyplot as plt
    from examples.util.panels import image_figure
    from examples.quadrics.glass_cyclide import render

    quadrics, solid_table = scenarios.scene_solids()
    turns, eyes = core.orbit_cameras(scenarios.target, scenarios.distance, scenarios.elevation, np.zeros(1))
    frame, = render.camera_frames(
        render.shader(), quadrics, solid_table, turns, eyes, scenarios.lamp, scenarios.half_view, (32, 48),
    )
    assert frame.shape == (32, 48, 3)
    figure = image_figure(frame)
    plt.close(figure)
