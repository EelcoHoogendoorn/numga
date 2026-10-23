"""Terrell rotation: the photograph taken at rest and carried over by the boost is the one where each
point's worldline crosses the camera's past light cone, a sphere's outline stays a circle on the sky,
and the scene draws."""

from __future__ import annotations

from itertools import islice

import numpy as np

from examples.relativity.terrell import core, scenarios

mv, time = core.mv, core.time
travel, line_of_sight, up = core.space


def test_photograph_at_rest_matches_the_past_light_cone():
    """A point of the moving body is seen where its worldline in the camera's frame meets the camera's
    past light cone, the earlier root of a quadratic in the point's proper time. The ray from there to
    the camera has the direction of the ray in the body's frame, boosted."""
    frame = core.boost(np.array([0.0, 0.6, 0.9, 0.99]))                       # [cases] Rotor
    centre = core.sighting(4.0, 2.2)                                          # [] Antivector
    point = 0.7 * travel + 0.4 * up - 0.3 * line_of_sight                       # [] Antivector
    # The point's event when the body's centre is at `centre`, and its velocity.
    start = centre + (frame >> point)                                         # [cases] Antivector
    velocity = frame >> time                                                  # [cases] Antivector
    lean = velocity | start                                                   # [cases] Scalar
    proper = lean - (lean.squared() + start.squared()).square_root()          # [cases] Scalar
    direct = -(start + velocity * proper)                                     # [cases] Antivector
    boosted = frame >> core.rays(core.rest_camera(frame, centre), point)      # [cases] Antivector
    # Both are rays arriving at the camera: equal once scaled to one unit of its time.
    difference = direct / (mv.t & direct) - boosted / (mv.t & boosted)        # [cases] Antivector
    np.testing.assert_allclose(difference.kernel, 0, atol=1e-12)


def test_outline_of_a_sphere_stays_a_circle():
    """The rays grazing a sphere lie in one hyperplane through the camera at any speed and any line of
    sight, so the join of any four of them, scaled to unit time, vanishes; four rays from points of the
    sphere off its outline do not."""
    frame = core.boost(np.array([0.5, 0.95]))                                 # [cases] Rotor
    camera = core.rest_camera(frame, core.sighting(3.0, np.array([0.6, 2.4])))   # [cases] Antivector
    centre = 0.5 * up + 0.2 * travel                                            # [] Antivector
    angles = np.array([0.0, 1.0, 2.5, 4.0])
    outline = frame[:, None] >> core.rays(camera[:, None], core.outline(camera, centre, 1.0, angles))
    points = outline / (mv.t & outline)                                       # [cases, 4] Antivector
    joined = points[:, 0] & points[:, 1] & points[:, 2] & points[:, 3]        # [cases] Scalar
    np.testing.assert_allclose(joined.to_array(), 0, atol=1e-12)
    # The point of the sphere nearest the camera is seen inside the outline, not on it.
    offset = camera - centre                                                  # [cases] Antivector
    nearest = frame >> core.rays(camera, centre + offset / offset.squared().square_root())   # [cases] Antivector
    off = points[:, 0] & points[:, 1] & points[:, 2] & (nearest / (mv.t & nearest))   # [cases] Scalar
    assert np.all(np.abs(off.to_array()) > 1e-6)


def test_scene_draws():
    assert len(list(scenarios.photographs(islice(scenarios.cameras(), 2)))) == 2
