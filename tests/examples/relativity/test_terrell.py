"""Coverage tests verify past-light-cone correspondence, circular sphere outlines, and GPU rendering for Terrell rotation."""

from __future__ import annotations

from itertools import islice

import numpy as np

from examples.relativity.terrell import core, scenarios

mv, time = core.mv, core.time
travel, line_of_sight, up = core.space


def test_photograph_at_rest_matches_past_light_cone():
    """Rest-frame rays boosted to the camera match direct past-light-cone intersections in the camera frame."""
    frames = core.boost_rotors(np.array([0.0, 0.6, 0.9, 0.99]))                 # [4] Rotor+
    centre = core.sighting_events(4.0, 2.2)                                     # [] Antivector-
    point = 0.7 * travel + 0.4 * up - 0.3 * line_of_sight                       # [] Antivector-
    start = centre + (frames >> point)                                          # [4] Antivector-
    velocity = frames >> time                                                   # [4] Antivector-
    lean = velocity | start                                                     # [4] Scalar+
    proper_time = lean - (lean.squared() + start.squared()).square_root()       # [4] Scalar+
    direct = -(start + velocity * proper_time)                                  # [4] Antivector-
    boosted = frames >> core.rest_rays(core.rest_camera(frames, centre), point)  # [4] Antivector-
    difference = direct / (mv.t & direct) - boosted / (mv.t & boosted)          # [4] Antivector-
    assert (difference.antinorm() < 1e-8).all()


def test_sphere_outline_stays_circular():
    """Four boosted grazing rays lie in a single sky hyperplane while an interior point lies off that hyperplane."""
    frames = core.boost_rotors(np.array([0.5, 0.95]))                           # [2] Rotor+
    camera = core.rest_camera(frames, core.sighting_events(3.0, np.array([0.6, 2.4])))  # [2] Antivector-
    centre = 0.5 * up + 0.2 * travel                                            # [] Antivector-
    angles = np.array([0.0, 1.0, 2.5, 4.0])
    grazing = core.grazing_outline(camera, centre, 1.0, angles)                 # [2, 4] Antivector-
    boosted = frames[:, None] >> core.rest_rays(camera[:, None], grazing)       # [2, 4] Antivector-
    sky_points = boosted / (mv.t & boosted)                                     # [2, 4] Antivector-
    four_ray_join = sky_points[:, 0] & sky_points[:, 1] & sky_points[:, 2] & sky_points[:, 3]  # [2] Scalar+
    assert (four_ray_join.abs() < 1e-8).all()

    offset = camera - centre                                                    # [2] Antivector-
    nearest = frames >> core.rest_rays(camera, centre + offset / offset.antinorm())  # [2] Antivector-
    interior_join = sky_points[:, 0] & sky_points[:, 1] & sky_points[:, 2] & (nearest / (mv.t & nearest))  # [2] Scalar+
    assert (interior_join.abs() > 1e-6).all()


def test_scene_renders_single_frame():
    """One streamed camera state renders a multi-lane photograph frame."""
    assert len(list(scenarios.photograph_frames(islice(scenarios.camera_stream(), 1)))) == 1
