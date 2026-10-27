"""Ray-traced rendering for epipolar geometry and two-view 3D reconstruction animates the cameras, the wireframe house, and the triangulated landmarks across the Gauss-Newton search.

This module constructs viewport geometry only; the mathematics in core never imports it.
"""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np

from numga import concatenate
from examples.estimation.epipolar.core import Motor, Point, mv
from examples.util.wgsl import solids

# Camera 1, blue, camera 2 where it is, grey, and where the search puts it, green; the true landmarks, grey.
camera_1, camera_2_true, camera_2_estimated, landmark = [0.15, 0.45, 0.75], [0.62, 0.64, 0.68], [0.15, 0.65, 0.45], [0.70, 0.71, 0.74]
house = [0.80, 0.81, 0.83]
background = np.array([1.0, 1.0, 1.0])
# The view's elevation and azimuth, the pixels across the scene, and the margin around what it shows as a
# fraction of its larger side.
elevation, azimuth = np.radians(24.0), np.radians(-35.0)
pixels = 560
margin = 0.08
# Radii of the landmarks, of a camera's edges and of the sight rays; a camera's depth, and its screen's half
# width and height, as drawn; the landmarks whose sight rays are drawn, every so many.
landmark_radius, edge_radius, ray_radius = 0.045, 0.012, 0.012
camera_depth, camera_half = 0.35, (0.5, 0.38)
ray_every = 4
# A quarter turn about x for display: the cameras look along z with y down; the view has z up.
display = (mv.yz * (np.pi / 4)).exp()                                        # [] Motor+


# --- plumbing -------------------------------------------------------------------------
def camera_edges(pose: Motor) -> tuple[Point, Point]:
    """The edges of a camera pyramid, placed by `pose` and turned for display, are paired arrays of start and end points."""
    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * camera_half * camera_depth
    local = mv.point(np.concatenate([[[0.0, 0.0, 0.0]], np.c_[corners, np.full(4, camera_depth)]]))
    vertices = display >> (pose >> local)                                    # [vertices] Point-
    eye, ring = vertices[:1], vertices[1:]
    return concatenate([eye.broadcast_to(ring.shape), ring]), concatenate([ring, ring[np.roll(np.arange(4), -1)]])


def animate_reconstruction(landmarks: Point, edges: np.ndarray, true_motor: Motor,
                           states: Iterator[tuple[Motor, Point]]) -> Iterator[np.ndarray]:
    """Render one ray-traced frame per state of the search, showing camera 1 in blue, camera 2's true pose in
    grey and estimated pose in green, the true wireframe house in grey, and the triangulated landmarks and
    sight rays in colour scaled about camera 1 to the true baseline."""
    n_landmarks = landmarks.shape[0]
    colours = plt.colormaps["turbo"].resampled(n_landmarks)(np.arange(n_landmarks))[:, :3]
    origin = mv.w.dual()                                                     # [] Point-
    truth = display >> landmarks                                             # [landmarks] Point-
    eye_1 = display >> origin                                                # [] Point-
    true_baseline = (origin & (true_motor >> origin)).norm()                 # [] Scalar+
    edges_1, edges_2 = camera_edges(mv.rotor()), camera_edges(true_motor)
    scene_solids = [(solids.ball(truth, landmark_radius)[:, None], landmark),
                    (solids.rods(*edges_1, edge_radius), camera_1), (solids.rods(*edges_2, edge_radius), camera_2_true),
                    (solids.rods(truth[edges[:, 0]], truth[edges[:, 1]], edge_radius), house)]
    view = solids.view([truth, *edges_1, *edges_2], landmark_radius, elevation, azimuth, margin, pixels)

    for motor, points in states:
        scale = true_baseline / (origin & (motor >> origin)).norm()          # [] Scalar+
        unit_points = (display >> points) / (mv.w & points)                  # [landmarks] Point-
        scaled_points = unit_points * scale + origin * (1.0 - scale)         # [landmarks] Point-
        starts_2, ends_2 = (ends * scale + origin * (1.0 - scale) for ends in camera_edges(motor))
        eye_2 = starts_2[0]                                                  # [] Point-
        ray_targets = scaled_points[::ray_every]                             # [sampled_landmarks] Point-
        state_solids = [(solids.ball(scaled_points, landmark_radius)[:, None], colours),
                        (solids.rods(starts_2, ends_2, edge_radius), camera_2_estimated),
                        (solids.rods(eye_1.broadcast_to(ray_targets.shape), ray_targets, ray_radius), colours[::ray_every]),
                        (solids.rods(eye_2.broadcast_to(ray_targets.shape), ray_targets, ray_radius), colours[::ray_every])]
        yield solids.draw(scene_solids + state_solids, view, background, inks=[], translucent=[], meshes=[])
