"""Ray-traced rendering animates the two cameras, the wireframe house, and the triangulated landmarks across the Gauss–Newton reconstruction."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np

from numga import concatenate
from examples.estimation.epipolar.core import Motor, Point, mv
from numga.topology import Topology
from numga.render import solids

# Camera 1 in blue, camera 2's true pose in grey and estimated pose in green, and the reference house and landmarks in grey.
camera_1_colour = [0.15, 0.45, 0.75]
camera_2_true_colour = [0.62, 0.64, 0.68]
camera_2_estimated_colour = [0.15, 0.65, 0.45]
landmark_colour = [0.70, 0.71, 0.74]
house_colour = [0.80, 0.81, 0.83]
background = np.array([1.0, 1.0, 1.0])
# Orthographic elevation, azimuth, pixel count, and fractional viewport margin.
elevation, azimuth = np.radians(24.0), np.radians(-35.0)
n_pixels = 560
margin = 0.08
# Landmark ball radius, camera and house rod radii, frustum dimensions, and landmark stride for drawn sight rays.
landmark_radius, edge_radius, ray_radius = 0.045, 0.012, 0.012
camera_depth, camera_half = 0.35, (0.5, 0.38)
ray_stride = 4
# Quarter turn about x reorients camera coordinates (+z forward, +y down) into display coordinates (+z up).
display_turn = (mv.yz * (np.pi / 4)).exp()                                    # [] Motor+


# --- plumbing -------------------------------------------------------------------------
def camera_edges(pose: Motor) -> tuple[Point, Point]:
    """Start and end points (`[8] Point-`, `[8] Point-`) trace the eight pyramid edges of a camera placed by `pose` and rotated into display orientation."""
    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * camera_half * camera_depth
    local_vertices = mv.point(np.concatenate([[[0.0, 0.0, 0.0]], np.c_[corners, np.full(4, camera_depth)]]))  # [5] Point-
    posed_vertices = pose >> local_vertices                                   # [5] Point-
    vertices = display_turn >> posed_vertices                                 # [5] Point-
    eye, ring = vertices[:1], vertices[1:]                                    # [1] Point-, [4] Point-
    tails, heads = (Topology.loops([4]).incidence[1, 0] * ring.field()).batch()  # [4] Point-, [4] Point-
    return concatenate([eye.broadcast_to(ring.shape), tails]), concatenate([ring, heads])


def reconstruction_animation(
    landmarks: Point,
    house_edges: np.ndarray,
    true_motor: Motor,
    states: Iterable[tuple[Motor, Point]],
) -> Iterator[np.ndarray]:
    """Yields ray-traced RGB frames (`[n_rows, n_pixels, 3]`) showing camera 1, camera 2's true and estimated poses, the wireframe house, and the triangulated landmarks scaled to the true baseline."""
    n_landmarks = landmarks.shape[0]
    colours = plt.colormaps["turbo"].resampled(n_landmarks)(np.arange(n_landmarks))[:, :3]
    origin = mv.w.dual()                                                      # [] Point-
    true_landmarks = display_turn >> landmarks                                # [n_landmarks] Point-
    eye_1 = display_turn >> origin                                            # [] Point-
    true_baseline = (origin & (true_motor >> origin)).norm()                  # [] Scalar+
    edges_1, edges_2 = camera_edges(mv.rotor()), camera_edges(true_motor)
    scene_solids = [
        (solids.ball(true_landmarks, landmark_radius)[:, None], landmark_colour),
        (solids.rods(*edges_1, edge_radius), camera_1_colour),
        (solids.rods(*edges_2, edge_radius), camera_2_true_colour),
        (solids.rods(true_landmarks[house_edges[:, 0]], true_landmarks[house_edges[:, 1]], edge_radius), house_colour),
    ]
    view = solids.view([true_landmarks, *edges_1, *edges_2], landmark_radius, elevation, azimuth, margin, n_pixels)

    for motor, points in states:
        scale = true_baseline / (origin & (motor >> origin)).norm()           # [] Scalar+
        unit_points = (display_turn >> points) / (mv.w & points)              # [n_landmarks] Point-
        scaled_points = unit_points * scale + origin * (1.0 - scale)          # [n_landmarks] Point-
        starts_2, ends_2 = (ends * scale + origin * (1.0 - scale) for ends in camera_edges(motor))
        eye_2 = starts_2[0]                                                   # [] Point-
        ray_targets = scaled_points[::ray_stride]                             # [...] Point-
        state_solids = [
            (solids.ball(scaled_points, landmark_radius)[:, None], colours),
            (solids.rods(starts_2, ends_2, edge_radius), camera_2_estimated_colour),
            (solids.rods(eye_1.broadcast_to(ray_targets.shape), ray_targets, ray_radius), colours[::ray_stride]),
            (solids.rods(eye_2.broadcast_to(ray_targets.shape), ray_targets, ray_radius), colours[::ray_stride]),
        ]
        yield solids.draw(scene_solids + state_solids, view, background, inks=[], translucent=[], meshes=[])
