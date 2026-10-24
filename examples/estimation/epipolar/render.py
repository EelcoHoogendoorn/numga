"""Drawing for epipolar geometry and two-view reconstruction: the search animated, a view of the cameras, the
house and the points the search triangulates, ray traced on the GPU.

Constructs viewport geometry only; the mathematics in core never imports this module.
"""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np

from numga import concatenate, stack
from examples.estimation.epipolar.core import Motor, Point, mv
from examples.util.wgsl import solids

# Camera 1, blue, camera 2 where it is, grey, and where the search puts it, green; the true landmarks, grey.
camera_1, camera_2_true, camera_2_found, landmark = [0.15, 0.45, 0.75], [0.62, 0.64, 0.68], [0.15, 0.65, 0.45], [0.70, 0.71, 0.74]
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
display = (mv.yz * (np.pi / 4)).exp()


# --- plumbing -------------------------------------------------------------------------
def camera_edges(pose: Motor) -> tuple[Point, Point]:
    """The edges of a camera drawn as a pyramid, placed by its pose and turned for display: from the eye to
    the corners of its screen, and around the screen; their starts and ends `[edges]`."""
    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * camera_half * camera_depth
    local = mv.point(np.concatenate([[[0.0, 0.0, 0.0]], np.c_[corners, np.full(4, camera_depth)]]))
    placed = display >> (pose >> local)                                      # [eye and corners] Point
    eye, ring = placed[:1], placed[1:]
    return concatenate([eye.broadcast_to(ring.shape), ring]), concatenate([ring, ring[np.roll(np.arange(4), -1)]])


def at_scale(points: Point, scale) -> Point:
    """Points scaled about camera 1's eye, at the origin."""
    origin = mv.w.dual()
    return points / (mv.w & points) * scale + origin * (1.0 - scale)


def animate_reconstruction(landmarks: Point, edges: np.ndarray, true_motor: Motor,
                           states: Iterator[tuple[Motor, Point]]) -> Iterator[np.ndarray]:
    """One frame per state of the search: camera 1, blue, camera 2 where it is, grey, and where the search
    puts it, green; the true landmarks, grey, with the house's edges between them by index pairs
    `[edges, 2]`; and the landmarks the pose triangulates, coloured, with the sight rays of a few from both
    cameras, which cross at them. The baseline's length is not observable, so what the search finds is drawn scaled
    about camera 1 to the true baseline."""
    n_landmarks = landmarks.shape[0]
    colours = plt.colormaps["turbo"].resampled(n_landmarks)(np.arange(n_landmarks))[:, :3]
    truth = display >> landmarks                                             # [landmarks] Point
    eye_1 = display >> mv.w.dual()                                           # [] Point
    true_baseline = (mv.w.dual() & (true_motor >> mv.w.dual())).norm()       # [] Scalar
    edges_1, edges_2 = camera_edges(mv.rotor()), camera_edges(true_motor)
    fixed = [(solids.ball(truth, landmark_radius)[:, None], landmark),
             (solids.rods(*edges_1, edge_radius), camera_1), (solids.rods(*edges_2, edge_radius), camera_2_true),
             (solids.rods(truth[edges[:, 0]], truth[edges[:, 1]], edge_radius), house)]
    view = solids.view([truth, *edges_1, *edges_2], landmark_radius, elevation, azimuth, margin, pixels)

    for motor, points in states:
        scale = true_baseline / (mv.w.dual() & (motor >> mv.w.dual())).norm()   # [] Scalar
        found = at_scale(display >> points, scale)                           # [landmarks] Point
        eye = at_scale(display >> (motor >> mv.w.dual()), scale)             # [] Point
        shown = found[::ray_every]
        moving = [(solids.ball(found, landmark_radius)[:, None], colours),
                  (solids.rods(*(at_scale(ends, scale) for ends in camera_edges(motor)), edge_radius), camera_2_found),
                  (solids.rods(eye_1.broadcast_to(shown.shape), shown, ray_radius), colours[::ray_every]),
                  (solids.rods(eye.broadcast_to(shown.shape), shown, ray_radius), colours[::ray_every])]
        yield solids.draw(fixed + moving, view, background, inks=[], translucent=[], meshes=[])
