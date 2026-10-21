"""Drawing for epipolar geometry and two-view reconstruction: the search animated, a ray-traced view of the
cameras, the house and the points the search triangulates.

Constructs viewport geometry only; the mathematics in core never imports this module.
"""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from examples import raytrace as rt
from examples.estimation.epipolar.core import Motor, Point, mv

# Camera 1, blue, camera 2 where it is, grey, and where the search puts it, green; the true landmarks, grey.
CAMERA_1, CAMERA_2_TRUE, CAMERA_2_FOUND, LANDMARK = [0.15, 0.45, 0.75], [0.62, 0.64, 0.68], [0.15, 0.65, 0.45], [0.70, 0.71, 0.74]
HOUSE = [0.80, 0.81, 0.83]
BACKGROUND = np.array([1.0, 1.0, 1.0])
# The view's angles, the pixels across the scene, and the margin around what it shows as a fraction of its
# larger side.
VIEW = (24, -125)
PIXELS = 560
MARGIN = 0.08
# Radii of the landmarks, of a camera's edges and of the sight rays; a camera's depth, and its screen's half
# width and height, as drawn; the landmarks whose sight rays are drawn, every so many.
LANDMARK_RADIUS, EDGE_RADIUS, RAY_RADIUS = 0.045, 0.012, 0.012
CAMERA_DEPTH, CAMERA_HALF = 0.35, (0.5, 0.38)
RAY_EVERY = 4
# A quarter turn about x for display: the cameras look along z with y down; the view has z up.
DISPLAY = (mv.yz * (np.pi / 4)).exp()


# --- plumbing -------------------------------------------------------------------------
def euclidean(points: Point) -> np.ndarray:
    """Coordinates of points: their pairings with the planes x, y and z, over their weights."""
    return ((stack((mv.x, mv.y, mv.z)) & points[..., None]) / (mv.w & points)[..., None]).to_array()


def camera_edges(pose: Motor) -> tuple[np.ndarray, np.ndarray]:
    """The edges of a camera drawn as a pyramid, placed by its pose and turned for display: from the eye to
    the corners of its screen, and around the screen; their starts and ends `[edges, 3]`."""
    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * CAMERA_HALF * CAMERA_DEPTH
    local = mv.point(np.concatenate([[[0.0, 0.0, 0.0]], np.c_[corners, np.full(4, CAMERA_DEPTH)]]))
    placed = euclidean(DISPLAY >> (pose >> local))                            # [eye and corners, 3]
    eye, ring = placed[0], placed[1:]
    return np.concatenate([np.broadcast_to(eye, ring.shape), ring]), np.concatenate([ring, np.roll(ring, -1, axis=0)])


def solids(matrices: list[np.ndarray], centres: list[np.ndarray], colours: list[np.ndarray]):
    """Quadrics, as one batch, with their colours."""
    return rt.quadric(np.concatenate(matrices), np.concatenate(centres)), np.concatenate(colours)


def spheres(centres: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
    """Spheres of a radius about centres `[n, 3]`: their matrices and centres."""
    return np.broadcast_to(np.eye(3) / radius**2, centres.shape + (3,)), centres


def animate_reconstruction(landmarks: Point, edges: np.ndarray, true_motor: Motor,
                           states: Iterator[tuple[Motor, Point]]) -> list[np.ndarray]:
    """One frame per state of the search: camera 1, blue, camera 2 where it is, grey, and where the search
    puts it, green; the true landmarks, grey, with the house's edges between them by index pairs
    `[edges, 2]`; and the landmarks the pose triangulates, coloured, with the sight rays of a few from both
    cameras, which cross at them."""
    n_landmarks = landmarks.shape[0]
    colours = plt.colormaps["turbo"].resampled(n_landmarks)(np.arange(n_landmarks))[:, :3]
    truth = euclidean(DISPLAY >> landmarks)                                   # [landmarks, 3]
    eye_1 = euclidean(DISPLAY >> mv.w.dual())

    # The scene that does not move, traced once.
    edges_1, edges_2 = camera_edges(mv.rotor()), camera_edges(true_motor)
    parts = [spheres(truth, LANDMARK_RADIUS), rt.segments(*edges_1, EDGE_RADIUS), rt.segments(*edges_2, EDGE_RADIUS),
             rt.segments(truth[edges[:, 0]], truth[edges[:, 1]], EDGE_RADIUS)]
    fixed, fixed_colours = solids([m for m, _ in parts], [c for _, c in parts],
                                  [np.tile(LANDMARK, (n_landmarks, 1)), np.tile(CAMERA_1, (8, 1)), np.tile(CAMERA_2_TRUE, (8, 1)),
                                   np.tile(HOUSE, (len(edges), 1))])
    framed = np.concatenate([rt.ellipsoid_points(*spheres(truth, LANDMARK_RADIUS), 24).reshape(-1, 3), *edges_1, *edges_2])
    middle, half = rt.framing(framed, *VIEW, MARGIN)
    view = (*VIEW, middle, half, PIXELS * rt.SUPERSAMPLE)
    still, still_depth = rt.scene(fixed, fixed_colours, view, BACKGROUND)

    frames = []
    for motor, points in states:
        found = euclidean(DISPLAY >> points)                                  # [landmarks, 3]
        eye = euclidean(DISPLAY >> (motor >> mv.w.dual()))
        found_edges = camera_edges(motor)
        shown = found[::RAY_EVERY]
        parts = [spheres(found, LANDMARK_RADIUS), rt.segments(*found_edges, EDGE_RADIUS),
                 rt.segments(np.broadcast_to(eye_1, shown.shape), shown, RAY_RADIUS),
                 rt.segments(np.broadcast_to(eye, shown.shape), shown, RAY_RADIUS)]
        moving, moving_colours = solids([m for m, _ in parts], [c for _, c in parts],
                                        [colours, np.tile(CAMERA_2_FOUND, (8, 1)), colours[::RAY_EVERY], colours[::RAY_EVERY]])
        rgb, depth = rt.scene(moving, moving_colours, view, BACKGROUND)
        image = rt.downsample(np.where((depth < still_depth)[..., None], rgb, still), rt.SUPERSAMPLE)
        frames.append((np.clip(image, 0, 1) * 255).astype(np.uint8))
    return frames
