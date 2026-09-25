"""Implicit renders: every pixel's ray meets the arm's boxes and the ellipsoids; nearest hit wins.

A box is the unit cube carried by its body map. Pulled back through the inverse map the ray is
unchanged in its parameter, and the cube's three slabs clip it. A quadric's form bound to the ray
is a quadratic in the same parameter.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.mechanics.manipulability import core

mv, w = core.mv, core.w
# The unit cube lies within half a unit of each of these planes.
SLABS = mv("x y z", np.eye(3))                   # [3] Plane
BOX_COLOURS = np.array([[0.45, 0.47, 0.52], [0.62, 0.64, 0.70], [0.78, 0.60, 0.30], [0.30, 0.55, 0.62], [0.55, 0.40, 0.60]])
VELOCITY_COLOUR, FORCE_COLOUR = np.array([0.25, 0.45, 0.85]), np.array([0.92, 0.55, 0.20])
QUADRIC_ALPHA = 0.55


def euclidean(points: core.Point) -> np.ndarray:
    """Euclidean coordinates of points, from their homogeneous components at unit weight."""
    homogeneous = points.dual().kernel
    return homogeneous[..., :3] / homogeneous[..., 3:]


def camera(elevation: float, azimuth: float, centre: np.ndarray, extent: float, pixels: int):
    """An orthographic camera: ray origins far out along the view, one per pixel, and the heading."""
    tilt, turn = np.radians(elevation), np.radians(azimuth)
    # Toward the viewer.
    back = np.array([np.cos(tilt) * np.cos(turn), np.cos(tilt) * np.sin(turn), np.sin(tilt)])
    right = np.array([-np.sin(turn), np.cos(turn), 0.0])
    up = np.cross(back, right)
    offsets = np.linspace(-extent, extent, pixels)
    screen = centre + offsets[None, :, None] * right + offsets[::-1, None, None] * up + 20 * extent * back
    lamp = back - 0.4 * right + 0.7 * up
    return core.arm.point(screen), mv("x y z", -back).dual(), lamp / np.linalg.norm(lamp)


def box_hits(body: core.arm.PointMap, origins: core.Point, heading: core.Point):
    """Ray parameter of the first hit on a box, infinite where it misses, and the world normal of the face hit."""
    to_box = body.inverse()
    local_origins, local_heading = to_box(origins), to_box(heading)
    weight = (w & local_origins).to_array()
    start = np.stack([(slab & local_origins).to_array() / weight for slab in SLABS], axis=-1)
    rate = np.stack([np.broadcast_to((slab & local_heading).to_array() / weight, weight.shape) for slab in SLABS], axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        low, high = (-0.5 - start) / rate, (0.5 - start) / rate
    near, far = np.minimum(low, high), np.maximum(low, high)
    enter, leave = near.max(axis=-1), far.min(axis=-1)
    distance = np.where((enter < leave) & (enter > 0), enter, np.inf)
    face = near.argmax(axis=-1)
    # Face normals, by the inverse transpose of the body's linear part.
    normal = np.eye(3)[face] @ np.linalg.inv(body.kernel[:3, :3])
    return distance, normal


def quadric_hits(surface: core.Quadric, origins: core.Point, heading: core.Point):
    """Ray parameter where each ray enters the solid quadric, and the normal there, from its polar
    plane. Bound to a ray in both slots, the form is a quadratic in the parameter; at the entering
    root it falls through zero."""
    a = (surface(heading) & heading).to_array()
    b = (surface(heading) & origins).to_array()
    c = (surface(origins) & origins).to_array()
    discriminant = b * b - a * c
    with np.errstate(divide="ignore", invalid="ignore"):
        distance = np.where(discriminant >= 0, (-b - np.sqrt(np.clip(discriminant, 0, None))) / a, np.inf)
    hits = origins + heading * mv.scalar(np.where(np.isfinite(distance), distance, 0.0)[..., None])
    return distance, surface(hits).kernel[..., :3]


def image(bodies, surface: core.Quadric, colour: np.ndarray, view, centre: np.ndarray, extent: float, pixels: int) -> np.ndarray:
    """One RGB image of the arm's boxes, opaque, under a translucent quadric: each pixel keeps the
    nearest box, shaded by how squarely its face meets the lamp, and the quadric is laid over it
    wherever the ray enters the quadric first, so the arm shows through."""
    origins, heading, lamp = camera(*view, centre, extent, pixels)
    hits = [box_hits(body, origins, heading) for body in bodies]
    depth = np.stack([distance for distance, _ in hits])                # [boxes, pixels, pixels]
    nearest = depth.argmin(axis=0)
    normal = np.choose(nearest[..., None], [normal / np.linalg.norm(normal, axis=-1, keepdims=True) for _, normal in hits])
    shade = 0.35 + 0.65 * np.abs(np.sum(normal * lamp, axis=-1))
    boxes = np.where(np.isfinite(depth.min(axis=0))[..., None], BOX_COLOURS[nearest] * shade[..., None], 1.0)

    distance, normal = quadric_hits(surface, origins, heading)
    normal = normal / (np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-12)
    shade = 0.45 + 0.55 * np.abs(np.sum(normal * lamp, axis=-1))
    over = (distance < depth.min(axis=0))[..., None]
    return np.where(over, (1 - QUADRIC_ALPHA) * boxes + QUADRIC_ALPHA * colour * shade[..., None], boxes)


def frame(pose, view, centre: np.ndarray, extent: float, pixels: int) -> np.ndarray:
    """One animation frame: the velocity ellipsoid beside the force ellipsoid, as 8-bit RGB."""
    bodies, _, velocity, force = pose
    gap = np.ones((pixels, pixels // 20, 3))
    panels = [image(bodies, velocity, VELOCITY_COLOUR, view, centre, extent, pixels), gap,
              image(bodies, force, FORCE_COLOUR, view, centre, extent, pixels)]
    return (np.clip(np.concatenate(panels, axis=1), 0, 1) * 255).astype(np.uint8)


def draw_ellipsoids(poses, view, centre: np.ndarray, extent: float, pixels: int) -> plt.Figure:
    """One row per pose: the velocity ellipsoid beside the force ellipsoid, at the gripper."""
    figure, axes = plt.subplots(len(poses), 1, figsize=(11, 5.5 * len(poses)))
    for ax, pose in zip(np.atleast_1d(axes), poses):
        ax.imshow(frame(pose, view, centre, extent, pixels), interpolation="bilinear")
        ax.set_axis_off()
    figure.tight_layout()
    return figure
