"""Implicit renders of the top on its ground: every pixel's ray meets the nearest quadric."""

import numpy as np
from examples.mechanics.spinning_top import core

mv, w, Point = core.mv, core.w, core.Point
Scalar, Direction, Quadric, Motor = core.Scalar, core.Direction, core.Quadric, core.Motor
# Colours of the disc, the stem and the tip; alternating sectors, drawn on the disc, show its spin.
PART_COLOURS = np.array([[0.85, 0.30, 0.25], [0.35, 0.35, 0.40], [0.80, 0.70, 0.35]])
SECTOR_CONTRAST = np.array([0.25, 0.0, 0.0])


def camera(elevation: float, azimuth: float, centre: np.ndarray, extent: float,
           pixels: int) -> tuple[Point, Direction, np.ndarray]:
    """Parallel rays for a square image of half-width extent about the centre, seen from the given
    elevation and azimuth in degrees: one origin per pixel on a screen far behind the scene, one
    shared heading, and the unit direction toward a lamp above the viewer's left shoulder."""
    tilt, turn = np.radians(elevation), np.radians(azimuth)
    # Toward the viewer.
    back = np.array([np.cos(tilt) * np.cos(turn), np.cos(tilt) * np.sin(turn), np.sin(tilt)])
    right = np.array([-np.sin(turn), np.cos(turn), 0.0])
    up = np.cross(back, right)
    offsets = np.linspace(-extent, extent, pixels)
    screen = centre + offsets[None, :, None] * right + offsets[::-1, None, None] * up + 20 * extent * back
    lamp = back - 0.3 * right + 0.8 * up
    return core.point(screen), mv(Direction, -back), lamp / np.linalg.norm(lamp)


def hits(surface: Quadric, origins: Point, heading: Direction) -> tuple[np.ndarray, Point]:
    """The distance along each ray's line to where it enters the solid quadric, infinite where it
    misses, and the points hit. Bound to a ray in both slots, the form is a quadratic in the distance;
    at the entering root it falls through zero, so the root is the one with
    `a * distance + b == -np.sqrt(discriminant)`, wherever along the line the ray starts."""
    a = (surface(heading) & heading).to_array()
    b = (surface(heading) & origins).to_array()
    c = (surface(origins) & origins).to_array()
    discriminant = b * b - a * c
    with np.errstate(divide="ignore", invalid="ignore"):
        distance = np.where(discriminant >= 0, (-b - np.sqrt(np.clip(discriminant, 0, None))) / a, np.inf)
    return distance, origins + heading * mv.scalar(np.where(np.isfinite(distance), distance, 0.0)[..., None])


def components(directions: Direction) -> np.ndarray:
    """Euclidean components of directions: their inner products with the axis directions."""
    return (directions[..., None].dual() | mv(Direction, np.eye(3)).dual()).to_array()


def normals(surface: Quadric, points: Point) -> np.ndarray:
    """Unit normals of the quadric at points on it: the normals of their polar planes."""
    normal = components(surface(points).dual().cast(Direction))
    return normal / (np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-12)


def coordinates(points: Point) -> np.ndarray:
    """Euclidean coordinates of points: their pairings with the coordinate planes, at unit weight."""
    return np.stack([((plane & points) / (w & points)).to_array() for plane in core.axes], axis=-1)


def frame(motor: Motor, parts: Quadric, ground: Quadric, view: tuple[float, float], centre: np.ndarray,
          extent: float, pixels: int) -> np.ndarray:
    """One RGB image of the top placed by the motor on the ground. Each pixel keeps the nearest hit
    over the ground and the parts, shaded by how squarely its surface faces the lamp; the ground
    carries a checkerboard, and each part alternating sectors about the top's axis, with its own
    contrast."""
    origins, heading, lamp = camera(*view, centre, extent, pixels)
    # The parts in the world.
    placed = motor >> parts(motor << Point)                         # [parts] Plane <- Point
    depth = np.full((pixels, pixels), np.inf)
    rgb = np.ones((pixels, pixels, 3))
    distance, hit = hits(ground, origins, heading)
    xy = coordinates(hit)[..., :2]
    checker = (np.floor(xy[..., 0] / 0.15) + np.floor(xy[..., 1] / 0.15)) % 2
    shade = 0.45 + 0.55 * np.abs(normals(ground, hit) @ lamp)
    ground_rgb = (0.62 + 0.18 * checker)[..., None] * np.array([0.95, 0.93, 0.88]) * shade[..., None]
    rgb = np.where(np.isfinite(distance)[..., None], ground_rgb, rgb)
    depth = np.where(np.isfinite(distance), distance, depth)
    for part, colour, contrast in zip(placed, PART_COLOURS, SECTOR_CONTRAST):
        distance, hit = hits(part, origins, heading)
        nearer = distance < depth
        # The hit in the top's own frame.
        local = coordinates(motor << hit)
        sector = (np.floor(np.arctan2(local[..., 1], local[..., 0]) / (np.pi / 4)) % 2)[..., None]
        tint = colour * (1 - contrast * (1 - sector))
        shade = 0.35 + 0.65 * np.abs(normals(part, hit) @ lamp)
        rgb = np.where(nearer[..., None], tint * shade[..., None], rgb)
        depth = np.where(nearer, distance, depth)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
