"""Ray tracing solid quadrics in PGA3D, for the examples' renders: parallel camera rays, where each ray enters
a quadric, the normal there from the polar plane, and shading by a lamp.

A quadric is a map from points to planes, `Plane <- Point`, zero on its surface where `quadric(point) & point`
vanishes and negative inside. Bound to a ray in both slots it is a quadratic in the distance along the ray, so a
ray meets it where that quadratic has a root.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext, stack
from numga.algebras import PGA3D as ga

mv = NumpyContext(ga).multivector
Point = ga.gatype.antivector()
Plane = ga.gatype.vector()
Quadric = ga.gatype((Plane, Point))                                   # Plane <- Point
# The coordinate planes.
AXES = stack((mv.x, mv.y, mv.z))                                      # [3] Plane
# The lamp's ambient share, its specular share and the specular exponent.
AMBIENT, SPECULAR, SHININESS = 0.3, 0.3, 40
# Traced pixels along each side of a shown one.
SUPERSAMPLE = 2


def ellipsoid(semi_axes: np.ndarray) -> Quadric:
    """The solid ellipsoid about the origin with the given semi-axes along x, y and z on the last axis: each
    coordinate plane weighted by the inverse square of its semi-axis, less the plane at infinity."""
    weights = 1 / np.asarray(semi_axes, dtype=float) ** 2
    return (AXES * (AXES & Point) * weights).sum(axis=-1) - mv.w * (mv.w & Point)


def quadric(matrix: np.ndarray, centre: np.ndarray) -> Quadric:
    """The solid quadric `(x - centre) @ matrix @ (x - centre) <= 1` for symmetric matrices `[..., 3, 3]` and
    centres `[..., 3]`, as a map from points to planes."""
    at_origin = (AXES[:, None] * (AXES & Point)[None, :] * matrix).sum(axis=(-1, -2)) - mv.w * (mv.w & Point)
    shift = ((stack((mv.xw, mv.yw, mv.zw)) * np.asarray(centre, dtype=float)).sum(axis=-1) / 2).exp()
    return shift >> at_origin(shift << Point)


def view_axes(elevation: float, azimuth: float) -> np.ndarray:
    """The unit directions screen right, screen up and toward the viewer `[3, 3]`, for a view from the given
    elevation and azimuth in degrees with z up."""
    tilt, turn = np.radians(elevation), np.radians(azimuth)
    back = np.array([np.cos(tilt) * np.cos(turn), np.cos(tilt) * np.sin(turn), np.sin(tilt)])
    right = np.array([-np.sin(turn), np.cos(turn), 0.0])
    return np.stack([right, np.cross(back, right), back])


def framing(points: np.ndarray, elevation: float, azimuth: float, margin: float) -> tuple[np.ndarray, np.ndarray]:
    """The centre and the half width and half height on the screen that frame points `[..., 3]` seen from the
    given elevation and azimuth, with a margin as a fraction of the larger side."""
    right, up, _ = view_axes(elevation, azimuth)
    screen = np.reshape(points, (-1, 3)) @ np.stack([right, up]).T
    low, high = screen.min(axis=0), screen.max(axis=0)
    half = (high - low) / 2 + margin * (high - low).max()
    return np.stack([right, up]).T @ ((low + high) / 2), half


def camera(elevation: float, azimuth: float, centre: np.ndarray, half: np.ndarray, pixels: int):
    """Parallel rays for an image of the given half width and half height about the centre, `pixels` across,
    seen from the given elevation and azimuth in degrees with z up: one origin per pixel on a screen far
    behind the scene, the shared heading, a direction, and the unit direction toward the lamp, above the
    viewer's left shoulder."""
    right, up, back = view_axes(elevation, azimuth)
    rows = max(1, int(round(pixels * half[1] / half[0])))
    across = np.linspace(-half[0], half[0], pixels)
    down = np.linspace(half[1], -half[1], rows)
    screen = centre + across[None, :, None] * right + down[:, None, None] * up + 20 * max(half) * back
    return mv.point(screen), mv.direction(-back), lamp(elevation, azimuth)


def lamp(elevation: float, azimuth: float) -> np.ndarray:
    """The unit direction toward the lamp, above the left shoulder of a viewer at the given elevation and
    azimuth."""
    right, up, back = view_axes(elevation, azimuth)
    toward = back - 0.4 * right + 0.9 * up
    return toward / np.linalg.norm(toward)


def ellipsoid_points(matrix: np.ndarray, centre: np.ndarray, n_samples: int) -> np.ndarray:
    """Points `[..., samples, 3]` on the surfaces of the solid quadrics `(x - centre) @ matrix @ (x - centre) <= 1`
    of positive definite matrices, for framing: the unit sphere, sampled on a spiral, carried by the inverse
    square root of each matrix."""
    index = np.arange(n_samples) + 0.5
    height = 1 - 2 * index / n_samples
    turn = np.pi * (1 + 5**0.5) * index
    sphere = np.stack([np.sqrt(1 - height**2) * np.cos(turn), np.sqrt(1 - height**2) * np.sin(turn), height], axis=-1)
    values, vectors = np.linalg.eigh(matrix)
    root = vectors @ (vectors / np.sqrt(values)[..., None, :]).swapaxes(-1, -2)
    return np.asarray(centre)[..., None, :] + sphere @ root.swapaxes(-1, -2)


def hits(surface: Quadric, origins: Point, heading: Point) -> tuple[np.ndarray, Point]:
    """The distance along each ray to where it enters the solid quadric, infinite where it misses or where the
    entry lies behind the origin, and the points hit. At the entering root the quadratic falls through zero."""
    a = (surface(heading) & heading).to_array()
    b = (surface(heading) & origins).to_array()
    c = (surface(origins) & origins).to_array()
    discriminant = b * b - a * c
    with np.errstate(divide="ignore", invalid="ignore"):
        distance = (-b - np.sqrt(np.clip(discriminant, 0, None))) / a
    distance = np.where((discriminant >= 0) & (a > 0) & (distance > 0), distance, np.inf)
    return distance, origins + heading * mv.scalar(np.where(np.isfinite(distance), distance, 0.0)[..., None])


def coordinates(points: Point) -> np.ndarray:
    """Euclidean coordinates of points `[..., 3]`: their pairings with the coordinate planes, over their weights."""
    return ((AXES & points[..., None]) / (mv.w & points)[..., None]).to_array()


def normals(surface: Quadric, points: Point) -> np.ndarray:
    """Unit outward normals `[..., 3]` of the quadric at points on it: the normals of their polar planes, read
    by the metric against the coordinate planes."""
    normal = (surface(points)[..., None] | AXES).to_array()
    return normal / (np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-12)


def shade(colour: np.ndarray, normal: np.ndarray, lamp: np.ndarray, toward_viewer: np.ndarray, lit: np.ndarray) -> np.ndarray:
    """Colours `[..., 3]` lit by the lamp: an ambient share everywhere, diffuse light and a highlight where the
    surface is lit and faces the lamp."""
    facing = np.clip(normal @ lamp, 0, None) * lit
    mirrored = 2 * (normal @ lamp)[..., None] * normal - lamp
    highlight = np.clip(mirrored @ toward_viewer, 0, None) ** SHININESS * (facing > 0)
    return colour * (AMBIENT + (1 - AMBIENT) * facing)[..., None] + SPECULAR * highlight[..., None]


def scene(surfaces: Quadric, colours: np.ndarray, view, background: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """An RGB image `[pixels, pixels, 3]` of solid quadrics on the last axis, each pixel the nearest one hit,
    in its colour `[surfaces, 3]`, shaded by the lamp, and the depth of that hit, infinite on the background."""
    origins, heading, lamp = camera(*view)
    distance, hit = hits(surfaces, origins[..., None], heading)              # [pixels, pixels, surfaces]
    nearest = distance.argmin(axis=-1)
    depth = np.take_along_axis(distance, nearest[..., None], axis=-1)[..., 0]
    normal = np.take_along_axis(normals(surfaces, hit), nearest[..., None, None], axis=-2)[..., 0, :]
    toward_viewer = view_axes(*view[:2])[2]
    rgb = shade(np.asarray(colours)[nearest], normal, lamp, toward_viewer, 1.0)
    return np.where(np.isfinite(depth)[..., None], rgb, background), depth


def segments(starts: np.ndarray, ends: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
    """Thin solid ellipsoids along segments between points `[..., 3]`, reaching from end to end with the given
    radius across: their matrices and centres, for `quadric`."""
    starts, ends = np.asarray(starts, dtype=float), np.asarray(ends, dtype=float)
    length = np.linalg.norm(ends - starts, axis=-1)[..., None, None]
    unit = (ends - starts) / length[..., 0]
    along = unit[..., :, None] * unit[..., None, :]
    return along / (length / 2) ** 2 + (np.eye(3) - along) / radius**2, (starts + ends) / 2


def downsample(image: np.ndarray, factor: int) -> np.ndarray:
    """An image `[rows, columns, channels]` traced at `factor` times the pixels it is shown at, each shown
    pixel the mean of its block of traced ones: supersampling, which smooths the edges."""
    rows, columns = (np.array(image.shape[:2]) // factor) * factor
    blocks = image[:rows, :columns].reshape(rows // factor, factor, columns // factor, factor, -1)
    return blocks.mean(axis=(1, 3))


def spheres(centres: np.ndarray, radius: float, colours: np.ndarray, view) -> tuple[np.ndarray, np.ndarray]:
    """An RGB image of many small spheres about centres `[n, 3]`, in colours `[n, 3]`, shaded by the lamp,
    and the depth along the rays of each pixel's nearest hit, infinite where none is: for parallel rays a
    sphere covers a disc on the screen, where its depth and normal have a closed form, so each sphere is
    drawn into its own pixels only."""
    elevation, azimuth, centre, half, pixels = view
    right, up, back = view_axes(elevation, azimuth)
    rows = max(1, int(round(pixels * half[1] / half[0])))
    scale = pixels / (2 * half[0])
    # Pixel coordinates, and the distance from the screen, of each centre.
    offset = np.asarray(centres) - centre
    column = (offset @ right + half[0]) * scale
    row = (half[1] - offset @ up) * scale
    distance = 20 * max(half) - offset @ back
    depth = np.full((rows, pixels), np.inf)
    normal = np.zeros((rows, pixels, 3))
    owner = np.zeros((rows, pixels), dtype=int)
    reach = int(np.ceil(radius * scale)) + 1
    for index in np.argsort(distance)[::-1]:
        low_row, low_column = max(int(row[index]) - reach, 0), max(int(column[index]) - reach, 0)
        rr, cc = np.mgrid[low_row:min(int(row[index]) + reach + 1, rows), low_column:min(int(column[index]) + reach + 1, pixels)]
        across = (cc - column[index]) / scale, (row[index] - rr) / scale
        inside = across[0] ** 2 + across[1] ** 2 < radius**2
        towards = np.sqrt(np.clip(radius**2 - across[0] ** 2 - across[1] ** 2, 0, None))
        hit = distance[index] - towards
        nearer = inside & (hit < depth[rr, cc])
        depth[rr[nearer], cc[nearer]] = hit[nearer]
        owner[rr[nearer], cc[nearer]] = index
        normal[rr[nearer], cc[nearer]] = ((across[0][..., None] * right + across[1][..., None] * up + towards[..., None] * back) / radius)[nearer]
    hit = np.isfinite(depth)
    rgb = np.zeros((rows, pixels, 3))
    rgb[hit] = shade(np.asarray(colours).reshape(-1, 3)[owner[hit]], normal[hit], lamp(elevation, azimuth), back, 1.0)
    return rgb, depth

