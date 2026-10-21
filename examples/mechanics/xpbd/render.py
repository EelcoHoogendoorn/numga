"""Drawing for the XPBD chain: every pixel's ray meets the nearest link, a solid ellipsoid placed by its motor,
over a floor that catches the links' shadows."""

from __future__ import annotations

import numpy as np

from examples import raytrace as rt
from examples.mechanics.xpbd.core import Motor

# Two alternating link colours, the pin's colour, the floor's two tiles and the background.
LINK_COLOURS = np.array([[0.86, 0.42, 0.20], [0.20, 0.52, 0.66]])
PIN_COLOUR = np.array([0.25, 0.25, 0.28])
FLOOR_COLOURS = np.array([[0.86, 0.85, 0.82], [0.76, 0.75, 0.72]])
BACKGROUND = np.array([0.97, 0.97, 0.96])
# The darker tone of a link's alternate octants, which shows it turning about its own axes.
OCTANT_CONTRAST = 0.22
# The view: its angles and its margin around the whole swing and its shadows, as a fraction of the larger side;
# the floor's depth below the swing's lowest point and its reach beyond the swing, and a floor tile's side,
# in link spacings; the pin's radius as a fraction of the spacing, and the pixels across.
ELEVATION, AZIMUTH = 14, -58
MARGIN = 0.08
FLOOR_DROP, FLOOR_REACH, TILE = 1.0, 12.0, 3.0
PIN_RADIUS = 0.18
PIXELS = 360


# --- plumbing -------------------------------------------------------------------------
def frame(links: Motor, shape: rt.Quadric, pin: rt.Quadric, floor: tuple[np.ndarray, np.ndarray], tile: float, view) -> np.ndarray:
    """One RGB image: the links and the pin over the floor, each pixel the nearest hit, shaded by the lamp;
    floor pixels whose path to the lamp passes through a link lie in its shadow."""
    origins, heading, lamp = rt.camera(*view)
    rays = origins[..., None]                                                 # [pixels, pixels, 1] Point
    placed = links >> shape(links << rt.Point)                                # [links] Plane <- Point
    distance, hit = rt.hits(placed, rays, heading)                            # [pixels, pixels, links]
    nearest = distance.argmin(axis=-1)
    depth = np.take_along_axis(distance, nearest[..., None], axis=-1)[..., 0]
    toward_viewer = -rt.coordinates(heading + rt.mv.w.dual())

    # The links: alternating colours, a darker tone on alternate octants of each link's own frame.
    normal = np.take_along_axis(rt.normals(placed, hit), nearest[..., None, None], axis=-2)[..., 0, :]
    local = np.take_along_axis(rt.coordinates(links << hit), nearest[..., None, None], axis=-2)[..., 0, :]
    octant = (local > 0).sum(axis=-1) % 2
    colour = LINK_COLOURS[nearest % 2] * (1 - OCTANT_CONTRAST * octant)[..., None]
    image = np.where(np.isfinite(depth)[..., None], rt.shade(colour, normal, lamp, toward_viewer, 1.0), BACKGROUND)

    # The pin, where it is nearer.
    pin_distance, pin_hit = rt.hits(pin, origins, heading)
    pin_rgb = rt.shade(PIN_COLOUR, rt.normals(pin, pin_hit), lamp, toward_viewer, 1.0)
    image = np.where((pin_distance < depth)[..., None], pin_rgb, image)
    depth = np.minimum(depth, pin_distance)

    # The floor, a square where nothing nearer covers it, checkered, and dark where a link blocks the lamp.
    low, high = floor
    start, direction = rt.coordinates(origins), -toward_viewer
    floor_distance = (low[2] - start[..., 2]) / direction[2]
    floor_xyz = start + floor_distance[..., None] * direction
    on_floor = np.all((floor_xyz[..., :2] > low[:2]) & (floor_xyz[..., :2] < high[:2]), axis=-1)
    tiles = (np.floor(floor_xyz[..., 0] / tile) + np.floor(floor_xyz[..., 1] / tile)) % 2
    shadow, _ = rt.hits(placed, rt.mv.point(floor_xyz)[..., None], rt.mv.direction(lamp))
    lit = 1.0 - 0.75 * np.isfinite(shadow).any(axis=-1)
    floor_rgb = rt.shade(FLOOR_COLOURS[tiles.astype(int)], np.array([0.0, 0.0, 1.0]), lamp, toward_viewer, lit)
    image = np.where((on_floor & (floor_distance < depth))[..., None], floor_rgb, image)
    return (np.clip(rt.downsample(image, rt.SUPERSAMPLE), 0, 1) * 255).astype(np.uint8)


def animate_chain(motors: Motor, semi_axes: np.ndarray, spacing: float) -> list[np.ndarray]:
    """One frame per step: the links swinging from the pin, the view framed by the whole swing."""
    centres = rt.coordinates(motors >> rt.mv.w.dual()).reshape(-1, 3)         # [steps + 1 * bodies, 3]
    low, high = centres.min(axis=0), centres.max(axis=0)
    # The floor: a square below the swing, reaching well beyond the frame.
    floor = (np.append(low[:2] - FLOOR_REACH * spacing, low[2] - FLOOR_DROP * spacing), high + FLOOR_REACH * spacing)
    # Frame the swing, each link's whole extent, and its shadow on the floor, cast along the lamp.
    toward_lamp = rt.lamp(ELEVATION, AZIMUTH)
    shadows = centres - toward_lamp * ((centres[:, 2] - floor[0][2]) / toward_lamp[2])[:, None]
    middle, half = rt.framing(np.concatenate([centres, shadows]), ELEVATION, AZIMUTH, MARGIN)
    view = (ELEVATION, AZIMUTH, middle, half + semi_axes.max(), PIXELS * rt.SUPERSAMPLE)
    # The pin sits at the joint of the fixed body and the first link.
    pin = rt.quadric(np.eye(3) / (PIN_RADIUS * spacing) ** 2, centres[0] + [spacing / 2, 0, 0])
    shape = rt.ellipsoid(semi_axes)
    return [frame(links, shape, pin, floor, TILE * spacing, view) for links in motors[:, 1:]]
