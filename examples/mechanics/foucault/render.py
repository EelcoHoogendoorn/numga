"""The earth turning a pendulum's floor underneath it, ray traced on the GPU: the globe translucent with its polar
cap, its equator and the latitude circle inked on it, the floor a slab on the globe with the swing and north on it
as rods."""

from __future__ import annotations

from collections.abc import Iterator

from matplotlib.colors import to_rgb
import numpy as np

from numga import concatenate, stack
from examples.mechanics.foucault import core
from examples.util.wgsl import solids

SWING_COLOUR = "#d06c36"
GLOBE_COLOUR = "0.85"
CAP_COLOUR = "#287fa0"
FLOOR_COLOUR = "#efe9dc"
COMPASS_COLOUR = "0.35"
FLOOR_EDGE_COLOUR = "0.5"
EQUATOR_COLOUR = "0.6"
# The half width of the floor drawn on the globe and its thickness, and how far north its compass reaches from the
# middle.
FLOOR, FLOOR_THICKNESS = 0.3, 0.02
COMPASS = 1.4
# The view of the globe, elevation and azimuth in radians, azimuth zero looking along +y; the margin around the
# globe as a share of its radius, and the pixels across a picture.
ELEVATION, AZIMUTH = np.radians([50.0, -35.0 + 90.0])
MARGIN = 0.05
PIXELS = 440
# How much of what lies behind them each surface of the globe and of its cap hides; the width of the circles, the
# radii of the floor's edges, of the swing and of the compass, in pixels.
GLOBE_OPACITY, CAP_OPACITY = 0.14, 0.3
CIRCLE_WIDTH, EDGE_RADIUS, SWING_RADIUS, COMPASS_RADIUS = 2.0, 0.8, 2.0, 1.0


# --- plumbing -------------------------------------------------------------------------
def globe(earth: core.Rotor, vertical: core.Point, swing: core.Point, north: core.Point, east: core.Point) -> np.ndarray:
    """The earth turned by a rotor: its equator and the site's latitude circle, the polar cap that circle
    bounds, and the site's floor with the line the pendulum swings along."""
    view = solids.view([solids.origin], 1.0 + FLOOR, ELEVATION, AZIMUTH, MARGIN, PIXELS)
    sphere = solids.ball(solids.origin, 1.0)[None, None]                      # [1, 1] Plane+ <- Point-
    # The equator and the site's latitude circle, inked where their planes across the axis cut the globe, and the cap:
    # the globe on the far side of the latitude circle's plane from the equator, the plane z moved to the site.
    on_latitude = solids.origin + solids.context.include_from_algebra(vertical)         # [] Point-
    latitude = solids.mv.z - solids.mv.w * (solids.mv.z & on_latitude)        # [] Plane+
    circles = solids.half_space(stack([solids.mv.z, latitude]))                 # [2] Plane+ <- Point-
    # The floor at the site, turned with the earth: a slab across the vertical, north and east along its sides, its top
    # lifted off the globe by the latitude circle's width, so that the circle through the site passes under it.
    site = vertical * (1 + CIRCLE_WIDTH * view.pixel)                          # [] Point-
    signs = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * FLOOR             # [corners, 2]
    # The swing and the compass lie on the floor, their axes their own radius above it.
    resting = site + vertical * (SWING_RADIUS * view.pixel)                   # [] Point-
    swing_ends, compass_ends, tile = (solids.origin + solids.context.include_from_algebra(earth >> points) for points in (
        stack([resting - swing * FLOOR, resting + swing * FLOOR]),
        stack([resting + north * FLOOR, resting + north * (COMPASS * FLOOR)]),
        stack([site + east * across + north * along for across, along in signs])))   # [2] Point-, [2] Point-, [corners] Point-
    # The floor's sides, turned with the earth: the directions from the earth's centre, its origin.
    sides = solids.context.include_from_algebra(earth >> stack([vertical, north, east]))   # [3] Point-
    middle = (solids.origin + solids.context.include_from_algebra(earth >> site)
              - sides[0] * (FLOOR_THICKNESS / 2))                              # [] Point-
    half_extents = np.array([FLOOR_THICKNESS / 2, FLOOR, FLOOR])
    # The slab: inside the planes square to each of its sides, as far from its middle as it reaches.
    across = concatenate([-sides.dual(), sides.dual()])                        # [6] Plane+
    floor = (across - solids.mv.w * (across & middle) - solids.mv.w * np.concatenate([half_extents, half_extents]))[None]   # [1, 6] Plane+
    # The swing through the middle of the floor, both ways, and north from its edge outwards.
    swing_rod = solids.rods(swing_ends[0], swing_ends[1], SWING_RADIUS * view.pixel)
    compass = solids.rods(compass_ends[0], compass_ends[1], COMPASS_RADIUS * view.pixel)
    edges, joints = solids.edges(tile, tile[[1, 2, 3, 0]], EDGE_RADIUS * view.pixel)
    floor_colour, swing_colour, compass_colour, edge_colour = (np.array(to_rgb(colour)) for colour in
                                                               (FLOOR_COLOUR, SWING_COLOUR, COMPASS_COLOUR, FLOOR_EDGE_COLOUR))
    parts = [(floor, floor_colour), (swing_rod[None], swing_colour), (compass[None], compass_colour), (edges, edge_colour),
             (joints, edge_colour)]
    # The globe is the first translucent part, after the opaque ones.
    inks = [solids.Ink(len(parts), solids.half_space(latitude)[None],
                       solids.half_space(-latitude)[None], np.array(to_rgb(CAP_COLOUR)), np.inf, CAP_OPACITY),
            solids.Ink(len(parts), circles, solids.everywhere[None],
                       np.array([to_rgb(EQUATOR_COLOUR), to_rgb(CAP_COLOUR)]), CIRCLE_WIDTH, 1.0)]
    return solids.draw(parts, view, np.ones(3), inks,
                       translucent=[(sphere, np.array(to_rgb(GLOBE_COLOUR)), GLOBE_OPACITY)], meshes=[])


def animate_day(earth: core.Rotor, swings: core.Point, vertical: core.Point, north: core.Point,
                east: core.Point) -> Iterator[np.ndarray]:
    """Per frame of the day: the earth turned so far, with the pendulum's floor and the line it swings along."""
    for index in range(len(earth) - 1):
        yield globe(earth[index], vertical, swings[index], north, east)
