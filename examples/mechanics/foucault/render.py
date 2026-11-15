"""GPU ray-traced globe depicts a pendulum's floor riding around its latitude circle above the
shaded polar cap while the swing rod turns relative to the floor's north rod."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from matplotlib.colors import to_rgb
import numpy as np

from numga import stack
from examples.mechanics.foucault import core
from numga.topology import Topology
from numga.render import solids

swing_colour = "#d06c36"
globe_colour = "0.85"
cap_colour = "#287fa0"
floor_colour = "#efe9dc"
compass_colour = "0.35"
floor_edge_colour = "0.5"
equator_colour = "0.6"
# Floor half-width, slab thickness, and outward reach of the north compass rod in globe radii.
floor_half, floor_thickness = 0.3, 0.02
compass_reach = 1.4
# Camera elevation and azimuth in radians (zero azimuth looks along +y), framing margin, and image width in pixels.
elevation, azimuth = np.radians([50.0, -35.0 + 90.0])
margin = 0.05
n_pixels = 440
# Opacities of the globe and polar cap, inked circle width, and rod radii in pixels.
globe_opacity, cap_opacity = 0.14, 0.3
circle_width, edge_radius, swing_radius, compass_radius = 2.0, 0.8, 2.0, 1.0


# --- plumbing -------------------------------------------------------------------------
def globe_image(earth: core.Rotor, vertical: core.Point, swing: core.Point, north: core.Point, east: core.Point) -> np.ndarray:
    """Ray-traces the `[rows, n_pixels, 3]` RGB globe turned by `earth`, with the inked equator, latitude circle, polar cap, and floor slab."""
    view = solids.view([solids.origin], 1.0 + floor_half, elevation, azimuth, margin, n_pixels)
    sphere = solids.ball(solids.origin, 1.0)[None, None]                      # [1, 1] Plane+ <- Point-
    # Shift the equatorial plane z to the site to ink the latitude circle and bound the polar cap above it.
    on_latitude = solids.origin + solids.context.include_from_algebra(vertical)  # [] Point-
    latitude = solids.mv.z - solids.mv.w * (solids.mv.z & on_latitude)        # [] Plane+
    circles = solids.half_space(stack([solids.mv.z, latitude]))               # [2] Plane+ <- Point-
    # Lift the floor top above the unit sphere by the inked circle width so the latitude circle passes cleanly underneath.
    site = vertical * (1 + circle_width * view.pixel)                         # [] Point-
    signs = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * floor_half       # [4, 2]
    spans = stack([east * across + north * forward for across, forward in signs])  # [4] Point-
    middle = site - vertical * (floor_thickness / 2)                          # [] Point-
    # Rest the swing and north rods on the floor top by offsetting their axes by one rod radius.
    resting = site + vertical * (swing_radius * view.pixel)                   # [] Point-
    # Push local offsets forward by the earth's rotor and embed them into projective space around the globe centre.
    swing_ends, compass_ends, tile, slab = (
        solids.origin + solids.context.include_from_algebra(earth >> points)
        for points in (
            stack([resting - swing * floor_half, resting + swing * floor_half]),
            stack([resting + north * floor_half, resting + north * (compass_reach * floor_half)]),
            site + spans,
            middle + spans,
        )
    )                                                                         # [2] Point-, [2] Point-, [4] Point-, [4] Point-
    floor_plate = solids.plates(slab, floor_thickness)[None]                  # [1, 6] Plane+
    swing_rod = solids.rods(swing_ends[0], swing_ends[1], swing_radius * view.pixel)  # [3] Plane+ <- Point-
    compass_rod = solids.rods(compass_ends[0], compass_ends[1], compass_radius * view.pixel)  # [3] Plane+ <- Point-
    rim = Topology.loops([4])
    tails, heads = rim.incidence[1, 0] * tile.field()                         # [2] Point-[E]
    rim_rods, rim_joints = solids.edges(tails.batch(), heads.batch(), edge_radius * view.pixel)
    floor_rgb, swing_rgb, compass_rgb, edge_rgb = (
        np.array(to_rgb(colour))
        for colour in (floor_colour, swing_colour, compass_colour, floor_edge_colour)
    )
    parts = [
        (floor_plate, floor_rgb),
        (swing_rod[None], swing_rgb),
        (compass_rod[None], compass_rgb),
        (rim_rods, edge_rgb),
        (rim_joints, edge_rgb),
    ]
    # Index len(parts) addresses the translucent globe placed immediately after the opaque parts.
    inks = [
        solids.Ink(
            len(parts),
            solids.half_space(latitude)[None],
            solids.half_space(-latitude)[None],
            np.array(to_rgb(cap_colour)),
            np.inf,
            cap_opacity,
        ),
        solids.Ink(
            len(parts),
            circles,
            solids.everywhere[None],
            np.array([to_rgb(equator_colour), to_rgb(cap_colour)]),
            circle_width,
            1.0,
        ),
    ]
    return solids.draw(
        parts,
        view,
        np.ones(3),
        inks,
        translucent=[(sphere, np.array(to_rgb(globe_colour)), globe_opacity)],
        meshes=[],
    )


def day_animation(
    vertical: core.Point,
    north: core.Point,
    east: core.Point,
    states: Iterable[tuple[core.Rotor, core.Point]],
) -> Iterator[np.ndarray]:
    """Yields `[rows, n_pixels, 3]` RGB frames across a looping day, omitting the closing duplicate of frame zero."""
    iterator = iter(states)
    previous = next(iterator, None)
    for current in iterator:
        earth, swing = previous
        yield globe_image(earth, vertical, swing, north, east)
        previous = current
