"""Drawing for a cellular flame: the disk of mixture seen from above, beside the same disk coloured by the energy its
parcels hold, at the start and as the flame spreads.

The flame is drawn as a sheet of paper burns: pale where the mixture is unburnt and charred dark where it has burnt, by
the heat its parcels hold, smouldering red where burnt mixture holds more heat than its fuel released, and with the rate
its fuel burns at glowing over that from deep red through orange to pale yellow, so that the front's cells shine and
the creases between them, where the flame all but goes out, show dark and leave dark trails. The energy is drawn red
where a parcel holds more than the mean unburnt mixture and blue where it holds less.

The disk's points are read by the planes x and y and weighed by their volume, `mv.v & point`, the algebra's own
projective weight; the parcels are read by the planes given, `mv.h` for their heat and the energy's plane for their
energy.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure

from numga.extensor import Extensor
from examples.util.wgsl import contours, image_figure, simplices
from examples.util.wgsl.camera import Camera

Topology = simplices.Topology
unburnt_colour, burnt_colour = np.array(to_rgb("#f4efe4")), np.array(to_rgb("#1c0a06"))
ember_colour = np.array(to_rgb("#9a2a0c"))
glow_map, energy_map = colormaps["afmhot"], colormaps["RdBu_r"]
# Each panel's pixels across, the room around the disk in its radii, and the columns between the panels.
n_pixels, margin, n_gap = 300, 0.02, 8
# The rate of burning per volume that glows brightest, the shares of the glow's colour map it runs between, and the
# power that brightens a faint glow.
brightest, glow_start, glow_end, glow_power = 3.0, 0.3, 0.8, 0.5
# The heat beyond what the fuel released at which burnt mixture smoulders brightest.
ember_span = 0.3
# The energy's departure from the mean unburnt mixture's at either end of its colour map, and the share of the map it spans.
energy_span, span = 0.3, 0.8


# --- plumbing -------------------------------------------------------------------------
def on_white(picture: np.ndarray) -> np.ndarray:
    """Colour and coverage `[rows, columns, 4]` laid over white, as bytes `[rows, columns, 3]`."""
    shown = picture[..., :3] + (1 - picture[..., 3:])
    return np.round(np.clip(shown, 0, 1) * 255).astype(np.uint8)


def side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Two pictures of one height, the first left of the second, with white columns between them."""
    return np.concatenate([left, np.full((left.shape[0], n_gap, 3), 255, dtype=np.uint8), right], axis=1)


def disk_camera(vertices: Extensor, radius: float) -> Camera:
    """The camera looking down on a disk of this radius, `n_pixels` across: points read by the planes x and y, weighed
    by their volume."""
    mv = vertices.context.multivector
    return Camera(mv.x, mv.y, 0 * mv.x, mv.v, 0.0, np.zeros(2), np.full(2, radius * (1 + margin)), n_pixels)


def burning_paper(points: Extensor, topology: Topology, camera: Camera, heat: np.ndarray,
                  burning: np.ndarray) -> np.ndarray:
    """The heat the mixture's parcels hold `[vertices]` and the rate their fuel burns at `[vertices]` as a sheet of
    paper burns: pale where unburnt, charred dark where burnt, smouldering where burnt mixture holds more heat than its
    fuel released, and the burning glowing over all of it, as bytes `[rows, columns, 3]`."""
    charred = np.clip(heat, 0, 1)[:, None]
    smouldering = np.clip((heat - 1) / ember_span, 0, 1)[:, None]
    glow = np.clip(burning / brightest, 0, 1)[:, None] ** glow_power
    base = (1 - charred) * unburnt_colour + charred * ((1 - smouldering) * burnt_colour + smouldering * ember_colour)
    flame = glow_map(glow_start + (glow_end - glow_start) * glow[:, 0])[:, :3]
    colours = (1 - glow) * base + glow * flame
    return on_white(simplices.composited([contours.picture(points, topology, charred[:, 0], colours, camera)]))


# --- drawing --------------------------------------------------------------------------
def energy_picture(points: Extensor, topology: Topology, camera: Camera, held: np.ndarray) -> np.ndarray:
    """The energy the mixture's parcels hold beyond the mean unburnt mixture's `[vertices]`, red where more and blue
    where less, as bytes `[rows, columns, 3]`."""
    colours = energy_map(np.clip(0.5 + span / 2 * held / energy_span, 0, 1))[:, :3]
    return on_white(simplices.composited([contours.picture(points, topology, held, colours, camera)]))


def draw_start(topology: Topology, vertices: Extensor, radius: float, energy: Extensor, parcels: Extensor) -> Figure:
    """The disk of this radius with the mixture's parcels `Point-[vertices]` at the start, before anything burns: the
    sparks charred dark in the pale unburnt mixture, beside the energy the plane `energy` `[] Plane+` reads off them."""
    mv = vertices.context.multivector
    camera = disk_camera(vertices, radius)
    points = vertices.batch()
    heat = np.asarray((mv.h & parcels).to_array())
    held = np.asarray((energy & parcels).to_array()) - 1
    return image_figure(side_by_side(burning_paper(points, topology, camera, heat, np.zeros_like(heat)),
                                     energy_picture(points, topology, camera, held)))


def animate(topology: Topology, vertices: Extensor, radius: float, energy: Extensor,
            frames: Iterable[tuple[Extensor, Extensor]]) -> Iterator[np.ndarray]:
    """The disk of this radius at every frame of the mixture's parcels `Point-[vertices]` and the rate their fuel burns
    at `Scalar+[vertices]` as they come: the flame, as a sheet of paper burns, beside the energy the plane `energy`
    `[] Plane+` reads off the parcels, red above the mean unburnt mixture's and blue below."""
    mv = vertices.context.multivector
    camera = disk_camera(vertices, radius)
    points = vertices.batch()
    for parcels, burning in frames:
        heat = np.asarray((mv.h & parcels).to_array())
        held = np.asarray((energy & parcels).to_array()) - 1
        yield side_by_side(burning_paper(points, topology, camera, heat, np.asarray(burning.to_array())),
                           energy_picture(points, topology, camera, held))
