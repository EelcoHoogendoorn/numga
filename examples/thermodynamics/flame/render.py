"""Rendering for a cellular flame draws the triangulated disc from above beside its enthalpy field at ignition and
across the flame's growth.

The left panel shades the mixture like a burning sheet of paper: pale where unburnt, dark char where burnt, smouldering
red where sensible heat exceeds the adiabatic rise, and glowing from deep red through orange to pale yellow with the
volumetric reaction rate so that cellular bulges shine and extinguished creases leave dark trails. The right panel
colours enthalpy red above the unburnt mixture's value and blue below it.

Vertex positions are plain spatial offsets read by pairing with `x` and `y` at unit spatial weight in `Camera`, while
parcels are read by pairing with `h` for sensible heat and with the enthalpy plane for total energy.
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
from examples.thermodynamics.flame import core

mv = core.mv
Topology = simplices.Topology
unburnt_colour, burnt_colour = np.array(to_rgb("#f4efe4")), np.array(to_rgb("#1c0a06"))
ember_colour = np.array(to_rgb("#9a2a0c"))
glow_map, energy_map = colormaps["afmhot"], colormaps["RdBu_r"]
# Panel width in pixels, radial margin fraction, and white gap between panels in pixels.
n_pixels, margin, n_gap = 300, 0.02, 8
# Peak reaction rate for full glow, active slice of the glow colormap, and gamma exponent brightening faint fronts.
brightest, glow_start, glow_end, glow_power = 3.0, 0.3, 0.8, 0.5
# Excess sensible heat above unity for full ember red, and enthalpy half-range and active slice of the diverging map.
ember_span = 0.3
energy_span, span = 0.3, 0.8


# --- plumbing -------------------------------------------------------------------------
def on_white(picture: np.ndarray) -> np.ndarray:
    """RGBA coverage composited over a white background as RGB bytes."""
    shown = picture[..., :3] + (1 - picture[..., 3:])
    return np.round(np.clip(shown, 0, 1) * 255).astype(np.uint8)


def side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Two equal-height RGB images joined horizontally with a narrow white gap."""
    return np.concatenate([left, np.full((left.shape[0], n_gap, 3), 255, dtype=np.uint8), right], axis=1)


def disc_camera(radius: float) -> Camera:
    """Top-down camera framing the disc, reading spatial offsets by pairing with `x` and `y` at unit weight."""
    return Camera(mv.x, mv.y, 0 * mv.x, 0 * mv.x, 1.0, np.zeros(2), np.full(2, radius * (1 + margin)), n_pixels)


def burning_paper(points: Extensor, topology: Topology, camera: Camera, heat: np.ndarray,
                  reaction_rate: np.ndarray) -> np.ndarray:
    """RGB image of the disc shaded as burning paper from sensible heat and overlaid with reaction-rate glow."""
    charred = np.clip(heat, 0, 1)[:, None]
    smouldering = np.clip((heat - 1) / ember_span, 0, 1)[:, None]
    glow = np.clip(reaction_rate / brightest, 0, 1)[:, None] ** glow_power
    base = (1 - charred) * unburnt_colour + charred * ((1 - smouldering) * burnt_colour + smouldering * ember_colour)
    flame = glow_map(glow_start + (glow_end - glow_start) * glow[:, 0])[:, :3]
    colours = (1 - glow) * base + glow * flame
    return on_white(simplices.composited([contours.picture(points, topology, charred[:, 0], colours, camera)]))


# --- drawing --------------------------------------------------------------------------
def energy_picture(points: Extensor, topology: Topology, camera: Camera, enthalpy_excess: np.ndarray) -> np.ndarray:
    """RGB image of the enthalpy departure from the unburnt mixture, coloured red where positive and blue where negative."""
    colours = energy_map(np.clip(0.5 + span / 2 * enthalpy_excess / energy_span, 0, 1))[:, :3]
    return on_white(simplices.composited([contours.picture(points, topology, enthalpy_excess, colours, camera)]))


def draw_start(topology: Topology, vertices: Extensor, radius: float, energy: Extensor, parcels: Extensor) -> Figure:
    """Side-by-side view of the initial charred sparks and uniform enthalpy field before ignition advances."""
    camera = disc_camera(radius)
    points = vertices.batch()
    heat = np.asarray((mv.h & parcels.batch()).to_array())
    enthalpy_excess = np.asarray((energy & parcels.batch()).to_array()) - 1
    return image_figure(side_by_side(burning_paper(points, topology, camera, heat, np.zeros_like(heat)),
                                     energy_picture(points, topology, camera, enthalpy_excess)))


def animate(topology: Topology, vertices: Extensor, radius: float, energy: Extensor,
            frames: Iterable[tuple[Extensor, Extensor]]) -> Iterator[np.ndarray]:
    """Yield side-by-side RGB frames of the burning-paper view and the enthalpy departure across the simulation."""
    camera = disc_camera(radius)
    points = vertices.batch()
    for parcels, reaction_rate in frames:
        heat = np.asarray((mv.h & parcels.batch()).to_array())
        enthalpy_excess = np.asarray((energy & parcels.batch()).to_array()) - 1
        yield side_by_side(burning_paper(points, topology, camera, heat, np.asarray(reaction_rate.batch().to_array())),
                           energy_picture(points, topology, camera, enthalpy_excess))
