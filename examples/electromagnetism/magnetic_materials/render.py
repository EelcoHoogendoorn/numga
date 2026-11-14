"""Magnetic field lines cross the permanent magnet, iron core, copper ring and surrounding space."""
from __future__ import annotations

from collections.abc import Iterable, Iterator
import matplotlib.pyplot as plt
import numpy as np
from numga import Extensor
from numga.render import Camera, contours, drawing
from numga.topology import Topology
from examples.util import charts
from examples.util.panels import image_figure
from examples.electromagnetism.magnetic_materials import core

mv = core.mv
n_pixels = 360
background = np.array([0.035, 0.055, 0.085])
iron_colour = np.array([0.22, 0.29, 0.35])
magnet_colour = np.array([0.30, 0.32, 0.34])
copper_colour = np.array([0.48, 0.29, 0.14])
coil_colour = np.array([0.25, 0.27, 0.29])
positive_current, negative_current = np.array([1.0, 0.30, 0.20]), np.array([0.12, 0.62, 1.0])
line_ink = np.array([0.92, 0.83, 0.55, 0.90])
edge_ink = np.array([0.63, 0.70, 0.75, 0.85])
line_spacing, line_width = 0.055, 1.3
current_scale = 1.5


def field_animation(topology: Topology, vertices: Extensor, fractions: Extensor, coils: Extensor,
                    states: Iterable[tuple[Extensor, Extensor]]) -> Iterator[np.ndarray]:
    """Yields RGB frames showing magnetic field lines and signed axial currents across the magnet, coil and copper."""
    camera = Camera(mv.x, mv.y, 0 * mv.z, mv.w, 0.0, np.zeros(2), np.array([1.67, 1.27]), n_pixels)
    points = vertices.batch()
    magnet, iron, copper = fractions.batch().to_array()
    material = magnet + iron + copper
    winding = np.clip(coils.batch().to_array().sum(axis=0), 0, 1)[:, None]
    base = (
        background
        + magnet[:, None] * (magnet_colour - background)
        + iron[:, None] * (iron_colour - background)
        + copper[:, None] * (copper_colour - background)
    )
    base = base * (1 - winding) + coil_colour * winding
    lines = (contours.Lines(line_spacing, line_spacing / 2, line_ink, line_width),)
    levels = (contours.Level(0.5, edge_ink, 1.2),)
    for potential, current in states:
        signed_current = (mv.z & current).batch().to_array()
        amplitude = np.arcsinh(np.abs(signed_current) / current_scale)
        amplitude = (amplitude / (1 + amplitude))[:, None]
        ink = np.where(signed_current[:, None] >= 0, positive_current, negative_current)
        colours = base * (1 - amplitude) + ink * amplitude
        layers = [
            contours.picture(points, topology, potential.batch().to_array(), colours, camera, lines=lines),
            contours.picture(points, topology, material, np.zeros(3), camera, levels=levels, opacity=0.0),
        ]
        image = drawing.composited(layers)
        rgb = image[..., :3] * image[..., 3:] + background * (1 - image[..., 3:])
        yield np.round(np.clip(rgb, 0, 1) * 255).astype(np.uint8)


@charts.style()
def relaxation_figure(heating: Extensor, step: float, pulse_end: float) -> plt.Figure:
    """Resistive dissipation continues after the coil pulse ends as induced currents relax."""
    power = heating.to_array()
    times = (np.arange(len(power)) + 0.5) * step
    figure, ax = charts.subplots(340, 190, layout="constrained")
    ax.plot(times, power, color="#bd642f")
    ax.axvline(pulse_end, color="#888888", linestyle=":")
    ax.axhline(0, color="#777777", linewidth=charts.FRAME_WIDTH)
    ax.set(xlabel="time", ylabel="Joule heating", xlim=(0, times[-1]), xticks=[0, pulse_end], yticks=[0])
    ax.spines[["top", "right"]].set_visible(False)
    return figure


def setup_figure(topology: Topology, vertices: Extensor, fractions: Extensor, coils: Extensor) -> plt.Figure:
    """Material regions and coil cross-sections frame the domain before solving the magnetic field."""
    potential = 0 * fractions[0]
    current = potential * core.axis
    return image_figure(next(field_animation(topology, vertices, fractions, coils, [(potential, current)])))
