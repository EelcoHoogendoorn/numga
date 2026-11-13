"""Magnetic field lines cross the magnet, iron, conductor and surrounding space."""
from __future__ import annotations

import numpy as np
from examples.util import charts
from numga.render import Camera, contours, drawing
from examples.util.panels import image_figure
from examples.electromagnetism.magnetic_materials import core

mv = core.mv
n_pixels = 360
background = np.array([.035, .055, .085])
iron_colour = np.array([.22, .29, .35])
magnet_colour = np.array([.30, .32, .34])
copper_colour = np.array([.48, .29, .14])
coil_colour = np.array([.25, .27, .29])
positive_current, negative_current = np.array([1., .30, .20]), np.array([.12, .62, 1.])
line_ink = np.array([.92, .83, .55, .90])
edge_ink = np.array([.63, .70, .75, .85])
line_spacing, line_width = .055, 1.3
current_scale = 1.5


def animate(topology, vertices, fractions, coils, states):
    """Draw every current along z, the magnet's bound current, the coil's and the conductor's, on one red/blue scale."""
    camera = Camera(mv.x, mv.y, 0 * mv.z, mv.w, 0., np.zeros(2), np.array([1.67, 1.27]), n_pixels)
    points = vertices.batch()
    magnet, iron, copper = fractions.batch().to_array()
    material = magnet + iron + copper
    winding = np.clip(coils.batch().to_array().sum(axis=0), 0, 1)[:, None]
    base = (background + magnet[:, None] * (magnet_colour - background)
            + iron[:, None] * (iron_colour - background) + copper[:, None] * (copper_colour - background))
    base = base * (1 - winding) + coil_colour * winding
    lines = (contours.Lines(line_spacing, line_spacing / 2, line_ink, line_width),)
    levels = (contours.Level(.5, edge_ink, 1.2),)
    for potential, current in states:
        signed = (mv.zt & current).batch().to_array()
        amplitude = np.arcsinh(np.abs(signed) / current_scale)
        amplitude = (amplitude / (1 + amplitude))[:, None]
        ink = np.where(signed[:, None] >= 0, positive_current, negative_current)
        colours = base * (1 - amplitude) + ink * amplitude
        layers = [contours.picture(points, topology, potential.batch().to_array(), colours, camera, lines=lines),
                  contours.picture(points, topology, material, np.zeros(3), camera, levels=levels, opacity=0.)]
        image = drawing.composited(layers)
        rgb = image[..., :3] * image[..., 3:] + background * (1 - image[..., 3:])
        yield np.round(np.clip(rgb, 0, 1) * 255).astype(np.uint8)


@charts.style()
def relaxation(history, mass, step, pulse_end):
    """Joule heating continues as the conductor relaxes after the coil pulse."""
    rate = (history[1:] - history[:-1]) / step
    rate = rate - (mass * rate).sites.sum() / mass.sites.sum()
    power = (mass * rate.squared()).sites.sum().to_array()
    times = (np.arange(len(power)) + .5) * step
    figure, ax = charts.subplots(340, 190, layout="constrained")
    ax.plot(times, power, color="#bd642f")
    ax.axvline(pulse_end, color="#888888", linestyle=":")
    ax.axhline(0, color="#777777", linewidth=charts.FRAME_WIDTH)
    ax.set(xlabel="time", ylabel="Joule heating", xlim=(0, times[-1]), xticks=[0, pulse_end], yticks=[0])
    ax.spines[["top", "right"]].set_visible(False)
    return figure


def setup(topology, vertices, fractions, coils):
    """Show the material regions before drawing their common field."""
    potential = 0 * fractions[0]
    current = potential * mv.zt.dual()
    return image_figure(next(animate(topology, vertices, fractions, coils, [(potential, current)])))
