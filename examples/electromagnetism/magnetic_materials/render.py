"""Magnetic field lines cross the magnet, iron, conductor and surrounding space."""
from __future__ import annotations

import numpy as np
from examples.util import charts
from examples.util.wgsl import contours, image_figure, simplices
from examples.util.wgsl.camera import Camera
from examples.electromagnetism.magnetic_materials import core

mv = core.mv
n_pixels = 360
background = np.array([.035, .055, .085])
iron_colour = np.array([.22, .29, .35])
magnetized_iron = np.array([.67, .74, .72])
copper_colour = np.array([.48, .29, .14])
north_colour, south_colour = np.array([.88, .27, .18]), np.array([.12, .43, .73])
positive_current, negative_current = np.array([1., .52, .20]), np.array([.18, .78, 1.])
line_ink = np.array([.92, .83, .55, .90])
edge_ink = np.array([.63, .70, .75, .85])
line_spacing, line_width = .035, 1.3
magnetization_scale, current_scale = .12, .2


def animate(topology, vertices, fractions, coils, peak_drive, states):
    """Draw one magnetic cross-section, with induced magnetization and current."""
    camera = Camera(mv.x, mv.y, 0 * mv.z, mv.w, 0., np.zeros(2), np.array([1.67, 1.27]), n_pixels)
    points = vertices.batch()
    magnet, iron, copper = fractions.batch().to_array()
    poles = (mv.y & vertices).batch().to_array()
    magnet_colour = np.where(poles[:, None] > 0, north_colour, south_colour)
    material = magnet + iron + copper
    coil_profiles = coils.batch().to_array()
    mean = topology.incidence[2, 0].mean(axis=0)
    lines = (contours.Lines(line_spacing, line_spacing / 2, line_ink, line_width),)
    levels = (contours.Level(.5, edge_ink, 1.2),)
    for potential, magnetization, current, drive in states:
        strength = magnetization.antinorm()
        nodal = (~mean * strength) / (~mean * (0 * strength + 1))
        level = np.arcsinh(nodal.batch().to_array() / magnetization_scale)
        level /= 1 + level
        iron_ink = iron_colour + level[:, None] * (magnetized_iron - iron_colour)
        signed = (mv.zt & current).batch().to_array()
        amplitude = np.arcsinh(np.abs(signed) / current_scale)
        amplitude /= 1 + amplitude
        current_ink = np.where(signed[:, None] >= 0, positive_current, negative_current)
        copper_ink = copper_colour * (1 - amplitude[:, None]) + current_ink * amplitude[:, None]
        colours = (background + magnet[:, None] * (magnet_colour - background)
                   + iron[:, None] * (iron_ink - background) + copper[:, None] * (copper_ink - background))
        coil_strength = .25 + .75 * abs(drive / peak_drive)
        coil_ink = np.stack((positive_current, negative_current)) if drive >= 0 else np.stack((negative_current, positive_current))
        for profile, ink in zip(coil_profiles, coil_ink):
            opacity = profile[:, None] * coil_strength
            colours = colours * (1 - opacity) + ink * opacity
        layers = [contours.picture(points, topology, potential.batch().to_array(), colours, camera, lines=lines),
                  contours.picture(points, topology, material, np.zeros(3), camera, levels=levels, opacity=0.)]
        image = simplices.composited(layers)
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


def setup(topology, vertices, fractions, coils, peak_drive):
    """Show the material regions before drawing their common field."""
    potential = 0 * fractions[0]
    magnetization = (0 * mv.xz.dual()).broadcast_to((topology.n_elements[2],)).field()
    current = potential * mv.zt.dual()
    return image_figure(next(animate(topology, vertices, fractions, coils, peak_drive,
                                    [(potential, magnetization, current, 0.)])))
