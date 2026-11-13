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
magnet_colour = np.array([.30, .32, .34])
copper_colour = np.array([.48, .29, .14])
coil_colour = np.array([.25, .27, .29])
positive_pole, negative_pole = np.array([1., .30, .20]), np.array([.12, .62, 1.])
positive_current, negative_current = np.array([.48, 1., .34]), np.array([.88, .40, 1.])
line_ink = np.array([.92, .83, .55, .90])
edge_ink = np.array([.63, .70, .75, .85])
line_spacing, line_width = .055, 1.3
pole_scale, pole_peak_scale, current_scale = 1., 30., .2


def animate(topology, vertices, fractions, coils, peak_drive, states):
    """Draw magnetic pole density in red/blue and electric current in green/violet."""
    camera = Camera(mv.x, mv.y, 0 * mv.z, mv.w, 0., np.zeros(2), np.array([1.67, 1.27]), n_pixels)
    points = vertices.batch()
    magnet, iron, copper = fractions.batch().to_array()
    material = magnet + iron + copper
    coil_profiles = coils.batch().to_array()
    lines = (contours.Lines(line_spacing, line_spacing / 2, line_ink, line_width),)
    levels = (contours.Level(.5, edge_ink, 1.2),)
    for potential, pole_density, current, drive in states:
        signed = (mv.zt & current).batch().to_array()
        amplitude = np.arcsinh(np.abs(signed) / current_scale)
        amplitude /= 1 + amplitude
        current_ink = np.where(signed[:, None] >= 0, positive_current, negative_current)
        copper_ink = copper_colour * (1 - amplitude[:, None]) + current_ink * amplitude[:, None]
        colours = (background + magnet[:, None] * (magnet_colour - background)
                   + iron[:, None] * (iron_colour - background) + copper[:, None] * (copper_ink - background))
        # One signed pole-density scale applies to the permanent magnet and induced magnetization.
        poles = pole_density.batch().to_array()
        level = np.arcsinh(np.abs(poles) / pole_scale)
        level /= 1 + level
        pole_ink = np.where(poles[:, None] >= 0, positive_pole, negative_pole)
        highlight = poles ** 2 / (poles ** 2 + pole_peak_scale ** 2)
        pole_ink += (1 - pole_ink) * highlight[:, None]
        colours = colours * (1 - level[:, None]) + pole_ink * level[:, None]
        coil_strength = abs(drive / peak_drive)
        coil_ink = np.stack((positive_current, negative_current)) if drive >= 0 else np.stack((negative_current, positive_current))
        for profile, ink in zip(coil_profiles, coil_ink):
            ink = coil_colour + coil_strength * (ink - coil_colour)
            opacity = profile[:, None]
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
    pole_density = 0 * potential
    current = potential * mv.zt.dual()
    return image_figure(next(animate(topology, vertices, fractions, coils, peak_drive,
                                    [(potential, pole_density, current, 0.)])))
