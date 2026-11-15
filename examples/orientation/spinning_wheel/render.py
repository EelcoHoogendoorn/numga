"""Planar GPU rendering draws a spinning wheel and its mirror image side by side through the planes `x` and `y`."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga import concatenate
from numga.render import drawing, simplices
from numga.topology import Topology
from examples.util.animation import capture
from examples.util.panels import offset_panel, overlay

mass_colours = np.array([to_rgb(colour) for colour in ("#df7246", "#238c9b", "#7662a1")])
push_colour, spoke_colour, rim_colour, mirror_colour = (
    to_rgb(colour) for colour in ("#262b32", "#75808a", "#c3ccd4", "#aebfca"))
# Width and height of the rendered figure in pixels.
figure_pixels = (420, 210)
dpi = 100
# Hub distance from the mirror and outer frame margin in rim radii.
separation, margin = 1.3, 0.3
# Rim sampling count, stroke widths, dash lengths, arrow chevron size, and marker radii in pixels.
n_rim = 96
mirror_width, mirror_dashes, rim_width, spoke_width = 1.5, (4.5, 5.5), 2.0, 1.5
arrow_width, arrow_chevron = 2.2, 8.0
hub_radius, mass_radius = 2.5, 4.5
# Arrow reaches and radial handle-push offset in rim radii.
velocity_reach, push_reach, push_offset = 0.45, 0.45, 0.12


# --- plumbing -------------------------------------------------------------------------
def pairs(n_pairs: int) -> Topology:
    """Each of `n_pairs` directed edges connects point `i` to point `i + n_pairs` in a `2 * n_pairs`-vertex topology."""
    return Topology(2 * n_pairs, np.stack([np.arange(n_pairs), n_pairs + np.arange(n_pairs)], axis=-1))


class WheelView:
    """A planar GPU view holds the static mirror line, wheel rims, and hubs while updating spokes, masses, and arrows."""

    def __init__(self, points, rates, masses: np.ndarray) -> None:
        mv = points.context.multivector
        n_wheels, n_masses = points.shape
        self.masses = masses
        # Place the two wheel hubs symmetrically to either side of the mirror x = 0.
        self.hubs = mv.x.dual() * (separation * np.array([-1.0, 1.0]))          # [n_wheels] Point-
        reach = 1 + margin
        self.figure, ax = plt.subplots(figsize=np.array(figure_pixels) / dpi, dpi=dpi)
        self.figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.set(xlim=(-separation - reach, separation + reach), ylim=(-reach, reach), aspect="equal")
        ax.set_axis_off()
        self.camera = offset_panel(ax, mv.x, mv.y)
        # Endpoints of the mirror plane x = 0 spanning the vertical frame.
        mirror = mv.y.dual() * np.array([-reach, reach])                        # [2] Point-
        # Carry the first mass around each wheel's turning plane to trace its rim.
        angles = np.linspace(0, 2 * np.pi, n_rim, endpoint=False)
        rounds = (rates.normalized()[:, None] * (-angles / 2)).exp()            # [n_wheels, n_rim] Rotor+
        rims = self.hubs[:, None] + (rounds >> points[:, :1])                   # [n_wheels, n_rim] Point-
        self.fixed = [simplices.strokes(mirror, Topology.chains([2]), mirror_colour, mirror_width, self.camera,
                                        dashes=mirror_dashes, cap=simplices.Cap.BUTT),
                      simplices.strokes(rims.reshape(-1), Topology.loops([n_rim] * n_wheels), rim_colour,
                                        rim_width, self.camera)]
        self.hub_marks = simplices.markers(self.hubs, spoke_colour, hub_radius, self.camera)
        self.spoke_hubs = self.hubs[np.repeat(np.arange(n_wheels), n_masses)]   # [n_wheels * n_masses] Point-
        self.image = overlay(ax, self.fixed + [self.hub_marks])

    def update(self, places, velocities, pushes) -> plt.Figure:
        """Draws the spokes, masses, velocity arrows, and handle pushes for one state (`[n_wheels, n_masses] Point-`)."""
        n_wheels, n_masses = places.shape
        drawn = (self.hubs[:, None] + places).reshape(-1)                       # [n_wheels * n_masses] Point-
        colours = np.tile(mass_colours[:n_masses], (n_wheels, 1))
        # Scale velocity and push arrows by their peak antinorms.
        tips = drawn + velocities.reshape(-1) * (velocity_reach / velocities.antinorm().max())   # [n_wheels * n_masses] Point-
        push_tails = self.hubs + (1 + push_offset) * places[:, 0]               # [n_wheels] Point-
        push_tips = push_tails + pushes * (push_reach / pushes.antinorm().max())   # [n_wheels] Point-
        self.image.set_data(drawing.composited(self.fixed + [
            simplices.strokes(concatenate([self.spoke_hubs, drawn]), pairs(n_wheels * n_masses), spoke_colour,
                              spoke_width, self.camera),
            self.hub_marks,
            simplices.strokes(concatenate([drawn, tips]), pairs(n_wheels * n_masses), colours, arrow_width,
                              self.camera, chevrons=np.ones(n_wheels * n_masses), chevron=arrow_chevron),
            simplices.strokes(concatenate([push_tails, push_tips]), pairs(n_wheels), push_colour, arrow_width,
                              self.camera, chevrons=np.ones(n_wheels), chevron=arrow_chevron),
            simplices.markers(drawn, colours, np.tile(mass_radius * np.sqrt(self.masses), n_wheels), self.camera)]))
        return self.figure


def draw_wheels(points, rates, masses: np.ndarray, state: tuple) -> plt.Figure:
    """Renders a still figure of the wheel and its mirror image (`[n_wheels, n_masses] Point-`) in one state."""
    return WheelView(points, rates, masses).update(*state)


def animate(points, rates, masses: np.ndarray, states: Iterable[tuple]) -> Iterator[np.ndarray]:
    """Yields RGB frames as the wheel and its mirror image turn through `states`."""
    view = WheelView(points, rates, masses)
    try:
        for state in states:
            yield capture(view.update(*state))
    finally:
        plt.close(view.figure)
