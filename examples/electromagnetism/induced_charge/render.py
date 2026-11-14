"""Charge density appears in red and blue with an electric-field texture and gold equipotentials."""
from __future__ import annotations

from collections.abc import Iterable, Iterator
import matplotlib.pyplot as plt
import numpy as np
from numga import concatenate, stack
from examples.electromagnetism.induced_charge import core
from examples.util import charts
from examples.util.panels import image_figure
from numga.render import contours, drawing, flow
from numga.render.camera import Camera
from numga.topology import Topology

mv = core.mv
n_pixels = 320
background = np.array([.055, .075, .11])
metal_colour = np.array([.25, .29, .34])
positive_colour = np.array([1., .34, .19])
negative_colour = np.array([.12, .66, 1.])
charge_scale, peak_scale = 12., 80.
line_spacing, line_width = .14, 1.2
line_ink = np.array([.88, .72, .36, .72])
edge_ink = np.array([.55, .61, .68, .55])
field_ink = np.array([.64, .79, .87])
texture_length, texture_grain, texture_speck = 64., 12., .7
texture_opacity, texture_reference, texture_fade = .55, 1., .2

camera = Camera(mv.x, mv.y, 0 * mv.x, mv.w, 0., np.zeros(2), np.full(2, 1.02), n_pixels)
edge = contours.Level(.5, edge_ink, 1.2)


def composited_frame(layers: list[np.ndarray]) -> np.ndarray:
    """Composites RGBA `layers` over the dark background into `[n_pixels, n_pixels, 3]` RGB bytes."""
    image = drawing.composited(layers)
    rgb = image[..., :3] * image[..., 3:] + background * (1 - image[..., 3:])
    return np.round(np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def relaxation_animation(topology: Topology, vertices: core.Point, metal: core.Scalar,
                         states: Iterable[tuple[core.Scalar, core.Scalar, core.Scalar, core.Point]]) -> Iterator[np.ndarray]:
    """Yields `[n_pixels, n_pixels, 3]` RGB frames of charge density, gold equipotentials, and ice-blue electric-field streaks."""
    V, E, F = topology.n_elements
    points = vertices.batch()                                       # [V] Point-
    fraction = np.asarray(metal.batch().to_array())                 # [V]
    corners = topology.incidence[2, 0]                              # [3] Scalar+[F <- V]
    n_corners = topology.n_dim + 1
    face_points = (corners * vertices).batch().reshape(-1)          # [3 * F] Point-
    split_topology = Topology(n_corners * F, np.arange(n_corners * F).reshape(n_corners, F).T)
    base = background + fraction[:, None] * (metal_colour - background)
    lines = contours.Lines(line_spacing, line_spacing / 2, line_ink, line_width)
    for potential, _, density, force in states:
        signed = np.asarray(density.batch().to_array())             # [V]
        strength = np.arcsinh(np.abs(signed) / charge_scale)
        saturation = strength / (1 + strength)
        ink = np.where(signed[:, None] >= 0, positive_colour, negative_colour)
        colours = base * (1 - saturation[:, None]) + ink * saturation[:, None]
        # High densities keep changing in lightness beyond the colour that identifies their sign.
        highlight = signed ** 2 / (signed ** 2 + peak_scale ** 2)
        colours += (1 - colours) * highlight[:, None]
        voltage = np.asarray(potential.batch().to_array())          # [V]
        layers = [contours.picture(points, topology, fraction, colours, camera)]
        # Streak length retains field magnitude on one scale throughout the animation.
        corner_force = concatenate([force.batch()] * n_corners)     # [3 * F] Point-
        field_layer = flow.layer(face_points, split_topology, corner_force, camera, texture_reference, field_ink,
                                 length=texture_length, grain_pixels=texture_grain,
                                 speck_pixels=texture_speck, ink_opacity=texture_opacity,
                                 downstream_fraction=.5)
        # Fade the seed ink with the local field too, so a vanishing field leaves no specks.
        magnitude = force.antinorm()                                # Scalar+[F]
        fade = concatenate([(magnitude / (magnitude + texture_fade)).batch()] * n_corners).to_array()  # [3 * F]
        coverage = drawing.composited([contours.picture(
            face_points, split_topology, fade, np.repeat(fade[:, None], 3, axis=1), camera)])
        field_layer[..., 3] *= coverage[..., 0]
        layers.append(field_layer)
        layers.append(contours.picture(points, topology, voltage, np.zeros(3), camera,
                                       lines=(lines,), opacity=0.))
        layers.append(contours.picture(points, topology, fraction, np.zeros(3), camera,
                                       levels=(edge,), opacity=0.))
        yield composited_frame(layers)


def setup_figure(topology: Topology, vertices: core.Point, metal: core.Scalar) -> plt.Figure:
    """Draws the conducting bodies and their resolved transition into the surrounding insulator."""
    points = vertices.batch()                                       # [V] Point-
    fraction = np.asarray(metal.batch().to_array())                 # [V]
    base = background + fraction[:, None] * (metal_colour - background)
    layers = [contours.picture(points, topology, fraction, base, camera),
              contours.picture(points, topology, fraction, np.zeros(3), camera, levels=(edge,), opacity=0.)]
    return image_figure(composited_frame(layers))


@charts.style()
def balance_figure(net_charge: core.Scalar, negative_charge: core.Scalar, frame_step: float) -> plt.Figure:
    """Plots the conserved net charge and the magnitude of negative charge on each conductor over time."""
    figure, ax = charts.subplots(340, 210, layout="constrained")
    n_frames = net_charge.shape[0]
    times = np.arange(n_frames) * frame_step
    for total, negative_total, colour in zip(net_charge.to_array().T, negative_charge.to_array().T, ("#2b84c5", "#cb593b")):
        ax.plot(times, negative_total, color=colour)
        ax.plot(times, total, color=colour, linestyle="--")
    ax.axhline(0, color="#777777", linewidth=charts.FRAME_WIDTH)
    ax.set(xlabel="time", ylabel="charge", xlim=(0, times[-1]), xticks=[0], yticks=[0])
    ax.spines[["top", "right"]].set_visible(False)
    return figure


@charts.style()
def corner_density_figure(vertices: core.Point, density: core.Scalar, centre: core.Point, turn,
                          half_lengths: np.ndarray, layer_width: float) -> plt.Figure:
    """Plots the peak charge-density magnitude across the rod's boundary layer along its length."""
    rod_axes = turn >> stack((mv.x, mv.y))                          # [2] Upright+
    readings = rod_axes & (vertices - centre)                       # [2] Scalar+[V]
    margin = 2 * layer_width
    selected = (readings[0].abs() - half_lengths[0]).abs() < margin
    selected &= readings[1].abs() < half_lengths[1] + margin
    positions = readings[1].batch()[selected].to_array()
    peak_candidates = density.batch()[selected].abs().to_array()
    n_bins = 35
    edges = np.linspace(-half_lengths[1] - margin, half_lengths[1] + margin, n_bins + 1)
    bins = np.clip(np.digitize(positions, edges) - 1, 0, n_bins - 1)
    peaks = np.zeros(n_bins)
    np.maximum.at(peaks, bins, peak_candidates)
    figure, ax = charts.subplots(340, 210, layout="constrained")
    ax.plot((edges[:-1] + edges[1:]) / 2, peaks, color="#cb593b")
    for end in (-half_lengths[1], half_lengths[1]):
        ax.axvline(end, color="#777777", linestyle=":", linewidth=charts.FRAME_WIDTH)
    ax.axhline(0, color="#777777", linewidth=charts.FRAME_WIDTH)
    ax.set(xlabel="position along rod", ylabel="peak charge density", xticks=[-half_lengths[1], 0, half_lengths[1]])
    ax.spines[["top", "right"]].set_visible(False)
    return figure
