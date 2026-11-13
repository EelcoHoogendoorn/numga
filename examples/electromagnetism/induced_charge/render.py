"""Charge density appears in red and blue with a total-field texture and gold equipotentials."""
from __future__ import annotations

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
positive = np.array([1., .34, .19])
negative = np.array([.12, .66, 1.])
charge_scale, peak_scale = 12., 80.
line_spacing, line_width = .14, 1.2
line_ink = np.array([.88, .72, .36, .72])
edge_ink = np.array([.55, .61, .68, .55])
field_ink = np.array([.64, .79, .87])
texture_length, texture_grain, texture_speck = 64., 12., .7
texture_opacity, texture_reference, texture_fade = .55, 1., .2


def animate(topology, vertices, metal, states, *, draw_fields=True):
    """Draw charge density, gold equipotentials and an ice-blue electric-field texture."""
    camera = Camera(mv.x, mv.y, 0 * mv.x, mv.w, 0., np.zeros(2), np.full(2, 1.02), n_pixels)
    points = vertices.batch()
    fraction = np.asarray(metal.batch().to_array())
    corners = topology.incidence[2, 0]
    n_faces = topology.n_elements[2]
    n_corners = topology.n_dim + 1
    face_points = (corners * vertices).batch().reshape(-1)
    split = Topology(n_corners * n_faces, np.arange(n_corners * n_faces).reshape(n_corners, n_faces).T)
    base = background + fraction[:, None] * (metal_colour - background)
    edge = contours.Level(.5, edge_ink, 1.2)
    lines = contours.Lines(line_spacing, line_spacing / 2, line_ink, line_width)
    for potential, charge, density, force in states:
        signed = np.asarray(density.batch().to_array())
        strength = np.arcsinh(np.abs(signed) / charge_scale)
        amount = strength / (1 + strength)
        ink = np.where(signed[:, None] >= 0, positive, negative)
        colours = base * (1 - amount[:, None]) + ink * amount[:, None]
        # High densities keep changing in lightness, beyond the colour that identifies their sign.
        highlight = signed ** 2 / (signed ** 2 + peak_scale ** 2)
        colours += (1 - colours) * highlight[:, None]
        voltage = np.asarray(potential.batch().to_array())
        layers = [contours.picture(points, topology, fraction, colours, camera)]
        if draw_fields:
            # Streak length retains field magnitude on one scale throughout the animation.
            corner_force = concatenate([force.batch()] * n_corners)
            field_layer = flow.layer(face_points, split, corner_force, camera, texture_reference, field_ink,
                                     length=texture_length, grain_pixels=texture_grain,
                                     speck_pixels=texture_speck, ink_opacity=texture_opacity,
                                     downstream_fraction=.5)
            # Fade the seed ink with the local field too, so a vanishing field leaves no specks.
            magnitude = force.antinorm()
            fade = concatenate([(magnitude / (magnitude + texture_fade)).batch()] * n_corners).to_array()
            coverage = drawing.composited([contours.picture(
                face_points, split, fade, np.repeat(fade[:, None], 3, axis=1), camera)])
            field_layer[..., 3] *= coverage[..., 0]
            layers.append(field_layer)
            layers.append(contours.picture(points, topology, voltage, np.zeros(3), camera,
                                           lines=(lines,), opacity=0.))
        layers.append(contours.picture(points, topology, fraction, np.zeros(3), camera,
                                       levels=(edge,), opacity=0.))
        image = drawing.composited(layers)
        rgb = image[..., :3] * image[..., 3:] + background * (1 - image[..., 3:])
        yield np.round(np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def setup(topology, vertices, metal):
    """Draw the conducting material and its transition into insulation."""
    zero = 0 * metal
    return image_figure(next(animate(topology, vertices, metal, [(zero, zero, zero, zero)], draw_fields=False)))


@charts.style()
def balance(net, negative, frame_step):
    """The total charge stays fixed while negative charge gathers on each conductor."""
    figure, ax = charts.subplots(340, 210, layout="constrained")
    n_frames = net.shape[0]
    times = np.arange(n_frames) * frame_step
    for total, amount, colour in zip(net.to_array().T, negative.to_array().T, ("#2b84c5", "#cb593b")):
        ax.plot(times, amount, color=colour)
        ax.plot(times, total, color=colour, linestyle="--")
    ax.axhline(0, color="#777777", linewidth=charts.FRAME_WIDTH)
    ax.set(xlabel="time", ylabel="charge", xlim=(0, times[-1]), xticks=[0], yticks=[0])
    ax.spines[["top", "right"]].set_visible(False)
    return figure


@charts.style()
def corner_density(vertices, density, centre, turn, half_lengths, layer_width):
    """The rod's peak layer density is shown along its length, with its corner positions marked."""
    axes = turn >> stack((mv.x, mv.y))
    readings = axes & (vertices - centre)
    margin = 2 * layer_width
    selected = (readings[0].abs() - half_lengths[0]).abs() < margin
    selected &= readings[1].abs() < half_lengths[1] + margin
    along = readings[1].batch()[selected].to_array()
    peak_candidates = density.batch()[selected].abs().to_array()
    n_bins = 35
    edges = np.linspace(-half_lengths[1] - margin, half_lengths[1] + margin, n_bins + 1)
    bins = np.clip(np.digitize(along, edges) - 1, 0, n_bins - 1)
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
