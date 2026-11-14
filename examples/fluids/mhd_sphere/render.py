"""WebGPU sphere shading and matplotlib energy charts display stepped radial current density, uniform-width magnetic field lines, and energy exchange."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import chain

import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure

from examples.fluids.mhd_sphere import core
from examples.util import charts
from numga.render import meshes, solids

surface_colour = np.array(to_rgb("#f6e7c8"))
positive_colour = np.array(to_rgb("#d9480f"))
negative_colour = np.array(to_rgb("#5f3dc4"))
line_colour = np.array(to_rgb("#2b1d0e"))
view_degrees = np.array([24.0, -30.0])
surface_reach = 0.04
margin = 0.04
n_pixels = 300
n_field_lines = 7
line_half_width = 0.012
current_scale = 29.0
n_current_steps = 8
background = np.ones(3)
kinetic_colour = "#1971c2"
magnetic_colour = "#d9480f"
axis_colour = "0.8"

# Stepped vertex current colours the sphere while normalized flux-phase distance draws even-width field lines.
field_line_shader = meshes.Colour("""
fn colour(values: vec4<f32>, face_values: vec4<f32>, corner: vec3<f32>, front: bool) -> vec4<f32> {
    let level = clamp(values.x, -1.0, 1.0);
    let tone = select(negative_colour, positive_colour, level >= 0.0);
    let strength = round(abs(level) * steps) / steps;
    let base = mix(surface_colour, tone, pow(strength, 0.7) * 0.9);
    let wave = abs(fract(values.y + 0.5) - 0.5);
    let distance = wave / max(face_values.x, 1e-5);
    return vec4<f32>(select(base, line_colour, distance < half_width), 1.0);
}
""", dict(surface_colour=surface_colour, positive_colour=positive_colour, negative_colour=negative_colour,
          line_colour=line_colour, half_width=line_half_width, steps=float(n_current_steps)))


# --- plumbing -------------------------------------------------------------------------
def camera_view(points: solids.Point) -> solids.View:
    """An elevated perspective camera frames `points` (`[V] Point-`) inside a square viewport."""
    elevation, azimuth = np.radians(view_degrees[0]), np.radians(view_degrees[1] + 90.0)
    return solids.view([points], surface_reach, elevation, azimuth, margin, n_pixels).squared()


def animation(topology: core.Topology, vertices: core.Point,
              frame_channels: Iterable[tuple[core.Scalar, core.Scalar, core.Scalar]]) -> Iterator[np.ndarray]:
    """WebGPU fragment shading yields RGB frames combining stepped vertex current colour with uniform-width flux contours calibrated to the initial flux range."""
    channel_stream = iter(frame_channels)
    first_frame = next(channel_stream)
    line_spacing = float(np.ptp(first_frame[1].to_array())) / n_field_lines
    points = solids.origin + solids.context.include_from_algebra(vertices.batch())   # [V] Point-
    smooth_normals = meshes.smooth(points, topology)
    frame = camera_view(points)
    for current, flux_function, strength in chain((first_frame,), channel_stream):
        channels = (current / current_scale, flux_function / line_spacing)           # Scalar+[V], Scalar+[V]
        corner_values = np.stack([(topology.incidence[2, 0] * channel).batch().to_array() for channel in channels],
                                 axis=-1).transpose(1, 0, 2)                         # [F, 3, 2]
        face_strength = (strength.to_array() / line_spacing)[:, None]                # [F, 1]
        surface = meshes.samples(points, topology.elements[2], corner_values, smooth_normals, frame,
                                 face_values=face_strength, colour=field_line_shader, gloss=0.0)
        yield solids.draw([], frame, background, inks=[], translucent=[], meshes=[(surface, 1.0)])


@charts.style()
def energy_figure(times: np.ndarray, kinetic: core.Scalar, magnetic: core.Scalar) -> Figure:
    """Normalized kinetic and magnetic energy curves (`[n_frames] Scalar+`) plot fluid-field exchange and resistive-viscous dissipation over `times`."""
    kinetic_values, magnetic_values = kinetic.to_array(), magnetic.to_array()
    total = kinetic_values[0] + magnetic_values[0]
    figure, axis = charts.subplots(300, 200, layout="constrained")
    axis.plot(times, kinetic_values / total, color=kinetic_colour)
    axis.plot(times, magnetic_values / total, color=magnetic_colour)
    axis.set(xlim=(times[0], times[-1]), ylim=(0, 1.05), xticks=[0, times[-1]], yticks=[0, 1], xlabel="time",
             ylabel="energy")
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color(axis_colour)
    return figure
