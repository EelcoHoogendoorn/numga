"""Drawing for Laplacian eigenfunction fluids on the Stanford bunny: GPU-shaded Laplace–Beltrami streamfunction
modes and evolving vorticity fields with uniform-width streamline contours normalized by the face velocity norm,
alongside kinetic energy and enstrophy conservation curves."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure

from numga import stack
from examples.fluids.eigen_fluids import core
from examples.util.wgsl import image_figure, meshes, solids

# Surface and diverging field palette, contour colour, curve colours, camera elevation and azimuth in degrees
# from x, viewport margin and pixel dimensions, contour density and width, and plot typography.
surface_colour = np.array([0.94, 0.92, 0.88])
positive_colour = np.array(to_rgb("#c84b31"))
negative_colour = np.array(to_rgb("#2d6a9f"))
contour_colour = np.array([0.12, 0.18, 0.26])
ink_colour = "#263b51"
energy_colour = "#148d88"
enstrophy_colour = "#c84b31"
view_degrees = np.array([18.0, -42.0])
margin = 0.04
n_mode_pixels = 380
n_flow_pixels = 440
n_mode_contours = 5.5
n_flow_contours = 7.5
contour_half_width = 0.011
contour_feather = 0.004
vorticity_exposure = 95.5
stream_exposure = 98.5
background = np.ones(3)
font_size, line_width = 7, 1.1

# WGSL fragment colouring that shades the surface by the first interpolated vertex channel (`values.x`,
# vorticity or mode amplitude) and draws uniform-width streamline contours by dividing the contour phase
# residual of the second vertex channel (`values.y`, streamfunction) by the face speed (`face_values.x`).
streamline_shader = meshes.Colour("""
fn colour(values: vec4<f32>, face_values: vec4<f32>, corner: vec3<f32>, front: bool) -> vec4<f32> {
    let level = clamp(values.x, -1.0, 1.0);
    let tone = select(negative_colour, positive_colour, level >= 0.0);
    let weight = pow(abs(level), 0.72) * 0.88;
    let base = mix(surface_colour, tone, weight);
    let wave = abs(fract(values.y + 0.5) - 0.5);
    let dist = wave / max(face_values.x, 1e-5);
    let line = smoothstep(half_width + feather, half_width - feather, dist);
    let shaded = mix(base, contour_colour, 0.54 * line);
    return vec4<f32>(shaded, 1.0);
}
""", dict(surface_colour=surface_colour, positive_colour=positive_colour, negative_colour=negative_colour,
          contour_colour=contour_colour, half_width=contour_half_width, feather=contour_feather))


# --- plumbing -------------------------------------------------------------------------
def coordinates(elements: core.Point) -> np.ndarray:
    """The Cartesian coordinates of points, antivectors, read by their pairings with the coordinate planes."""
    mv = core.mv
    return (stack((mv.x, mv.y, mv.z)) & elements.batch()[..., None]).to_array()


def plain_axes(axis: plt.Axes) -> None:
    """A quantity plot's labels and ticks in the plots' text size, with faint left and bottom spines only."""
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("0.8")
    axis.tick_params(color="0.8", labelsize=font_size)
    axis.xaxis.label.set_fontsize(font_size)
    axis.yaxis.label.set_fontsize(font_size)


def camera_view(points: solids.Point, n_pixels: int) -> solids.View:
    """A square camera view framing the surface vertices at `n_pixels` across."""
    elevation, azimuth = np.radians(view_degrees[0]), np.radians(view_degrees[1] + 90.0)
    return solids.view([points], 0.04, elevation, azimuth, margin, n_pixels).squared()


def surface_panel(mesh: core.Mesh, points: solids.Point, smooth_normals: dict, frame: solids.View,
                field: core.Scalar, stream: core.Scalar, velocity: core.Point,
                field_scale: float, stream_scale: float, n_contours: float) -> np.ndarray:
    """One GPU-rendered panel of the surface coloured by `field` with uniform-width streamline contours of `stream`."""
    scale = max(stream_scale, 1e-12)
    scaled_field = field.to_array() / max(field_scale, 1e-12)                    # [vertices]
    scaled_stream = (stream.to_array() / scale) * n_contours                     # [vertices]
    face_speed = ((velocity.norm().to_array() / scale) * n_contours)[:, None]    # [faces, 1]
    corner_values = np.stack([scaled_field, scaled_stream], axis=-1)[mesh.faces] # [faces, 3, 2]
    surface = meshes.samples(points, mesh.faces, corner_values, smooth_normals, frame,
                             face_values=face_speed, colour=streamline_shader, gloss=solids.gloss)
    return solids.draw([], frame, background, inks=[], translucent=[], meshes=[(surface, 1.0)])


def draw_modes(mesh: core.Mesh, vertices: core.Point, modes: core.Scalar,
               mode_velocities: core.Point) -> Figure:
    """The Stanford bunny side by side for each selected Laplace–Beltrami streamfunction mode and its streamlines."""
    points = solids.mv.point(coordinates(vertices))
    smooth_normals = meshes.smooth(points, mesh.faces)
    frame = camera_view(points, n_mode_pixels)
    panels = []
    for mode, velocity in zip(modes, mode_velocities):
        values = np.abs(mode.to_array())
        field_scale = float(np.percentile(values, vorticity_exposure))
        stream_scale = float(np.percentile(values, stream_exposure))
        panels.append(surface_panel(mesh, points, smooth_normals, frame, mode, mode, velocity,
                                  field_scale, stream_scale, n_mode_contours))
    return image_figure(np.concatenate(panels, axis=1))


def draw_conservation(times: np.ndarray, inviscid_energy: core.Scalar, viscous_energy: core.Scalar,
                      inviscid_enstrophy: core.Scalar, viscous_enstrophy: core.Scalar,
                      initial_vorticity: core.Scalar, inviscid_vorticity: core.Scalar, viscous_vorticity: core.Scalar) -> Figure:
    """Normalized kinetic energy and enstrophy over time for inviscid and viscous flow, beside the cumulative
    modal enstrophy distribution across the Laplace–Beltrami spectrum."""
    energy_inv = inviscid_energy.to_array()
    energy_vis = viscous_energy.to_array()
    enstrophy_inv = inviscid_enstrophy.to_array()
    enstrophy_vis = viscous_enstrophy.to_array()
    initial_modes = initial_vorticity.squared().to_array() / (2 * enstrophy_inv[0])
    final_inv_modes = inviscid_vorticity.squared().to_array() / (2 * enstrophy_inv[0])
    final_vis_modes = viscous_vorticity.squared().to_array() / (2 * enstrophy_inv[0])
    mode_indices = np.arange(1, len(initial_modes) + 1)

    figure, (history, spectrum) = plt.subplots(1, 2, figsize=(4.8, 1.95), layout="constrained")
    history.plot(times, energy_inv / energy_inv[0], color=ink_colour, linewidth=line_width)
    history.plot(times, enstrophy_inv / enstrophy_inv[0], color=ink_colour, linestyle="--", linewidth=0.9)
    history.plot(times, energy_vis / energy_inv[0], color=energy_colour, linewidth=line_width)
    history.plot(times, enstrophy_vis / enstrophy_inv[0], color=enstrophy_colour, linewidth=line_width)
    history.set(xlim=(times[0], times[-1]), ylim=(0, 1.08), xticks=[0, times[-1]], yticks=[0, 1],
                xlabel="time", ylabel=r"relative $E(t),\, Z(t)$")
    plain_axes(history)

    spectrum.plot(mode_indices, np.cumsum(initial_modes), color="0.65", linestyle="--", linewidth=0.9)
    spectrum.plot(mode_indices, np.cumsum(final_inv_modes), color=ink_colour, linewidth=line_width)
    spectrum.plot(mode_indices, np.cumsum(final_vis_modes), color=enstrophy_colour, linewidth=line_width)
    spectrum.set(xlim=(1, mode_indices[-1]), ylim=(0, 1.08), xticks=[1, mode_indices[-1]], yticks=[0, 1],
                 xlabel="mode index", ylabel="cumulative enstrophy")
    plain_axes(spectrum)
    return figure


def animate_flow(mesh: core.Mesh, vertices: core.Point,
                 states: Iterable[tuple[core.Scalar, core.Scalar, core.Point]]) -> Iterator[np.ndarray]:
    """Draw each arriving surface state with colour and contour scales fixed by its first frame."""
    states = iter(states)
    first = next(states)
    field_scale = float(np.percentile(np.abs(first[0].to_array()), vorticity_exposure))
    stream_scale = float(np.percentile(np.abs(first[1].to_array()), stream_exposure))
    points = solids.mv.point(coordinates(vertices))
    smooth_normals = meshes.smooth(points, mesh.faces)
    frame = camera_view(points, n_flow_pixels)
    for vorticity, stream, velocity in chain((first,), states):
        yield surface_panel(mesh, points, smooth_normals, frame, vorticity, stream, velocity,
                          field_scale, stream_scale, n_flow_contours)
