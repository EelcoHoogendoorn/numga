"""Drawing for magnetohydrodynamics on a sphere: the current density on a lit sphere, in flat steps, with the field
lines, the flux function's contours, drawn at an even width across it."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import chain

import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure

from examples.fluids.mhd_sphere import core
from examples.util import charts
from examples.util.wgsl import meshes, solids

# The surface's colour where no current flows, the colours of the two senses of current, and the field lines' colour.
surface_colour = np.array(to_rgb("#f6e7c8"))
positive_colour = np.array(to_rgb("#d9480f"))
negative_colour = np.array(to_rgb("#5f3dc4"))
line_colour = np.array(to_rgb("#2b1d0e"))
# The camera's elevation and azimuth in degrees, the reach framed around each vertex, the margin and the pixels across.
view_degrees = np.array([24.0, -30.0])
surface_reach = 0.04
margin = 0.04
n_pixels = 300
# How many field lines span the first frame's range of the flux function, and their half width; the current that
# saturates the colour, the brightest percent of the winding run's current sheets, and the steps the current's strength
# is shown in, each way.
n_field_lines = 7.0
line_half_width = 0.012
current_scale = 29.0
n_current_steps = 8.0
background = np.ones(3)
# The colours of the kinetic and magnetic energy curves, and of the axes.
kinetic_colour = "#1971c2"
magnetic_colour = "#d9480f"
axis_colour = "0.8"

# WGSL fragment colouring: the surface coloured by the first vertex channel (`values.x`, the current), in flat steps of
# its strength, so that weak currents leave the surface plain, and field lines of even width from the phase of the
# second (`values.y`, the flux function) divided by the face's field strength (`face_values.x`), their edges smoothed
# by the samples of each pixel alone.
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
          line_colour=line_colour, half_width=line_half_width, steps=n_current_steps))


# --- plumbing -------------------------------------------------------------------------
def camera_view(points: solids.Point) -> solids.View:
    """A square camera view framing the sphere from above its equator."""
    elevation, azimuth = np.radians(view_degrees[0]), np.radians(view_degrees[1] + 90.0)
    return solids.view([points], surface_reach, elevation, azimuth, margin, n_pixels).squared()


def animate(topology: core.Topology, vertices: core.Point,
            frames: Iterable[tuple[core.Scalar, core.Scalar, core.Scalar]]) -> Iterator[np.ndarray]:
    """The sphere coloured by its current, with its field lines, at every frame as it comes, the field lines spaced by
    the first frame's range of the flux function."""
    frames = iter(frames)
    first = next(frames)
    line_spacing = float(np.ptp(first[1].to_array())) / n_field_lines
    points = solids.origin + solids.context.include_from_algebra(vertices.batch())   # [vertices] Point-
    smooth_normals = meshes.smooth(points, topology)
    frame = camera_view(points)
    for current, flux_function, strength in chain((first,), frames):
        channels = (current / current_scale, flux_function / line_spacing)       # Scalar+[vertices], Scalar+[vertices]
        # Each channel at the corners of each face, faces first, for the shader.
        corner_values = np.stack([(topology.incidence[2, 0] * channel).batch().to_array() for channel in channels],
                                 axis=-1).transpose(1, 0, 2)                      # [faces, 3, 2]
        face_strength = (strength.to_array() / line_spacing)[:, None]            # [faces, 1]
        surface = meshes.samples(points, topology.elements[2], corner_values, smooth_normals, frame,
                                 face_values=face_strength, colour=field_line_shader, gloss=0.0)
        yield solids.draw([], frame, background, inks=[], translucent=[], meshes=[(surface, 1.0)])


@charts.style()
def draw_energies(times: np.ndarray, kinetic: core.Scalar, magnetic: core.Scalar) -> Figure:
    """The kinetic and magnetic energy over time, as fractions of the starting total: the flow's energy wound into the
    field, traded back and forth, and dissipated."""
    # Each energy is an amount, read as a number by its pairing with the unit scalar.
    unit = core.mv.scalar([1.0])
    kinetic, magnetic = (kinetic & unit).to_array(), (magnetic & unit).to_array()
    total = kinetic[0] + magnetic[0]
    figure, axis = charts.subplots(300, 200, layout="constrained")
    axis.plot(times, kinetic / total, color=kinetic_colour, label="kinetic")
    axis.plot(times, magnetic / total, color=magnetic_colour, label="magnetic")
    axis.set(xlim=(times[0], times[-1]), ylim=(0, 1.05), xticks=[0, times[-1]], yticks=[0, 1], xlabel="time",
             ylabel="energy")
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color(axis_colour)
    axis.legend(frameon=False)
    return figure
