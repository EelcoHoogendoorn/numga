"""GPU rendering draws subdivided surfaces alongside their dashed control nets and vertex frame axes."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np

from examples.surfaces.motor_loop import core
from numga.render import meshes, solids

surface_colour = np.array([0.82, 0.85, 0.9])
axis_colours = np.array([[0.85, 0.15, 0.15], [0.2, 0.65, 0.2], [0.15, 0.3, 0.85]])
control_colour = np.full(3, 0.15)
background = np.ones(3)
# Spherical camera angles in radians (zero azimuth looks along +y), fractional margin, and horizontal pixel width.
elevation, azimuth = np.radians(12.0), np.radians(30.0)
margin, n_pixels = 0.2, 320
# Dash count along each control-net edge, rod widths, and tip-ball radius in pixels.
n_dashes = 4
net_width, axis_width, tip_radius = 2.4, 3.0, 4.0


def surface_panel(frames: core.Point, surface: core.Point, control_topology: core.Topology,
                  refined_topology: core.Topology, view: solids.View) -> np.ndarray:
    """Renders a smooth-shaded surface (`Point-[V_refined]`) over its dashed control net and `[4] Point-[V]` frame arrows."""
    V, E, F = control_topology.n_elements
    pixel = view.pixel
    faces = refined_topology.elements[2]
    sheet = meshes.samples(surface, faces, surface_colour, meshes.smooth(surface, refined_topology), view,
                           face_values=np.zeros(4), colour=meshes.plain, gloss=solids.gloss)
    origins, tips = frames[0], frames[1:].batch()                           # Point-[V], [3, V] Point-
    # Divide each control edge into 2 * n_dashes - 1 equal segments so alternating segments form n_dashes dashes.
    starts, ends = (control_topology.incidence[1, 0] * origins).batch()     # [E] Point-, [E] Point-
    shares = np.linspace(0, 1, 2 * n_dashes).reshape(n_dashes, 2)           # [n_dashes, 2]
    dashes = starts[:, None, None] * (1 - shares) + ends[:, None, None] * shares  # [E, n_dashes, 2] Point-
    shafts = solids.rods(origins.batch()[None], tips, axis_width / 2 * pixel)  # [3, V, 3] Plane+ <- Point-
    balls = solids.ball(tips.reshape(-1), tip_radius * pixel)[:, None]      # [3 * V, 1] Plane+ <- Point-
    colours = np.repeat(axis_colours, V, axis=0)                            # [3 * V, 3]
    marks = [(solids.rods(dashes[..., 0], dashes[..., 1], net_width / 2 * pixel).reshape(-1, 3), control_colour),
             (shafts.reshape(-1, 3), colours), (balls, colours)]
    return solids.draw(marks, view, background, inks=[], translucent=[], meshes=[(sheet, 1.0)])


def ribbon_view(bounds: core.Point) -> solids.View:
    """Orthographic camera bounds framing `bounds` keep the viewport fixed across streamed frames."""
    return solids.view([bounds.batch()], 0.0, elevation, azimuth, margin, n_pixels)


def surface_animation(control_topology: core.Topology, refined_topology: core.Topology, view: solids.View,
                      states: Iterable[tuple[core.Point, core.Point]]) -> Iterator[np.ndarray]:
    """Yields side-by-side RGB panels comparing point-subdivided and motor-subdivided surfaces for each `(frames, surfaces)` state."""
    for frames, surfaces in states:
        yield np.concatenate([surface_panel(frames, surface, control_topology, refined_topology, view)
                              for surface in surfaces.batch()], axis=1)
