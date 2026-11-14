"""Drawing for Jeffery orbits: long-axis paths on the sphere of directions and rigid ellipsoids tumbling in shear flow.

The flow runs to the right, its velocity gradient points up, and the normal to the shear plane points out of the picture. Each body is traced on the GPU as the solid ellipsoid bounded by its dual quadric, cut by its three principal planes into eight octants painted light and dark by parity so rigid turning reads cleanly.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import to_rgb

from numga import concatenate, stack
from examples.fluids.jeffery import core
from examples.util import charts
from numga.topology import Topology
from numga.render import Scene, simplices, solids
from examples.util.panels import labels, show

# Start-tilt colours, body octant colours, tip trail colour, flow arrow colour, and axis label colour.
start_colours = cm.viridis(np.linspace(0.1, 0.9, 5))
body_light, body_dark, tip_colour, flow_colour, label_colour = "#7dd3fc", "#0369a1", "#f97316", "#94a3b8", "#475569"
view_angles = (18, 28)
mark_end = 1.15
frame_margin = 0.08
# Octant sign combinations along the three body axes and their alternating colours.
octant_signs = np.array(list(product((1, -1), repeat=3)))                     # [8, 3]
octant_colours = np.where((octant_signs.prod(axis=-1) > 0)[:, None], to_rgb(body_dark), to_rgb(body_light))
tip_rgb = np.array(to_rgb(tip_colour))
path_width, mark_width = 2.0, 3.0
# Unit direction sphere styling, coordinate-frame arrows, and orbit tube width.
sphere_rgb, sphere_opacity = np.array(to_rgb("#cbd5e1")), 0.15
grid_width, grid_opacity = 1.5, 0.3
flow_rgb = np.array(to_rgb(flow_colour))
arrow_length, arrow_width, arrow_chevron, label_distance = 1.35, 3.0, 12.0, 1.47
sphere_arrow_width, sphere_arrow_tip = 1.5, 2.4
orbit_width, orbit_margin, orbit_pixels = 1.5, 0.05, 260

# Rotor cycling core axes into the scene frame so the flow runs right, the gradient points up, and the vortex axis points forward.
scene_turn = ((solids.mv.x - solids.mv.y) * (solids.mv.y - solids.mv.z)).normalized()   # [] Rotor+


def direction_sphere(view: solids.View) -> tuple[list, list, list]:
    """Opaque frame arrows, translucent unit ball, and latitude-longitude grid inks for a direction-sphere panel."""
    arrow_tips = solids.origin + solids.axes.dual() * arrow_length            # [3] Point-
    parts = [(solids.rods(solids.origin, arrow_tips, sphere_arrow_width / 2 * view.pixel), flow_rgb),
             (solids.ball(arrow_tips, sphere_arrow_tip * view.pixel)[:, None], flow_rgb)]
    ball = [(solids.ball(solids.origin, 1.0)[None, None], sphere_rgb, sphere_opacity)]
    # Latitude planes through turned pole points and longitude planes turned around the vertical axis.
    polar_rotors = (solids.mv.zx * (-np.radians(np.arange(30, 180, 30)) / 2)).exp()     # [5] Rotor+
    parallel_points = polar_rotors >> (solids.origin + solids.mv.z.dual())              # [5] Point-
    parallels = solids.mv.z - solids.mv.w * (solids.mv.z & parallel_points)             # [5] Plane+
    meridian_rotors = (solids.mv.xy * (-np.radians(np.arange(0, 180, 30)) / 2)).exp()   # [6] Rotor+
    meridians = meridian_rotors >> solids.mv.x                                          # [6] Plane+
    grid_planes = concatenate([parallels, meridians])                                   # [11] Plane+
    grid = solids.Ink(len(parts), solids.half_space(grid_planes), solids.everywhere[None].broadcast_to(grid_planes.shape),
                      sphere_rgb, grid_width, grid_opacity)
    return parts, ball, [grid]


@charts.style()
def draw_orbits(trails: core.Point) -> plt.Figure:
    """Render both antipodal ends of each body's long-axis trajectory across the translucent sphere of directions."""
    axis_ends = trails * np.array([1, -1])[:, None, None, None]                         # [2, ...] Point-
    scene_offsets = scene_turn >> solids.context.include_from_algebra(axis_ends)        # [2, ...] Point-
    ends = solids.origin + scene_offsets                                                # [2, ...] Point-
    n_ends, n_points, n_bodies, n_starts = ends.shape
    vertex_indices = np.arange(n_ends * n_points * n_starts).reshape(n_ends, n_points, n_starts)
    paths = Topology(vertex_indices.size,
                               np.stack([vertex_indices[:, :-1], vertex_indices[:, 1:]], axis=-1).reshape(-1, 2))
    path_colours = np.broadcast_to(start_colours[:, :3], vertex_indices[:, 1:].shape + (3,)).reshape(-1, 3)
    view = solids.View(solids.aim(solids.origin, np.radians(view_angles[0]), np.radians(view_angles[1] + 90)),
                       np.full(2, label_distance * (1 + orbit_margin)), orbit_pixels)
    figure = plt.figure(figsize=(n_bodies * orbit_pixels / 100, orbit_pixels / 100), dpi=100)
    parts, ball, inks = direction_sphere(view)
    for body_index in range(n_bodies):
        orbits = simplices.tubes(ends[:, :, body_index].reshape(-1), paths, path_colours, orbit_width, view,
                                 gloss=solids.gloss)
        ax = figure.add_axes((body_index / n_bodies, 0, 1 / n_bodies, 1))
        show(ax, solids.draw(parts, view, np.ones(3), inks, translucent=ball, meshes=[(orbits, 1.0)]), view)
        labels(ax, view, solids.origin + solids.axes.dual() * label_distance, ["z", "x", "y"], ha="center",
               va="center", color=label_colour, fontstyle="italic")
    return figure


def octant_solids(stretches: core.Stretch, orientations: core.Rotor) -> solids.Quadric:
    """Solid ellipsoids obtained by polarity inversion of their dual quadrics and clipped into eight body-frame octants."""
    shapes = solids.context.include_from_algebra(core.body_shapes(stretches, orientations))   # [...] Point- <- Plane+
    ellipsoids = solids.placed(solids.solid_of_dual_quadric(shapes), scene_turn)              # [...] Plane+ <- Point-
    body_planes = orientations[..., None] >> core.axes                                        # [..., 3] Plane+
    normals = scene_turn >> solids.context.include_from_algebra(body_planes)                  # [..., 3] Plane+
    sides = solids.half_space(normals[..., None, :] * -octant_signs)                          # [..., 8, 3] Plane+ <- Point-
    return concatenate([ellipsoids[..., None, None].broadcast_to(sides.shape[:-1] + (1,)), sides], axis=-1)


def animate_tumbling(stretches: core.Stretch, heights: core.Point, velocities: core.Point, n_pixels: int,
                     states: Iterable[tuple[core.Rotor, core.Point]]) -> Iterator[np.ndarray]:
    """Yield RGB frames of the ellipsoids tumbling side by side in the shear flow with accumulating tip trails."""
    orientations, tips = (stack(list(part)) for part in zip(*states))                   # [..., ...] Rotor+, Point-
    n_frames, n_bodies = tips.shape
    radii = tips.norm().reshape(-1)                                                     # [...] Scalar+
    limit = solids.context.include_from_algebra(radii.max()) * (mark_end * (1 + frame_margin))
    camera = solids.aim(solids.origin, np.radians(view_angles[0]), np.radians(view_angles[1] + 90))
    view = solids.View(camera, (limit * np.array([n_bodies, 1.0])).to_array(), n_bodies * n_pixels)
    panel_offsets = (camera >> solids.mv.x.dual()) * (2 * limit * (np.arange(n_bodies) - (n_bodies - 1) / 2))   # [n_bodies] Point-
    panel_motors = solids.translator(panel_offsets)                                     # [n_bodies] Motor+
    panel_bodies = solids.placed(octant_solids(stretches, orientations), panel_motors[:, None, None])
    colours = np.tile(octant_colours, (n_bodies, 1))
    # Ambient shear velocity arrows centred at each sampled height behind every body.
    arrow_starts, arrow_ends = (
        panel_motors[:, None] >> (solids.origin + (scene_turn >> solids.context.include_from_algebra(offsets)))
        for offsets in (heights - velocities / 2, heights + velocities / 2)
    )                                                                                   # [n_bodies, ...] Point-, Point-
    n_arrows = int(np.prod(arrow_starts.shape))
    flow = simplices.tubes(stack((arrow_starts, arrow_ends), axis=-1).reshape(-1),
                           Topology.chains([2] * n_arrows), flow_rgb,
                           arrow_width, view, gloss=solids.gloss, chevrons=np.ones(n_arrows), chevron=arrow_chevron)
    tip_ends = tips * np.array([1, -1])[:, None, None]                                  # [2, n_frames, n_bodies] Point-
    tip_offsets = scene_turn >> solids.context.include_from_algebra(tip_ends)           # [2, n_frames, n_bodies] Point-
    axis_ends = panel_motors >> (solids.origin + tip_offsets)                           # [2, n_frames, n_bodies] Point-
    mark_ends = panel_motors >> (solids.origin + tip_offsets * mark_end)                # [2, n_frames, n_bodies] Point-
    vertex_indices = np.arange(np.prod(axis_ends.shape)).reshape(axis_ends.shape)
    edge_pairs = np.stack([vertex_indices[:, :-1], vertex_indices[:, 1:]], axis=-1)
    mark_chains = Topology.chains([2] * (2 * n_bodies))
    scene = Scene(view, np.ones(3), [], [(flow, 1.0)], inks=[], translucent=[])
    for frame_index in range(n_frames):
        paths = simplices.tubes(axis_ends.reshape(-1),
                                Topology(vertex_indices.size, edge_pairs[:, :frame_index].reshape(-1, 2)),
                                tip_rgb, path_width, view, gloss=solids.gloss)
        marks = simplices.tubes(stack([axis_ends[:, frame_index], mark_ends[:, frame_index]], axis=-1).reshape(-1),
                                mark_chains, tip_rgb, mark_width, view, gloss=solids.gloss)
        yield scene.draw([(panel_bodies[frame_index].reshape(-1, 4), colours)], inks=[], translucent=[],
                         meshes=[(paths, 1.0), (marks, 1.0)])

