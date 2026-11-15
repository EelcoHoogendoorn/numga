"""GPU rendering draws relativistic particle trails between the two current coils across a meridional
sheet coloured by magnetic field strength."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from examples.electromagnetism.magnetic_bottle import core
from numga import stack
from numga.render import Scene, meshes, simplices, solids
from numga.render.drawing import Draw
from numga.topology import Topology

mv = core.mv
sm = solids.mv
# Intersecting each event's laboratory worldline with the constant-time hyperplane reads its spatial offset.
space_slice = mv.t ^ (core.Antivector & core.time)                            # [] Antivector- <- Antivector-
# A one-third spatial turn cycles z to x, x to y, and y to z so the bottle axis lies horizontally.
scene_turn = (-(sm.xy + sm.yz + sm.zx) * (np.pi / (3 * np.sqrt(3)))).exp()    # [] Rotor+

strength_colours = plt.get_cmap("Blues")
wire_colour = "#f59e0b"
particle_colours = plt.get_cmap("plasma")
n_trail = 60
elevation, azimuth = np.radians([22.0, -28.0])
margin, n_pixels = 0.03, 320
sheet_opacity = 0.55
track_width, wire_width, particle_radius = 1.2, 2.6, 2.2
n_around_coil = 96


# --- plumbing -------------------------------------------------------------------------
def sheet(points: core.Antivector, strength: core.Scalar, view: solids.View) -> Draw:
    """Logarithmic field `strength` (`[...] Scalar+`) colours a triangulated surface over the meridional
    grid `points` (`[...] Antivector-`)."""
    log_strength = strength.log()                                             # [...] Scalar+
    span = log_strength.max() - log_strength.min()                            # [] Scalar+
    normalized_strength = (log_strength - log_strength.min()) / span          # [...] Scalar+
    shades = strength_colours(0.15 + 0.75 * normalized_strength.to_array())[..., :3]
    # Split each grid quad into two triangles indexed row by row.
    indices = np.arange(points.shape[0] * points.shape[1]).reshape(points.shape[:2])
    quads = np.stack([indices[:-1, :-1], indices[:-1, 1:], indices[1:, 1:], indices[1:, :-1]], axis=-1)
    triangles = quads[..., [[0, 1, 2], [0, 2, 3]]].reshape(-1, 3)
    offsets = scene_turn >> solids.context.include_from_algebra(space_slice(points.reshape(-1)))
    places = solids.origin + offsets                                          # [...] Point-
    return meshes.samples(
        places, triangles, shades.reshape(-1, 3)[triangles], meshes.flat(places, triangles), view,
        face_values=np.zeros(4), colour=meshes.plain, gloss=solids.gloss,
    )


def animate_flights(
    pitches: np.ndarray,
    points: core.Antivector,
    strength: core.Scalar,
    radius: float,
    heights: np.ndarray,
    n_steps_per_frame: int,
    states: Iterable[tuple[core.Antivector, core.Antivector]],
) -> Iterator[np.ndarray]:
    """Yields RGB frames of the `n_particles` particles and their trailing tracks across the meridional
    field sheet as `states` streams in."""
    n_coils = len(heights)
    n_particles = len(pitches)
    colours = particle_colours(0.15 + 0.7 * (pitches - pitches.min()) / np.ptp(pitches))[:, :3]
    # Frame a bounding box as deep across the bottle axis as the meridional grid is wide.
    corners = points[[0, -1]][:, [0, -1]]                                     # [2, 2] Antivector-
    offsets = corners - corners.mean(axis=0)                                  # [2, 2] Antivector-
    quarter_turn = (mv.xy * (-np.pi / 4)).exp()                               # [] Rotor+
    depths = quarter_turn >> offsets                                          # [2, 2] Antivector-
    box = corners[..., None] + depths[..., None] * np.array([1, -1])          # [2, 2, 2] Antivector-
    scene_box = solids.origin + (scene_turn >> solids.context.include_from_algebra(space_slice(box)))
    view = solids.view([scene_box], 0.0, elevation, azimuth, margin, n_pixels)
    # Build the static meridional sheet and circular coil tubes once before streaming particle steps.
    coil_points = core.z_axis * heights + core.y_axis * radius                # [n_coils] Antivector-
    rims = solids.origin + (scene_turn >> solids.context.include_from_algebra(space_slice(coil_points)))   # [n_coils] Point-
    bottle_axis = scene_turn >> solids.context.include_from_algebra(mv.xy)    # [] Line+
    rings = meshes.circles(rims, bottle_axis, n_around_coil)                  # [n_coils, n_around_coil] Point-
    wire = simplices.tubes(
        rings.reshape(-1), Topology.loops([n_around_coil] * n_coils),
        to_rgb(wire_colour), wire_width, view, gloss=solids.gloss,
    )
    scene = Scene(
        view, np.ones(3), [], [(sheet(points, strength, view), sheet_opacity), (wire, 1.0)],
        inks=[], translucent=[],
    )
    trail_window: deque = deque(maxlen=n_trail)
    for step_index, (event, _) in enumerate(states, start=1):
        point = solids.origin + (scene_turn >> solids.context.include_from_algebra(space_slice(event)))   # [n_particles] Point-
        trail_window.append(point)
        if step_index % n_steps_per_frame == 0:
            trail = stack(list(trail_window))                                 # [..., n_particles] Point-
            node_indices = np.arange(trail.shape[0] * n_particles).reshape(trail.shape)
            edges = np.stack([node_indices[:-1], node_indices[1:]], axis=-1).reshape(-1, 2)
            chains = Topology(node_indices.size, edges)
            trails = simplices.tubes(
                trail.reshape(-1), chains, np.tile(colours, (len(trail) - 1, 1)), track_width, view,
                gloss=solids.gloss,
            )
            particles = solids.ball(point, particle_radius * view.pixel)[:, None]
            yield scene.draw([(particles, colours)], inks=[], translucent=[], meshes=[(trails, 1.0)])
