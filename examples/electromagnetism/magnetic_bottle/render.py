"""Drawing for the magnetic bottle, on the GPU: the particles' tracks in space, trailing behind them, through a sheet
of the field's strength in the plane through the axis, between the coils' wire."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga import stack
from examples.electromagnetism.magnetic_bottle import core
from examples.util.wgsl import meshes, solids, strokes
from examples.util.wgsl.scene import Scene

mv = core.mv
# The colours of the field's strength, the coils' wire, and the particles by their angle to the axis.
strength_colours = plt.get_cmap("Blues")
wire_colour = "#f59e0b"
particle_colours = plt.get_cmap("plasma")
# The steps of each track drawn behind its particle.
n_trail = 60
# The view, elevation and azimuth in radians, azimuth zero looking along +y, with the bottle's axis along x; its margin
# as a share of its larger side and its pixels across; how much of what lies behind it the sheet of the field's
# strength hides; in pixels, the widths of the tracks and of the wire and the radius of the particles; and the points
# around a coil.
elevation, azimuth = np.radians([22.0, -28.0])
margin, n_pixels = 0.03, 320
sheet_opacity = 0.55
track_width, wire_width, particle_radius = 1.2, 2.6, 2.2
n_around_coil = 96


# --- plumbing -------------------------------------------------------------------------
def space(points: core.Antivector) -> np.ndarray:
    """Points in the view's space, on the last axis: along the bottle's axis, then across it twice, turned on from z,
    x and y, so that the tracks keep the sense they turn in."""
    return (stack((mv.z, mv.x, mv.y)) & points[..., None]).to_array()


def sheet(points: core.Antivector, strength: core.Scalar, view: solids.View) -> meshes.Samples:
    """The field's strength over its grid in the plane through the axis `[across, along]`, coloured by its logarithm,
    as a mesh of the grid's squares, as `meshes.samples`."""
    grid = space(points)                                                      # [across, along, 3]
    logarithm = np.log(strength.to_array())
    shades = strength_colours(0.15 + 0.75 * (logarithm - logarithm.min()) / np.ptp(logarithm))[..., :3]
    index = np.arange(grid[..., 0].size).reshape(grid.shape[:2])
    quads = np.stack([index[:-1, :-1], index[:-1, 1:], index[1:, 1:], index[1:, :-1]], axis=-1)
    triangles = quads[..., [[0, 1, 2], [0, 2, 3]]].reshape(-1, 3)
    places = solids.mv.point(grid.reshape(-1, 3))
    return meshes.samples(places, triangles, shades.reshape(-1, 3)[triangles], meshes.flat(places, triangles), view,
                          face_values=np.zeros(4), colour=meshes.plain, gloss=solids.gloss)


def animate_flights(events: core.Antivector, pitches: np.ndarray, points: core.Antivector, strength: core.Scalar,
                    radius: float, heights: np.ndarray, n_steps_per_frame: int) -> Iterator[np.ndarray]:
    """The particles' tracks `[steps, particles]`, coloured by their angle to the axis, each trailing behind its
    particle, through the field's strength at the points of a grid in the plane through the axis `[across, along]`,
    between the coils of a radius at their heights."""
    tracks = solids.mv.point(space(events))                                   # [steps, particles] Point-
    grid = space(points)                                                      # [across, along, 3]
    colours = particle_colours(0.15 + 0.7 * (pitches - pitches.min()) / np.ptp(pitches))[:, :3]
    # The box of the grid, as deep across the axis as it is wide.
    half_extent = np.abs(grid).max(axis=(0, 1))
    box = np.stack(np.meshgrid(*[[-1, 1]] * 3, indexing="ij"), axis=-1).reshape(-1, 3) * half_extent[[0, 1, 1]]
    view = solids.view([solids.mv.point(box)], 0.0, elevation, azimuth, margin, n_pixels)
    # What stays put, set up once: the sheet and the coils' wire, each coil a ring about the axis, x, through the point
    # a radius above the axis at its height.
    rims = solids.origin + solids.mv.x.dual() * heights + solids.mv.z.dual() * radius   # [coils] Point-
    rings = meshes.circles(rims, solids.mv.yz, n_around_coil)                 # [coils, around + 1] Point-
    wire = strokes.polylines([(ring, to_rgb(wire_colour)) for ring in rings], wire_width, view)
    scene = Scene(view, np.ones(3), [], [(sheet(points, strength, view), sheet_opacity), (wire, 1.0)], inks=[],
                  translucent=[])
    # What moves, frame by frame: the particles and their trails.
    for end in range(n_steps_per_frame, len(tracks) + 1, n_steps_per_frame):
        trail = tracks[max(0, end - n_trail):end]                             # [trail, particles] Point-
        trails = strokes.polylines([(trail[:, particle], colour) for particle, colour in enumerate(colours)],
                                   track_width, view)
        particles = solids.ball(tracks[end - 1], particle_radius * view.pixel)[:, None]
        yield scene.draw([(particles, colours)], inks=[], translucent=[], meshes=[(trails, 1.0)])
