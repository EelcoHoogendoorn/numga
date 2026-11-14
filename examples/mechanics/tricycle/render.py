"""Planar and spatial GPU views draw the tricycle's frame, wheels, axle extensions to the turning centre, and rear-axle trail.

Both views prepare their geometry on algebraic points in the drive's own algebra: the top-down planar view reads points directly with a camera of planes `x`, `y`, and `w`, while the perspective spatial view passes its `PGA3D` solids and rim tubes straight to the GPU ray tracer."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga import Extensor, concatenate, stack
from examples.util.animation import capture
from numga.topology import Topology
from numga.render import meshes, simplices, solids
from examples.util.panels import overlay, panel

frame_colour = "#2c3e50"
wheel_colour = "#222222"
axle_colour = "#c0392b"
centre_colour = "#c0392b"
trail_colour = "#7f8c8d"
wheel_radius = 0.25
# Half-width of the square view window following the midpoint of the rear axle.
view_half_width = 3.0
background = np.ones(3)
# Pixel stroke widths, dash pattern, and marker radius in the top-down planar view.
plane_trail_width, axle_width, axle_dashes, plane_frame_width, wheel_width = 1.0, 1.5, (6.0, 4.5), 2.0, 4.0
centre_radius = 2.3
# Perspective camera elevation and azimuth (zero looking along +y), framing margin, pixel width,
# spatial axle dash length, rim sample count, and stroke widths in pixels.
elevation, azimuth = np.radians(35.0), np.radians(25.0)
margin, n_pixels = 0.04, 300
dash_length = 0.12
n_rim = 48
dash_width, frame_width, post_width, rim_width, trail_width = 2.0, 3.0, 2.0, 2.0, 2.0
# Unit-square corner offsets for the ground patch tracked by the spatial camera.
ground_corners = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])


# --- plumbing -------------------------------------------------------------------------
def axle_origins(contacts: Extensor, centre: Extensor) -> Extensor:
    """The nearer of the two rear wheel contacts in `contacts` (`[3] Point-`) to unit-weight `centre` (`[] Point-`) pairs with the front wheel contact (`[2] Point-`) so each axle line is drawn once to `centre`."""
    # Line norm between unit-weight points measures Euclidean distance along the ground.
    nearer = (contacts[:2] & centre).norm().argmin()
    return contacts[[nearer, 2]]                                                # [2] Point-


def frame_bars(contacts: Extensor) -> tuple[Extensor, Extensor]:
    """Wheel ground contacts `contacts` (`[3] Point-`) span the rear axle bar and the main spine from the rear-axle midpoint to the front wheel (`[2] Point-`, `[2] Point-`)."""
    middle = (contacts[0] + contacts[1]) / 2                                    # [] Point-
    return stack([contacts[0], middle]), stack([contacts[1], contacts[2]])      # [2] Point-, [2] Point-


def wheel_steps(contacts: Extensor, forward: Extensor) -> Extensor:
    """Scaling `forward - contacts` by `wheel_radius` over the join norm `(contacts & forward).norm()` gives each wheel's one-radius ground step (`[3] Point-`)."""
    return (forward - contacts) * (wheel_radius / (contacts & forward).norm())  # [3] Point-


def segment_pairs(n_pairs: int) -> Topology:
    """Index pairs `[k, n_pairs + k]` form a 1D `Topology` of `n_pairs` disjoint segments across `2 * n_pairs` concatenated endpoints."""
    return Topology(2 * n_pairs, np.stack([np.arange(n_pairs), n_pairs + np.arange(n_pairs)], axis=-1))


def plane(contacts: Extensor, forward: Extensor, centre: Extensor, middle: Extensor) -> np.ndarray:
    """Renders a top-down RGB frame (`[rows, columns, 3]`) of the planar tricycle in its own algebra."""
    mv = centre.context.multivector
    # Reorient the extrinsic meet into the unit-weight intrinsic point where the axles cross.
    centre = (centre * mv.antipseudoscalar([1])).normalized()                   # [] Point-
    steps = wheel_steps(contacts, forward)                                      # [3] Point-
    across, up = following_limits(middle[-1])
    figure, ax = plt.subplots(figsize=(3, 3))
    ax.set(xlim=across, ylim=up, aspect="equal")
    ax.axis("off")
    figure.tight_layout()
    camera = panel(ax, mv.x, mv.y, mv.w)
    origins = axle_origins(contacts, centre)                                    # [2] Point-
    spokes = Topology(len(origins) + 1, np.stack([np.arange(len(origins)), np.full(len(origins), len(origins))], -1))
    starts, ends = frame_bars(contacts)                                         # [2] Point-, [2] Point-
    overlay(ax, [
        simplices.strokes(middle, Topology.chains([len(middle)]), to_rgb(trail_colour), plane_trail_width, camera),
        simplices.strokes(concatenate([origins, centre[None]]), spokes, to_rgb(axle_colour), axle_width, camera,
                          dashes=axle_dashes, cap=simplices.Cap.BUTT),
        simplices.markers(centre[None], to_rgb(centre_colour), centre_radius, camera),
        simplices.strokes(concatenate([starts, ends]), segment_pairs(len(starts)), to_rgb(frame_colour),
                          plane_frame_width, camera),
        simplices.strokes(concatenate([contacts - steps, contacts + steps]), segment_pairs(len(contacts)),
                          to_rgb(wheel_colour), wheel_width, camera)])
    pixels = capture(figure)
    plt.close(figure)
    return pixels


def space(contacts: solids.Point, forward: solids.Point, centre: solids.Line, middle: solids.Point) -> np.ndarray:
    """Renders a perspective RGB frame (`[rows, columns, 3]`) of the spatial tricycle on the GPU, with dashed axle rods, frame rods, a vertical post at the turning-centre line, wheel rims, and the rear-axle trail."""
    mv = centre.context.multivector
    # Ideal vertical direction and unit-weight ground point where the vertical turning-centre line meets z = 0.
    up = mv.z.dual()                                                            # [] Point-
    foot = ((centre ^ mv.z) * mv.antipseudoscalar([1])).normalized()            # [] Point-
    view = space_view(middle[-1])
    origins = axle_origins(contacts, foot)                                      # [2] Point-
    lengths = (origins & foot).norm()                                           # [2] Scalar+
    directions = (foot - origins) / lengths                                     # [2] Point-
    shares = [np.arange(0, length - dash_length, 2 * dash_length) for length in lengths.to_array()]
    starts = concatenate([start + direction * share for start, direction, share in zip(origins, directions, shares)])
    dash_ends = concatenate([start + direction * (share + dash_length)
                             for start, direction, share in zip(origins, directions, shares)])
    dashes = solids.rods(starts + up * wheel_radius, dash_ends + up * wheel_radius, dash_width / 2 * view.pixel)
    hubs = contacts + up * wheel_radius                                         # [3] Point-
    bars = solids.rods(*(ends + up * wheel_radius for ends in frame_bars(contacts)), frame_width / 2 * view.pixel)
    post = solids.rods(foot, foot + up * (4 * wheel_radius), post_width / 2 * view.pixel)
    parts = [(dashes, to_rgb(axle_colour)), (bars, to_rgb(frame_colour)), (post[None], to_rgb(centre_colour))]
    # Quarter-turn each rolling step in the ground plane to form the wheel's axle line through its hub.
    steps = wheel_steps(contacts, forward)                                      # [3] Point-
    sideways = steps.commutator(mv.xy)                                          # [3] Point-
    rims = meshes.circles(hubs + steps, (hubs & sideways) * solids.flip, n_rim)  # [3, n_rim] Point-
    tyres = simplices.tubes(rims.reshape(-1), Topology.loops([rims.shape[1]] * rims.shape[0]), to_rgb(wheel_colour),
                            rim_width, view, gloss=solids.gloss)
    trail = simplices.tubes(middle, Topology.chains([len(middle)]), to_rgb(trail_colour), trail_width, view,
                            gloss=solids.gloss)
    return solids.draw(parts, view, background, inks=[], translucent=[], meshes=[(tyres, 1.0), (trail, 1.0)])


def space_view(place: solids.Point) -> solids.View:
    """Frames the square ground patch of half-width `view_half_width` around `place` (`[] Point-`) up to the top of the turning-centre post."""
    mv = place.context.multivector
    corners = place + (mv.x.dual() * ground_corners[:, 0] + mv.y.dual() * ground_corners[:, 1]) * view_half_width  # [4] Point-
    return solids.view([corners, corners + mv.z.dual() * (4 * wheel_radius)], 0.0, elevation, azimuth, margin,
                       n_pixels)


def following_limits(place: Extensor) -> np.ndarray:
    """Pairing `place` (`[] Point-`) with the coordinate planes `x` and `y` over its weight gives `[2, 2]` axis limits spanning `view_half_width` to either side."""
    mv = place.context.multivector
    middle = (stack((mv.x, mv.y)) & place) / (mv.w & place)                     # [2] Scalar+
    return stack([middle - view_half_width, middle + view_half_width], axis=-1).to_array()


def animate(draw, scenes) -> Iterator[np.ndarray]:
    """Yields one rendered RGB frame (`[rows, columns, 3]`) per scene tuple in `scenes`."""
    for scene in scenes:
        yield draw(*scene)
