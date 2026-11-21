"""GPU solid scenes display GPS constellations, signal rays, and one-sigma spatial uncertainty ellipsoids."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga import concatenate, dim, stack
from examples.relativity.gps import core
from examples.util import shapes
from numga.topology import Topology
from numga.render import simplices, solids
from examples.util.panels import image_figure

mv = core.mv
blue_colour, orange_colour, earth_colour, grid_colour, ink_colour = (
    np.array(to_rgb(colour)) for colour in
    ("#3196b4", "#de9439", "#c7dce3", "#8faeb9", "#485961"))
white_colour = np.ones(3)
n_pixels = 320
elevation, azimuth = np.radians([22, 32])
# Viewport margin as a fraction of half the bounding extent across the full animation.
margin = 0.2
# Satellite central cube half-extents, solar-panel half-extents, and polar-axis panel offset in Earth radii.
body_half_extents = np.full(3, 0.065)
panel_half_extents = np.array([0.105, 0.065, 0.012])
panel_offset = 0.19


# --- plumbing -------------------------------------------------------------------------
def earth_grid() -> solids.Ink:
    """Inked latitude parallels and longitude meridians on the unit Earth sphere with polar axis along `x`."""
    sm = solids.mv
    lat_rotors = (sm.xy * (np.radians([-60, -30, 0, 30, 60]) / 2)).exp()          # [5] Rotor+
    latitudes = lat_rotors >> (solids.origin + sm.y.dual())                       # [5] Point-
    lon_rotors = (sm.yz * (-np.radians([0, 30, 60, 90, 120, 150]) / 2)).exp()     # [6] Rotor+
    meridians = lon_rotors >> sm.y                                                # [6] Plane+
    planes = concatenate((sm.x - sm.w * (sm.x & latitudes), meridians))           # [11] Plane+
    return solids.Ink(0, solids.half_space(planes),
                      solids.everywhere[None].broadcast_to(planes.shape), grid_colour, 0.9, 0.65)


def rest_slice_points(events: core.Event) -> solids.Point:
    """Projecting spacetime `events` (`[...] Event-`) onto the rest-frame slice embeds them into 3D `solids`."""
    space_slice = mv.t ^ (core.Event & -mv.t.dual())                              # [] Event- <- Event-
    return solids.context.include_from_algebra(space_slice(events))


def fixed_view(receiver: core.Event, orbit_radius: core.Scalar) -> solids.View:
    """Fixed-scale camera view framing the orbital shell above `receiver`."""
    radius_value = float((orbit_radius / dim.length).to_array())
    centre = solids.origin + (rest_slice_points(receiver) - solids.origin) * (radius_value * 0.3)
    return solids.View(solids.aim(centre, elevation, azimuth), np.full(2, radius_value * 0.95), n_pixels)


def swing_view(states: list) -> solids.View:
    """Shared camera view framing Earth, all `n_frames` satellite constellations, and every uncertainty ellipsoid."""
    emitted = stack([transmissions for transmissions, _, _, _ in states])         # [n_frames, n_satellites] Event-
    estimates = stack([estimate for _, _, estimate, _ in states])                 # [n_frames] Event-
    semi_axes = stack([axes for _, _, _, axes in states])                         # [n_frames, 3] Event-
    topology, coordinates = shapes.sphere(2)
    V, E, F = topology.n_elements
    earth = solids.mv.point(coordinates)                                          # [V] Point-
    span = solids.mv.x.dual() * ((panel_offset + panel_half_extents[0]) * np.array([-1, 1]))  # [2] Point-
    tips = rest_slice_points(emitted)[..., None] + span                           # [n_frames, n_satellites, 2] Point-
    centres = rest_slice_points(estimates)[:, None]                               # [n_frames, 1] Point-
    reach = rest_slice_points(semi_axes)                                          # [n_frames, 3] Point-
    ends = concatenate((centres - reach, centres + reach), axis=-1)               # [n_frames, 6] Point-
    return solids.view([earth, tips, ends], 0.0, elevation, azimuth, margin, n_pixels)


def constellation_solids(emitted: core.Event, receiver: core.Event, view: solids.View) -> tuple[list, list]:
    """Solid parts for Earth, satellite bodies and solar panels, `receiver`, and signal ray tubes."""
    satellite_points = rest_slice_points(emitted)                                 # [n_satellites] Point-
    receiver_point = rest_slice_points(receiver)                                  # [] Point-

    bodies = solids.boxes(solids.translator(satellite_points - solids.origin), body_half_extents)
    panel_shifts = solids.mv.x.dual() * np.array([-panel_offset, panel_offset])   # [2] Point-
    panels = solids.boxes(solids.translator(satellite_points[:, None] + panel_shifts - solids.origin),
                          panel_half_extents).reshape(-1, 6)
    parts = [(solids.ball(solids.origin, 1)[None, None], earth_colour),
             (bodies, orange_colour), (panels, blue_colour),
             (solids.ball(receiver_point, 2.7 * view.pixel)[None, None], orange_colour)]
    signals = stack((satellite_points, receiver_point.broadcast_to(satellite_points.shape)), axis=-1)  # [n_satellites, 2] Point-
    rays = [(simplices.tubes(signals.reshape(-1), Topology.chains([2] * len(satellite_points)), blue_colour, 0.9, view,
                             solids.gloss), 0.35)]
    return parts, rays


def constellation_image(emitted: core.Event, receiver: core.Event, view: solids.View) -> np.ndarray:
    """Rendered RGB frame of Earth, the transmitting satellites, and their signal rays to `receiver`."""
    parts, rays = constellation_solids(emitted, receiver, view)
    return solids.draw(parts, view, white_colour, inks=[earth_grid()], translucent=[], meshes=rays)


def solution_image(emitted: core.Event, receiver: core.Event, estimate: core.Event, semi_axes: core.Event,
                   view: solids.View) -> np.ndarray:
    """Rendered RGB frame adding the fitted reception `estimate`, its uncertainty ellipsoid, and principal diameters."""
    parts, rays = constellation_solids(emitted, receiver, view)
    centre = rest_slice_points(estimate)                                          # [] Point-
    reach = rest_slice_points(semi_axes)                                          # [3] Point-
    # Sum the semi-axes' dyads into the origin-centred dual quadric and translate its inverse to `centre`.
    dual_quadric = (
        (reach * (solids.Plane & reach)).sum(axis=0) - solids.origin * (solids.Plane & solids.origin)
    )                                                                             # [] Point- <- Plane+
    ellipsoid = solids.placed(dual_quadric.inverse(), solids.translator(centre - solids.origin))
    diameters = stack((centre - reach, centre + reach), axis=-1)                  # [3, 2] Point-
    principal = simplices.tubes(diameters.reshape(-1), Topology.chains([2] * 3), blue_colour, 0.9, view, solids.gloss)
    parts = parts + [(solids.ball(centre, 2.0 * view.pixel)[None, None], blue_colour)]
    return solids.draw(parts, view, white_colour, inks=[earth_grid()],
                       translucent=[(ellipsoid[None, None], blue_colour, 0.3)],
                       meshes=rays + [(principal, 0.8)])


def constellation_figure(emitted: core.Event, receiver: core.Event, orbit_radius: core.Scalar) -> plt.Figure:
    """Static figure of the satellite constellation and signal paths prior to positioning."""
    return image_figure(constellation_image(emitted, receiver, fixed_view(receiver, orbit_radius)))


def positioning_figure(emitted: core.Event, receiver: core.Event, estimate: core.Event, semi_axes: core.Event,
                       orbit_radius: core.Scalar) -> plt.Figure:
    """Static figure of the satellite constellation with the fitted position and one-sigma uncertainty ellipsoid."""
    return image_figure(solution_image(emitted, receiver, estimate, semi_axes, fixed_view(receiver, orbit_radius)))


def geometry_animation(states: Iterable) -> Iterator[np.ndarray]:
    """Yields `[rows, columns, 3]` RGB frames (forward and back) of the constellation and position ellipsoid."""
    collected = list(states)
    view = swing_view(collected)
    forward_frames = []
    for emitted, receiver, estimate, semi_axes in collected:
        frame = solution_image(emitted, receiver, estimate, semi_axes, view)
        forward_frames.append(frame)
        yield frame
    for frame in forward_frames[-2:0:-1]:
        yield frame
