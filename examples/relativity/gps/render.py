"""Satellite geometry and position uncertainty, drawn as solids at fixed scales."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga import concatenate, stack
from examples.relativity.gps import core
from examples.util.wgsl import image_figure, solids, strokes

BLUE, ORANGE, EARTH, GRID, INK = (
    np.array(to_rgb(colour)) for colour in
    ("#3196b4", "#de9439", "#c7dce3", "#8faeb9", "#485961"))
WHITE = np.ones(3)
N_PIXELS = 320
ELEVATION, AZIMUTH = np.radians([22, 32])


def gpu_points(events: core.Event) -> solids.Point:
    """The GPU library's points at the places of events of unit weight: their spatial coordinates, read by pairing
    with the three spatial planes, are the one hand-off from spacetime to the renderer's space."""
    mv = events.context.multivector
    return solids.mv.point((stack((mv.x, mv.y, mv.z)) & events[..., None]).to_array())


def earth_grid() -> solids.Ink:
    """Latitude and longitude curves on the unit Earth, its axis along x: a latitude's plane lies across the axis,
    through a point of the equator turned towards the pole, and a meridian's plane through the axis."""
    sm = solids.mv
    latitudes = (sm.xy * (np.radians([-60, -30, 0, 30, 60]) / 2)).exp() >> (solids.origin + sm.y.dual())   # [latitudes] Point-
    meridians = (sm.yz * (-np.radians([0, 30, 60, 90, 120, 150]) / 2)).exp() >> sm.y      # [meridians] Plane+
    planes = concatenate((sm.x - sm.w * (sm.x & latitudes), meridians))          # [curves] Plane+
    return solids.Ink(0, solids.half_space(planes),
                      solids.everywhere[None].broadcast_to(planes.shape), GRID, 0.9, 0.65)


def constellation(emitted: core.Event, receiver: core.Event, orbit_radius: float) -> tuple[solids.View, list, list]:
    """The view at a fixed orbital scale, the solids of Earth, the transmitting satellites and the receiver, and the
    signals' paths to it."""
    satellite_points = gpu_points(emitted)
    receiver_point = gpu_points(receiver)
    centre = solids.origin + (receiver_point - solids.origin) * (orbit_radius * 0.3)
    camera = solids.aim(centre, ELEVATION, AZIMUTH)
    view = solids.View(camera, np.full(2, orbit_radius * 0.95), N_PIXELS)

    # The symbols show each satellite's central body and pair of solar panels.
    bodies = solids.boxes(solids.translator(satellite_points - solids.origin), np.array([0.065, 0.065, 0.065]))
    panel_offsets = solids.mv.x.dual() * np.array([-0.19, 0.19])
    panels = solids.boxes(solids.translator(satellite_points[:, None] + panel_offsets - solids.origin),
                          np.array([0.105, 0.065, 0.012])).reshape(-1, 6)
    parts = [(solids.ball(solids.origin, 1)[None, None], EARTH),
             (bodies, ORANGE), (panels, BLUE),
             (solids.ball(receiver_point, 2.7 * view.pixel)[None, None], ORANGE)]
    rays = [(strokes.segments(satellite_points, receiver_point.broadcast_to(satellite_points.shape), BLUE, 0.9, view),
             0.35)]
    return view, parts, rays


def constellation_image(emitted: core.Event, receiver: core.Event, orbit_radius: float) -> np.ndarray:
    """Earth, transmitting satellites and their paths to the receiver, at a fixed orbital scale."""
    view, parts, rays = constellation(emitted, receiver, orbit_radius)
    return solids.draw(parts, view, WHITE, inks=[earth_grid()], translucent=[], meshes=rays)


def solution_image(emitted: core.Event, receiver: core.Event, estimate: core.Event, semi_axes: core.Event,
                   orbit_radius: float) -> np.ndarray:
    """The constellation with the fitted reception, its position's one-standard-deviation ellipsoid about it, and the
    ellipsoid's semi-axes `[3]` as diameters through it."""
    view, parts, rays = constellation(emitted, receiver, orbit_radius)
    centre = gpu_points(estimate)
    reach = gpu_points(estimate + semi_axes) - centre                            # [3] Point-
    # The semi-axes' dyads sum to the spatial covariance, the dual quadric of the ellipsoid about the origin; its
    # inverse bounds the ellipsoid, moved to the fitted position.
    dual = (reach * (solids.Plane & reach)).sum(axis=0) - solids.origin * (solids.Plane & solids.origin)   # [] Point- <- Plane+
    ellipsoid = solids.placed(dual.inverse(), solids.translator(centre - solids.origin))
    principal = strokes.segments(centre - reach, centre + reach, BLUE, 0.9, view)
    parts = parts + [(solids.ball(centre, 2.0 * view.pixel)[None, None], BLUE)]
    return solids.draw(parts, view, WHITE, inks=[earth_grid()], translucent=[(ellipsoid[None, None], BLUE, 0.3)],
                       meshes=rays + [(principal, 0.8)])


def draw_constellation(emitted: core.Event, receiver: core.Event, orbit_radius: float) -> plt.Figure:
    """The satellite arrangement before solving for the receiver."""
    return image_figure(constellation_image(emitted, receiver, orbit_radius))


def draw_solution(emitted: core.Event, receiver: core.Event, estimate: core.Event, semi_axes: core.Event,
                  orbit_radius: float) -> plt.Figure:
    """The satellite geometry with the estimated position and its uncertainty, given by the semi-axes of its
    ellipsoid."""
    return image_figure(solution_image(emitted, receiver, estimate, semi_axes, orbit_radius))


def animate_geometry(states: Iterable, orbit_radius: float) -> Iterator[np.ndarray]:
    """A stream of satellite arrangements and their fitted position uncertainty, the view held fixed."""
    for emitted, receiver, estimate, semi_axes in states:
        yield solution_image(emitted, receiver, estimate, semi_axes, orbit_radius)
