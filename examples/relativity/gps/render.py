"""Satellite geometry and position uncertainty, drawn as solids at fixed scales."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga import concatenate, stack
from examples.relativity.gps import core
from examples.util import shapes
from examples.util.wgsl import image_figure, simplices, solids

mv = core.mv
BLUE, ORANGE, EARTH, GRID, INK = (
    np.array(to_rgb(colour)) for colour in
    ("#3196b4", "#de9439", "#c7dce3", "#8faeb9", "#485961"))
WHITE = np.ones(3)
N_PIXELS = 320
ELEVATION, AZIMUTH = np.radians([22, 32])
# The animation's margin at each edge, as a share of half the larger extent of all it frames.
MARGIN = 0.2
# A satellite's body, a cube, and its two solar panels on either side of it along the polar axis: their half extents,
# and the panels' offset from the body, in Earth radii.
BODY = np.full(3, 0.065)
PANEL = np.array([0.105, 0.065, 0.012])
PANEL_OFFSET = 0.19


def earth_grid() -> solids.Ink:
    """Latitude and longitude curves on the unit Earth, its axis along x: a latitude's plane lies across the axis,
    through a point of the equator turned towards the pole, and a meridian's plane through the axis."""
    sm = solids.mv
    latitudes = (sm.xy * (np.radians([-60, -30, 0, 30, 60]) / 2)).exp() >> (solids.origin + sm.y.dual())   # [latitudes] Point-
    meridians = (sm.yz * (-np.radians([0, 30, 60, 90, 120, 150]) / 2)).exp() >> sm.y      # [meridians] Plane+
    planes = concatenate((sm.x - sm.w * (sm.x & latitudes), meridians))          # [curves] Plane+
    return solids.Ink(0, solids.half_space(planes),
                      solids.everywhere[None].broadcast_to(planes.shape), GRID, 0.9, 0.65)


def sliced(events: core.Event) -> solids.Point:
    """Events as the observer at rest sees them, in the projective algebra of space."""
    # The observer sees the point where its time line meets the spatial slice.
    space_slice = mv.t ^ (core.Event & -mv.t.dual())
    # The projective embedding of the events as the observer slices them: the projective algebra of spacetime holds
    # the one of space.
    return solids.context.include_from_algebra(space_slice(events))


def fixed_view(receiver: core.Event, orbit_radius: float) -> solids.View:
    """The stills' view, at a fixed orbital scale."""
    centre = solids.origin + (sliced(receiver) - solids.origin) * (orbit_radius * 0.3)
    return solids.View(solids.aim(centre, ELEVATION, AZIMUTH), np.full(2, orbit_radius * 0.95), N_PIXELS)


def swing_view(states: list) -> solids.View:
    """The animation's view, framing Earth and, over every frame, the satellites with their panels and the fitted
    position's semi-axes as diameters through it."""
    emitted = stack([transmissions for transmissions, _, _, _ in states])     # [frames, satellites] Event-
    estimates = stack([estimate for _, _, estimate, _ in states])             # [frames] Event-
    semi_axes = stack([axes for _, _, _, axes in states])                     # [frames, 3] Event-
    # Points on Earth's surface, the vertices of a finely split icosahedron.
    _, coordinates = shapes.sphere(2)
    earth = solids.mv.point(coordinates)                                      # [vertices] Point-
    # Each satellite's span, from tip to tip of its panels along the polar axis.
    span = solids.mv.x.dual() * ((PANEL_OFFSET + PANEL[0]) * np.array([-1, 1]))   # [2] Point-
    tips = sliced(emitted)[..., None] + span                                  # [frames, satellites, 2] Point-
    centres, reach = sliced(estimates)[:, None], sliced(semi_axes)            # [frames, 1] Point-, [frames, 3] Point-
    ends = concatenate((centres - reach, centres + reach), axis=-1)           # [frames, 6] Point-
    return solids.view([earth, tips, ends], 0.0, ELEVATION, AZIMUTH, MARGIN, N_PIXELS)


def constellation(emitted: core.Event, receiver: core.Event, view: solids.View) -> tuple[list, list]:
    """The solids of Earth, the transmitting satellites and the receiver, and the signals' paths to it."""
    satellite_points = sliced(emitted)
    receiver_point = sliced(receiver)

    # The symbols show each satellite's central body and pair of solar panels.
    bodies = solids.boxes(solids.translator(satellite_points - solids.origin), BODY)
    panel_offsets = solids.mv.x.dual() * np.array([-PANEL_OFFSET, PANEL_OFFSET])
    panels = solids.boxes(solids.translator(satellite_points[:, None] + panel_offsets - solids.origin),
                          PANEL).reshape(-1, 6)
    parts = [(solids.ball(solids.origin, 1)[None, None], EARTH),
             (bodies, ORANGE), (panels, BLUE),
             (solids.ball(receiver_point, 2.7 * view.pixel)[None, None], ORANGE)]
    # Each signal's path an edge from its satellite to the receiver.
    signals = stack((satellite_points, receiver_point.broadcast_to(satellite_points.shape)), axis=-1)   # [satellites, 2] Point-
    rays = [(simplices.tubes(signals.reshape(-1), simplices.Topology.chains([2] * len(satellite_points)), BLUE, 0.9, view,
                             solids.gloss), 0.35)]
    return parts, rays


def constellation_image(emitted: core.Event, receiver: core.Event, view: solids.View) -> np.ndarray:
    """Earth, transmitting satellites and their paths to the receiver."""
    parts, rays = constellation(emitted, receiver, view)
    return solids.draw(parts, view, WHITE, inks=[earth_grid()], translucent=[], meshes=rays)


def solution_image(emitted: core.Event, receiver: core.Event, estimate: core.Event, semi_axes: core.Event,
                   view: solids.View) -> np.ndarray:
    """The constellation with the fitted reception, its position's one-standard-deviation ellipsoid about it, and the
    ellipsoid's semi-axes `[3]` as diameters through it."""
    parts, rays = constellation(emitted, receiver, view)
    # The sliced estimate, and its semi-axes, directions at infinity.
    centre = sliced(estimate)                                                 # [] Point-
    reach = sliced(semi_axes)                                                 # [3] Point-
    # The semi-axes' dyads sum to the spatial covariance, the dual quadric of the ellipsoid about the origin; its
    # inverse bounds the ellipsoid, moved to the fitted position.
    dual = (reach * (solids.Plane & reach)).sum(axis=0) - solids.origin * (solids.Plane & solids.origin)   # [] Point- <- Plane+
    ellipsoid = solids.placed(dual.inverse(), solids.translator(centre - solids.origin))
    diameters = stack((centre - reach, centre + reach), axis=-1)               # [3, 2] Point-
    principal = simplices.tubes(diameters.reshape(-1), simplices.Topology.chains([2] * 3), BLUE, 0.9, view, solids.gloss)
    parts = parts + [(solids.ball(centre, 2.0 * view.pixel)[None, None], BLUE)]
    return solids.draw(parts, view, WHITE, inks=[earth_grid()], translucent=[(ellipsoid[None, None], BLUE, 0.3)],
                       meshes=rays + [(principal, 0.8)])


def draw_constellation(emitted: core.Event, receiver: core.Event, orbit_radius: float) -> plt.Figure:
    """The satellite arrangement before solving for the receiver."""
    return image_figure(constellation_image(emitted, receiver, fixed_view(receiver, orbit_radius)))


def draw_solution(emitted: core.Event, receiver: core.Event, estimate: core.Event, semi_axes: core.Event,
                  orbit_radius: float) -> plt.Figure:
    """The satellite geometry with the estimated position and its uncertainty, given by the semi-axes of its
    ellipsoid."""
    return image_figure(solution_image(emitted, receiver, estimate, semi_axes, fixed_view(receiver, orbit_radius)))


def animate_geometry(states: Iterable) -> Iterator[np.ndarray]:
    """A stream of satellite arrangements and their fitted position uncertainty, the view held fixed on all of them."""
    states = list(states)
    view = swing_view(states)
    for emitted, receiver, estimate, semi_axes in states:
        yield solution_image(emitted, receiver, estimate, semi_axes, view)
