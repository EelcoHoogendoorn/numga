"""Satellite geometry and position uncertainty, drawn as solids at fixed scales."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.font_manager import FontProperties, findfont
from PIL import Image, ImageDraw, ImageFont

from numga import Extensor, concatenate, stack
from examples.util.wgsl import image_figure, solids, strokes

BLUE, ORANGE, EARTH, GRID, INK = (
    np.array(to_rgb(colour)) for colour in
    ("#3196b4", "#de9439", "#c7dce3", "#8faeb9", "#485961"))
WHITE = np.ones(3)
N_PIXELS = 480
ELEVATION, AZIMUTH = np.radians([22, 32])
FONT = ImageFont.truetype(findfont(FontProperties(family="DejaVu Sans")), 12)


def coordinates(events: Extensor) -> np.ndarray:
    """Spatial event coordinates, read by the three spatial planes at the drawing boundary."""
    mv = events.context.multivector
    return (stack((mv.x, mv.y, mv.z)) & events[..., None]).to_array()


def spatial_covariance(covariance: Extensor, earth_radius_metres: float) -> np.ndarray:
    """The marginal position covariance in square metres, with the clock coordinate integrated out."""
    mv = covariance.context.multivector
    readouts = stack((mv.x, mv.y, mv.z))
    return ((readouts[:, None] & covariance[None](readouts)[None, :]).to_array()
            * earth_radius_metres**2)


def earth_grid() -> solids.Ink:
    """Latitude and longitude curves on the unit Earth."""
    latitudes = np.sin(np.radians([-60, -30, 0, 30, 60]))
    meridians = (solids.mv.yz * (-np.radians([0, 30, 60, 90, 120, 150]) / 2)).exp() >> solids.mv.y
    planes = concatenate((solids.mv.x - solids.mv.w * latitudes, meridians))
    return solids.Ink(0, solids.half_space(planes),
                      solids.everywhere[None].broadcast_to(planes.shape), GRID, 1.3, 0.65)


def constellation_image(emitted: Extensor, receiver: Extensor, orbit_radius: float) -> np.ndarray:
    """Earth, transmitting satellites and their paths to the receiver, at a fixed orbital scale."""
    satellite_points = solids.mv.point(coordinates(emitted))
    receiver_point = solids.mv.point(coordinates(receiver))
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
             (solids.ball(receiver_point, 4 * view.pixel)[None, None], ORANGE)]
    rays = strokes.segments(satellite_points, receiver_point.broadcast_to(satellite_points.shape),
                            BLUE, 1.4, view)
    return solids.draw(parts, view, WHITE, inks=[earth_grid()], translucent=[], meshes=[(rays, 0.35)])


def uncertainty_image(receiver: Extensor, estimate: Extensor, covariance: Extensor,
                      earth_radius_metres: float, uncertainty_extent_metres: float) -> np.ndarray:
    """The position marginal's one-standard-deviation ellipsoid, centred on the estimate, beside the true location."""
    position_covariance = spatial_covariance(covariance, earth_radius_metres)
    variances, principal_axes = np.linalg.eigh(position_covariance)
    semiaxes = np.sqrt(variances)
    offset = coordinates(estimate - receiver) * earth_radius_metres
    centre = solids.mv.point(offset)
    camera = solids.aim(solids.origin, ELEVATION, AZIMUTH)
    view = solids.View(camera, np.full(2, uncertainty_extent_metres), N_PIXELS)

    # The inverse of the spatial covariance bounds the ellipsoid; its centre is the fitted position.
    directions = solids.axes.dual()
    dual = ((directions[:, None] * position_covariance
             * (solids.Plane & directions)[None, :]).sum(axis=(-1, -2))
            - solids.origin * (solids.Plane & solids.origin))
    ellipsoid = solids.placed(dual.inverse(), solids.translator(centre - solids.origin))
    parts = [(solids.ball(solids.origin, 3.2 * view.pixel)[None, None], ORANGE),
             (solids.ball(centre, 2.2 * view.pixel)[None, None], BLUE)]

    # The principal diameters make the narrow and weakly constrained directions visible through the surface.
    diameters = solids.mv.direction((principal_axes * semiaxes).T)
    principal = strokes.segments(centre - diameters, centre + diameters, BLUE, 1.1, view)
    axes_extent = uncertainty_extent_metres * 0.18
    axis_ends = solids.origin + directions * axes_extent
    axis_starts = solids.origin - directions * axes_extent
    axes = strokes.segments(axis_starts, axis_ends, GRID, 0.8, view)
    principal_planes = (solids.axes * principal_axes.T).sum(axis=-1)
    principal_planes = solids.translator(centre - solids.origin) >> principal_planes
    ink = solids.Ink(len(parts), solids.half_space(principal_planes),
                     solids.everywhere[None].broadcast_to(principal_planes.shape), BLUE, 1.1, 0.5)
    image = solids.draw(parts, view, WHITE, inks=[ink], translucent=[(ellipsoid[None, None], BLUE, 0.16)],
                        meshes=[(axes, 0.45), (principal, 0.8)])

    # A fixed physical scale keeps changing uncertainties comparable between frames.
    canvas = Image.fromarray(image)
    pen = ImageDraw.Draw(canvas)
    scale_metres = uncertainty_extent_metres / 2
    scale_pixels = round(scale_metres / view.pixel)
    left, bottom = 25, N_PIXELS - 26
    colour = tuple(np.rint(INK * 255).astype(int))
    pen.line([(left, bottom), (left + scale_pixels, bottom)], fill=colour, width=2)
    pen.line([(left, bottom - 3), (left, bottom + 3)], fill=colour)
    pen.line([(left + scale_pixels, bottom - 3), (left + scale_pixels, bottom + 3)], fill=colour)
    pen.text((left + scale_pixels / 2, bottom - 6), f"{scale_metres:g} m", fill=colour, font=FONT, anchor="mb")
    for position, name in zip(view.across(axis_ends), ("x", "y", "z")):
        pixel = ((position[0] + view.half[0]) / view.pixel,
                 (view.half[1] - position[1]) / view.pixel - 8)
        pen.text(pixel, name, fill=colour, font=FONT, anchor="mm")
    return np.asarray(canvas)


def draw_constellation(emitted: Extensor, receiver: Extensor, orbit_radius: float) -> plt.Figure:
    """The satellite arrangement before solving for the receiver."""
    return image_figure(constellation_image(emitted, receiver, orbit_radius))


def draw_solution(emitted: Extensor, receiver: Extensor, estimate: Extensor, covariance: Extensor,
                  orbit_radius: float, earth_radius_metres: float, uncertainty_extent_metres: float) -> plt.Figure:
    """The satellite geometry beside the estimated position and its uncertainty."""
    image = np.concatenate((constellation_image(emitted, receiver, orbit_radius),
                            uncertainty_image(receiver, estimate, covariance,
                                              earth_radius_metres, uncertainty_extent_metres)), axis=1)
    return image_figure(image)


def animate_geometry(states: Iterable, orbit_radius: float, earth_radius_metres: float,
                     uncertainty_extent_metres: float) -> Iterator[np.ndarray]:
    """A stream of satellite arrangements and their fitted position uncertainty, with both views held fixed."""
    for emitted, receiver, estimate, covariance in states:
        yield np.concatenate((constellation_image(emitted, receiver, orbit_radius),
                              uncertainty_image(receiver, estimate, covariance,
                                                earth_radius_metres, uncertainty_extent_metres)), axis=1)


def measurements(receiver: Extensor, estimate: Extensor, clock_reading: Extensor,
                 covariance: Extensor, earth_radius_metres: float, light_speed: float) -> str:
    """Position error, recovered clock offset, and their one-standard-deviation uncertainties."""
    mv = estimate.context.multivector
    position_error = np.linalg.norm(coordinates(estimate - receiver)) * earth_radius_metres
    position_sigma = np.sqrt(np.diag(spatial_covariance(covariance, earth_radius_metres)))
    clock_offset = float((clock_reading - (mv.t & estimate)).to_array()) * earth_radius_metres / light_speed
    clock_sigma = np.sqrt(float((mv.t & covariance(mv.t)).to_array())) * earth_radius_metres / light_speed
    return (f"Recovered clock offset: {clock_offset * 1e6:.6f} µs\n"
            f"Clock uncertainty (1σ): {clock_sigma * 1e9:.2f} ns\n"
            f"Position error: {position_error:.2f} m\n"
            f"Position uncertainty (1σ, x/y/z): "
            f"{position_sigma[0]:.2f} / {position_sigma[1]:.2f} / {position_sigma[2]:.2f} m")


def candidate_readout(candidates: Extensor, reference: Extensor,
                      earth_radius_metres: float, light_speed: float) -> str:
    """The cone intersections' distances from a reference and their reception coordinate times."""
    mv = candidates.context.multivector
    separation = candidates - reference
    distances = np.linalg.norm(coordinates(separation), axis=-1) * earth_radius_metres / 1000
    times = (mv.t & candidates).to_array() * earth_radius_metres / light_speed * 1000
    heading = "  candidate    distance from reference (km)    reception time (ms)"
    rows = [f"{index + 1:11d}  {distance:30.3f}  {time:28.6f}"
            for index, (distance, time) in enumerate(zip(distances, times))]
    return "\n".join([heading, *rows])
