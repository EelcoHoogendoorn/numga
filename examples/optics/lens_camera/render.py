"""This module draws the lens camera's sensor images and side views, and nothing else.

The sensor images are drawn on the GPU by a shader in numga's dialect: each sensor pixel is a point of the sensor
plane, and every scene point's image cone lights it by its form there, `cone(point) & point`, negative inside the
blur disc. The side view hands the GPU's solids the lenses' rims, the sensor's corners and the rays' legs as points.
"""

from __future__ import annotations

from collections.abc import Iterator
from functools import cache

import matplotlib.pyplot as plt
import numpy as np

from numga import stack

from examples.optics.lens_camera.core import Motor, Plane, Point, Quadric, ga, home, mv, origin, point, unit
from examples.util.wgsl import Shader, image_figure, solids, supersampled
from examples.util.wgsl.scene import Scene

# Sensor pixels, rows by columns.
resolution = (168, 216)
# How bright a disc's light shows, per unit of its energy over the sensor's area, and the width of a disc's edge, a
# crude diffraction limit, in pixels.
brightness = 1.36e-3
edge_pixels = 1.0
# Half extents of the sensor window in its own frame, and the width of a disc's edge on it.
sensor_half_extents = (0.225, 0.175)
edge_width = edge_pixels * 2 * sensor_half_extents[1] / resolution[0]
# The planes whose pairings with a point of the sensor in its home frame are how far right and up it lies; the point
# so far along each is `origin + (axes.dual() * place).sum()`.
axes = stack((mv.y, mv.z))                                                      # [2] Plane+
# The aperture whose exposure is the reference brightness, and the half-height the rear lens is drawn with in the
# side view; the front lens is drawn as wide as the aperture.
reference_aperture = 0.45
rear_half_height = 0.6
# The colour each depth layer of the scene lights on the sensor, red, green and blue, in the order of the layers, and
# the white between the stills.
layer_colours = np.eye(3)
gap_pixels = 40
# The side view, drawn on the GPU: its elevation and azimuth in radians, azimuth zero looking along +y once the
# scene's y is turned up; the box of space it frames, along the optical axis x, up y and across z; its pixels across;
# and in pixels, the radii of the scene's points and of the rays; the corners around a lens's rim, the thickness of
# the lenses and the sensor, and how much of what lies behind it a lens's face hides; the colours of the lenses, the
# sensor and the rays.
side_elevation, side_azimuth = np.radians([14.0, -18.0])
side_box = ((-3.6, 2.6), (-1.0, 1.0), (-0.6, 0.6))
side_pixels = 216
point_radius, ray_radius = 2.0, 0.6
n_rim, thickness, lens_opacity = 32, 0.02, 0.3
lens_colour, sensor_colour, ray_colour = (0.55, 0.7, 0.85), (0.3, 0.3, 0.32), (1.0, 0.5, 0.05)
# A quarter turn about the optical axis for display: the scene has y up, the view z up.
display = (mv.yz * (np.pi / 4)).exp()


sensor_source = """
alias Sample = vec3<f32>;
var<uniform> frame: Motor;

// A scene point's blur: its image cone, the colour of its layer, and its light per unit area of the sensor.
struct Disc {
    cone: Quadric,
    colour: vec3<f32>,
    light: f32,
}

var<storage, read> discs: array<Disc>;

fn sample(place: vec2<f32>) -> Sample {
    // The sensor plane and the point of it at the place, both carried from home by the frame.
    let sensor = frame >> home;
    let point = frame >> (origin + (axes.dual() * (screen(place) * half_height)).sum());
    var colour = vec3<f32>(0.0);
    for (var index = 0u; index < arrayLength(&discs); index++) {
        let disc = discs[index];
        let polar = disc.cone(point);
        // The form's gradient is twice the polar, and its slope along the sensor the norm of the polar's meet with the
        // sensor: the form over that slope is the distance to the disc's rim, to first order, negative inside.
        let distance = (polar & point) / (2 * (polar ^ sensor).norm());
        // A logistic edge, `edge` wide.
        colour += disc.colour * disc.light / (1 + exp(distance / edge));
    }
    return colour;
}
"""


# --- plumbing -------------------------------------------------------------------------
@cache
def sensor_shader() -> Shader:
    """The shader of a sensor image, its frame of axes, the sensor's half height and the width of a disc's edge given
    as its constants."""
    return Shader(supersampled() + sensor_source, ga, dict(Point=Point, Plane=Plane, Motor=Motor, Quadric=Quadric),
                  dict(origin=origin, home=home, axes=axes, half_height=sensor_half_extents[1], edge=edge_width))


def disc_light(frame: Motor, cones: Quadric, energy: float) -> np.ndarray:
    """Each scene point's light per unit area of the sensor `[points]`: its energy, scaled by the aperture area, spread
    over its blur disc and the spot its edge spreads a point over.

    The disc's area comes from the determinants of the cone's form on the sensor's points and on its directions, each
    the join of those points paired with the meet of their polars: an ellipse's area is pi times the first over the
    second to the power three halves. A logistic edge about the first-order distance spreads a point over two thirds
    of pi cubed edge widths squared.
    """
    centre, right, up = frame >> origin, frame >> mv.y.dual(), frame >> mv.z.dual()     # [] Point-, Point-, Point-
    whole = (centre & right & up) & (cones(centre) ^ cones(right) ^ cones(up))         # [depths, heights, widths] Scalar+
    directions = (right & up) & (cones(right) ^ cones(up))                             # [depths, heights, widths] Scalar+
    area = np.pi * np.abs(whole.to_array()) / directions.to_array()**1.5
    spot = 2 / 3 * np.pi**3 * edge_width**2
    return brightness * energy * 4 * sensor_half_extents[0] * sensor_half_extents[1] / (area + spot)


def expose_sensor(frame: Motor, cones: Quadric, energy: float) -> np.ndarray:
    """The sensor image, `[rows, columns, 3]` bytes: every scene point's blur disc in the colour of its depth layer,
    the light of all of them summed. Each disc deposits the same total energy, scaled by the aperture area, so a point
    in focus is a bright dot and a defocused one a dim wide disc."""
    layers = np.broadcast_to(layer_colours.reshape((3,) + (1,) * (cones.ndim - 1) + (3,)), cones.shape + (3,))
    discs = dict(cone=cones.reshape(-1), colour=layers.reshape(-1, 3), light=disc_light(frame, cones, energy).reshape(-1))
    image = sensor_shader()(resolution, frame=frame, discs=discs)["image"]
    return np.round(np.clip(image, 0, 1) * 255).astype(np.uint8)


def side_view() -> solids.View:
    """The view of the box of space around the camera, turned for display."""
    corners = point(np.stack(np.meshgrid(*side_box, indexing="ij"), axis=-1).reshape(-1, 3))
    return solids.view([display >> corners], 0.0, side_elevation, side_azimuth, 0.02, side_pixels)


def scene_points(view: solids.View, scene: Point) -> tuple[solids.Quadric, np.ndarray]:
    """The scene's points as the side view shows them, coloured by their layers."""
    layers = np.broadcast_to(layer_colours.reshape((3,) + (1,) * (scene.ndim - 1) + (3,)), scene.shape + (3,))
    return solids.ball(display >> scene.reshape(-1), point_radius * view.pixel)[:, None], layers.reshape(-1, 3)


def draw_side(side: Scene, planes: Plane, radius: float, frame: Motor, legs: Point) -> np.ndarray:
    """The camera from the side, among the scene's points the side scene holds: the front lens as wide as the
    aperture and the rear lens, glass discs on their planes, the sensor where the frame places it, and the fan of rays
    from the subject through the aperture's rim, leg by leg; `[rows, columns, 3]` bytes."""
    # Each lens a disc about the axis on its plane: its centre where the plane meets the axis, its rim the directions
    # across the axis, turned around it, at the lens's half-height.
    centres = unit(planes[:2] ^ mv.y ^ mv.z)                                    # [lenses] Point-
    around = np.linspace(0, 2 * np.pi, n_rim, endpoint=False)
    turns = (mv.yz * (-around / 2)).exp()                                       # [rim] Motor+
    spokes = turns >> mv.y.dual()                                               # [rim] Point-
    radii = np.array([radius, rear_half_height])
    rims = centres[:, None] + spokes * radii[:, None]                           # [lenses, rim] Point-
    lenses = solids.plates(display >> rims, thickness)                          # [lenses, rim + 2] Plane+
    corners = frame >> point(np.array([[0, -1, -1], [0, 1, -1], [0, 1, 1], [0, -1, 1]]) * [0, *sensor_half_extents])
    sensor = solids.plates((display >> corners)[None], thickness)
    displayed_legs = display >> legs                                            # [planes + 1, rays] Point-
    rays = solids.rods(displayed_legs[:-1].reshape(-1), displayed_legs[1:].reshape(-1), ray_radius * side.view.pixel)
    return side.draw([(sensor, sensor_colour), (rays, ray_colour)], inks=[],
                     translucent=[(lenses, np.array(lens_colour), lens_opacity)], meshes=[])


def draw_stills(exposures: tuple[tuple, ...]) -> plt.Figure:
    """The sensor image of each still setting, side by side."""
    images = [expose_sensor(frame, cones, 1.0) for collineation, camera, frame, cones, planes, legs in exposures]
    gap = np.full((resolution[0], gap_pixels, 3), 255, dtype=np.uint8)
    return image_figure(np.concatenate([part for image in images for part in (gap, image)][1:], axis=1))


def animate_camera(states: Iterator[tuple], scene: Point) -> Iterator[np.ndarray]:
    """Frames of the sensor image over the camera seen from the side, one per state, among the scene's points, which
    are set up once."""
    view = side_view()
    side = Scene(view, np.ones(3), [scene_points(view, scene)], [], inks=[], translucent=[])
    for rear_at, focus_at, radius, (collineation, camera, frame, cones, planes, legs) in states:
        sensor = expose_sensor(frame, cones, (radius / reference_aperture)**2)
        side_image = draw_side(side, planes, radius, frame, legs)
        # The sensor image in the middle above the side view, on a white band as wide.
        pad = (side_image.shape[1] - sensor.shape[1]) // 2
        sensor = np.pad(sensor, ((0, 0), (pad, side_image.shape[1] - sensor.shape[1] - pad), (0, 0)), constant_values=255)
        yield np.concatenate([sensor, side_image], axis=0)
