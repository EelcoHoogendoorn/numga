"""Sensor images and side views of the lens camera: drawing only.

The sensor images are drawn on the GPU by a shader in numga's dialect: each sensor pixel is a point of the sensor
plane, and every scene point's image cone lights it by its form there, `cone(point) & point`, negative inside the
blur disc. Coordinates leave the algebra here: a point is read by its pairings with the planes x, y and z, over its
weight.
"""

from __future__ import annotations

from collections.abc import Iterator
from functools import cache

import matplotlib.pyplot as plt
import numpy as np

from numga import stack

from examples.optics.lens_camera.core import Motor, Plane, Point, Quadric, ga, home, mv, origin, point
from examples.util.wgsl import Shader, image_figure, solids, supersampled
from examples.util.wgsl.scene import Scene

# Sensor pixels, rows by columns.
RESOLUTION = (168, 216)
# How bright a disc's light shows, per unit of its energy over the sensor's area, and the width of a disc's edge, a
# crude diffraction limit, in pixels.
BRIGHTNESS = 1.36e-3
EDGE_PIXELS = 1.0
# Half extents of the sensor window in its own frame, and the width of a disc's edge on it.
SENSOR = (0.225, 0.175)
EDGE = EDGE_PIXELS * 2 * SENSOR[1] / RESOLUTION[0]
# The planes whose pairings with a point of the sensor in its home frame are how far right and up it lies; the point
# so far along each is `origin + (axes.dual() * place).sum()`.
axes = stack((mv.y, mv.z))                                                      # [2] Plane
# The aperture whose exposure is the reference brightness, and the half-heights the front and rear lens
# are drawn with in the side view.
REFERENCE_APERTURE = 0.45
LENS_HALF_HEIGHTS = [0.6, 0.35]
# The colour each depth layer of the scene lights on the sensor, red, green and blue, in the order of the layers, and
# the white between the stills.
LAYER_COLOURS = np.eye(3)
GAP_PIXELS = 40
# The side view, drawn on the GPU: its elevation and azimuth in radians, azimuth zero looking along +y once the
# scene's y is turned up; the box of space it frames, along the optical axis x, up y and across z; its pixels across;
# and in pixels, the radii of the scene's points and of the rays; the corners around a lens's rim, the thickness of
# the lenses and the sensor, and how much of what lies behind it a lens's face hides; the colours of the lenses, the
# sensor and the rays.
SIDE_ELEVATION, SIDE_AZIMUTH = np.radians([14.0, -18.0])
SIDE_BOX = ((-3.6, 2.6), (-1.0, 1.0), (-0.6, 0.6))
SIDE_PIXELS = 216
POINT_RADIUS, RAY_RADIUS = 2.0, 0.6
N_RIM, THICKNESS, LENS_OPACITY = 32, 0.02, 0.3
LENS, SENSOR_COLOUR, RAY = (0.55, 0.7, 0.85), (0.3, 0.3, 0.32), (1.0, 0.5, 0.05)
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
def xyz(points: Point) -> np.ndarray:
    """(..., 3) Euclidean coordinates of finite points: their pairings with the planes x, y and z, over
    their weights."""
    return ((stack((mv.x, mv.y, mv.z)) & points[..., None]) / (mv.w & points)[..., None]).to_array()


@cache
def sensor_shader() -> Shader:
    """The shader of a sensor image, its frame of axes, the sensor's half height and the width of a disc's edge given
    as its constants."""
    return Shader(supersampled() + sensor_source, ga, dict(Point=Point, Plane=Plane, Motor=Motor, Quadric=Quadric),
                  dict(origin=origin, home=home, axes=axes, half_height=SENSOR[1], edge=EDGE))


def disc_light(frame: Motor, cones: Quadric, energy: float) -> np.ndarray:
    """Each scene point's light per unit area of the sensor `[points]`: its energy, scaled by the aperture area, spread
    over its blur disc and the spot its edge spreads a point over.

    The disc's area comes from the determinants of the cone's form on the sensor's points and on its directions, each
    the join of those points paired with the meet of their polars: an ellipse's area is pi times the first over the
    second to the power three halves. A logistic edge about the first-order distance spreads a point over two thirds
    of pi cubed edge widths squared.
    """
    centre, right, up = frame >> origin, frame >> mv.y.dual(), frame >> mv.z.dual()     # [] Point
    whole = (centre & right & up) & (cones(centre) ^ cones(right) ^ cones(up))         # [depth, height, width] Scalar
    directions = (right & up) & (cones(right) ^ cones(up))                             # [depth, height, width] Scalar
    area = np.pi * np.abs(whole.to_array()) / directions.to_array()**1.5
    spot = 2 / 3 * np.pi**3 * EDGE**2
    return BRIGHTNESS * energy * 4 * SENSOR[0] * SENSOR[1] / (area + spot)


def expose_sensor(frame: Motor, cones: Quadric, energy: float) -> np.ndarray:
    """The sensor image, `[rows, columns, 3]` bytes: every scene point's blur disc in the colour of its depth layer,
    the light of all of them summed. Each disc deposits the same total energy, scaled by the aperture area, so a point
    in focus is a bright dot and a defocused one a dim wide disc."""
    layers = np.broadcast_to(LAYER_COLOURS.reshape((3,) + (1,) * (cones.ndim - 1) + (3,)), cones.shape + (3,))
    discs = dict(cone=cones.reshape(-1), colour=layers.reshape(-1, 3), light=disc_light(frame, cones, energy).reshape(-1))
    image = sensor_shader()(RESOLUTION, frame=frame, discs=discs)["image"]
    return np.round(np.clip(image, 0, 1) * 255).astype(np.uint8)


def side_view() -> solids.View:
    """The view of the box of space around the camera, turned for display."""
    corners = point(np.stack(np.meshgrid(*SIDE_BOX, indexing="ij"), axis=-1).reshape(-1, 3))
    return solids.view([display >> corners], 0.0, SIDE_ELEVATION, SIDE_AZIMUTH, 0.02, SIDE_PIXELS)


def scene_points(view: solids.View, scene: Point) -> tuple[solids.Quadric, np.ndarray]:
    """The scene's points as the side view shows them, coloured by their layers."""
    layers = np.broadcast_to(LAYER_COLOURS.reshape((3,) + (1,) * (scene.ndim - 1) + (3,)), scene.shape + (3,))
    return solids.ball(display >> scene.reshape(-1), POINT_RADIUS * view.pixel)[:, None], layers.reshape(-1, 3)


def draw_side(side: Scene, planes: Plane, radius: float, frame: Motor, legs: Point) -> np.ndarray:
    """The camera from the side, among the scene's points the side scene holds: the front lens as wide as the
    aperture and the rear lens, glass discs on their planes, the sensor where the frame places it, and the fan of rays
    from the subject through the aperture's rim, leg by leg; `[rows, columns, 3]` bytes."""
    # Each lens a disc about the axis on its plane: its centre where the plane meets the axis.
    centres = xyz(planes[:2] ^ mv.y ^ mv.z)                                 # [lenses, 3]
    around = np.linspace(0, 2 * np.pi, N_RIM, endpoint=False)
    radii = np.array([radius, LENS_HALF_HEIGHTS[0]])
    rims = centres[:, None] + radii[:, None, None] * np.stack([np.zeros_like(around), np.cos(around), np.sin(around)], axis=-1)
    lenses = solids.plates(display >> point(rims), THICKNESS)               # [lenses, corners + 2] Plane
    corners = frame >> point(np.array([[0, -1, -1], [0, 1, -1], [0, 1, 1], [0, -1, 1]]) * [0, *SENSOR])
    sensor = solids.plates((display >> corners)[None], THICKNESS)
    shown = display >> legs                                                    # [plane + 1, ray] Point
    rays = solids.rods(shown[:-1].reshape(-1), shown[1:].reshape(-1), RAY_RADIUS * side.view.pixel)
    return side.draw([(sensor, SENSOR_COLOUR), (rays, RAY)], inks=[],
                     translucent=[(lenses, np.array(LENS), LENS_OPACITY)], meshes=[])


def draw_stills(exposures) -> plt.Figure:
    """The sensor image of each still setting, side by side."""
    images = [expose_sensor(frame, cones, 1.0) for collineation, camera, frame, cones, planes, legs in exposures]
    gap = np.full((RESOLUTION[0], GAP_PIXELS, 3), 255, dtype=np.uint8)
    return image_figure(np.concatenate([part for image in images for part in (gap, image)][1:], axis=1))


def animate_camera(states, scene: Point) -> Iterator[np.ndarray]:
    """Frames of the sensor image over the camera seen from the side, one per state, among the scene's points, which
    are set up once."""
    view = side_view()
    side = Scene(view, np.ones(3), [scene_points(view, scene)], [], inks=[], translucent=[])
    for rear_at, focus_at, radius, (collineation, camera, frame, cones, planes, legs) in states:
        sensor = expose_sensor(frame, cones, (radius / REFERENCE_APERTURE)**2)
        seen = draw_side(side, planes, radius, frame, legs)
        # The sensor image in the middle above the side view, on a white band as wide.
        pad = (seen.shape[1] - sensor.shape[1]) // 2
        sensor = np.pad(sensor, ((0, 0), (pad, seen.shape[1] - sensor.shape[1] - pad), (0, 0)), constant_values=255)
        yield np.concatenate([sensor, seen], axis=0)
