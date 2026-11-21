"""GPU sensor shader and three-dimensional side-view rendering for the two-lens zoom camera.

Each sensor pixel maps to a point on the sensor plane, and every scene point's image cone illuminates the pixel according to its implicit quadric form `cone(point) & point`, which is negative inside the blur conic. Below the sensor strip, the side view renders the lens discs, tilted sensor plate, scene layers, and traced aperture rays directly in projective geometric algebra.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from functools import cache

import matplotlib.pyplot as plt
import numpy as np

from examples.optics.lens_camera.core import Motor, Plane, Point, Quadric, Scalar, ga, home, mv, origin, scene_point, unit_point
from examples.util.panels import image_figure
from numga import dim, stack
from numga.render import Scene, solids
from numga.wgsl import Shader, supersampled

# Sensor resolution in rows by columns, exposure brightness scale, and diffraction-like logistic edge width in pixels.
resolution = (168, 216)
brightness = 1.36e-3
edge_pixels = 1.0
sensor_half_extents = (0.225, 0.175)
edge_width = edge_pixels * 2 * sensor_half_extents[1] / resolution[0]
# Home-frame sensor coordinate planes along y (right) and z (up).
sensor_axes = stack((mv.y, mv.z))                                               # [2] Plane+
reference_aperture = 0.45
rear_half_height = 0.6
# Depth-layer RGB colours (red, green, blue) and white separator width between stills.
layer_colours = np.eye(3)
gap_pixels = 40
# Side-view camera angles, bounding box, resolution, element geometry, and colours.
side_elevation, side_azimuth = np.radians([14.0, -18.0])
side_box = ((-3.6, 2.6), (-1.0, 1.0), (-0.6, 0.6))
side_pixels = 216
point_radius, ray_radius = 2.0, 0.6
n_rim, thickness, lens_opacity = 32, 0.02, 0.3
lens_colour, sensor_colour, ray_colour = (0.55, 0.7, 0.85), (0.3, 0.3, 0.32), (1.0, 0.5, 0.05)
# Quarter turn about the optical axis mapping the scene's +y vertical onto the side view's +z vertical.
display_turn = (mv.yz * (np.pi / 4)).exp()                                      # [] Rotor+


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
    """Compiles the supersampled WGSL sensor shader with the home sensor geometry bound as constants."""
    return Shader(
        supersampled() + sensor_source,
        ga,
        dict(Point=Point, Plane=Plane, Motor=Motor, Quadric=Quadric),
        dict(origin=origin, home=home, axes=sensor_axes, half_height=sensor_half_extents[1], edge=edge_width),
    )


def disc_light(frame: Motor, cones: Quadric, energy: float) -> np.ndarray:
    """Dividing `energy` by each blur ellipse's area (from the 3D and 2D determinants of `cones` on `frame`) plus the logistic spot floor gives sensor irradiance (`[...]`)."""
    centre, right, up = frame >> origin, frame >> mv.y.dual(), frame >> mv.z.dual()  # [] Point-, Point-, Point-
    whole = (cones(centre) ^ cones(right) ^ cones(up)) & (centre & right & up)  # [...] Scalar+
    directions = (cones(right) ^ cones(up)) & (right & up)                      # [...] Scalar+
    area = np.pi * np.abs(whole.to_array()) / directions.to_array()**1.5
    spot = 2 / 3 * np.pi**3 * edge_width**2
    return brightness * energy * 4 * sensor_half_extents[0] * sensor_half_extents[1] / (area + spot)


def sensor_image(frame: Motor, cones: Quadric, energy: float) -> np.ndarray:
    """Renders `[rows, columns, 3]` `uint8` RGB sensor pixels summing the equal-energy blur discs of `cones` across the three coloured depth layers."""
    layers = np.broadcast_to(layer_colours.reshape((3,) + (1,) * (cones.ndim - 1) + (3,)), cones.shape + (3,))
    discs = dict(
        cone=cones.reshape(-1),
        colour=layers.reshape(-1, 3),
        light=disc_light(frame, cones, energy).reshape(-1),
    )
    image = sensor_shader()(resolution, frame=frame, discs=discs)["image"]
    return np.round(np.clip(image, 0, 1) * 255).astype(np.uint8)


def side_view() -> solids.View:
    """Frames an orthographic 3D `solids.View` around `side_box` rotated by `display_turn`."""
    corners = scene_point(np.stack(np.meshgrid(*side_box, indexing="ij"), axis=-1).reshape(-1, 3))  # [8] Point-
    return solids.view([display_turn >> corners], 0.0, side_elevation, side_azimuth, 0.02, side_pixels)


def scene_spheres(view: solids.View, scene: Point) -> tuple[solids.Quadric, np.ndarray]:
    """Builds coloured 3D sphere quadrics for `scene` (`[...] Point-`) in the side view."""
    layers = np.broadcast_to(layer_colours.reshape((3,) + (1,) * (scene.ndim - 1) + (3,)), scene.shape + (3,))
    return solids.ball(display_turn >> scene.reshape(-1), point_radius * view.pixel)[:, None], layers.reshape(-1, 3)


def side_image(side: Scene, planes: Plane, radius: Scalar, frame: Motor, ray_vertices: Point) -> np.ndarray:
    """Renders `[rows, columns, 3]` `uint8` RGB pixels of the two translucent lenses, the sensor plate, and the traced ray segments inside `side`."""
    centres = unit_point(planes[:2] ^ mv.y ^ mv.z)                              # [2] Point-, 1 | length
    around = np.linspace(0, 2 * np.pi, n_rim, endpoint=False)
    turns = (mv.yz * (-around / 2)).exp()                                       # [n_rim] Rotor+
    spokes = turns >> mv.y.dual()                                               # [n_rim] Point-
    radii = stack((radius, mv.scalar(rear_half_height, unit=dim.length)))       # [2] Scalar+, length
    rims = centres[:, None] + spokes * radii[:, None]                           # [2, n_rim] Point-, 1 | length
    lenses = solids.plates(display_turn >> rims, thickness)                     # [2, n_rim + 2] Plane+
    corners = frame >> scene_point(
        np.array([[0, -1, -1], [0, 1, -1], [0, 1, 1], [0, -1, 1]]) * [0, *sensor_half_extents]
    )                                                                           # [4] Point-
    sensor = solids.plates((display_turn >> corners)[None], thickness)          # [1, 6] Plane+
    displayed_vertices = display_turn >> ray_vertices                           # [4, ...] Point-
    rays = solids.rods(
        displayed_vertices[:-1].reshape(-1), displayed_vertices[1:].reshape(-1), ray_radius * side.view.pixel,
    )
    return side.draw(
        [(sensor, sensor_colour), (rays, ray_colour)],
        inks=[],
        translucent=[(lenses, np.array(lens_colour), lens_opacity)],
        meshes=[],
    )


def stills_figure(exposures: tuple[tuple, ...]) -> plt.Figure:
    """Renders the sensor images of `exposures` side by side separated by white margins."""
    images = [sensor_image(frame, cones, 1.0) for _, _, frame, cones, _, _ in exposures]
    gap = np.full((resolution[0], gap_pixels, 3), 255, dtype=np.uint8)
    return image_figure(np.concatenate([part for image in images for part in (gap, image)][1:], axis=1))


def camera_animation(scene: Point, states: Iterable[tuple]) -> Iterator[np.ndarray]:
    """Yields composite RGB frames stacking the live sensor image above the 3D side view for each state in `states`."""
    view = side_view()
    side = Scene(view, np.ones(3), [scene_spheres(view, scene)], [], inks=[], translucent=[])
    reference = mv.scalar(reference_aperture, unit=dim.length)
    for _, _, radius, (_, _, frame, cones, planes, ray_vertices) in states:
        sensor = sensor_image(frame, cones, float(((radius / reference)**2).to_array()))
        below = side_image(side, planes, radius, frame, ray_vertices)
        pad = (below.shape[1] - sensor.shape[1]) // 2
        sensor = np.pad(sensor, ((0, 0), (pad, below.shape[1] - sensor.shape[1] - pad), (0, 0)), constant_values=255)
        yield np.concatenate([sensor, below], axis=0)
