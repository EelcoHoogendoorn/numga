"""Top-down rendering composites a GPU-shaded rolling ball over a static flat table and loop.

Each pixel over the ball reconstructs the point on the upper hemisphere and pulls it back into the body frame via `turn << point`, where pairings with the body planes colour the octant pattern, great circles, and contact trail.
"""

from __future__ import annotations

from collections.abc import Iterator
from functools import cache

import numpy as np

from numga import NumpyContext, stack
from numga.algebras import VGA3D
from numga.render import Camera, drawing, simplices
from numga.topology import Topology
from numga.wgsl import Shader, supersampled

ga = VGA3D
mv = NumpyContext(ga).multivector
Point = ga.gatype.antivector()                                                  # Antivector-
Rotor = ga.gatype.rotor()                                                       # Rotor+

table_colour = np.array([0.93, 0.91, 0.86])
loop_colour = np.array([0.55, 0.53, 0.50])
light_colour = np.array([0.96, 0.80, 0.30])
dark_colour = np.array([0.25, 0.45, 0.70])
circle_colour = np.array([0.15, 0.15, 0.18])
trail_colour = np.array([0.85, 0.20, 0.20])
n_trail_segments = 64
n_pixels = 300
loop_width = 1.0
margin = 1.3
ambient = 0.4

axes = stack((mv.x, mv.y))                                                      # [2] Vector+
planes = stack((mv.x, mv.y, mv.z))                                              # [3] Vector+
upward = mv.z.dual()                                                            # [] Point-
lamp = (mv.x * -0.4 + mv.y * 0.4 + mv.z * 0.82).dual()                          # [] Point-

source = '''
var<uniform> turn: Rotor;
var<uniform> centre: Point;
var<uniform> radius: f32;
var<uniform> half_width: f32;
var<storage, read> marks: array<Point>;

alias Sample = vec4<f32>;

// Project the offset from `start` onto `segment` by pairing with its orthogonal plane `segment.dual()`.
fn segment_distance(point: Point, start: Point, end: Point) -> f32 {
    let segment = end - start;
    let share = clamp((segment.dual() & (point - start)) / max(segment.dual() & segment, 1e-12), 0.0, 1.0);
    return (point - start - segment * share).norm();
}

fn sample(place: vec2<f32>) -> Sample {
    let offset = ((axes.dual() * (screen(place) * half_width)).sum() - centre) / radius;
    let offset_squared = offset.scalar_norm_squared();
    if (offset_squared > 1) {
        return vec4<f32>(0.0);
    }
    // Lift the table sample onto the upper hemisphere and pull it back into the ball's body frame.
    let surface_point = offset + upward * sqrt(1 - offset_squared);
    let body_point = turn << surface_point;
    let coordinates = planes & body_point;
    var colour = select(dark, light, coordinates.x * coordinates.y * coordinates.z > 0);
    let seam = min(abs(coordinates.x), min(abs(coordinates.y), abs(coordinates.z)));
    colour = mix(colour, circle, clamp(1 - seam / 0.04, 0.0, 1.0));
    var nearest = 2.0;
    for (var index = 1u; index < arrayLength(&marks); index++) {
        nearest = min(nearest, segment_distance(body_point, marks[index - 1], marks[index]));
    }
    colour = mix(colour, trail, clamp(1 - nearest / 0.05, 0.0, 1.0));
    // Pair the outward tangent plane `surface_point.dual()` with the lamp direction for diffuse shading.
    let illumination = ambient + (1 - ambient) * clamp(surface_point.dual() & lamp, 0.0, 1.0);
    return vec4<f32>(colour * illumination, 1.0);
}
'''


@cache
def ball_shader() -> Shader:
    """The cached WGSL shader shades the ball's octant pattern, great circles, and contact trail."""
    return Shader(
        supersampled() + source,
        ga,
        dict(Point=Point, Rotor=Rotor),
        dict(
            axes=axes,
            planes=planes,
            upward=upward,
            lamp=lamp,
            ambient=ambient,
            light=light_colour,
            dark=dark_colour,
            circle=circle_colour,
            trail=trail_colour,
        ),
    )


def table_background(path: Point, half_width: float) -> np.ndarray:
    """The background pass draws the static table and loop within `half_width` of the origin (`[n_pixels, n_pixels, 3]`)."""
    camera = Camera(mv.x, mv.y, 0 * mv.x, 0 * mv.x, 1.0, np.zeros(2), np.full(2, half_width), n_pixels)
    loop = drawing.composited([simplices.strokes(path, Topology.chains([len(path)]), loop_colour, loop_width, camera)])
    return table_colour * (1 - loop[..., 3:]) + loop[..., :3] * loop[..., 3:]


def ball_frame(
    background: np.ndarray,
    half_width: float,
    position: Point,
    turn: Rotor,
    trail: Point,
    radius: float,
) -> np.ndarray:
    """Alpha compositing layers the shaded ball at `position` and `turn` with `trail` over `background` (`[n_pixels, n_pixels, 3]`)."""
    indices = np.linspace(0, len(trail) - 1, min(len(trail), n_trail_segments + 1)).astype(int)
    marks = trail[indices]                                                      # [...] Point-
    ball = ball_shader()(
        (n_pixels, n_pixels),
        turn=turn,
        centre=position,
        radius=radius,
        half_width=half_width,
        marks=marks,
    )["image"]
    composite = ball[..., :3] + (1 - ball[..., 3:]) * background
    return np.round(np.clip(composite, 0, 1) * 255).astype(np.uint8)


def rolling_frames(
    path: Point,
    states: Iterator[tuple[Point, Rotor, Point]],
    radius: float,
) -> Iterator[np.ndarray]:
    """Yields `[n_pixels, n_pixels, 3]` RGB frames as the ball rolls through `states` along `path`."""
    # Pair the loop points with the horizontal coordinate planes to size the square view window.
    half_width = float((axes[:, None] & path.reshape(-1)).abs().max().to_array()) + margin * radius
    background = table_background(path, half_width)
    for position, turn, trail in states:
        yield ball_frame(background, half_width, position, turn, trail, radius)
