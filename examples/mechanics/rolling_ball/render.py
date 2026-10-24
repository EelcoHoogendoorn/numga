"""Drawing for the rolling ball: the table seen from above, with the loop and the ball on it; the ball
painted on the GPU with a pattern fixed to it, and with the trail its contact point has left on it.

Each sample of the picture is a point of the table seen from above; under the ball it is a point of the ball's upper
half, and carried back into the ball's own frame by its turn, `turn << point`, the point of the ball that is there,
whose pairings with the planes of the ball's own frame pick its colour. The table and the loop do not move: they are drawn once, and every frame lays the
ball over them.
"""

from __future__ import annotations

from collections.abc import Iterator
from functools import cache

import matplotlib.pyplot as plt
import numpy as np

from numga import NumpyContext, stack
from numga.algebras import VGA3D
from examples.util.animation import capture
from examples.util.wgsl import Shader, supersampled

mv = NumpyContext(VGA3D).multivector

TABLE = np.array([0.93, 0.91, 0.86])
LOOP = np.array([0.55, 0.53, 0.50])
# The ball's two pattern colours, one for each sign of the product of its three coordinates, the dark
# great circles where a coordinate vanishes, and the trail of its contact point.
LIGHT = np.array([0.96, 0.80, 0.30])
DARK = np.array([0.25, 0.45, 0.70])
CIRCLE = np.array([0.15, 0.15, 0.18])
TRAIL = np.array([0.85, 0.20, 0.20])
# At most this many segments of the trail are painted, evenly spaced along it; the pixels across the picture, and the
# margin around the loop and the ball, as a share of the ball's radius.
TRAIL_SEGMENTS = 64
N_PIXELS = 500
MARGIN = 1.3
# The planes whose pairings with a point of the table are how far across and up the picture it lies; the point so far
# along each is `(axes.dual() * at).sum()`.
axes = stack((mv.x, mv.y))                                                    # [2] Vector
# The direction the light comes from, the upper left, and the share of the light that falls everywhere.
lamp = (mv.x * -0.4 + mv.y * 0.4 + mv.z * 0.82).dual()                        # [] Point
AMBIENT = 0.4

source = """
// The ball's turn, its centre on the table and its radius; half the width of the table seen; and the points of the
// ball its contact point has met, in the ball's own frame, a segment between each and the next.
var<uniform> turn: Rotor;
var<uniform> centre: Point;
var<uniform> radius: f32;
var<uniform> half_width: f32;
var<storage, read> marks: array<Point>;

// The colour, and its coverage.
alias Sample = vec4<f32>;

// How far a point is from the segment between two others: how far along the segment it lies, as a share of the
// segment, is its offset from the start paired with the plane square to the segment, `along.dual()`.
fn from_segment(point: Point, start: Point, end: Point) -> f32 {
    let along = end - start;
    let share = clamp((along.dual() & (point - start)) / max(along.dual() & along, 1e-12), 0.0, 1.0);
    return (point - start - along * share).norm();
}

fn sample(place: vec2<f32>) -> Sample {
    // Across the ball from its centre, in radii.
    let across = ((axes.dual() * (screen(place) * half_width)).sum() - centre) / radius;
    let squared = across.scalar_norm_squared();
    if (squared > 1) {
        return vec4<f32>(0.0);
    }
    // The point of the ball's upper half under the sample, and the point of the ball that is there, in its frame,
    // read by its pairings with the planes of that frame.
    let point = across + mv.z.dual() * sqrt(1 - squared);
    let body = turn << point;
    let x = mv.x & body;
    let y = mv.y & body;
    let z = mv.z & body;
    var colour = select(dark, light, x * y * z > 0);
    // The great circles where a coordinate vanishes, a little wide.
    colour = mix(colour, circle, clamp(1 - min(abs(x), min(abs(y), abs(z))) / 0.04, 0.0, 1.0));
    // The trail: the nearest of its segments, within its width.
    var nearest = 2.0;
    for (var index = 1u; index < arrayLength(&marks); index++) {
        nearest = min(nearest, from_segment(body, marks[index - 1], marks[index]));
    }
    colour = mix(colour, trail, clamp(1 - nearest / 0.05, 0.0, 1.0));
    // Lit by the lamp, the ball's tangent plane at the point, square to it, paired with the lamp's direction; and a
    // share of the light everywhere.
    let lit = ambient + (1 - ambient) * clamp(point.dual() & lamp, 0.0, 1.0);
    return vec4<f32>(colour * lit, 1.0);
}
"""


@cache
def shader() -> Shader:
    """The shader of the ball: the frame of the picture, the lamp and the colours, given as its constants."""
    return Shader(supersampled() + source, VGA3D, dict(Point=VGA3D.gatype.antivector(), Rotor=VGA3D.gatype.rotor()),
                  dict(axes=axes, lamp=lamp, ambient=AMBIENT, light=LIGHT, dark=DARK, circle=CIRCLE, trail=TRAIL))


def coordinates(points) -> np.ndarray:
    """How far points lie across and up the table, their pairings with its axes, along the last axis."""
    return (axes & points[..., None]).to_array()


def table(path, reach: float) -> np.ndarray:
    """The table from above with the loop on it, `[pixels, pixels, 3]` in [0, 1], within `reach` of its middle: drawn
    once, for every frame to lay the ball over."""
    xy = coordinates(path)
    figure, ax = plt.subplots(figsize=(N_PIXELS / 100, N_PIXELS / 100), dpi=100, facecolor=TABLE)
    ax.set_facecolor(TABLE)
    ax.plot(*xy.T, color=LOOP, linewidth=1.0)
    ax.set(xlim=(-reach, reach), ylim=(-reach, reach), aspect="equal")
    ax.axis("off")
    figure.subplots_adjust(0, 0, 1, 1)
    image = capture(figure) / 255
    plt.close(figure)
    return image


def frame(background: np.ndarray, reach: float, position, turn, trail, radius: float) -> np.ndarray:
    """One frame: the ball, painted where it stands on the table, over the table and the loop, `[pixels, pixels, 3]`
    bytes."""
    marks = trail[np.linspace(0, len(trail) - 1, min(len(trail), TRAIL_SEGMENTS + 1)).astype(int)]
    ball = shader()((N_PIXELS, N_PIXELS), turn=turn, centre=position, radius=radius, half_width=reach,
                    marks=marks)["image"]
    # The ball's colour already weighted by its coverage, over what the table shows where it does not cover.
    shown = ball[..., :3] + (1 - ball[..., 3:]) * background
    return np.round(np.clip(shown, 0, 1) * 255).astype(np.uint8)


def animate(path, rolling, radius: float) -> Iterator[np.ndarray]:
    """One frame per state of the ball: its position, its turn and its contact point's trail, all over one view of the
    table, the loop drawn once."""
    reach = np.abs(coordinates(path)).max() + MARGIN * radius
    background = table(path, reach)
    for position, turn, trail in rolling:
        yield frame(background, reach, position, turn, trail, radius)
