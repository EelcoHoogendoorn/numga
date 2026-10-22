"""Drawing for the rolling ball: the table seen from above, with the loop and the ball on it; the ball
painted on the GPU with a pattern fixed to it, and with the trail its contact point has left on it.

Each sample of the ball's picture, seen from above, is a point of the ball's upper half; carried back into the ball's
own frame by its turn, `turn << point`, it is the point of the ball that is there, whose coordinates pick its colour.
"""

from __future__ import annotations

from functools import cache

import matplotlib.pyplot as plt
import numpy as np

from numga.algebras import VGA3D
from examples.animation import capture
from examples.wgsl import Shader, supersampled, vec3

TABLE = np.array([0.93, 0.91, 0.86])
LOOP = np.array([0.55, 0.53, 0.50])
# The ball's two pattern colours, one for each sign of the product of its three coordinates, the dark
# great circles where a coordinate vanishes, and the trail of its contact point.
LIGHT = np.array([0.96, 0.80, 0.30])
DARK = np.array([0.25, 0.45, 0.70])
CIRCLE = np.array([0.15, 0.15, 0.18])
TRAIL = np.array([0.85, 0.20, 0.20])
# At most this many trail points are painted, evenly spaced along it; the pixels across the ball's picture.
TRAIL_POINTS = 240
N_PIXELS = 300

source = supersampled() + """
// The ball's turn, and the points of the ball its contact point has met, in the ball's own frame.
var<uniform> turn: Rotor;
var<storage, read> marks: array<Vector>;

// The colour, and its coverage.
alias Sample = vec4<f32>;

fn sample(place: vec2<f32>) -> Sample {
    let across = screen(place);
    let squared = dot(across, across);
    if (squared > 1) {
        return vec4<f32>(0.0);
    }
    // The point of the ball's upper half under the sample, and the point of the ball that is there, in its frame.
    let point = mv.x * across.x + mv.y * across.y + mv.z * sqrt(1 - squared);
    let body = turn << point;
    let x = body.dot(mv.x);
    let y = body.dot(mv.y);
    let z = body.dot(mv.z);
    var colour = select(DARK, LIGHT, x * y * z > 0);
    // The great circles where a coordinate vanishes, a little wide.
    colour = mix(colour, CIRCLE, clamp(1 - min(abs(x), min(abs(y), abs(z))) / 0.04, 0.0, 1.0));
    // The trail: the nearest mark, within its width.
    var nearest = 4.0;
    for (var index = 0u; index < arrayLength(&marks); index++) {
        let offset = body - marks[index];
        nearest = min(nearest, offset.dot(offset));
    }
    colour = mix(colour, TRAIL, clamp(1 - sqrt(nearest) / 0.05, 0.0, 1.0));
    // Lit from the upper left, and a share of the light everywhere.
    let lit = 0.4 + 0.6 * clamp(point.dot(mv.x * -0.4 + mv.y * 0.4 + mv.z * 0.82), 0.0, 1.0);
    return vec4<f32>(colour * lit, 1.0);
}
"""
source = (source.replace("LIGHT", vec3(LIGHT)).replace("DARK", vec3(DARK)).replace("CIRCLE", vec3(CIRCLE))
          .replace("TRAIL", vec3(TRAIL)))


@cache
def shader() -> Shader:
    return Shader(source, VGA3D, dict(Vector=VGA3D.gatype.vector(), Rotor=VGA3D.gatype.rotor()))


def coordinates(points) -> np.ndarray:
    """The x, y and z coefficients of vectors, along the last axis."""
    return points.cast(points.algebra.subspace("x y z")).kernel


def ball(turn, trail) -> np.ndarray:
    """The visible half of the ball as RGBA pixels: each pixel coloured by the point of the ball under it, in the
    ball's own frame, and the trail `[points]` of its contact point; transparent outside the ball."""
    marks = trail[np.linspace(0, len(trail) - 1, min(len(trail), TRAIL_POINTS)).astype(int)]
    image = shader()((N_PIXELS, N_PIXELS), turn=turn, marks=marks)["image"]
    alpha = image[..., 3:]
    return np.concatenate([image[..., :3] / np.maximum(alpha, 1e-12), alpha], axis=-1)


def scene(path, position, image: np.ndarray, radius: float) -> plt.Figure:
    """The table from above: the loop, and the ball's image at its position."""
    xy = coordinates(path)[:, :2]
    centre = coordinates(position)[:2]
    reach = np.abs(xy).max() + 1.3 * radius
    figure, ax = plt.subplots(figsize=(5, 5), facecolor=TABLE)
    ax.set_facecolor(TABLE)
    ax.plot(*xy.T, color=LOOP, linewidth=1.0)
    ax.imshow(image, extent=(centre[0] - radius, centre[0] + radius, centre[1] - radius, centre[1] + radius),
              interpolation="bilinear", zorder=3)
    ax.set(xlim=(-reach, reach), ylim=(-reach, reach), aspect="equal")
    ax.axis("off")
    figure.subplots_adjust(0, 0, 1, 1)
    return figure


def animate(path, rolling, radius: float) -> list[np.ndarray]:
    """One frame per state of the ball: its position, its turn and its contact point's trail."""
    frames = []
    for position, turn, trail in rolling:
        figure = scene(path, position, ball(turn, trail), radius)
        frames.append(capture(figure))
        plt.close(figure)
    return frames
