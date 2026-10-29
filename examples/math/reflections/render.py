"""Drawing for the reflection notebooks: a kaleidoscope painted from its folded points, a sphere and
disks of tiles coloured by how many reflections folded each pixel, and the 600-cell projected from
the three-sphere."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import stack
from examples.util.wgsl import solids, strokes

background = np.array([0.06, 0.07, 0.09])
# The two tiles of a reflection: one colour for an even number of reflections, one for an odd number.
even = np.array([0.93, 0.83, 0.62])
odd = np.array([0.35, 0.55, 0.72])
line = np.array([0.12, 0.12, 0.16])
# The pixels of background between panels side by side.
gap = 6
# The colours of a shader's tiles, and the width of their sides, as a share of a point's pairings with the three
# mirrors, and their darkness, as the shader's constants.
tiling_colours = dict(even=even, odd=odd, line=line, background=background, side_width=0.02, side_strength=0.5)
# Beads tumbling in the kaleidoscope, scattered over the disk: centre radius, angle, speed and size of
# each, and its colour, drawn from a short palette.
n_beads = 40
_sampler = np.random.default_rng(7)
beads = np.stack([
    np.sqrt(_sampler.uniform(0.0, 1.0, n_beads)), _sampler.uniform(0.0, 2 * np.pi, n_beads),
    _sampler.choice([-2, -1, 1, 2], n_beads), _sampler.uniform(0.04, 0.13, n_beads),
], axis=-1)
palette = np.array([
    [0.91, 0.30, 0.24], [0.98, 0.77, 0.19], [0.18, 0.62, 0.48], [0.56, 0.27, 0.68],
    [0.20, 0.47, 0.85], [0.95, 0.55, 0.65], [0.99, 0.97, 0.90],
])
bead_colours = palette[np.arange(n_beads) % len(palette)]
# The 600-cell: its vertices' colour; its edges' width and its vertices' radius in pixels; the pixels across its
# picture, half the picture's width about the middle, in the projected space, and the tilt of the view.
vertex_colour = odd
edge_width, vertex_radius = 0.8, 1.6
n_pixels, half_width = 300, 0.6
tilt = 0.45


def disk(resolution: int) -> tuple[np.ndarray, np.ndarray]:
    """Pixel centres of a square view of the unit disk: the two coordinates, each [rows, columns]."""
    centres = (np.arange(resolution) + 0.5) / resolution * 2 - 1
    u, v = np.meshgrid(centres, -centres)
    return u, v


def to_image(colours: np.ndarray) -> np.ndarray:
    """RGB in [0, 1] to 8-bit pixels."""
    return (np.clip(colours, 0.0, 1.0) * 255).round().astype(np.uint8)


def side_by_side(images: np.ndarray) -> np.ndarray:
    """Panels along the leading axis joined left to right, with a gap of background between them."""
    spacer = np.broadcast_to(to_image(background), (images.shape[1], gap, 3))
    return np.concatenate([part for image in images for part in (image, spacer)][:-1], axis=1)


# --- kaleidoscope -----------------------------------------------------------------------
def kaleidoscope(folded, time: float) -> np.ndarray:
    """The beads at the given time, painted at every folded point: [..., rows, columns, 3] pixels."""
    mv = folded.context.multivector
    xy = (stack((mv.x, mv.y)) & folded[..., None]).to_array().astype(np.float32)   # [..., rows, columns, 2]
    # The beads from the top down: each is painted over the ones listed before it.
    radius, angle, speed, size = beads[::-1].T.astype(np.float32)
    centres = radius[:, None] * np.stack([np.cos(angle + speed * time), np.sin(angle + speed * time)], axis=-1)
    # Squared distances to the bead centres through one matrix product over all pixels.
    squared = (xy ** 2).sum(axis=-1, keepdims=True) - 2 * xy @ centres.T + (centres ** 2).sum(axis=-1)
    distance = np.sqrt(np.maximum(squared, 0.0))                               # [..., rows, columns, beads]
    cover = np.clip((size - distance) / 0.02, 0.0, 1.0)
    # What shows of a bead is its cover times what the beads above it leave uncovered, and the background
    # shows where all of them leave it uncovered.
    uncovered = np.cumprod(1 - cover, axis=-1)                                 # [..., rows, columns, beads]
    shown = cover * np.concatenate([np.ones_like(uncovered[..., :1]), uncovered[..., :-1]], axis=-1)
    return to_image(uncovered[..., -1:] * background + shown @ bead_colours[::-1])


def animate_kaleidoscope(folded, frames: int) -> Iterator[np.ndarray]:
    """The beads tumbling once around, each frame the kaleidoscopes side by side."""
    for time in np.linspace(0, 2 * np.pi, frames, endpoint=False):
        yield side_by_side(kaleidoscope(folded, time))


# --- tiles ------------------------------------------------------------------------------
def tiles(flips, nearness: np.ndarray) -> np.ndarray:
    """Colours of tiles by the parity of their reflections, darkened along the mirrors, where the
    nearness to the closest mirror runs from one on it to zero away from it."""
    parity = (np.round(flips.to_array()) % 2)[..., None]
    colours = even + parity * (odd - even)
    return colours + nearness[..., None] * (line - colours)


def sphere(screen, folded, flips, mirrors, outline) -> np.ndarray:
    """The visible half of a sphere tiled by reflections, lit from the viewer's upper left, with the
    tiles' sides drawn thin and the sides on the outlining mirror, one of the mirrors, drawn bold."""
    mv = screen.context.multivector
    view = (stack((mv.x, mv.y, mv.z)) & screen[..., None]).to_array()          # [rows, columns, 3]
    lit = 0.35 + 0.65 * np.clip(view @ np.array([-0.4, 0.5, 0.77]), 0.0, 1.0)
    sides = np.abs((mirrors & folded[..., None]).to_array())                  # [rows, columns, mirrors]
    thin = np.clip(1 - sides.min(axis=-1) / 0.012, 0.0, 1.0)
    bold = np.clip(1 - np.abs((outline & folded).to_array()) / 0.03, 0.0, 1.0)
    colours = tiles(flips, np.maximum(0.6 * thin, bold)) * lit[..., None]
    inside = (view[..., :2] ** 2).sum(axis=-1) < 1
    return to_image(np.where(inside[..., None], colours, background))


def animate_sphere(screen, turning, mirrors, outline) -> Iterator[np.ndarray]:
    """One frame per folded view of the turning sphere."""
    for folded, flips in turning:
        yield sphere(screen, folded, flips, mirrors, outline)


def tiling_colouring() -> str:
    """WGSL for the colour of a tiling's sample, `tile`, from the constants `tiling_colours` names, and the
    `background` outside the disk."""
    return """
// The colour of a tile folded by a count of reflections, by its parity, darkened along the tile's sides, where the
// share of the nearest mirror in a point's pairings with the three falls towards zero.
fn tile(flips: u32, share: f32) -> vec3<f32> {
    let colour = mix(even, odd, f32(flips % 2u));
    return mix(colour, line, side_strength * clamp(1 - share / side_width, 0.0, 1.0));
}
"""


def animate_tilings(moving) -> Iterator[np.ndarray]:
    """One frame per step of the motion: the shaders' disks of tiles of every geometry side by side."""
    for images in moving:
        yield side_by_side(to_image(np.stack(images)))


# --- the 600-cell -----------------------------------------------------------------------
def distinct(points: np.ndarray) -> np.ndarray:
    """The distinct rows of an array of points, up to round-off: one of each set of rows equal to six
    decimals, in sorted order."""
    rounded = np.round(points * 1e6).astype(np.int64)                          # [points, coordinates]
    # Equal rows sort next to each other, the first of them first; a row starts a new set where it differs
    # from the one before.
    order = np.lexsort(rounded.T[::-1])
    first = order[np.concatenate([[True], (np.diff(rounded[order], axis=0) != 0).any(axis=-1)])]
    return points[np.sort(first)]


def polytope(spinors) -> np.ndarray:
    """The distinct unit spinors as points of the three-sphere, in perspective from beyond the scalar
    axis into three dimensions and from there seen from slightly above, `[pixels, pixels, 3]` bytes:
    the points traced as balls, and edges joining nearest neighbours drawn as strokes, lit on the GPU."""
    mv = spinors.context.multivector
    # A spinor's coefficients on one and on the planes, read by its scalar product with their reverses.
    blades = stack((mv.scalar([1]), mv.yz, mv.zx, mv.xy)).reverse()
    points = distinct(blades.scalar_product(spinors[..., None]).to_array())   # [vertices, 4]
    gaps = np.linalg.norm(points[:, None] - points[None], axis=-1)
    nearest = gaps[gaps > 1e-9].min()
    first, second = np.nonzero(np.triu(np.abs(gaps - nearest) < 1e-6))
    # From beyond the scalar axis, each chord of the three-sphere projects to a straight segment.
    space = points[:, 1:] / (2.4 - points[:, :1])                              # [vertices, 3]
    # Seen from slightly above: the camera turned about x by the tilt, looking along its own -z.
    view = solids.View((solids.mv.yz * (tilt / 2)).exp(), np.full(2, half_width), n_pixels)
    places = solids.mv.point(space)                                            # [vertices] Point-
    edges = strokes.segments(places[first], places[second], even, edge_width, view)
    vertices = solids.ball(places, vertex_radius * view.pixel)[:, None]       # [vertices, 1] Plane+ <- Point-
    return solids.draw([(vertices, vertex_colour)], view, background, inks=[], translucent=[], meshes=[(edges, 1.0)])


def animate_polytope(turning) -> Iterator[np.ndarray]:
    """One frame per batch of turned spinors."""
    for spinors in turning:
        yield polytope(spinors)
