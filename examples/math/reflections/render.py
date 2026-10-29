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
# Beads tumbling in the kaleidoscope, scattered over the disk: the distance of each centre from the middle, its
# angle, speed and size, and its colour, drawn from a short palette.
n_beads = 40
_sampler = np.random.default_rng(7)
bead_distances = np.sqrt(_sampler.uniform(0.0, 1.0, n_beads))
bead_angles = _sampler.uniform(0.0, 2 * np.pi, n_beads)
bead_speeds = _sampler.choice([-2, -1, 1, 2], n_beads)
bead_sizes = _sampler.uniform(0.04, 0.13, n_beads)
palette = np.array([
    [0.91, 0.30, 0.24], [0.98, 0.77, 0.19], [0.18, 0.62, 0.48], [0.56, 0.27, 0.68],
    [0.20, 0.47, 0.85], [0.95, 0.55, 0.65], [0.99, 0.97, 0.90],
])
bead_colours = palette[np.arange(n_beads) % len(palette)]
# The lamp lighting the sphere from the viewer's upper left, as the gradient of its light over the sphere's points.
lamp = np.array([-0.4, 0.5, 0.77])
# The 600-cell: its vertices' colour; its edges' width and its vertices' radius in pixels; the pixels across its
# picture, half the picture's width about the middle, in the projected space, and the tilt of the view; the eye on
# the scalar axis, beyond the three-sphere, that the spinors are seen from.
vertex_colour = odd
edge_width, vertex_radius = 0.8, 1.6
n_pixels, half_width = 300, 0.6
tilt = 0.45
eye_distance = 2.4


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
    # The beads from the top down: each is painted over the ones listed before it, its centre a point on the x-axis
    # turned about the middle to its angle at the time.
    turns = (mv.xy * (-(bead_angles + bead_speeds * time)[::-1] / 2)).exp()   # [beads] Rotor+
    centres = bead_distances[::-1] * (turns >> mv.x.dual())                  # [beads] Antivector-
    distance = (folded[..., None] - centres).norm().to_array()               # [..., rows, columns, beads]
    cover = np.clip((bead_sizes[::-1] - distance) / 0.02, 0.0, 1.0)
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
    # The light on each point of the sphere, by the lamp's gradient paired with it.
    lit = 0.35 + 0.65 * np.clip((mv.vector(lamp) & screen).to_array(), 0.0, 1.0)
    sides = np.abs((mirrors & folded[..., None]).to_array())                  # [rows, columns, mirrors]
    thin = np.clip(1 - sides.min(axis=-1) / 0.012, 0.0, 1.0)
    bold = np.clip(1 - np.abs((outline & folded).to_array()) / 0.03, 0.0, 1.0)
    colours = tiles(flips, np.maximum(0.6 * thin, bold)) * lit[..., None]
    # The sphere covers the pixels whose points stand out towards the viewer.
    inside = (mv.z & screen) > 0
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
def gpu_points(points) -> solids.Point:
    """The GPU library's own points at the places of the spinors' plane parts: their coefficients on the planes, read
    by the scalar product with the planes' reverses, are the one hand-off from this algebra to the renderer's."""
    mv = points.context.multivector
    planes = stack((mv.yz, mv.zx, mv.xy)).reverse()
    return solids.mv.point(planes.scalar_product(points[..., None]).to_array())


def distinct(spinors):
    """The distinct spinors, up to round-off: one of each set whose coefficients are equal to six decimals, in sorted
    order. The comparison is a numeric step on the coefficients, read by the scalar product with the blades'
    reverses."""
    mv = spinors.context.multivector
    blades = stack((mv.scalar([1]), mv.yz, mv.zx, mv.xy)).reverse()
    rounded = np.round(blades.scalar_product(spinors[..., None]).to_array() * 1e6).astype(np.int64)   # [spinors, 4]
    # Equal rows sort next to each other, the first of them first; a row starts a new set where it differs
    # from the one before.
    order = np.lexsort(rounded.T[::-1])
    first = order[np.concatenate([[True], (np.diff(rounded[order], axis=0) != 0).any(axis=-1)])]
    return spinors[np.sort(first)]


def polytope(spinors) -> np.ndarray:
    """The distinct unit spinors as points of the three-sphere, in perspective from beyond the scalar
    axis into three dimensions and from there seen from slightly above, `[pixels, pixels, 3]` bytes:
    the points traced as balls, and edges joining nearest neighbours drawn as strokes, lit on the GPU."""
    vertices = distinct(spinors)                                               # [vertices] Even+
    gaps = (vertices[:, None] - vertices[None]).norm()                         # [vertices, vertices] Scalar+
    nearest = gaps.to_array()[gaps > 1e-9].min()
    first, second = np.nonzero(np.triu((gaps - nearest).abs() < 1e-6))
    # From the eye beyond the scalar axis, each chord of the three-sphere projects to a straight segment: the plane
    # part, scaled down by the distance from the eye along the scalar axis.
    space = vertices.select_grade(2) / (eye_distance - vertices.select_grade(0))   # [vertices] Bivector+
    # Seen from slightly above: the camera turned about x by the tilt, looking along its own -z.
    view = solids.View((solids.mv.yz * (tilt / 2)).exp(), np.full(2, half_width), n_pixels)
    places = gpu_points(space)                                                 # [vertices] Point-
    edges = strokes.segments(places[first], places[second], even, edge_width, view)
    balls = solids.ball(places, vertex_radius * view.pixel)[:, None]           # [vertices, 1] Plane+ <- Point-
    return solids.draw([(balls, vertex_colour)], view, background, inks=[], translucent=[], meshes=[(edges, 1.0)])


def animate_polytope(turning) -> Iterator[np.ndarray]:
    """One frame per batch of turned spinors."""
    for spinors in turning:
        yield polytope(spinors)
