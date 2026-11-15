"""Drawing for the reflection notebooks paints a kaleidoscope from its folded points, colours a
sphere and disks of tiles by reflection parity, and projects the 600-cell from the three-sphere."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import stack
from numga.topology import Topology
from numga.render import simplices, solids

background = np.array([0.06, 0.07, 0.09])
# Even and odd reflection counts alternate between two tile colours.
even = np.array([0.93, 0.83, 0.62])
odd = np.array([0.35, 0.55, 0.72])
line = np.array([0.12, 0.12, 0.16])
# Width in pixels of background between adjacent panels.
gap = 6
# Shader constants for tile colours and mirror-edge darkening.
tiling_colours = dict(even=even, odd=odd, line=line, background=background, side_width=0.02, side_strength=0.5)
# Tumbling kaleidoscope beads scattered across the disk with fixed radii, angles, speeds, and colours.
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
# Illumination gradient lighting the sphere from the viewer's upper left.
lamp = np.array([-0.4, 0.5, 0.77])
# Rendering geometry and perspective eye distance along the scalar axis for the 600-cell.
vertex_colour = odd
edge_width, vertex_radius = 0.8, 1.6
n_pixels, half_width = 300, 0.6
tilt = 0.45
eye_distance = 2.4


def disk(n_resolution: int) -> tuple[np.ndarray, np.ndarray]:
    """Pixel centres across a square `[n_resolution, n_resolution]` view of the unit disk."""
    centres = (np.arange(n_resolution) + 0.5) / n_resolution * 2 - 1
    horizontal, vertical = np.meshgrid(centres, -centres)
    return horizontal, vertical


def image_bytes(colours: np.ndarray) -> np.ndarray:
    """Clamping `[..., 3]` RGB floats in `[0, 1]` and scaling to `[0, 255]` produces `uint8` pixels."""
    return (np.clip(colours, 0.0, 1.0) * 255).round().astype(np.uint8)


def side_by_side(images: np.ndarray) -> np.ndarray:
    """Panels along the leading axis concatenate left to right with a `gap`-pixel background strip between neighbours."""
    spacer = np.broadcast_to(image_bytes(background), (images.shape[1], gap, 3))
    return np.concatenate([part for image in images for part in (image, spacer)][:-1], axis=1)


# --- kaleidoscope -----------------------------------------------------------------------
def kaleidoscope_frame(folded_points, time: float) -> np.ndarray:
    """Paints the tumbling beads at `time` across `folded_points` into `[..., n_rows, n_columns, 3]` `uint8` pixels."""
    mv = folded_points.context.multivector
    # Beads composite top-down, each centred on a turned radial ray at its instantaneous angle.
    turns = (mv.xy * (-(bead_angles + bead_speeds * time)[::-1] / 2)).exp()       # [n_beads] Rotor+
    unit_centres = turns >> mv.x.dual()                                           # [n_beads] Antivector-
    centres = bead_distances[::-1] * unit_centres                                 # [n_beads] Antivector-
    distance = (folded_points[..., None] - centres).norm().to_array()             # [..., n_beads]
    cover = np.clip((bead_sizes[::-1] - distance) / 0.02, 0.0, 1.0)
    # Each bead contributes its cover times the fraction left uncovered by earlier beads.
    uncovered = np.cumprod(1 - cover, axis=-1)                                    # [..., n_beads]
    shown = cover * np.concatenate([np.ones_like(uncovered[..., :1]), uncovered[..., :-1]], axis=-1)
    return image_bytes(uncovered[..., -1:] * background + shown @ bead_colours[::-1])


def kaleidoscope_animation(folded_points, n_frames: int) -> Iterator[np.ndarray]:
    """Yields side-by-side kaleidoscope frames across `n_frames` steps of one full revolution."""
    for time in np.linspace(0, 2 * np.pi, n_frames, endpoint=False):
        yield side_by_side(kaleidoscope_frame(folded_points, time))


# --- tiles ------------------------------------------------------------------------------
def tile_colours(flip_counts, nearness: np.ndarray) -> np.ndarray:
    """Reflection parity selects `even` or `odd` tile colours and `nearness` darkens mirror boundaries toward `line`."""
    parity = (np.round(flip_counts.to_array()) % 2)[..., None]
    colours = even + parity * (odd - even)
    return colours + nearness[..., None] * (line - colours)


def sphere_frame(screen_points, folded_points, flip_counts, mirrors, outline_mirror) -> np.ndarray:
    """Shades the visible hemisphere tiled by `mirrors` with thin mirror lines and bold `outline_mirror` edges."""
    mv = screen_points.context.multivector
    # Pairing the lamp's gradient vector with each point on the sphere sets diffuse shading.
    lit = 0.35 + 0.65 * np.clip((mv.vector(lamp) & screen_points).to_array(), 0.0, 1.0)
    sides = np.abs((mirrors & folded_points[..., None]).to_array())               # [..., 3]
    thin = np.clip(1 - sides.min(axis=-1) / 0.012, 0.0, 1.0)
    bold = np.clip(1 - np.abs((outline_mirror & folded_points).to_array()) / 0.03, 0.0, 1.0)
    colours = tile_colours(flip_counts, np.maximum(0.6 * thin, bold)) * lit[..., None]
    inside = (mv.z & screen_points) > 0
    return image_bytes(np.where(inside[..., None], colours, background))


def sphere_animation(screen_points, mirrors, outline_mirror, states) -> Iterator[np.ndarray]:
    """Yields shaded sphere frames for each `(folded_points, flip_counts)` state."""
    for folded_points, flip_counts in states:
        yield sphere_frame(screen_points, folded_points, flip_counts, mirrors, outline_mirror)


def tiling_colouring() -> str:
    """WGSL helper `tile` colouring a sample by reflection parity and darkening mirror edges."""
    return (
        "// Tile colour from reflection parity, darkened where the nearest mirror's share approaches zero.\n"
        "fn tile(flips: u32, share: f32) -> vec3<f32> {\n"
        "    let colour = mix(even, odd, f32(flips % 2u));\n"
        "    return mix(colour, line, side_strength * clamp(1 - share / side_width, 0.0, 1.0));\n"
        "}\n"
    )


def tilings_animation(states) -> Iterator[np.ndarray]:
    """Yields side-by-side RGB frames joining the three geometries' rendered disks."""
    for images in states:
        yield side_by_side(image_bytes(np.stack(images)))


# --- the 600-cell -----------------------------------------------------------------------
def distinct_spinors(spinors):
    """Deduplicates `spinors` up to six decimal places in their four even-grade coefficients."""
    mv = spinors.context.multivector
    blades = stack((mv.scalar([1]), mv.yz, mv.zx, mv.xy)).reverse()
    rounded = np.round(blades.scalar_product(spinors[..., None]).to_array() * 1e6).astype(np.int64)   # [..., 4]
    # Equal rows sort adjacent so each distinct spinor appears at the first index where its row changes.
    order = np.lexsort(rounded.T[::-1])
    first = order[np.concatenate([[True], (np.diff(rounded[order], axis=0) != 0).any(axis=-1)])]
    return spinors[np.sort(first)]


def polytope_frame(spinors) -> np.ndarray:
    """Projects distinct unit spinors from the three-sphere into 3D perspective and renders vertex balls and nearest-neighbour edges."""
    vertices = distinct_spinors(spinors)                                          # [n_vertices] Even+
    n_vertices = vertices.shape[0]
    gaps = (vertices[:, None] - vertices[None]).norm()                            # [n_vertices, n_vertices] Scalar+
    distinct_gaps = gaps[gaps > 1e-9]
    nearest = distinct_gaps.min()
    first, second = np.nonzero(np.triu((gaps - nearest).abs() < 1e-6))
    # Perspective projection from `eye_distance` on the scalar axis maps chords of the three-sphere to straight segments.
    space = vertices.select_grade(2) / (eye_distance - vertices.select_grade(0))  # [n_vertices] Bivector+
    patches = space * vertices.context.multivector.antipseudoscalar([1])          # [n_vertices] Antivector-
    view = solids.View((solids.mv.yz * (tilt / 2)).exp(), np.full(2, half_width), n_pixels)
    places = solids.origin + solids.context.include_from_algebra(patches)         # [n_vertices] Point-
    neighbours = Topology(n_vertices, np.stack([first, second], axis=-1))
    edges = simplices.tubes(places, neighbours, even, edge_width, view, gloss=solids.gloss)
    balls = solids.ball(places, vertex_radius * view.pixel)[:, None]              # [n_vertices, 1] Plane+ <- Point-
    return solids.draw([(balls, vertex_colour)], view, background, inks=[], translucent=[], meshes=[(edges, 1.0)])


def polytope_animation(states) -> Iterator[np.ndarray]:
    """Yields rendered 600-cell frames for each batch of turned spinors."""
    for spinors in states:
        yield polytope_frame(spinors)
