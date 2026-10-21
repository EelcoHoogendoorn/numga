"""Drawing for the symmetry example, ray traced: heat-flow ellipsoids, the cubic lattice and its responses, and
rotary powers on the sphere of directions."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from examples import raytrace as rt
from examples.raster import rasterise, screen, view_frame
from examples.mechanics.symmetry.core import Conductivity, Direction, Field, Point, Scalar, mv, pga_mv

DRIVING_COLOUR, FLOW_COLOUR = "#262b32", "#d44e1a"
SURFACE_COLOUR = np.array([0.36, 0.60, 0.80])
STIFFNESS_COLOUR = np.array([0.88, 0.52, 0.30])
SITE_COLOURS = {"centre": [0.20, 0.22, 0.26], "axial": [0.88, 0.45, 0.22], "diagonal": [0.30, 0.56, 0.75], "other": [0.78, 0.78, 0.80]}
BACKGROUND = np.array([1.0, 1.0, 1.0])
# The view's angles, the pixels across a panel, the margin around what it shows as a fraction of its larger
# side, and the surface samples it is framed by.
VIEW = (23, -55)
PIXELS = 360
MARGIN = 0.08
N_FRAMING = 400
# The lattice's site and bond radii, in lattice spacings.
SITE_RADIUS, BOND_RADIUS = 0.13, 0.035


# --- plumbing -------------------------------------------------------------------------
def euclidean(points: Point) -> np.ndarray:
    """Coordinates of points: their pairings with the planes x, y and z, over their weights."""
    return ((stack((pga_mv.x, pga_mv.y, pga_mv.z)) & points[..., None]) / (pga_mv.w & points)[..., None]).to_array()


def along(directions: Direction) -> np.ndarray:
    """The components of directions along x, y and z: their pairings with the planes x, y and z."""
    return (stack((mv.x, mv.y, mv.z)) & directions[..., None]).to_array()


def across(fields: Field) -> np.ndarray:
    """The components of fields along x, y and z: their pairings with the directions x, y and z."""
    return (fields[..., None] & stack((mv.x.dual(), mv.y.dual(), mv.z.dual()))).to_array()


def matrices(conductivities: Conductivity) -> np.ndarray:
    """The conductivities' matrices `[..., 3, 3]`: column j the flow driven by the unit field along axis j."""
    return np.swapaxes(along(conductivities[..., None](stack((mv.x, mv.y, mv.z)))), -1, -2)


def image_of_sphere(matrix: np.ndarray) -> rt.Quadric:
    """The solid ellipsoid a map with these matrices makes of the unit ball: its points x satisfy
    `x @ inv(matrix @ matrix.T) @ x <= 1`."""
    return rt.quadric(np.linalg.inv(matrix @ np.swapaxes(matrix, -1, -2)), np.zeros(matrix.shape[:-2] + (3,)))


def panel(ax: plt.Axes, rgb: np.ndarray, middle: np.ndarray, half: np.ndarray) -> None:
    """An image on screen coordinates, framed about the middle with the given half width and height, without
    axes."""
    x, y = on_screen(middle)
    ax.imshow(np.clip(rt.downsample(rgb, rt.SUPERSAMPLE), 0, 1), extent=(x - half[0], x + half[0], y - half[1], y + half[1]), interpolation="bilinear")
    ax.set_axis_off()


def figure(halves: list[np.ndarray], height: float) -> tuple[plt.Figure, list[plt.Axes]]:
    """A row of panels with the given half widths and heights, each as wide as its aspect asks, filling the
    figure's height, with a thin gap between them."""
    widths = [height * half[0] / half[1] for half in halves]
    fig = plt.figure(figsize=(sum(widths) * 1.02, height), dpi=120)
    grid = fig.add_gridspec(1, len(halves), width_ratios=widths, left=0, right=1, bottom=0, top=1, wspace=0.02)
    return fig, [fig.add_subplot(cell) for cell in grid]


def on_screen(points: np.ndarray) -> np.ndarray:
    """Points `[..., 3]` in screen coordinates about the view's centre at the origin."""
    return points @ rt.view_axes(*VIEW)[:2].T


def mesh_image(vertices: np.ndarray, colour: np.ndarray, middle: np.ndarray, extent: float) -> np.ndarray:
    """A square RGB image, of half width extent about the middle, of a closed surface sampled on a
    latitude-longitude grid `[latitudes, longitudes, 3]`, smooth shaded by normals interpolated from the corners, each the mean of its triangles' normals."""
    rows, cols = vertices.shape[:2]
    index = np.arange(rows * cols).reshape(rows, cols)
    corner = index[:-1, :-1].ravel(), index[:-1, 1:].ravel(), index[1:, 1:].ravel(), index[1:, :-1].ravel()
    faces = np.concatenate([np.stack([corner[0], corner[1], corner[2]], -1), np.stack([corner[0], corner[2], corner[3]], -1)])
    flat = vertices.reshape(-1, 3)
    normal = np.cross(flat[faces[:, 1]] - flat[faces[:, 0]], flat[faces[:, 2]] - flat[faces[:, 0]])
    # Outward: away from the centre.
    normal *= np.sign(np.sum(normal * flat[faces].mean(axis=1), axis=-1, keepdims=True))
    corner_normal = np.zeros_like(flat)
    np.add.at(corner_normal, faces.ravel(), np.repeat(normal, 3, axis=0))
    corner_normal /= np.linalg.norm(corner_normal, axis=-1, keepdims=True) + 1e-12
    frame = view_frame(*VIEW)
    with np.errstate(divide="ignore", invalid="ignore"):
        pixel, owner, weight, _ = rasterise(screen(flat, frame, middle, extent, PIXELS * rt.SUPERSAMPLE), faces, PIXELS * rt.SUPERSAMPLE)
    smooth = np.einsum("pk,pkc->pc", weight, corner_normal[faces[owner]])
    smooth /= np.linalg.norm(smooth, axis=-1, keepdims=True)
    size = PIXELS * rt.SUPERSAMPLE
    rgb = np.ones((size * size, 3)) * BACKGROUND
    rgb[pixel] = rt.shade(colour, smooth, rt.lamp(*VIEW), frame[2], 1.0)
    return rgb.reshape(size, size, 3)[::-1]


def draw_conduction(conductivities: Conductivity, driving: Field, fluxes: Direction) -> plt.Figure:
    """For the candidate and each group's average, the image of the unit ball of driving fields, a heat-flow
    ellipsoid, with one driving field, dark, and the heat flow it drives, orange."""
    matrix = matrices(conductivities)                                         # [panels, 3, 3]
    flow_matrix = matrix @ np.swapaxes(matrix, -1, -2)
    largest = np.sqrt(np.linalg.eigvalsh(flow_matrix).max())
    # The unit driving field is drawn at the largest gain, the scale the flows are drawn at.
    driving_tip, flows = across(driving) * largest, along(fluxes)
    surfaces = rt.ellipsoid_points(np.linalg.inv(flow_matrix), np.zeros(3), N_FRAMING)
    middle, half = rt.framing(np.concatenate([surfaces.reshape(-1, 3), driving_tip[None], flows]), *VIEW, MARGIN)
    fig, axes = figure([half] * len(matrix), 3.6)
    for ax, panel_matrix, flow in zip(axes, matrix, flows):
        rgb, _ = rt.scene(image_of_sphere(panel_matrix)[None], SURFACE_COLOUR[None], (*VIEW, middle, half, PIXELS * rt.SUPERSAMPLE), BACKGROUND)
        panel(ax, rgb, middle, half)
        for vector, colour, width in ((driving_tip, DRIVING_COLOUR, 1.8), (flow, FLOW_COLOUR, 2.6)):
            ax.annotate("", xy=on_screen(vector), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=colour, lw=width))
    return fig


def lattice_scene(sites: Point, axial: Point, diagonal: Point) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The lattice's sites as spheres, the centre dark, its axial neighbours orange, its diagonal ones blue,
    and the bonds from the centre to them as thin ellipsoids: their matrices, centres and colours."""
    positions, axial_ends, diagonal_ends = euclidean(sites), euclidean(axial), euclidean(diagonal)
    kind = np.where(np.linalg.norm(positions, axis=-1) < 1e-9, "centre", "other").astype(object)
    for ends, name in ((axial_ends, "axial"), (diagonal_ends, "diagonal")):
        kind[np.isclose(positions[:, None], ends[None]).all(axis=-1).any(axis=1)] = name
    ends = np.concatenate([axial_ends, diagonal_ends])
    length = np.linalg.norm(ends, axis=-1, keepdims=True)
    unit = ends / length
    # A bond: a cigar from the centre to its end, long along the bond and thin across it.
    across_bond = (np.eye(3) - unit[:, :, None] * unit[:, None, :]) / BOND_RADIUS**2
    bond_matrix = unit[:, :, None] * unit[:, None, :] / (length[..., None] / 2) ** 2 + across_bond
    matrix = np.concatenate([np.broadcast_to(np.eye(3) / SITE_RADIUS**2, (len(positions), 3, 3)), bond_matrix])
    centre = np.concatenate([positions, ends / 2])
    colours = np.array([SITE_COLOURS[name] for name in kind] + [SITE_COLOURS["axial"]] * len(axial_ends)
                       + [SITE_COLOURS["diagonal"]] * len(diagonal_ends))
    return matrix, centre, colours


def draw_crystal(sites: Point, axial: Point, diagonal: Point, conductivity: Conductivity, stiffness: Direction) -> plt.Figure:
    """The lattice with its bonds; the lattice's conduction, the image of the unit ball of fields, a sphere;
    and its stiffness along every direction as a radius, a cube with rounded faces."""
    lattice_matrix, lattice_centre, colours = lattice_scene(sites, axial, diagonal)
    lattice_view = rt.framing(rt.ellipsoid_points(lattice_matrix, lattice_centre, N_FRAMING), *VIEW, MARGIN)
    matrix = matrices(conductivity)
    conduction_view = rt.framing(rt.ellipsoid_points(np.linalg.inv(matrix @ matrix.T), np.zeros(3), N_FRAMING), *VIEW, MARGIN)
    vertices = along(stiffness)
    stiffness_middle, stiffness_half = rt.framing(vertices, *VIEW, MARGIN)
    stiffness_half = np.full(2, stiffness_half.max())
    fig, axes = figure([lattice_view[1], conduction_view[1], stiffness_half], 4.0)
    lattice, _ = rt.scene(rt.quadric(lattice_matrix, lattice_centre), colours, (*VIEW, *lattice_view, PIXELS * rt.SUPERSAMPLE), BACKGROUND)
    panel(axes[0], lattice, *lattice_view)
    conduction, _ = rt.scene(image_of_sphere(matrix)[None], SURFACE_COLOUR[None], (*VIEW, *conduction_view, PIXELS * rt.SUPERSAMPLE), BACKGROUND)
    panel(axes[1], conduction, *conduction_view)
    panel(axes[2], mesh_image(vertices, STIFFNESS_COLOUR, stiffness_middle, stiffness_half[0]), stiffness_middle, stiffness_half)
    return fig


def draw_optical_activity(rotary: Scalar) -> plt.Figure:
    """For the candidate and each group's average, the sphere of light directions coloured by the rotary
    power along each, red for one sense and blue for the other, shaded by the lamp."""
    middle, half = rt.framing(rt.ellipsoid_points(np.eye(3), np.zeros(3), N_FRAMING), *VIEW, MARGIN)
    origins, heading, lamp = rt.camera(*VIEW, middle, half, PIXELS * rt.SUPERSAMPLE)
    distance, hit = rt.hits(rt.ellipsoid(np.ones(3)), origins, heading)
    normal = rt.normals(rt.ellipsoid(np.ones(3)), hit)                         # [pixels, pixels, 3]
    directions = mv.vector(normal).dual()                                     # [pixels, pixels] Direction
    powers = rotary[:, None, None](directions[None], directions[None]).to_array()   # [panels, pixels, pixels]
    limit = np.abs(powers).max()
    colours = plt.get_cmap("RdBu_r")((powers / limit + 1) / 2)[..., :3]
    toward_viewer = rt.view_axes(*VIEW)[2]
    fig, axes = figure([half] * len(powers), 2.6)
    for ax, colour in zip(axes, colours):
        rgb = np.where(np.isfinite(distance)[..., None], rt.shade(colour, normal, lamp, toward_viewer, 1.0), BACKGROUND)
        panel(ax, rgb, middle, half)
    return fig
