"""A shear acting on a square, its grid and the two basis vectors."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga import concatenate, stack
from numga.extensor import Extensor

from examples.math.doubled_algebra import core
from examples.util.wgsl import contours, simplices
from examples.util.wgsl.camera import panel

reference_colour, grid_colour, boundary_colour, fill_colour, dot_colour = (
    np.array(to_rgb(colour)) for colour in ("#cbd2d8", "#3885a0", "#cc793e", "#fff1df", "#3b4146"))
basis_colours = np.array([to_rgb("#168b82"), to_rgb("#8050ad")])
# The figure's size in inches, at a hundred pixels an inch; the widths of its lines, the dashes and gaps of the
# reference square's outline, the arrows' heads and the dot at the origin's radius, in pixels.
figure_size = (3.2, 2.2)
grid_width, reference_width, boundary_width, arrow_width = 1.0, 1.4, 2.0, 2.5
reference_dashes = (4.0, 4.0)
head_length, dot_radius = 11.0, 2.2


# --- plumbing -------------------------------------------------------------------------
def readers(vectors: core.Vector) -> Extensor:
    """The planes that read a vector's x and y coefficients, `[2]`: the directions along x and y, which a vector
    pairs with first, `vector & mv.x.dual()` being `mv.x | vector`."""
    mv = vectors.context.multivector
    return stack((mv.x, mv.y)).dual()


def coordinates(vectors: core.Vector) -> np.ndarray:
    """The x and y coefficients of vectors, read by pairing them with the planes that read them."""
    return (readers(vectors) & vectors[..., None]).to_array()


def loop(n_corners: int) -> simplices.Mesh:
    """The closed outline through corners in their order, as a mesh of edges."""
    corners = np.arange(n_corners)
    return simplices.Mesh(n_corners, np.stack([corners, np.roll(corners, -1)], axis=-1))


def draw_shear(
    reference_grid: core.Vector,
    grid: core.Vector,
    reference_boundary: core.Vector,
    boundary: core.Vector,
    arrows: core.Vector,
) -> plt.Figure:
    """The reference square in grey and its sheared grid, with the images of the two basis vectors."""
    origin = 0 * arrows[0]                                                     # [] Vector+
    framed = coordinates(concatenate([reference_grid.reshape(-1), grid.reshape(-1), reference_boundary, boundary,
                                      arrows, origin[None]]))
    lower = framed.min(axis=0)
    upper = framed.max(axis=0)
    margin = (upper - lower).max() * 0.08

    figure, ax = plt.subplots(figsize=figure_size, layout="constrained")
    ax.set(xlim=(lower[0] - margin, upper[0] + margin),
           ylim=(lower[1] - margin, upper[1] + margin), aspect="equal")
    ax.set_axis_off()
    # The camera reads this algebra's places, vectors, as they are, by the planes that read their x and y.
    camera = panel(ax, *readers(boundary))
    n_lines = np.prod(grid.shape[:-1])
    lines = simplices.Mesh.chains([grid.shape[-1]] * n_lines)
    square = simplices.Mesh(len(boundary), np.array([[0, 1, 2], [0, 2, 3]]))
    # Each basis vector's image an arrow from the origin.
    arrow_ends = stack((origin.broadcast_to(arrows.shape), arrows), axis=-1).reshape(-1)   # [arrows * ends] Vector+
    simplices.overlay(ax, [
        contours.picture(boundary, square, np.zeros(len(boundary)), fill_colour, camera),
        simplices.strokes(reference_grid.reshape(-1), lines, reference_colour, grid_width, camera),
        simplices.strokes(reference_boundary, loop(len(reference_boundary)), reference_colour,
                          reference_width, camera, dashes=reference_dashes),
        simplices.strokes(grid.reshape(-1), lines, grid_colour, grid_width, camera),
        simplices.strokes(boundary, loop(len(boundary)), boundary_colour, boundary_width, camera),
        simplices.strokes(arrow_ends, simplices.Mesh.chains([2] * len(arrows)), basis_colours, arrow_width,
                          camera, heads=np.ones(len(arrows)), head=head_length),
        simplices.markers(origin[None], dot_colour, dot_radius, camera)])
    return figure
