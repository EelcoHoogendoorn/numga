"""A viscous channel flow past an offset circular obstacle."""

from __future__ import annotations

import numpy as np

from numga import concatenate, stack
from numga.algebras import VGA2D as ga
from examples import instantiate

mesh_module = instantiate("examples.util.mesh", ga)
Mesh, Point, Scalar, mv = mesh_module.Mesh, mesh_module.Point, mesh_module.Scalar, mesh_module.mv

LENGTH, HEIGHT = 6.0, 2.0
OBSTACLE_CENTRE = (2.5, 0.2)
OBSTACLE_RADIUS = 0.45
N_DIVISIONS = 18
VISCOSITY, MEAN_SPEED = 1.0, 1.0
N_FRAMES, DURATION_MS = 84, 40
MESH_SEED = 8294


# --- plumbing -------------------------------------------------------------------------
def channel_vertices(n_divisions: int) -> tuple[Point, Point, np.ndarray, np.ndarray]:
    """The rectangular boundary and a slightly perturbed triangular lattice inside it."""
    spacing = HEIGHT / n_divisions
    n_length = int(np.ceil(LENGTH / spacing))
    along = np.linspace(0, LENGTH, n_length + 1)
    across = np.linspace(-HEIGHT / 2, HEIGHT / 2, n_divisions + 1)
    horizontal, vertical = mv.x.dual(), mv.y.dual()
    boundary = concatenate((horizontal * along - vertical * HEIGHT / 2,
                            horizontal * along + vertical * HEIGHT / 2,
                            vertical * across[1:-1],
                            horizontal * LENGTH + vertical * across[1:-1]))
    left_start = 2 * (n_length + 1)
    right_start = left_start + n_divisions - 1
    inlet = np.concatenate(([0, n_length + 1], np.arange(left_start, right_start)))
    outlet = np.concatenate(([n_length, 2 * n_length + 1],
                             np.arange(right_start, right_start + n_divisions - 1)))

    # A gap beside the straight walls keeps each boundary edge's circumcentric dual inside.
    wall_spacing, jitter = 0.85, 0.08
    rows = np.arange(wall_spacing * spacing, HEIGHT - 0.7 * spacing, np.sqrt(3) * spacing / 2)
    columns = np.arange(wall_spacing * spacing, LENGTH - 0.7 * spacing, spacing)
    along, across = np.meshgrid(columns, rows)
    along += (np.arange(len(rows)) % 2)[:, None] * spacing / 2
    random = np.random.default_rng(MESH_SEED)
    along += random.uniform(-jitter, jitter, along.shape) * spacing
    across += random.uniform(-jitter, jitter, across.shape) * spacing
    inside_mask = along < LENGTH - 0.7 * spacing
    interior = horizontal * along[inside_mask] + vertical * (across[inside_mask] - HEIGHT / 2)
    return boundary, interior, inlet, outlet


def triangulate(vertices: Point, hole: np.ndarray, inlet: np.ndarray, outlet: np.ndarray
                ) -> tuple[Mesh, Point, np.ndarray, np.ndarray, np.ndarray]:
    """Delaunay connectivity with a polygonal hole and the three sets of boundary edges."""
    from scipy.spatial import Delaunay

    coordinates = (stack((mv.x, mv.y)) & vertices.batch()[..., None]).to_array()
    faces = Delaunay(coordinates).simplices
    # The obstacle's boundary edges are present in the triangulation; its interior consists
    # exactly of triangles with all three corners on that boundary.
    faces = faces[~np.isin(faces, hole).all(axis=1)]
    mesh = Mesh(len(coordinates), faces)
    boundary = mesh.boundary_edges
    inlet_edges = boundary[np.isin(mesh.edges[boundary], inlet).all(axis=1)]
    outlet_edges = boundary[np.isin(mesh.edges[boundary], outlet).all(axis=1)]
    wall_edges = np.setdiff1d(boundary, np.concatenate((inlet_edges, outlet_edges)))
    return mesh, vertices, inlet_edges, outlet_edges, wall_edges


def channel(n_divisions: int) -> tuple[Mesh, Point, np.ndarray, np.ndarray, np.ndarray]:
    """A Delaunay mesh of the channel around the circular obstacle, with its boundary edges."""
    boundary, interior, inlet, outlet = channel_vertices(n_divisions)
    spacing = HEIGHT / n_divisions
    centre = mv.x.dual() * OBSTACLE_CENTRE[0] + mv.y.dual() * OBSTACLE_CENTRE[1]
    n_obstacle = int(np.ceil(2 * np.pi * OBSTACLE_RADIUS / spacing))
    angles = np.arange(n_obstacle) * (2 * np.pi / n_obstacle)
    obstacle = centre + ((mv.xy * (-angles / 2)).exp() >> mv.x.dual()) * OBSTACLE_RADIUS
    # The surrounding ring leaves each obstacle chord's diameter circle empty, so Delaunay
    # triangulation includes every obstacle segment. A varying radius avoids cocircular cells.
    ring_radii = OBSTACLE_RADIUS + spacing * (0.85 + 0.06 * np.sin(3 * angles + 0.3))
    ring = centre + ((mv.xy * (-(angles + np.pi / n_obstacle) / 2)).exp() >> mv.x.dual()) * ring_radii
    interior = interior[(interior - centre).norm() > OBSTACLE_RADIUS + 1.55 * spacing]
    hole = np.arange(len(boundary), len(boundary) + n_obstacle)
    vertices = concatenate((boundary, obstacle, ring, interior)).field()
    return triangulate(vertices, hole, inlet, outlet)


def plain_channel(n_divisions: int) -> tuple[Mesh, Point, np.ndarray, np.ndarray, np.ndarray]:
    """The same channel without the obstacle, for its parabolic-flow solution."""
    boundary, interior, inlet, outlet = channel_vertices(n_divisions)
    vertices = concatenate((boundary, interior)).field()
    return triangulate(vertices, np.array([], dtype=int), inlet, outlet)


def main() -> None:
    from numga.sparse import SparseExtensor
    from examples.fluids.dec_stokes.core import Operators, stokes
    from examples.fluids.dec_stokes import render
    from examples.util.animation import save_animation, save_figure

    mesh, vertices, inlet, outlet, walls = channel(N_DIVISIONS)
    operators = Operators(mesh, vertices)

    # The inlet and outlet carry the same parabolic flow. Its integral between each
    # edge's ends gives the edge flux; the solid walls carry no normal flow.
    ordinate = mv.y & vertices                                               # Scalar+[vertices]
    streamfunction = 1.5 * MEAN_SPEED * (ordinate - 4 * ordinate.squared() * ordinate / (3 * HEIGHT**2))  # Scalar+[vertices]
    ends = SparseExtensor.selection(mesh_module.ctx, np.concatenate((inlet, outlet)), len(mesh.edges))
    prescribed_flux = ~ends * (ends * (mesh.d0 * streamfunction))            # Scalar+[edges]
    boundary_circulation = operators.dual_patches * 0                        # Patch-[vertices]
    flux, pressure, vorticity, velocity = stokes(operators, VISCOSITY, prescribed_flux,
                                                 mesh.boundary_edges, boundary_circulation)
    streamfunction = (mesh.d0 * Scalar).lstsq(flux)                          # Scalar+[vertices]

    save_figure(render.draw_mesh(mesh, vertices), "dec_stokes_mesh")
    save_figure(render.draw_complex(mesh, vertices, operators.circumcentres, operators.dual_edges), "dec_stokes_complex")
    save_figure(render.draw_fields(mesh, vertices, pressure, vorticity, streamfunction), "dec_stokes_fields")
    save_animation(render.animate_flow(mesh, vertices, pressure, streamfunction, velocity, N_FRAMES),
                   "dec_stokes_flow", DURATION_MS)


if __name__ == "__main__":
    main()
