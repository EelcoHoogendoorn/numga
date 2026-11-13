"""A viscous channel flow past an offset circular obstacle."""

from __future__ import annotations

import numpy as np

from numga import NumpyContext, concatenate, stack
from numga.algebras import VGA2D as ga
from examples.fluids.dec_stokes import render
from examples.fluids.dec_stokes.core import Operators, stokes_flow
from examples.util import shapes
from examples.util.animation import save_figure
from examples.util.topology import Topology

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                                  # Scalar+
Point = ga.gatype.antivector()                                               # Antivector-

length, height = 6.0, 2.0
obstacle_centre = (2.5, 0.2)
obstacle_radius = 0.45
n_divisions = 18
viscosity, mean_speed = 1.0, 1.0
triangulation_seed = 8294


# --- plumbing -------------------------------------------------------------------------
def channel_vertices(n_divisions: int) -> tuple[Point, Point, np.ndarray, np.ndarray]:
    """The rectangular boundary and a slightly perturbed triangular lattice inside it."""
    spacing = height / n_divisions
    n_length = int(np.ceil(length / spacing))
    abscissae = np.linspace(0, length, n_length + 1)
    ordinates = np.linspace(-height / 2, height / 2, n_divisions + 1)
    horizontal, vertical = mv.x.dual(), mv.y.dual()
    boundary = concatenate((horizontal * abscissae - vertical * height / 2,
                            horizontal * abscissae + vertical * height / 2,
                            vertical * ordinates[1:-1],
                            horizontal * length + vertical * ordinates[1:-1]))
    left_start = 2 * (n_length + 1)
    right_start = left_start + n_divisions - 1
    inlet = np.concatenate(([0, n_length + 1], np.arange(left_start, right_start)))
    outlet = np.concatenate(([n_length, 2 * n_length + 1],
                             np.arange(right_start, right_start + n_divisions - 1)))

    # A gap beside the straight walls keeps each boundary edge's circumcentric dual inside.
    wall_spacing, jitter = 0.85, 0.08
    rows = np.arange(wall_spacing * spacing, height - 0.7 * spacing, np.sqrt(3) * spacing / 2)
    columns = np.arange(wall_spacing * spacing, length - 0.7 * spacing, spacing)
    abscissae, ordinates = np.meshgrid(columns, rows)
    abscissae += (np.arange(len(rows)) % 2)[:, None] * spacing / 2
    generator = np.random.default_rng(triangulation_seed)
    abscissae += generator.uniform(-jitter, jitter, abscissae.shape) * spacing
    ordinates += generator.uniform(-jitter, jitter, ordinates.shape) * spacing
    inside_mask = abscissae < length - 0.7 * spacing
    interior = horizontal * abscissae[inside_mask] + vertical * (ordinates[inside_mask] - height / 2)
    return boundary, interior, inlet, outlet


def triangulation(vertices: Point, hole: np.ndarray, inlet: np.ndarray, outlet: np.ndarray
                  ) -> tuple[Topology, Point, np.ndarray, np.ndarray, np.ndarray]:
    """The Delaunay connectivity with a polygonal hole, and the three sets of boundary edges."""
    # The vertices read across and up by the planes x and y, one step outside the algebra.
    delaunay, _ = shapes.delaunay((stack((mv.x, mv.y)) & vertices.batch()[..., None]).to_array())
    # The obstacle's boundary edges are present in the triangulation; its interior consists
    # exactly of triangles with all three corners on that boundary.
    triangles = delaunay.elements[2]
    topology = delaunay.subset(~np.isin(triangles, hole).all(axis=1), -1)
    edges, boundary = topology.elements[1], topology.exterior[1]
    inlet_edges = boundary[np.isin(edges[boundary], inlet).all(axis=1)]
    outlet_edges = boundary[np.isin(edges[boundary], outlet).all(axis=1)]
    wall_edges = np.setdiff1d(boundary, np.concatenate((inlet_edges, outlet_edges)))
    return topology, vertices, inlet_edges, outlet_edges, wall_edges


def channel(n_divisions: int) -> tuple[Topology, Point, np.ndarray, np.ndarray, np.ndarray]:
    """The Delaunay triangulation of the channel around the circular obstacle, with its boundary edges."""
    boundary, interior, inlet, outlet = channel_vertices(n_divisions)
    spacing = height / n_divisions
    centre = mv.x.dual() * obstacle_centre[0] + mv.y.dual() * obstacle_centre[1]
    n_obstacle = int(np.ceil(2 * np.pi * obstacle_radius / spacing))
    angles = np.arange(n_obstacle) * (2 * np.pi / n_obstacle)
    obstacle = centre + ((mv.xy * (-angles / 2)).exp() >> mv.x.dual()) * obstacle_radius
    # The surrounding ring leaves each obstacle chord's diameter circle empty, so Delaunay
    # triangulation includes every obstacle segment. A varying radius avoids cocircular cells.
    ring_radii = obstacle_radius + spacing * (0.85 + 0.06 * np.sin(3 * angles + 0.3))
    ring = centre + ((mv.xy * (-(angles + np.pi / n_obstacle) / 2)).exp() >> mv.x.dual()) * ring_radii
    interior = interior[(interior - centre).norm() > obstacle_radius + 1.55 * spacing]
    hole = np.arange(len(boundary), len(boundary) + n_obstacle)
    vertices = concatenate((boundary, obstacle, ring, interior)).field()
    return triangulation(vertices, hole, inlet, outlet)


def plain_channel(n_divisions: int) -> tuple[Topology, Point, np.ndarray, np.ndarray, np.ndarray]:
    """The same channel without the obstacle, for its parabolic-flow solution."""
    boundary, interior, inlet, outlet = channel_vertices(n_divisions)
    vertices = concatenate((boundary, interior)).field()
    return triangulation(vertices, np.array([], dtype=int), inlet, outlet)


def boundary_flux(topology: Topology, vertices: Point, inlet: np.ndarray, outlet: np.ndarray) -> Scalar:
    """The parabolic flux through the inlet and outlet edges, zero on every other edge."""
    # The integral of the parabolic profile between each edge's ends gives the edge flux.
    ordinate = mv.y & vertices                                               # Scalar+[V]
    streamfunction = 1.5 * mean_speed * (ordinate - 4 * ordinate.squared() * ordinate / (3 * height**2))  # Scalar+[V]
    ends = topology.reading(np.concatenate((inlet, outlet)), 1)              # Scalar+[E_ends <- E]
    return ~ends * (ends * (topology.d[0] * streamfunction))


def main() -> None:
    topology, vertices, inlet, outlet, walls = channel(n_divisions)
    operators = Operators(topology, vertices)

    # The inlet and outlet carry the same parabolic flow; the solid walls carry no normal flow.
    prescribed_flux = boundary_flux(topology, vertices, inlet, outlet)       # Scalar+[E]
    boundary_circulation = operators.dual_patches * 0                        # Patch-[V]
    flux, pressure, vorticity, velocity = stokes_flow(operators, viscosity, prescribed_flux,
                                                      topology.exterior[1], boundary_circulation)
    streamfunction = (topology.d[0] * Scalar).lstsq(flux)                    # Scalar+[V]

    save_figure(render.draw_triangulation(topology, vertices), "dec_stokes_triangulation")
    save_figure(render.draw_complex(topology, vertices, operators.circumcentres, operators.dual_edges), "dec_stokes_complex")
    save_figure(render.draw_fields(topology, vertices, pressure, vorticity, streamfunction), "dec_stokes_fields")
    save_figure(render.draw_flow(topology, vertices, pressure, velocity), "dec_stokes_flow")

    # checks
    # Every triangle passes on what it receives, and no fluid crosses the walls.
    assert (operators.d1(flux).abs() < 1e-9).all()
    assert ((topology.reading(walls, 1) * flux).abs() < 1e-9).all()
    # Viscous dissipation, summed over the dual cells, is positive.
    assert (vorticity.norm_squared() * operators.dual_areas).sites.sum() > 0


if __name__ == "__main__":
    main()
