"""One vertex of the triangulated sphere moves along its ray while diffusion stays on the unit sphere."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np

from numga import stack
from numga.algebras import VGA3D, PGA2D
from examples import instantiate
from examples.util import shapes

core = instantiate("examples.fluids.projective_sphere_flow.core", VGA3D)
planar = instantiate("examples.fluids.projective_sphere_flow.core", PGA2D)
mv = core.mv
n_levels = 1
n_quadrature_divisions = 5
n_display_divisions = 9
n_frames = 90
frame_duration_ms = 40
log_radius_amplitude = 0.65
corner_scales = np.array([0.45, 1.7, 1.0])
corner_values = np.array([0.0, 1.0, 0.25])
planar_corners = planar.mv.point([[-0.9, -0.65], [0.9, -0.5], [-0.1, 0.95]])


# --- plumbing -------------------------------------------------------------------------
def triangle_grid(n_divisions: int) -> tuple[core.Topology, np.ndarray]:
    """The triangles of a lattice of samples on the reference triangle, and the barycentric weights of the samples per
    corner, independent of any geometric realization."""
    steps = np.arange(n_divisions + 1)
    first, second = np.meshgrid(steps, steps, indexing="ij")
    inside = first + second <= n_divisions
    barycentric = np.stack((n_divisions - first - second, first, second))[:, inside] / n_divisions
    # The index of each lattice node among the samples, and the lattice cells' upward and downward triangles.
    lookup = np.cumsum(inside).reshape(inside.shape) - 1
    corner, along_first, along_second, opposite = lookup[:-1, :-1], lookup[1:, :-1], lookup[:-1, 1:], lookup[1:, 1:]
    layers = first[:-1, :-1] + second[:-1, :-1]
    upward = np.stack((corner, along_first, along_second), axis=-1)[layers < n_divisions]
    downward = np.stack((along_first, opposite, along_second), axis=-1)[layers < n_divisions - 1]
    return core.Topology(inside.sum(), np.concatenate((upward, downward))), barycentric


# --- math -----------------------------------------------------------------------------
def spherical_frames(topology: core.Topology, vertices: core.Point, radii: Iterable[np.ndarray],
                     quadrature: core.Scalar, display: np.ndarray
                     ) -> Iterator[tuple[core.Point, core.Point, core.Point, core.Scalar]]:
    """Yield the representatives, the face poles, the display points and their temperatures, as each frame's corner scales
    choose a new interpolation for the same source and sink."""
    corners = topology.incidence[2, 0] * vertices                          # [3] Point-[F]
    volume = corners[0] & corners[1] & corners[2]                          # Antiscalar-[F]
    n_samples = quadrature.shape[-1]
    quadrature_chords = (corners[:, None] * quadrature).sum(axis=0)        # [n_samples] Point-[F]
    samples = quadrature_chords.normalized()                               # [n_samples] Point-[F]
    # Radial projection expands chord area by the inverse cube of the distance.
    measures = volume.norm() / (2 * n_samples * quadrature_chords.norm() ** 3)   # [n_samples] Scalar+[F]
    source = 2 * (mv.z & samples)                                                # [n_samples] Scalar+[F]
    conductivity = core.Plane.dual()                                       # [] Point- <- Plane+
    display_points = (corners[:, None] * display).sum(axis=0).normalized() # [...] Point-[F]
    for scales in radii:
        representatives = vertices * mv.scalar(scales[..., None]).field()  # Point-[V]
        _, numerator, section = core.diffusion(topology, representatives, samples, measures, source, conductivity)
        values = (numerator & display_points) / (section & display_points)       # [...] Scalar+[F]
        yield representatives, section.dual().normalized(), display_points, values


def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.fluids.projective_sphere_flow import render

    topology, coordinates = shapes.sphere(n_levels)
    vertices = mv.vector(coordinates).dual().field()                            # Point-[V]
    quadrature_grid, quadrature_weights = triangle_grid(n_quadrature_divisions)
    # Each small triangle's centroid samples its share of the face, at the mean of its corners' weights.
    corner_mean = quadrature_grid.incidence[2, 0].mean(axis=0)            # Scalar+[F <- V]
    quadrature = (corner_mean * mv.scalar(quadrature_weights[..., None]).field()).batch()  # [3, ...] Scalar+
    display_grid, display = triangle_grid(n_display_divisions)
    moving_vertex = int((mv.x & vertices.batch()).argmax())
    phases = np.linspace(0, 2 * np.pi, n_frames, endpoint=False)
    radii = np.exp(log_radius_amplitude * np.sin(phases[:, None]) * (np.arange(topology.n_vertices) == moving_vertex))
    frames = spherical_frames(topology, vertices, radii, quadrature, display)
    save_animation(render.animate_sphere(topology, display_grid, frames), "projective_sphere_weights", frame_duration_ms)

    # The same projective quotient uses the degenerate algebra's finite affine slice.
    triangle = planar.Topology(3, [[0, 1, 2]])
    points = (planar_corners[:, None] * display).sum(axis=0)               # [...] Point-
    representatives = (planar_corners * corner_scales).field()             # Point-[V]
    tails, heads = triangle.incidence[1, 0] * representatives              # Point-[E], Point-[E]
    reciprocals = ((triangle.boundary[2, 1] * (tails & heads)) / (representatives.batch()[0] & representatives.batch()[1] & representatives.batch()[2])).batch()[:, 0]   # [3] Plane+
    section = reciprocals.sum(axis=0)                                      # [] Plane+
    barycentrics = (reciprocals[:, None] & points) / (section & points)    # [3, ...] Scalar+
    gradients = (reciprocals[:, None] - barycentrics * section) / (section & points)   # [3, ...] Plane+
    potential = (barycentrics * corner_values[:, None]).sum(axis=0)        # [...] Scalar+
    gradient = (gradients * corner_values[:, None]).sum(axis=0)            # [...] Plane+
    # The material pairs spatial directions only; it never inverts the null metric.
    directions = stack((planar.mv.x.dual(), planar.mv.y.dual()))           # [2] Point-
    conductivity = (directions * (planar.Plane & directions)).sum(axis=0)  # [] Point- <- Plane+
    current = -conductivity(gradient)                                      # [...] Point-
    save_figure(render.draw_planar(points, display_grid, potential, current), "projective_affine_slice")

    # checks
    # The corner functions partition unity, and their simplex gradients see no motion along the ray.
    assert ((barycentrics.sum(axis=0) - 1).abs() < 1e-12).all()
    assert ((gradients & points).abs() < 1e-12).all()
    # The current is a direction, with no part along the homogeneous coordinate.
    assert ((planar.mv.w & current).abs() < 1e-12).all()


if __name__ == "__main__":
    main()
