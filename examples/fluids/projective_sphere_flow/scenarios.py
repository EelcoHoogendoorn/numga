"""One mesh corner moves along its ray while diffusion stays on the unit sphere."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np

from numga import stack
from numga.algebras import VGA3D, PGA2D
from examples import instantiate

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
def triangle_grid(n_divisions: int) -> tuple[core.Mesh, np.ndarray]:
    """The triangles of a lattice of samples on the reference triangle, and the barycentric weights of the samples
    `[corners, samples]`, independent of any geometric realization."""
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
    return core.Mesh(inside.sum(), np.concatenate((upward, downward))), barycentric


# --- math -----------------------------------------------------------------------------
def spherical_frames(mesh: core.Mesh, vertices: core.Point, radii: Iterable[np.ndarray],
                     quadrature: core.Scalar, display: np.ndarray
                     ) -> Iterator[tuple[core.Point, core.Point, core.Point, core.Scalar]]:
    """Each frame solves the same source-and-sink problem again, under the interpolation its corner scales choose."""
    corners = mesh.incidence[2, 0] * vertices                              # [corners] Point-[faces]
    first, second, third = corners                                         # Point-[faces], Point-[faces], Point-[faces]
    volume = first & second & third                                        # Volume-[faces]
    quadrature_chords = (corners[:, None] * quadrature).sum(axis=0)        # [samples] Point-[faces]
    samples = quadrature_chords.normalized()                               # [samples] Point-[faces]
    # Radial projection expands chord area by the inverse cube of the distance.
    n_samples = quadrature.shape[-1]
    measures = volume.norm() / (2 * n_samples * quadrature_chords.norm() ** 3)   # [samples] Scalar+[faces]
    source = 2 * (mv.z & samples)                                          # [samples] Scalar+[faces]
    conductivity = core.Plane.dual()                                       # [] Point- <- Plane+
    display_points = (corners[:, None] * display).sum(axis=0).normalized() # [samples] Point-[faces]
    for scales in radii:
        lifted = vertices * core.as_scalar(scales).field()                 # Point-[vertices]
        lifted_corners = mesh.incidence[2, 0] * lifted                     # [corners] Point-[faces]
        potential, numerator = core.diffusion(mesh, lifted_corners, samples, measures, source, conductivity)
        first, second, third = lifted_corners                              # Point-[faces], Point-[faces], Point-[faces]
        reciprocals = stack((second & third, third & first, first & second)) / (first & second & third)   # [corners] Plane+[faces]
        section = reciprocals.sum(axis=0)                                  # Plane+[faces]
        values = (numerator & display_points) / (section & display_points) # [samples] Scalar+[faces]
        yield lifted, section.dual().normalized(), display_points, values


def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.fluids.projective_sphere_flow import render

    mesh, vertices = core.Mesh.icosphere(n_levels)
    quadrature_grid, quadrature_weights = triangle_grid(n_quadrature_divisions)
    # Each small triangle's centroid samples its share of the face, at the mean of its corners' weights.
    corner_mean = quadrature_grid.incidence[2, 0].mean(axis=0)            # Scalar+[faces <- vertices]
    quadrature = (corner_mean * core.as_scalar(quadrature_weights).field()).batch()   # [corners, samples] Scalar+
    display_grid, display = triangle_grid(n_display_divisions)
    chosen_vertex = int((mv.x & vertices.batch()).argmax())
    phases = np.linspace(0, 2 * np.pi, n_frames, endpoint=False)
    radii = np.exp(log_radius_amplitude * np.sin(phases[:, None]) * (np.arange(mesh.n_vertices) == chosen_vertex))
    frames = spherical_frames(mesh, vertices, radii, quadrature, display)
    save_animation(render.animate_sphere(mesh, display_grid, frames), "projective_sphere_weights", frame_duration_ms)

    # The same projective quotient uses the degenerate algebra's finite affine slice.
    points = (planar_corners[:, None] * display).sum(axis=0)               # [samples] Point-
    lifted = planar_corners * corner_scales                                # [corners] Point-
    first, second, third = lifted                                          # [] Point-, Point-, Point-
    reciprocals = stack((second & third, third & first, first & second)) / (first & second & third)   # [corners] Plane+
    section = reciprocals.sum(axis=0)                                      # [] Plane+
    hats = (reciprocals[:, None] & points) / (section & points)            # [corners, samples] Scalar+
    gradients = (reciprocals[:, None] - hats * section) / (section & points)   # [corners, samples] Plane+
    potential = (hats * corner_values[:, None]).sum(axis=0)                # [samples] Scalar+
    gradient = (gradients * corner_values[:, None]).sum(axis=0)            # [samples] Plane+
    # The material pairs spatial directions only; it never inverts the null metric.
    directions = stack((planar.mv.x.dual(), planar.mv.y.dual()))           # [axes] Point-
    conductivity = (directions * (planar.Plane & directions)).sum(axis=0)  # [] Point- <- Plane+
    current = -conductivity(gradient)                                      # [samples] Point-
    save_figure(render.draw_planar(points, display_grid, potential, current), "projective_affine_slice")

    # checks
    np.testing.assert_allclose(hats.sum(axis=0).to_array(), 1, atol=1e-12)
    np.testing.assert_allclose((gradients & points).to_array(), 0, atol=1e-12)
    np.testing.assert_allclose((planar.mv.w & current).to_array(), 0, atol=1e-12)


if __name__ == "__main__":
    main()
