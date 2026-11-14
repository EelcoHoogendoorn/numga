"""Scenes for spin transformations: a sphere given a dipole change in mean curvature, Spot the cow
rounded by conformal curvature flow, and the Dirac spheres."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from examples import MODEL_DIR
from examples.surfaces.spin_transformations import core
from examples.util import shapes

# How often the sphere's icosahedron is split, and how strongly its mean curvature changes along y.
n_subdivisions = 3
dipole_strength = 0.5
# Spot the cow from the shared models, and the two quarter turns that stand her up facing along x: her y
# axis, up in the file, turned onto z, and then x onto y.
spot_model = MODEL_DIR / "spot.obj"
# The share of each face's curvature departure each step of Spot's flow removes, the number of steps,
# and the duration of a frame.
flow_rate = 0.05
n_steps = 60
frame_duration_ms = 40
# The finer sphere the Dirac spheres are spun from, the eigenvalues shown, and the states shown of each.
n_dirac_subdivisions = 4
eigenvalues = (1, 2, 3, -2, -3)
n_states = 3


# --- plumbing -------------------------------------------------------------------------
def spot() -> tuple[core.Topology, core.Point]:
    """Spot the cow by Keenan Crane (CC0, https://www.cs.cmu.edu/~kmcrane/Projects/ModelRepository/):
    her triangulated surface and her vertices (`Point-[V]`) stood upright with z up, centred, and scaled
    to unit reach."""
    topology, coordinates = shapes.surface(spot_model, 2.0)
    z_turn = (core.mv.yz * (-np.pi / 4)).exp()                                 # [] Rotor+
    upright = (core.mv.xy * (-np.pi / 4)).exp() * z_turn                       # [] Rotor+
    vertices = core.mv.vector(coordinates).dual().field()                      # Point-[V]
    return topology, upright >> vertices                                       # Point-[V]


def dipole_scene(n_splits: int = n_subdivisions) -> tuple[core.Topology, core.Point, core.Scalar, core.Point]:
    """An icosahedron subdivided `n_splits` times into a sphere (`Point-[V]`), a dipole mean-curvature
    change (`Scalar+[F]`) growing along y, and the resulting deformed surface (`Point-[V]`)."""
    topology, coordinates = shapes.sphere(n_splits)
    vertices = core.mv.vector(coordinates).dual().field()                      # Point-[V]
    centroids = topology.incidence[2, 0].mean(axis=0) * vertices               # Point-[F]
    # Pairing the plane normal to y with each centroid reads its height along y.
    curvature_change = (core.mv.y & centroids) * dipole_strength               # Scalar+[F]
    return topology, vertices, curvature_change, core.deformed_surface(topology, vertices, curvature_change)


def rounding_states() -> tuple[core.Topology, Iterator[core.Point]]:
    """Spot's triangulated surface and the generator streaming her vertices (`Point-[V]`) under conformal
    curvature flow."""
    topology, vertices = spot()
    return topology, core.curvature_flow_states(topology, vertices, n_steps, flow_rate)


def dirac_scene(n_splits: int = n_dirac_subdivisions, n_modes: int = n_states) -> tuple[core.Topology, core.Point, dict[int, core.Point], core.Scalar]:
    """An icosahedron subdivided `n_splits` times into a sphere (`Point-[V]`) and `n_modes` Dirac
    spheres (`Point-[V]`) per eigenvalue along with their residual energies (`[..., n_modes] Scalar+`)."""
    topology, coordinates = shapes.sphere(n_splits)
    vertices = core.mv.vector(coordinates).dual().field()                      # Point-[V]
    energies, dirac_vertices = core.dirac_spheres(topology, vertices, np.array(eigenvalues, dtype=float), n_modes)  # [..., n_modes] Scalar+, [..., n_modes] Point-[V]
    return topology, vertices, dict(zip(eigenvalues, dirac_vertices)), energies


def main() -> None:
    from examples.surfaces.spin_transformations import render
    from examples.util.animation import save_animation, save_figure

    topology, vertices, curvature_change, transformed_vertices = dipole_scene()
    save_figure(render.dipole_figure(topology, vertices, curvature_change, transformed_vertices), "spin_transformation")
    sphere, sphere_vertices, gallery, energies = dirac_scene()
    save_figure(render.dirac_figure(sphere, gallery, sphere_vertices), "dirac_spheres")
    spot_topology, states = rounding_states()
    kept_frames: list[core.Point] = []

    def frame_stream() -> Iterator[core.Point]:
        for state in states:
            kept_frames.append(state)
            yield state

    save_animation(render.rounding_animation(spot_topology, frame_stream()), "spin_rounding", frame_duration_ms)

    # checks
    # Every corner keeps its angle up to triangle resolution, and mean curvature rises where prescribed.
    assert ((core.corner_cosines(topology, transformed_vertices) - core.corner_cosines(topology, vertices)).abs() < 0.05).all()
    curvature_rise = core.mean_curvature(topology, core.recentred_vertices(transformed_vertices)) - core.mean_curvature(topology, vertices)
    centred_rise = curvature_rise - curvature_rise.sites.mean()
    centred_target = curvature_change - curvature_change.sites.mean()
    assert (centred_rise * centred_target).sites.sum() > 0.5 * ((centred_rise.squared().sites.sum() * centred_target.squared().sites.sum()).square_root())
    # Spot rounds: the spread of vertex radii drops tenfold while the mean corner-cosine drift stays small.
    assert core.surface_roundness(spot_topology, kept_frames[-1]) < 0.1 * core.surface_roundness(spot_topology, kept_frames[0])
    assert (core.corner_cosines(spot_topology, kept_frames[-1]) - core.corner_cosines(spot_topology, kept_frames[0])).abs().sites.mean().mean() < 0.05
    # Integer Dirac eigenvalues yield zero-energy eigenfields up to discretization, preserving angles away from zeros.
    assert (energies.abs() < 0.05).all()
    sphere_cosines = core.corner_cosines(sphere, sphere_vertices)
    for spheres in gallery.values():
        for dirac_sphere in spheres:
            assert ((core.corner_cosines(sphere, dirac_sphere) - sphere_cosines).abs() < 0.1).mean() > 0.65


if __name__ == "__main__":
    main()
