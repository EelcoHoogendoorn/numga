"""Tests for Laplacian eigenfunction fluids compute four Laplace–Beltrami modes and their structure constants on a
coarse triangulated sphere, step a shear layer once, and render two frames."""

from __future__ import annotations

from examples.fluids.eigen_fluids import core, render, scenarios
from examples.util import shapes


def test_a_coarse_sphere_steps_once_and_draws():
    n_modes, n_pixels = 4, 32
    topology, coordinates = shapes.sphere(1)
    vertices = core.mv.vector(coordinates).dual().field()                        # Point-[V]
    # Opposite triangle sides and oriented face patches determine the corner simplex gradients.
    edges = topology.d[0] * vertices                                             # Point-[E]
    sides = topology.boundary[2, 1] * edges                                      # [3] Point-[F]
    patches = (sides[0] & sides[1]) / 2                                          # Patch-[F]
    areas = patches.norm()                                                       # Scalar+[F]
    normals = patches / areas                                                    # Patch-[F]
    corner_gradients = (sides | patches.inverse()) / 2                           # [3] Plane+[F]
    gradient = (core.spdiag(corner_gradients * core.Scalar) * topology.incidence[2, 0]).sum(axis=0)   # (Plane+ <- Scalar+)[F <- V]
    dual_areas = ~topology.incidence[2, 0].mean(axis=0) * areas                  # Scalar+[V]

    # Dirichlet stiffness and lumped vertex mass define the generalized Laplace–Beltrami eigenproblem.
    stiffness = gradient.adjugate()((areas * core.Plane.dual())(gradient))       # (Antiscalar- <- Scalar+)[V <- V]
    mass = core.spdiag(dual_areas * core.Scalar.dual())                          # (Antiscalar- <- Scalar+)[V <- V]
    eigenvalues, eigenfields = stiffness.eigh(mass, n_modes + 1)                 # [n_modes + 1] Scalar+, [n_modes + 1] Scalar+[V]
    eigenvalues, modes = eigenvalues[1:].field(), eigenfields[1:]                # Scalar+[n_modes], [n_modes] Scalar+[V]
    superposition = (modes * core.Scalar).batch().field(0, 1).adjoint()          # (Scalar+ <- Scalar+)[V <- n_modes]

    structure = core.structure_constants(topology, patches, gradient, modes)     # (Scalar+ <- (Scalar+, Scalar+))[n_modes <- (n_modes, n_modes)]
    layers = core.shear_layers(vertices, dual_areas, scenarios.sphere_latitude, scenarios.sphere_width,
                               scenarios.sphere_strength, scenarios.sphere_wave_amplitude)   # Scalar+[V]
    start = (modes & mass(layers)).sites.sum().field()                           # Scalar+[n_modes]
    states = core.advect(start, structure, eigenvalues, scenarios.sphere_viscosity, scenarios.sphere_dt, 1)
    frames = core.reconstruct(states, superposition, eigenvalues, gradient, normals)
    list(render.animate_flow(topology, vertices, frames, n_pixels))
