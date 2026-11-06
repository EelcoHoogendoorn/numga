"""Laplacian eigenfunction fluids: a coarse sphere's few modes, their structure constants, and a shear layer stepped
once, its two frames drawn."""

from __future__ import annotations

from examples.fluids.eigen_fluids import core, render, scenarios
from examples.util import shapes


def test_a_coarse_sphere_steps_once_and_draws():
    n_modes, n_pixels = 4, 32
    topology, coordinates = shapes.sphere(1)
    vertices = core.mv.vector(coordinates).dual().field()                       # Point-[vertices]
    # Each opposite edge and its face patch determine the corner hat gradient.
    edges = topology.d[0] * vertices                                             # Point-[edges]
    sides = topology.boundary[2, 1] * edges                                      # [corners] Point-[faces]
    patches = (sides[0] & sides[1]) / 2                                          # Patch-[faces]
    areas = patches.norm()                                                       # Scalar+[faces]
    normals = patches / areas                                                    # Patch-[faces]
    corner_gradients = (sides | patches.inverse()) / 2                           # [corners] Plane+[faces]
    gradient = (core.spdiag(corner_gradients * core.Scalar) * topology.incidence[2, 0]).sum(axis=0)  # (Plane+ <- Scalar+)[faces <- vertices]
    dual_areas = ~topology.incidence[2, 0].mean(axis=0) * areas                 # Scalar+[vertices]

    # The stiffness and vertex mass determine the surface's spatial modes.
    stiffness = gradient.adjugate()((areas * core.Plane.dual())(gradient))       # (Amount- <- Scalar+)[vertices <- vertices]
    mass = core.spdiag(dual_areas * core.Scalar.dual())                          # (Amount- <- Scalar+)[vertices <- vertices]
    eigenvalues, eigenfields = stiffness.eigh(mass, n_modes + 1)                 # [modes + 1] Scalar+, [modes + 1] Scalar+[vertices]
    eigenvalues, modes = eigenvalues[1:].field(), eigenfields[1:]                # Scalar+[modes], [modes] Scalar+[vertices]
    superposition = (modes * core.Scalar).batch().field(0, 1).adjoint()          # (Scalar+ <- Scalar+)[vertices <- modes]

    structure = core.structure_constants(topology, patches, gradient, modes)     # (Scalar+ <- (Scalar+, Scalar+))[modes <- (modes, modes)]
    layers = core.shear_layers(vertices, dual_areas, scenarios.sphere_latitude, scenarios.sphere_width,
                               scenarios.sphere_strength, scenarios.sphere_wave_amplitude)   # Scalar+[vertices]
    start = (modes & mass(layers)).sites.sum().field()                           # Scalar+[modes]
    states = core.advect(start, structure, eigenvalues, scenarios.sphere_viscosity, scenarios.sphere_dt, 1)
    frames = core.reconstruct(states, superposition, eigenvalues, gradient, normals)
    list(render.animate_flow(topology, vertices, frames, n_pixels))
