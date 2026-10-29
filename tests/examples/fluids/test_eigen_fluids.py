"""Laplacian eigenfunction fluids: Whitney gradients and spherical harmonic eigenvalues on the unit sphere,
total antisymmetry of the Lie–Poisson structure constants on the Stanford bunny, inviscid conservation of
kinetic energy and enstrophy against selective viscous decay, and GPU rendering."""

from __future__ import annotations

from itertools import islice

import matplotlib.pyplot as plt
import numpy as np

from numga import stack

from examples.fluids.eigen_fluids import core, render, scenarios


def test_whitney_gradients_and_spherical_harmonic_spectrum():
    """On the unit sphere, the Whitney gradient of a linear height field equals the tangential part of its
    plane, the induced velocity is tangent to every face, and the first three non-constant Laplace–Beltrami
    eigenvalues form the l = 1 dipole triplet near l(l + 1) = 2."""
    sphere, vertices = core.Mesh.icosphere(2)
    # Each opposite edge and its face patch determine the corner hat gradient.
    edges = sphere.d[0] * vertices                                               # Point-[edges]
    sides = sphere.boundary[2, 1] * edges                                        # [corners] Point-[faces]
    patches = (sides[0] & sides[1]) / 2                                          # Patch-[faces]
    areas = patches.norm()                                                      # Scalar+[faces]
    normals = patches / areas                                                   # Patch-[faces]
    corner_gradients = (sides | patches.inverse()) / 2                           # [corners] Plane+[faces]
    gradient = (core.spdiag(corner_gradients * core.Scalar) * sphere.incidence[2, 0]).sum(axis=0)  # (Plane+ <- Scalar+)[faces <- vertices]
    thirds = sphere.incidence[2, 0].mean(axis=0)                                 # Scalar+[faces <- vertices]

    # The stiffness and vertex mass determine the surface's spatial modes.
    stiffness = gradient.adjugate()(core.spdiag(areas * core.Plane.dual())(gradient))  # (Amount- <- Scalar+)[vertices <- vertices]
    mass = core.spdiag((~thirds * areas) * core.Scalar.dual())                # (Amount- <- Scalar+)[vertices <- vertices]
    eigenvalues, eigenfields = stiffness.eigh(mass, 8 + 1)                # [modes + 1] Scalar+, [modes + 1] Scalar+[vertices]
    eigenvalues, modes = eigenvalues[1:], eigenfields[1:]                          # [modes] Scalar+, [modes] Scalar+[vertices]

    height = core.mv.z & vertices                                                # Scalar+[vertices]
    face_gradient = gradient(height)                                             # Plane+[faces]
    velocity = face_gradient ^ normals                                           # Point-[faces]
    np.testing.assert_allclose((velocity - (core.mv.z ^ normals)).kernel, 0.0, atol=1e-13)
    np.testing.assert_allclose((velocity & patches).to_array(), 0.0, atol=1e-14)

    values = eigenvalues.to_array()
    np.testing.assert_allclose(values[:3], 2.0, rtol=2e-2)
    np.testing.assert_allclose(values[3:8], 6.0, rtol=4e-2)
    gram = (modes[:, None] & mass(modes[None, :])).sites.sum().to_array()
    np.testing.assert_allclose(gram, np.eye(8), atol=1e-12)


def test_bunny_structure_antisymmetry_conservation_and_rendering():
    """On the Stanford bunny, the Lie–Poisson structure constants are totally antisymmetric in their three modes,
    inviscid RK4 conserves both kinetic energy and enstrophy while viscosity damps enstrophy faster than
    kinetic energy, and the figures and animation frames render."""
    mesh, vertices = scenarios.bunny()
    # Each opposite edge and its face patch determine the corner hat gradient.
    edges = mesh.d[0] * vertices                                                 # Point-[edges]
    sides = mesh.boundary[2, 1] * edges                                          # [corners] Point-[faces]
    patches = (sides[0] & sides[1]) / 2                                          # Patch-[faces]
    areas = patches.norm()                                                      # Scalar+[faces]
    normals = patches / areas                                                   # Patch-[faces]
    corner_gradients = (sides | patches.inverse()) / 2                           # [corners] Plane+[faces]
    gradient = (core.spdiag(corner_gradients * core.Scalar) * mesh.incidence[2, 0]).sum(axis=0)  # (Plane+ <- Scalar+)[faces <- vertices]
    thirds = mesh.incidence[2, 0].mean(axis=0)                                   # Scalar+[faces <- vertices]

    # The stiffness and vertex mass determine the surface's spatial modes.
    stiffness = gradient.adjugate()(core.spdiag(areas * core.Plane.dual())(gradient))  # (Amount- <- Scalar+)[vertices <- vertices]
    mass = core.spdiag((~thirds * areas) * core.Scalar.dual())                # (Amount- <- Scalar+)[vertices <- vertices]
    eigenvalues, eigenfields = stiffness.eigh(mass, scenarios.n_modes + 1)                # [modes + 1] Scalar+, [modes + 1] Scalar+[vertices]
    eigenvalues, modes = eigenvalues[1:].field(), eigenfields[1:]                  # Scalar+[modes], [modes] Scalar+[vertices]
    superposition = (modes * core.Scalar).batch().field(0, 1).adjoint()          # (Scalar+ <- Scalar+)[vertices <- modes]
    # Meeting each mode's gradient plane with the unit tangent patch gives its divergence-free velocity on each face.
    mode_velocities = gradient(modes[scenarios.shown_modes]) ^ normals                     # [shown] Point-[faces]

    structure = core.structure_constants(mesh, patches, gradient, modes)         # (Scalar+ <- (Scalar+, Scalar+))[modes <- (modes, modes)]
    start = scenarios.initial_vorticity(vertices, modes, eigenvalues, mass)                # Scalar+[modes]
    inviscid = stack(tuple(core.advect(start, structure, eigenvalues, 0.0, scenarios.dt, scenarios.n_steps)))      # [steps + 1] Scalar+[modes]
    viscous = stack(tuple(core.advect(start, structure, eigenvalues, scenarios.viscosity, scenarios.dt, scenarios.n_steps))) # [steps + 1] Scalar+[modes]

    # Modal kinetic energy and enstrophy at every step of the inviscid and viscous runs.
    inviscid_energy = (inviscid.squared() / eigenvalues).sites.sum() / 2         # [steps + 1] Scalar+
    viscous_energy = (viscous.squared() / eigenvalues).sites.sum() / 2           # [steps + 1] Scalar+
    inviscid_enstrophy = inviscid.squared().sites.sum() / 2                      # [steps + 1] Scalar+
    viscous_enstrophy = viscous.squared().sites.sum() / 2                        # [steps + 1] Scalar+

    # Reconstruct vertex vorticity, streamfunction, and face velocity fields at every sampled frame of the inviscid run.
    sampled = inviscid[::scenarios.stride]                                                 # [frames] Scalar+[modes]
    vorticity_frames = superposition(sampled)                                    # [frames] Scalar+[vertices]
    stream_frames = superposition(sampled / eigenvalues)                         # [frames] Scalar+[vertices]
    velocity_frames = gradient(stream_frames) ^ normals                          # [frames] Point-[faces]

    gram = (modes[:, None] & mass(modes[None, :])).sites.sum().to_array()
    np.testing.assert_allclose(gram, np.eye(scenarios.n_modes), atol=1e-12)
    np.testing.assert_allclose((mode_velocities & patches).to_array(), 0.0, atol=1e-14)
    np.testing.assert_allclose((velocity_frames & patches).to_array(), 0.0, atol=1e-13)

    face_gradients = gradient(modes[:6])                                         # [subset] Plane+[faces]
    wedge_brackets = (face_gradients[:, None] ^ face_gradients[None, :]) & patches   # [subset, subset] Scalar+[faces]
    directional = (face_gradients[:, None] & (face_gradients[None, :] ^ normals)) * areas   # [subset, subset] Scalar+[faces]
    np.testing.assert_allclose((wedge_brackets - directional).to_array(), 0.0, atol=1e-14)

    constants = structure.kernel[..., 0, 0, 0]                                   # [modes, modes, modes]
    np.testing.assert_allclose(constants + constants.swapaxes(1, 2), 0.0, atol=1e-13)
    np.testing.assert_allclose(constants + constants.swapaxes(0, 1), 0.0, atol=1e-13)

    initial_rate = core.vorticity_rate(inviscid[0], structure, eigenvalues, 0.0) # Scalar+[modes]
    np.testing.assert_allclose(((inviscid[0] / eigenvalues) * initial_rate).sites.sum().to_array(), 0.0, atol=1e-11)
    np.testing.assert_allclose((inviscid[0] * initial_rate).sites.sum().to_array(), 0.0, atol=1e-11)

    energy_trace = inviscid_energy.to_array()
    enstrophy_trace = inviscid_enstrophy.to_array()
    np.testing.assert_allclose(energy_trace, energy_trace[0], rtol=5e-5)
    np.testing.assert_allclose(enstrophy_trace, enstrophy_trace[0], rtol=5e-5)

    viscous_energy_trace = viscous_energy.to_array()
    viscous_enstrophy_trace = viscous_enstrophy.to_array()
    assert np.all(np.diff(viscous_energy_trace) < 0)
    assert np.all(np.diff(viscous_enstrophy_trace) < 0)
    assert viscous_enstrophy_trace[-1] / viscous_enstrophy_trace[0] < viscous_energy_trace[-1] / viscous_energy_trace[0]

    times = np.arange(scenarios.n_steps + 1) * scenarios.dt
    plt.close(render.draw_modes(mesh, vertices, modes[scenarios.shown_modes], mode_velocities))
    plt.close(render.draw_conservation(times, inviscid_energy, viscous_energy, inviscid_enstrophy,
                                       viscous_enstrophy, start, inviscid[-1], viscous[-1]))
    frames = list(islice(render.animate_flow(mesh, vertices, zip(vorticity_frames[:2], stream_frames[:2], velocity_frames[:2])), 2))
    assert len(frames) == 2 and frames[0].shape == (render.n_flow_pixels, render.n_flow_pixels, 3)
