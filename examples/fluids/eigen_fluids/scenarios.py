"""Laplacian eigenfunction fluid scenarios evolve interacting vortex dipoles on the Stanford bunny and rippled shear
layers on a triangulated sphere."""

from __future__ import annotations

from itertools import islice

import numpy as np

from numga import stack

from examples.fluids.eigen_fluids import core
from examples.util import shapes

mv = core.mv
# Bounding-box size of the Stanford bunny, count of non-constant Laplace–Beltrami modes, and representative mode
# indices shown in the four-panel figure (from a global dipole up to a fine standing wave).
bunny_size = 2.0
n_modes = 128
n_shown_modes = 4
shown_modes = np.array([0, 7, 31, 119])
# Opposing vortex dipoles placed across the bunny's shoulder, chest, flank, haunch, neck, and back, spectrally tapered
# so they collide, exchange partners, and sweep across the surface without Gibbs ringing.
n_vortices = 12
vortex_centres = mv.vector([
    [0.35, -0.54, -0.03],
    [0.40, -0.69, -0.29],
    [-0.08, -0.52, 0.03],
    [-0.16, -0.58, -0.27],
    [-0.54, -0.48, 0.02],
    [-0.45, -0.58, -0.39],
    [0.34, -0.31, 0.26],
    [0.09, -0.40, 0.20],
    [0.22, 0.34, -0.16],
    [-0.14, 0.28, -0.08],
    [-0.48, 0.26, -0.14],
    [0.06, 0.47, -0.30],
]).dual()                                           # [n_vortices] Point-
vortex_strengths = np.array([75, -75, -75, 75, 70, -70, -65, 65, 72, -72, 68, -68])
vortex_radius = 0.12
eigenvalue_cutoff = 140.0
# Fourth-order Runge–Kutta step size, total step count, frame stride, kinematic viscosity, and frame duration.
dt = 0.007
n_steps = 800
stride = 8
playback_rate = 0.5
playback_stride = int(stride * playback_rate)
n_flow_steps = int(n_steps * playback_rate)
viscosity = 0.0009
frame_duration_ms = 40
# Gallery animation pixel width, step span, and palette colour count.
n_gallery_pixels = 300
n_gallery_steps = 640
n_gallery_colours = 64


# Spherical harmonic bands through degree eight on a four-fold subdivided icosphere.
n_sphere_levels = 4
n_sphere_modes = 80
sphere_latitude = 0.44
sphere_width = 0.16
sphere_strength = 9.0
sphere_wave_amplitude = 0.065
sphere_dt = 0.025
n_sphere_steps = 400
sphere_stride = 4
sphere_viscosity = 0.001


# --- plumbing -------------------------------------------------------------------------
def bunny() -> tuple[core.Topology, core.Point]:
    """The triangulated Stanford bunny stands upright at the scenario scale."""
    return core.bunny(bunny_size)


# --- math -----------------------------------------------------------------------------
def initial_vorticity(vertices: core.Point, modes: core.Scalar, eigenvalues: core.Scalar,
                      mass: core.Mass) -> core.Scalar:
    """Localized Gaussian vortex dipoles project onto modal vorticity amplitudes with a smooth high-wavenumber taper."""
    squared_distances = (vertices - vortex_centres).norm_squared()               # [n_vortices] Scalar+[V]
    spots = ((squared_distances * (-0.5 / vortex_radius ** 2)).exp() * vortex_strengths).sum(axis=0)   # Scalar+[V]
    taper = (-0.5 * (eigenvalues / eigenvalue_cutoff) ** 2).exp()                # Scalar+[n_modes]
    return (modes & mass(spots)).sites.sum().field() * taper                     # Scalar+[n_modes]


def sphere_demo() -> None:
    """Rippled zonal shear layers roll into vortices on a triangulated sphere."""
    from examples.util.animation import save_animation
    from examples.fluids.eigen_fluids import render

    topology, coordinates = shapes.sphere(n_sphere_levels)
    vertices = mv.vector(coordinates).dual().field()                             # Point-[V]
    # Opposite triangle sides and oriented face patches determine the corner simplex gradients.
    edges = topology.d[0] * vertices                                             # Point-[E]
    sides = topology.boundary[2, 1] * edges                                      # [3] Point-[F]
    patches = (sides[0] & sides[1]) / 2                                          # Patch-[F]
    areas = patches.norm()                                                       # Scalar+[F]
    normals = patches / areas                                                    # Patch-[F]
    corner_gradients = (sides | patches.inverse()) / 2                           # [3] Plane+[F]
    gradient = (core.spdiag(corner_gradients * core.Scalar) * topology.incidence[2, 0]).sum(axis=0)   # (Plane+ <- Scalar+)[F <- V]
    # Face-to-corner averaging and lumped dual vertex areas (one third of each incident triangle).
    corner_mean = topology.incidence[2, 0].mean(axis=0)                          # Scalar+[F <- V]
    dual_areas = corner_mean.adjugate() * areas                                            # Scalar+[V]

    # Dirichlet stiffness and lumped vertex mass define the generalized Laplace–Beltrami eigenproblem.
    medium = areas * core.Plane.dual()                                           # (Point- <- Plane+)[F]
    stiffness = gradient.adjugate()(medium(gradient))                            # (Antiscalar- <- Scalar+)[V <- V]
    mass = core.spdiag(dual_areas * core.Scalar.dual())                          # (Antiscalar- <- Scalar+)[V <- V]
    eigenvalues, eigenfields = stiffness.eigh(mass, n_sphere_modes + 1)          # [n_sphere_modes + 1] Scalar+, [n_sphere_modes + 1] Scalar+[V]
    # Drop the constant null mode and turn the eigenvalues into a field over the coupled mode space.
    eigenvalues, modes = eigenvalues[1:].field(), eigenfields[1:]                # Scalar+[n_sphere_modes], [n_sphere_modes] Scalar+[V]
    superposition = (modes * core.Scalar).batch().field(0, 1).adjoint()          # (Scalar+ <- Scalar+)[V <- n_sphere_modes]

    structure = core.structure_constants(topology, patches, gradient, modes)     # (Scalar+ <- (Scalar+, Scalar+))[n_sphere_modes <- (n_sphere_modes, n_sphere_modes)]
    initial_field = core.shear_layers(vertices, dual_areas, sphere_latitude, sphere_width,
                                      sphere_strength, sphere_wave_amplitude)    # Scalar+[V]
    start = (modes & mass(initial_field)).sites.sum().field()                    # Scalar+[n_sphere_modes]
    states = islice(core.advect(start, structure, eigenvalues, sphere_viscosity, sphere_dt, n_sphere_steps),
                    0, None, sphere_stride)
    frames = core.reconstruct(states, superposition, eigenvalues, gradient, normals)
    save_animation(render.animate_flow(topology, vertices, frames, render.n_flow_pixels), "eigen_fluids_sphere",
                   frame_duration_ms)


def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.fluids.eigen_fluids import render

    topology, vertices = bunny()
    # Opposite triangle sides and oriented face patches determine the corner simplex gradients.
    edges = topology.d[0] * vertices                                             # Point-[E]
    sides = topology.boundary[2, 1] * edges                                      # [3] Point-[F]
    patches = (sides[0] & sides[1]) / 2                                          # Patch-[F]
    areas = patches.norm()                                                       # Scalar+[F]
    normals = patches / areas                                                    # Patch-[F]
    corner_gradients = (sides | patches.inverse()) / 2                           # [3] Plane+[F]
    gradient = (core.spdiag(corner_gradients * core.Scalar) * topology.incidence[2, 0]).sum(axis=0)   # (Plane+ <- Scalar+)[F <- V]
    # Face-to-corner averaging and lumped dual vertex areas (one third of each incident triangle).
    corner_mean = topology.incidence[2, 0].mean(axis=0)                          # Scalar+[F <- V]
    dual_areas = corner_mean.adjugate() * areas                                            # Scalar+[V]

    # Dirichlet stiffness and lumped vertex mass define the generalized Laplace–Beltrami eigenproblem.
    medium = areas * core.Plane.dual()                                           # (Point- <- Plane+)[F]
    stiffness = gradient.adjugate()(medium(gradient))                            # (Antiscalar- <- Scalar+)[V <- V]
    mass = core.spdiag(dual_areas * core.Scalar.dual())                          # (Antiscalar- <- Scalar+)[V <- V]
    eigenvalues, eigenfields = stiffness.eigh(mass, n_modes + 1)                 # [n_modes + 1] Scalar+, [n_modes + 1] Scalar+[V]
    # Drop the constant null mode and turn the eigenvalues into a field over the coupled mode space.
    eigenvalues, modes = eigenvalues[1:].field(), eigenfields[1:]                # Scalar+[n_modes], [n_modes] Scalar+[V]
    # Modal superposition maps modal amplitudes onto vertex scalar fields.
    superposition = (modes * core.Scalar).batch().field(0, 1).adjoint()          # (Scalar+ <- Scalar+)[V <- n_modes]
    # Meeting each mode's gradient plane with the unit tangent patch gives its divergence-free face velocity.
    mode_velocities = gradient(modes[shown_modes]) ^ normals                     # [n_shown_modes] Point-[F]

    save_figure(render.draw_modes(topology, vertices, modes[shown_modes], mode_velocities), "eigen_fluids_modes")

    structure = core.structure_constants(topology, patches, gradient, modes)     # (Scalar+ <- (Scalar+, Scalar+))[n_modes <- (n_modes, n_modes)]
    start = initial_vorticity(vertices, modes, eigenvalues, mass)                # Scalar+[n_modes]
    inviscid = stack(tuple(core.advect(start, structure, eigenvalues, 0.0, dt, n_steps)))   # [n_steps + 1] Scalar+[n_modes]
    frames = core.reconstruct(inviscid[:n_flow_steps + 1:playback_stride], superposition, eigenvalues, gradient, normals)
    save_animation(render.animate_flow(topology, vertices, frames, render.n_flow_pixels), "eigen_fluids_bunny",
                   frame_duration_ms)

    # Compare the inviscid trajectory against a viscous trajectory from the same initial state, saving its animation too.
    viscous = stack(tuple(core.advect(start, structure, eigenvalues, viscosity, dt, n_steps)))   # [n_steps + 1] Scalar+[n_modes]
    viscous_frames = core.reconstruct(viscous[:n_flow_steps + 1:playback_stride], superposition, eigenvalues, gradient,
                                      normals)
    save_animation(render.animate_flow(topology, vertices, viscous_frames, render.n_flow_pixels),
                   "eigen_fluids_bunny_viscous", frame_duration_ms)
    runs = stack((inviscid, viscous))                                            # [2, n_steps + 1] Scalar+[n_modes]
    energies = (runs.squared() / eigenvalues).sites.sum() / 2                    # [2, n_steps + 1] Scalar+
    enstrophies = runs.squared().sites.sum() / 2                                 # [2, n_steps + 1] Scalar+
    inviscid_energy, viscous_energy = energies                                   # [n_steps + 1] Scalar+, [n_steps + 1] Scalar+
    inviscid_enstrophy, viscous_enstrophy = enstrophies                          # [n_steps + 1] Scalar+, [n_steps + 1] Scalar+

    times = np.arange(n_steps + 1) * dt
    save_figure(render.draw_conservation(times, inviscid_energy, viscous_energy, inviscid_enstrophy,
                                         viscous_enstrophy, start, inviscid[-1], viscous[-1]),
                "eigen_fluids_conservation")

    sphere_demo()

    # checks
    # Laplace–Beltrami eigenmodes are orthonormal under the lumped vertex mass form.
    gram = (modes[:, None] & mass(modes[None, :])).sites.sum()                   # [n_modes, n_modes] Scalar+
    assert np.all((gram - np.eye(n_modes)).abs() < 1e-12)
    # Every mode velocity and reconstructed flow velocity lies in its triangle's tangent plane.
    assert np.all((mode_velocities & patches).norm() < 1e-14)
    final_velocity = gradient(superposition(inviscid[-1] / eigenvalues)) ^ normals   # Point-[F]
    assert np.all((final_velocity & patches).norm() < 1e-13)
    # On every triangle, the Jacobian wedge pairing equals the area-weighted directional derivative.
    n_checked_modes = 6
    face_gradients = gradient(modes[:n_checked_modes])                           # [n_checked_modes] Plane+[F]
    wedge_brackets = (face_gradients[:, None] ^ face_gradients[None, :]) & patches   # [n_checked_modes, n_checked_modes] Scalar+[F]
    directional = (face_gradients[:, None] & (face_gradients[None, :] ^ normals)) * areas   # [n_checked_modes, n_checked_modes] Scalar+[F]
    assert np.all((wedge_brackets - directional).abs() < 1e-14)
    # The Lie–Poisson structure constants are totally antisymmetric in all three mode indices to round-off.
    constants = structure.kernel[..., 0, 0, 0]                                   # [n_modes, n_modes, n_modes]
    np.testing.assert_allclose(constants + constants.swapaxes(1, 2), 0.0, atol=1e-13)
    np.testing.assert_allclose(constants + constants.swapaxes(0, 1), 0.0, atol=1e-13)
    # Instantaneous inviscid advection conserves kinetic energy and enstrophy, and RK4 keeps both flat.
    initial_rate = core.vorticity_rate(start, structure, eigenvalues, 0.0)       # Scalar+[n_modes]
    assert (((start / eigenvalues) * initial_rate).sites.sum()).abs() < 1e-11
    assert ((start * initial_rate).sites.sum()).abs() < 1e-11
    assert np.all((inviscid_energy - inviscid_energy[0]).abs() < 5e-5 * inviscid_energy[0])
    assert np.all((inviscid_enstrophy - inviscid_enstrophy[0]).abs() < 5e-5 * inviscid_enstrophy[0])
    # Viscosity dissipates both invariants, damping high-wavenumber enstrophy faster than kinetic energy.
    assert np.all(viscous_energy[1:] < viscous_energy[:-1])
    assert np.all(viscous_enstrophy[1:] < viscous_enstrophy[:-1])
    assert viscous_enstrophy[-1] / viscous_enstrophy[0] < viscous_energy[-1] / viscous_energy[0]


def pinned() -> None:
    """The pinned gallery animation shows inviscid vortices advecting one another across the Stanford bunny."""
    from examples.util.animation import pin_animation
    from examples.fluids.eigen_fluids import render

    topology, vertices = bunny()
    # Opposite triangle sides and oriented face patches determine the corner simplex gradients.
    edges = topology.d[0] * vertices                                             # Point-[E]
    sides = topology.boundary[2, 1] * edges                                      # [3] Point-[F]
    patches = (sides[0] & sides[1]) / 2                                          # Patch-[F]
    areas = patches.norm()                                                       # Scalar+[F]
    normals = patches / areas                                                    # Patch-[F]
    corner_gradients = (sides | patches.inverse()) / 2                           # [3] Plane+[F]
    gradient = (core.spdiag(corner_gradients * core.Scalar) * topology.incidence[2, 0]).sum(axis=0)   # (Plane+ <- Scalar+)[F <- V]
    corner_mean = topology.incidence[2, 0].mean(axis=0)                          # Scalar+[F <- V]
    dual_areas = corner_mean.adjugate() * areas                                            # Scalar+[V]
    medium = areas * core.Plane.dual()                                           # (Point- <- Plane+)[F]
    stiffness = gradient.adjugate()(medium(gradient))                            # (Antiscalar- <- Scalar+)[V <- V]
    mass = core.spdiag(dual_areas * core.Scalar.dual())                          # (Antiscalar- <- Scalar+)[V <- V]
    eigenvalues, eigenfields = stiffness.eigh(mass, n_modes + 1)                 # [n_modes + 1] Scalar+, [n_modes + 1] Scalar+[V]
    eigenvalues, modes = eigenvalues[1:].field(), eigenfields[1:]                # Scalar+[n_modes], [n_modes] Scalar+[V]
    superposition = (modes * core.Scalar).batch().field(0, 1).adjoint()          # (Scalar+ <- Scalar+)[V <- n_modes]
    structure = core.structure_constants(topology, patches, gradient, modes)     # (Scalar+ <- (Scalar+, Scalar+))[n_modes <- (n_modes, n_modes)]
    start = initial_vorticity(vertices, modes, eigenvalues, mass)                # Scalar+[n_modes]
    # Step the inviscid trajectory over the gallery span and reconstruct every displayed frame.
    states = islice(core.advect(start, structure, eigenvalues, 0.0, dt, int(n_gallery_steps * playback_rate)),
                    0, None, playback_stride)
    frames = core.reconstruct(states, superposition, eigenvalues, gradient, normals)
    pin_animation(render.animate_flow(topology, vertices, frames, n_gallery_pixels), "eigen_fluids", frame_duration_ms,
                  n_gallery_colours)


if __name__ == "__main__":
    main()
