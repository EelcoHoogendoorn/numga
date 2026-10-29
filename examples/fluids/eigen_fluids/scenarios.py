"""Reduced surface flows: interacting bunny vortices and rippled shear layers on a sphere."""

from __future__ import annotations

from itertools import islice

import numpy as np

from numga import stack

from examples.fluids.eigen_fluids import core

mv = core.mv
# Bounding-box size of the Stanford bunny, count of non-constant Laplace–Beltrami modes, and the modes
# shown in the gallery (from a global dipole up to a fine-scale standing wave).
bunny_size = 2.0
n_modes = 128
shown_modes = np.array([0, 7, 31, 119])
# Opposing vortex dipoles distributed across the bunny's shoulder, chest, flank, haunch, neck, and back,
# spectrally tapered so they collide, exchange partners, and swirl across the whole surface without Gibbs ringing.
vortex_centres = mv.antivector([
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
])                                                  # [vortices] Point-
vortex_strengths = np.array([75, -75, -75, 75, 70, -70, -65, 65, 72, -72, 68, -68])
vortex_radius = 0.12
eigenvalue_cutoff = 140.0
# Fourth-order Runge–Kutta step size, total step count, frame stride, kinematic viscosity, and frame duration.
dt = 0.007
n_steps = 800
stride = 8
viscosity = 0.0009
frame_duration_ms = 40


# Complete spherical harmonic bands through degree eight, on a finer geometric mesh.
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
def bunny() -> tuple[core.Mesh, core.Point]:
    """The upright Stanford bunny mesh and its vertex positions."""
    return core.bunny(bunny_size)


# --- math -----------------------------------------------------------------------------
def initial_vorticity(vertices: core.Point, modes: core.Scalar, eigenvalues: core.Scalar,
                      mass: core.Mass) -> core.Scalar:
    """Localized vortex dipoles have modal amplitudes tapered towards the finer modes."""
    squared_distances = (vertices.batch()[:, None] - vortex_centres[None, :]).norm_squared()   # [vertices, vortices] Scalar+
    spots = ((squared_distances * (-0.5 / vortex_radius**2)).exp() * vortex_strengths).sum(axis=-1).field()   # Scalar+[vertices]
    taper = (-0.5 * (eigenvalues / eigenvalue_cutoff) ** 2).exp()                # Scalar+[modes]
    return (modes & mass(spots)).sites.sum().field() * taper                     # Scalar+[modes]


def sphere_demo() -> None:
    """The reduced integrator carries latitude bands into rolling vortices on a triangulated sphere."""
    from examples.util.animation import save_animation
    from examples.fluids.eigen_fluids import render

    mesh, vertices = core.Mesh.icosphere(n_sphere_levels)
    # Each opposite edge and its face patch determine the corner hat gradient.
    edges = mesh.d[0] * vertices                                                 # Point-[edges]
    sides = mesh.boundary[2, 1] * edges                                          # [corners] Point-[faces]
    patches = (sides[0] & sides[1]) / 2                                          # Patch-[faces]
    areas = patches.norm()                                                       # Scalar+[faces]
    normals = patches / areas                                                    # Patch-[faces]
    corner_gradients = (sides | patches.inverse()) / 2                           # [corners] Plane+[faces]
    gradient = (core.spdiag(corner_gradients * core.Scalar) * mesh.incidence[2, 0]).sum(axis=0)  # (Plane+ <- Scalar+)[faces <- vertices]
    # Each face's mean over its corners; its reverse hands each corner a third of every face around it.
    corner_mean = mesh.incidence[2, 0].mean(axis=0)                              # Scalar+[faces <- vertices]
    dual_areas = ~corner_mean * areas                                            # Scalar+[vertices]

    # The stiffness and vertex mass determine the surface's spatial modes.
    stiffness = gradient.adjugate()(core.spdiag(areas * core.Plane.dual())(gradient))   # (Amount- <- Scalar+)[vertices <- vertices]
    mass = core.spdiag(dual_areas * core.Scalar.dual())                          # (Amount- <- Scalar+)[vertices <- vertices]
    eigenvalues, eigenfields = stiffness.eigh(mass, n_sphere_modes + 1)          # [modes + 1] Scalar+, [modes + 1] Scalar+[vertices]
    # The eigenvalues and the amplitudes of the modes are fields over the modes, which the structure constants couple.
    eigenvalues, modes = eigenvalues[1:].field(), eigenfields[1:]                # Scalar+[modes], [modes] Scalar+[vertices]
    superposition = (modes * core.Scalar).batch().field(0, 1).adjoint()         # (Scalar+ <- Scalar+)[vertices <- modes]

    structure = core.structure_constants(mesh, patches, gradient, modes)         # (Scalar+ <- (Scalar+, Scalar+))[modes <- (modes, modes)]
    initial_field = core.shear_layers(vertices, dual_areas, sphere_latitude, sphere_width,
                                     sphere_strength, sphere_wave_amplitude)     # Scalar+[vertices]
    start = (modes & mass(initial_field)).sites.sum().field()                    # Scalar+[modes]
    states = islice(core.advect(start, structure, eigenvalues, sphere_viscosity, sphere_dt, n_sphere_steps),
                    0, None, sphere_stride)
    frames = core.reconstruct(states, superposition, eigenvalues, gradient, normals)
    save_animation(render.animate_flow(mesh, vertices, frames), "eigen_fluids_sphere", frame_duration_ms)


def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.fluids.eigen_fluids import render

    mesh, vertices = bunny()
    # Each opposite edge and its face patch determine the corner hat gradient.
    edges = mesh.d[0] * vertices                                                 # Point-[edges]
    sides = mesh.boundary[2, 1] * edges                                          # [corners] Point-[faces]
    patches = (sides[0] & sides[1]) / 2                                          # Patch-[faces]
    areas = patches.norm()                                                       # Scalar+[faces]
    normals = patches / areas                                                    # Patch-[faces]
    corner_gradients = (sides | patches.inverse()) / 2                           # [corners] Plane+[faces]
    gradient = (core.spdiag(corner_gradients * core.Scalar) * mesh.incidence[2, 0]).sum(axis=0)  # (Plane+ <- Scalar+)[faces <- vertices]
    # Each face's mean over its corners; its reverse hands each corner a third of every face around it.
    corner_mean = mesh.incidence[2, 0].mean(axis=0)                              # Scalar+[faces <- vertices]
    dual_areas = ~corner_mean * areas                                            # Scalar+[vertices]

    # The stiffness and vertex mass determine the surface's spatial modes.
    stiffness = gradient.adjugate()(core.spdiag(areas * core.Plane.dual())(gradient))   # (Amount- <- Scalar+)[vertices <- vertices]
    mass = core.spdiag(dual_areas * core.Scalar.dual())                          # (Amount- <- Scalar+)[vertices <- vertices]
    eigenvalues, eigenfields = stiffness.eigh(mass, n_modes + 1)                 # [modes + 1] Scalar+, [modes + 1] Scalar+[vertices]
    # The eigenvalues and the amplitudes of the modes are fields over the modes, which the structure constants couple.
    eigenvalues, modes = eigenvalues[1:].field(), eigenfields[1:]                # Scalar+[modes], [modes] Scalar+[vertices]
    # The surface field of modal amplitudes superposes the modes they weigh.
    superposition = (modes * core.Scalar).batch().field(0, 1).adjoint()         # (Scalar+ <- Scalar+)[vertices <- modes]
    # Meeting each mode's gradient plane with the unit tangent patch gives its divergence-free velocity on each face.
    mode_velocities = gradient(modes[shown_modes]) ^ normals                     # [shown] Point-[faces]

    save_figure(render.draw_modes(mesh, vertices, modes[shown_modes], mode_velocities), "eigen_fluids_modes")

    structure = core.structure_constants(mesh, patches, gradient, modes)         # (Scalar+ <- (Scalar+, Scalar+))[modes <- (modes, modes)]
    start = initial_vorticity(vertices, modes, eigenvalues, mass)                # Scalar+[modes]
    inviscid = stack(tuple(core.advect(start, structure, eigenvalues, 0.0, dt, n_steps)))   # [steps + 1] Scalar+[modes]
    frames = core.reconstruct(inviscid[::stride], superposition, eigenvalues, gradient, normals)
    save_animation(render.animate_flow(mesh, vertices, frames), "eigen_fluids_bunny", frame_duration_ms)

    # The inviscid run of the animation, against the same start under viscosity.
    viscous = stack(tuple(core.advect(start, structure, eigenvalues, viscosity, dt, n_steps)))   # [steps + 1] Scalar+[modes]
    runs = stack((inviscid, viscous))                                            # [cases, steps + 1] Scalar+[modes]
    energies = (runs.squared() / eigenvalues).sites.sum() / 2                    # [cases, steps + 1] Scalar+
    enstrophies = runs.squared().sites.sum() / 2                                 # [cases, steps + 1] Scalar+
    inviscid_energy, viscous_energy = energies                                   # [steps + 1] Scalar+, Scalar+
    inviscid_enstrophy, viscous_enstrophy = enstrophies                          # [steps + 1] Scalar+, Scalar+

    times = np.arange(n_steps + 1) * dt
    save_figure(render.draw_conservation(times, inviscid_energy, viscous_energy, inviscid_enstrophy,
                                         viscous_enstrophy, start, inviscid[-1], viscous[-1]), "eigen_fluids_conservation")

    sphere_demo()

    # checks
    # The Laplace–Beltrami eigenmodes are orthonormal with respect to the lumped vertex mass form.
    gram = (modes[:, None] & mass(modes[None, :])).sites.sum().to_array()
    np.testing.assert_allclose(gram, np.eye(n_modes), atol=1e-12)
    # Every mode velocity and reconstructed flow velocity lies in its triangle's tangent plane.
    np.testing.assert_allclose((mode_velocities & patches).to_array(), 0.0, atol=1e-14)
    _, _, final_velocity = next(core.reconstruct(inviscid[-1:], superposition, eigenvalues, gradient, normals))
    np.testing.assert_allclose((final_velocity & patches).to_array(), 0.0, atol=1e-13)
    # On every triangle, the Jacobian wedge pairing equals the area-weighted directional derivative.
    face_gradients = gradient(modes[:6])                                         # [subset] Plane+[faces]
    wedge_brackets = (face_gradients[:, None] ^ face_gradients[None, :]) & patches   # [subset, subset] Scalar+[faces]
    directional = (face_gradients[:, None] & (face_gradients[None, :] ^ normals)) * areas   # [subset, subset] Scalar+[faces]
    np.testing.assert_allclose((wedge_brackets - directional).to_array(), 0.0, atol=1e-14)
    # The Lie–Poisson structure constants are totally antisymmetric in their three modes to machine precision.
    constants = structure.kernel[..., 0, 0, 0]                                   # [modes, modes, modes]
    np.testing.assert_allclose(constants + constants.swapaxes(1, 2), 0.0, atol=1e-13)
    np.testing.assert_allclose(constants + constants.swapaxes(0, 1), 0.0, atol=1e-13)
    # Instantaneous inviscid advection produces zero kinetic energy and zero enstrophy, and RK4 keeps both flat.
    initial_rate = core.vorticity_rate(start, structure, eigenvalues, 0.0)       # Scalar+[modes]
    np.testing.assert_allclose(((start / eigenvalues) * initial_rate).sites.sum().to_array(), 0.0, atol=1e-11)
    np.testing.assert_allclose((start * initial_rate).sites.sum().to_array(), 0.0, atol=1e-11)
    energy_trace = inviscid_energy.to_array()
    enstrophy_trace = inviscid_enstrophy.to_array()
    np.testing.assert_allclose(energy_trace, energy_trace[0], rtol=5e-5)
    np.testing.assert_allclose(enstrophy_trace, enstrophy_trace[0], rtol=5e-5)
    # Viscosity dissipates both invariants, damping high-wavenumber enstrophy faster than kinetic energy.
    viscous_energy_trace = viscous_energy.to_array()
    viscous_enstrophy_trace = viscous_enstrophy.to_array()
    assert np.all(np.diff(viscous_energy_trace) < 0)
    assert np.all(np.diff(viscous_enstrophy_trace) < 0)
    assert viscous_enstrophy_trace[-1] / viscous_enstrophy_trace[0] < viscous_energy_trace[-1] / viscous_energy_trace[0]


if __name__ == "__main__":
    main()
