"""Eddy-current braking of rotating conducting discs with isotropic, radial and circumferential conductivity."""

from collections.abc import Iterator

import numpy as np

from numga import stack
from examples.electromagnetism.eddy_brake import core

Mesh = core.Mesh

radius = 0.1                                       # m
thickness = 0.001                                  # m
bulk_conductivity = 5.8e7                               # S/m
# Isotropic, radial-fibre and circumferential-fibre discs: the same mean conductivity in every
# case, redistributed between two directions.
radial_conductivity = bulk_conductivity * np.array([1.0, 1.9, 0.1])
circumferential_conductivity = bulk_conductivity * np.array([1.0, 0.1, 1.9])
angular_speed = 1.0                                # rad/s
field_strength = 0.2                               # T
field_width = 0.02                                 # m, Gaussian standard deviation
magnet_offset = 0.055                              # m, along x
n_divisions = 16
inertia = 0.005                                    # kg m², disc and attached rotor
initial_speed = 2.0                               # rad/s
initial_angle = 0.0                               # rad, straight fibres initially along x
along_conductivity = bulk_conductivity * 1.9
across_conductivity = bulk_conductivity * 0.1
elapsed = 12.0                                    # s
n_steps = 120
time_step = elapsed / n_steps
duration_ms = 80
n_fibre_lines = 12
n_fibre_samples = 64
fibre_inner_radius = 0.2 * radius
fibre_outer_radius = 0.85 * radius
material_radii = np.array([fibre_inner_radius, fibre_outer_radius, radius])
permeability = 4 * np.pi * 1e-7                   # H/m, nonmagnetic conductor in free space
n_modes = 63
n_harmonics = 16
inductive_field_strength = 1.0                    # T
inductive_initial_speed = 0.0                     # rad/s
inductive_target_speed = 600.0                    # rad/s, unloaded motor pulse
spinup_time = 0.012                               # s
inductive_elapsed = 0.1                           # s
n_inductive_steps = 120
inductive_time_step = inductive_elapsed / n_inductive_steps
n_substeps = 8


# --- math -----------------------------------------------------------------------------
def materials() -> tuple[core.Disc, core.Conductivity]:
    """The disc and its three material fields, each with the same mean conductivity."""
    disc = core.disc(*Mesh.concentric_disk(material_radii, n_divisions))
    principal = thickness * np.stack((radial_conductivity, circumferential_conductivity), axis=-1)  # [cases, axes]
    # Fibres occupy an annulus; the centre and rim conduct equally in both directions.
    conductivity = core.fibre_conductivity(disc, principal, fibre_inner_radius, fibre_outer_radius,
                                           thickness * bulk_conductivity)  # [cases] Conductivity[F]
    return disc, conductivity


def scene(disc: core.Disc, conductivity: core.Conductivity) -> tuple[core.Response, core.Point]:
    """The steady currents, heating and torque of all three material fields."""
    centre = (core.mv.x * magnet_offset).dual()                       # [] Point
    magnetic = core.mv.xy * field_strength * (-(disc.edge_midpoints - centre).squared() / (2 * field_width ** 2)).exp()   # Bivector[E]
    spin = angular_speed * core.mv.xy                                       # [] Bivector
    response = core.solve(disc, magnetic, spin, conductivity)
    return response, centre


def spin_down() -> tuple[core.Disc, Iterator[core.Motion], core.Point]:
    """A disc with straight fibres, released to slow under its own magnetic drag."""
    disc = core.disc(*Mesh.triangular_disk(radius, n_divisions))
    directions = stack((core.mv.x.dual(), core.mv.y.dual()))                 # [axes] Current
    principal = thickness * np.array([along_conductivity, across_conductivity])
    conductivity = (directions * (core.Field & (directions & core.time)) * principal).sum(axis=-1)[None]   # [cases] Conductivity
    centre = (core.mv.x * magnet_offset).dual()                       # [] Point
    orientation = (core.mv.xy * (-initial_angle / 2)).exp()[None]             # [cases] Rotor
    spin = (core.mv.xy * initial_speed)[None]                                # [cases] Bivector
    heat = (disc.triangle_areas * 0)[None]                                  # [cases] Scalar[F]
    frames = core.braking(disc, centre, core.mv.xy, field_strength, field_width,
                          conductivity, inertia, orientation, spin, heat, time_step, n_steps)
    return disc, frames, centre


def materials_braking(disc: core.Disc, conductivity: core.Conductivity) -> tuple[Iterator[core.Motion], core.Point]:
    """The three material discs released at the same speed, retaining their Joule heat on the
    rotating material triangles."""
    centre = (core.mv.x * magnet_offset).dual()                      # [] Point
    orientation = core.mv.rotor().broadcast_to(conductivity.shape)          # [cases] Rotor
    spin = (core.mv.xy * initial_speed).broadcast_to(orientation.shape)     # [cases] Bivector
    heat = (disc.triangle_areas * 0).broadcast_to(conductivity.shape)       # [cases] Scalar[F]
    frames = core.braking(disc, centre, core.mv.xy, field_strength, field_width,
                          conductivity, inertia, orientation, spin, heat, time_step, n_steps)
    return frames, centre


def inductive_materials(disc: core.Disc, conductivity: core.Conductivity) -> tuple[Iterator[core.InductiveMotion], core.Point]:
    """Current build-up and magnetic braking, with inductive memory on the rotating discs."""
    modes = core.current_modes(disc, conductivity, permeability, n_modes)
    centre = (core.mv.x * magnet_offset).dual()                      # [] Point
    orientation = core.mv.rotor().broadcast_to(conductivity.shape)          # [cases] Rotor
    spin = (core.mv.xy * inductive_initial_speed).broadcast_to(orientation.shape)  # [cases] Bivector
    heat = (disc.triangle_areas * 0).broadcast_to(conductivity.shape)       # [cases] Scalar[F]
    amplitudes = modes.decay * 0                                           # [cases] Scalar[modes]
    forcing = core.periodic_drive(disc, modes, centre, core.mv.xy,
                                   inductive_field_strength, field_width, orientation, n_harmonics)
    # A half-sine torque pulse accelerates the rotor, then releases it to magnetic braking.
    times = (np.arange(n_inductive_steps * n_substeps) + 0.5) * inductive_time_step / n_substeps
    phase = np.clip(times / spinup_time, 0, 1)
    acceleration = inductive_target_speed * np.pi / (2 * spinup_time) * np.sin(np.pi * phase) * (times < spinup_time)
    # The motor's torque about the turning plane, in its sense.
    motor_torque = core.mv.scalar(inertia * acceleration.reshape(n_inductive_steps, n_substeps, 1, 1))
    frames = core.inductive_braking(disc, modes, forcing, core.mv.xy, inertia,
                                     orientation, spin, heat, amplitudes,
                                     motor_torque, inductive_time_step)
    return frames, centre


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.electromagnetism.eddy_brake import render

    disc, conductivity = materials()
    response, centre = scene(disc, conductivity)
    save_figure(render.draw(disc.mesh, disc.vertices, disc.triangle_areas, response, centre, field_width),
                "eddy_brake_materials")
    straight, frames, centre = spin_down()
    parallel = (render.parallel_fibres(radius, n_fibre_lines, n_fibre_samples),)
    save_animation(render.animate(straight.mesh, straight.vertices, straight.triangle_areas, frames, centre, field_width,
                                  parallel, np.ones((1, 1))),
                   "eddy_brake_spin_down", duration_ms)
    fibres = (render.radial_fibres(fibre_inner_radius, fibre_outer_radius, n_fibre_lines, n_fibre_samples),
              render.circular_fibres(fibre_inner_radius, fibre_outer_radius, n_fibre_lines, n_fibre_samples))
    # Each disc shows the guides of the direction it conducts along best; the isotropic disc has none.
    shown = np.stack((radial_conductivity > circumferential_conductivity,
                      circumferential_conductivity > radial_conductivity), axis=-1)  # [cases, families]
    frames, centre = materials_braking(disc, conductivity)
    braked = list(frames)
    save_animation(render.animate(disc.mesh, disc.vertices, disc.triangle_areas, braked, centre, field_width, fibres,
                                  shown),
                   "eddy_brake_materials_spin_down", duration_ms)
    save_animation(render.animate_heat(disc.mesh, disc.vertices, disc.triangle_areas, braked, centre, field_width, fibres,
                                       shown),
                   "eddy_brake_materials_heat", duration_ms)
    frames, centre = inductive_materials(disc, conductivity)
    inductive = list(frames)
    save_animation(render.animate_induction(disc.mesh, disc.vertices, disc.triangle_areas, inductive, centre, field_width,
                                            fibres, shown),
                   "eddy_brake_inductive_currents", duration_ms)
    save_animation(render.animate_induction_heat(disc.mesh, disc.vertices, disc.triangle_areas, inductive, centre, field_width,
                                                 fibres, shown),
                   "eddy_brake_inductive_heat", duration_ms)

    # --- checks
    # The steady currents turn the mechanical power into heat. The braking discs conserve their
    # rotational energy plus heat; the inductive discs add the magnetic energy and the motor's work.
    spin = angular_speed * core.mv.xy
    np.testing.assert_allclose((-(spin & response.torque)).kernel, response.heating.sites.sum().kernel, rtol=1e-12)
    energy = stack([state.heat.sites.sum() + inertia / 2 * state.spin.scalar_norm_squared() for state in braked])
    np.testing.assert_allclose((energy - energy[0]).kernel, 0, atol=1e-13)
    energy = stack([state.heat.sites.sum() + state.magnetic_energy - state.motor_work
                    + inertia / 2 * state.spin.scalar_norm_squared() for state in inductive])
    np.testing.assert_allclose((energy - energy[0]).kernel, 0, atol=1e-8)


if __name__ == "__main__":
    main()
