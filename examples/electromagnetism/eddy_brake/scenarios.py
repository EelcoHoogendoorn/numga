"""Rotating conducting discs with isotropic, radial and circumferential conductivity."""

from collections.abc import Iterator

import numpy as np

from numga import stack
from examples.electromagnetism.eddy_brake import core
from examples.mesh import Mesh

RADIUS = 0.1                                       # m
THICKNESS = 0.001                                  # m
CONDUCTIVITY = 5.8e7                               # S/m
# Isotropic, radial-fibre and circumferential-fibre discs: the same mean conductivity in every
# case, redistributed between two directions.
RADIAL_CONDUCTIVITY = CONDUCTIVITY * np.array([1.0, 1.9, 0.1])
CIRCUMFERENTIAL_CONDUCTIVITY = CONDUCTIVITY * np.array([1.0, 0.1, 1.9])
ANGULAR_SPEED = 1.0                                # rad/s
FIELD_STRENGTH = 0.2                               # T
FIELD_WIDTH = 0.02                                 # m, Gaussian standard deviation
MAGNET_OFFSET = 0.055                              # m, along x
DIVISIONS = 16
INERTIA = 0.005                                    # kg m², disc and attached rotor
INITIAL_SPEED = 2.0                               # rad/s
INITIAL_ANGLE = 0.0                               # rad, straight fibres initially along x
ALONG_CONDUCTIVITY = CONDUCTIVITY * 1.9
ACROSS_CONDUCTIVITY = CONDUCTIVITY * 0.1
ELAPSED = 12.0                                    # s
STEPS = 120
STEP = ELAPSED / STEPS
DURATION_MS = 80
FIBRE_LINES = 12
FIBRE_SAMPLES = 64
FIBRE_INNER_RADIUS = 0.2 * RADIUS
FIBRE_OUTER_RADIUS = 0.85 * RADIUS
MATERIAL_RADII = np.array([FIBRE_INNER_RADIUS, FIBRE_OUTER_RADIUS, RADIUS])
PERMEABILITY = 4 * np.pi * 1e-7                   # H/m, nonmagnetic conductor in free space
CURRENT_MODES = 63
DRIVE_HARMONICS = 16
INDUCTIVE_FIELD_STRENGTH = 1.0                    # T
INDUCTIVE_INITIAL_SPEED = 0.0                     # rad/s
INDUCTIVE_TARGET_SPEED = 600.0                    # rad/s, unloaded motor pulse
SPINUP_TIME = 0.012                               # s
INDUCTIVE_ELAPSED = 0.1                           # s
INDUCTIVE_STEPS = 120
INDUCTIVE_STEP = INDUCTIVE_ELAPSED / INDUCTIVE_STEPS
ELECTRICAL_SUBSTEPS = 8


# --- math -----------------------------------------------------------------------------
def materials() -> tuple[Mesh, core.Conductivity]:
    """The disc mesh and its three material fields, each with the same mean conductivity."""
    mesh = Mesh.concentric_disk(MATERIAL_RADII, DIVISIONS)
    principal = THICKNESS * np.stack((RADIAL_CONDUCTIVITY, CIRCUMFERENTIAL_CONDUCTIVITY), axis=-1)  # [cases, axes]
    # Fibres occupy an annulus; the centre and rim conduct equally in both directions.
    conductivity = core.fibre_conductivity(mesh, principal, FIBRE_INNER_RADIUS, FIBRE_OUTER_RADIUS,
                                           THICKNESS * CONDUCTIVITY)  # [cases] Conductivity[F]
    return mesh, conductivity


def scene(mesh: Mesh, conductivity: core.Conductivity) -> tuple[core.Response, core.Vector]:
    """The steady currents, heating and torque of all three material fields."""
    centre = core.mv.x * MAGNET_OFFSET                                      # [] Vector
    magnetic = core.field(mesh.edge_midpoints, centre, core.mv.xy,
                          FIELD_STRENGTH, FIELD_WIDTH)                      # Bivector[E]
    spin = ANGULAR_SPEED * core.mv.xy                                       # [] Bivector
    response = core.solve(mesh, magnetic, spin, conductivity)
    return response, centre


def spin_down() -> tuple[Mesh, Iterator[core.Motion], core.Vector]:
    """A disc with straight fibres, released to slow under its own magnetic drag."""
    mesh = Mesh.triangular_disk(RADIUS, DIVISIONS)
    directions = stack((core.mv.x, core.mv.y))                               # [axes] Vector
    principal = THICKNESS * np.array([ALONG_CONDUCTIVITY, ACROSS_CONDUCTIVITY])
    conductivity = (directions * (directions | core.Vector) * principal).sum(axis=-1)[None]  # [cases] Conductivity
    centre = core.mv.x * MAGNET_OFFSET                                      # [] Vector
    orientation = (core.mv.xy * (-INITIAL_ANGLE / 2)).exp()[None]             # [cases] Rotor
    spin = (core.mv.xy * INITIAL_SPEED)[None]                                # [cases] Bivector
    heat = (mesh.triangle_areas * 0)[None]                                  # [cases] Scalar[F]
    frames = core.braking(mesh, centre, core.mv.xy, FIELD_STRENGTH, FIELD_WIDTH,
                          conductivity, INERTIA, orientation, spin, heat, STEP, STEPS)
    return mesh, frames, centre


def materials_braking(mesh: Mesh, conductivity: core.Conductivity) -> tuple[Iterator[core.Motion], core.Vector]:
    """The three material discs released at the same speed, retaining their Joule heat on the
    rotating material triangles."""
    centre = core.mv.x * MAGNET_OFFSET                                     # [] Vector
    orientation = core.mv.rotor().broadcast_to(conductivity.shape)          # [cases] Rotor
    spin = (core.mv.xy * INITIAL_SPEED).broadcast_to(orientation.shape)     # [cases] Bivector
    heat = (mesh.triangle_areas * 0).broadcast_to(conductivity.shape)       # [cases] Scalar[F]
    frames = core.braking(mesh, centre, core.mv.xy, FIELD_STRENGTH, FIELD_WIDTH,
                          conductivity, INERTIA, orientation, spin, heat, STEP, STEPS)
    return frames, centre


def inductive_materials(mesh: Mesh, conductivity: core.Conductivity) -> tuple[Iterator[core.InductiveMotion], core.Vector]:
    """Current build-up and magnetic braking, with inductive memory on the rotating discs."""
    modes = core.current_modes(mesh, conductivity, PERMEABILITY, CURRENT_MODES)
    centre = core.mv.x * MAGNET_OFFSET                                     # [] Vector
    orientation = core.mv.rotor().broadcast_to(conductivity.shape)          # [cases] Rotor
    spin = (core.mv.xy * INDUCTIVE_INITIAL_SPEED).broadcast_to(orientation.shape)  # [cases] Bivector
    heat = (mesh.triangle_areas * 0).broadcast_to(conductivity.shape)       # [cases] Scalar[F]
    amplitudes = modes.decay * 0                                           # [cases] Scalar[modes]
    forcing = core.periodic_drive(mesh, modes, centre, core.mv.xy,
                                   INDUCTIVE_FIELD_STRENGTH, FIELD_WIDTH, orientation, DRIVE_HARMONICS)
    # A half-sine torque pulse accelerates the rotor, then releases it to magnetic braking.
    times = (np.arange(INDUCTIVE_STEPS * ELECTRICAL_SUBSTEPS) + 0.5) * INDUCTIVE_STEP / ELECTRICAL_SUBSTEPS
    phase = np.clip(times / SPINUP_TIME, 0, 1)
    acceleration = INDUCTIVE_TARGET_SPEED * np.pi / (2 * SPINUP_TIME) * np.sin(np.pi * phase) * (times < SPINUP_TIME)
    motor_torque = core.mv.xy * (INERTIA * acceleration.reshape(INDUCTIVE_STEPS, ELECTRICAL_SUBSTEPS, 1))
    frames = core.inductive_braking(mesh, modes, forcing, core.mv.xy, INERTIA,
                                     orientation, spin, heat, amplitudes,
                                     motor_torque, INDUCTIVE_STEP)
    return frames, centre


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.animation import save_animation, save_figure
    from examples.electromagnetism.eddy_brake import render

    mesh, conductivity = materials()
    response, centre = scene(mesh, conductivity)
    save_figure(render.draw(mesh, response, centre, FIELD_WIDTH), "eddy_brake_materials")
    straight, frames, centre = spin_down()
    parallel = (render.parallel_fibres(RADIUS, FIBRE_LINES, FIBRE_SAMPLES),)
    save_animation(render.animate(straight, frames, centre, FIELD_WIDTH, parallel, np.ones((1, 1))),
                   "eddy_brake_spin_down", DURATION_MS)
    fibres = (render.radial_fibres(FIBRE_INNER_RADIUS, FIBRE_OUTER_RADIUS, FIBRE_LINES, FIBRE_SAMPLES),
              render.circular_fibres(FIBRE_INNER_RADIUS, FIBRE_OUTER_RADIUS, FIBRE_LINES, FIBRE_SAMPLES))
    # Each disc shows the guides of the direction it conducts along best; the isotropic disc has none.
    shown = np.stack((RADIAL_CONDUCTIVITY > CIRCUMFERENTIAL_CONDUCTIVITY,
                      CIRCUMFERENTIAL_CONDUCTIVITY > RADIAL_CONDUCTIVITY), axis=-1)  # [cases, families]
    frames, centre = materials_braking(mesh, conductivity)
    braked = list(frames)
    save_animation(render.animate(mesh, braked, centre, FIELD_WIDTH, fibres, shown),
                   "eddy_brake_materials_spin_down", DURATION_MS)
    save_animation(render.animate_heat(mesh, braked, centre, FIELD_WIDTH, fibres, shown),
                   "eddy_brake_materials_heat", DURATION_MS)
    frames, centre = inductive_materials(mesh, conductivity)
    inductive = list(frames)
    save_animation(render.animate_inductive(mesh, inductive, centre, FIELD_WIDTH, fibres, shown),
                   "eddy_brake_inductive_currents", DURATION_MS)
    save_animation(render.animate_heat(mesh, inductive, centre, FIELD_WIDTH, fibres, shown),
                   "eddy_brake_inductive_heat", DURATION_MS)

    # --- checks
    # The steady currents turn the mechanical power into heat. The braking discs conserve their
    # rotational energy plus heat; the inductive discs add the magnetic energy and the motor's work.
    spin = ANGULAR_SPEED * core.mv.xy
    np.testing.assert_allclose((spin | response.torque).kernel, response.heating.sites.sum().kernel, rtol=1e-12)
    energy = stack([state.heat.sites.sum() + INERTIA / 2 * state.spin.scalar_norm_squared() for state in braked])
    np.testing.assert_allclose((energy - energy[0]).kernel, 0, atol=1e-13)
    energy = stack([state.heat.sites.sum() + state.magnetic_energy - state.motor_work
                    + INERTIA / 2 * state.spin.scalar_norm_squared() for state in inductive])
    np.testing.assert_allclose((energy - energy[0]).kernel, 0, atol=1e-8)


if __name__ == "__main__":
    main()
