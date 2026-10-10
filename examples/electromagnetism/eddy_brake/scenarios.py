"""Rotating conducting discs with isotropic, radial and circumferential conductivity."""

from collections.abc import Iterator

import numpy as np

from numga import stack
from examples.electromagnetism.eddy_brake import core
from examples.mesh import Mesh, as_scalar

RADIUS = 0.1                                       # m
THICKNESS = 0.001                                  # m
CONDUCTIVITY = 5.8e7                               # S/m
LABELS = ("Isotropic", "Radial fibres", "Circumferential fibres")
# The same mean conductivity in every case, redistributed between two directions.
RADIAL_CONDUCTIVITY = CONDUCTIVITY * np.array([1.0, 1.9, 0.1])
CIRCUMFERENTIAL_CONDUCTIVITY = CONDUCTIVITY * np.array([1.0, 0.1, 1.9])
ANGULAR_SPEED = 1.0                                # rad/s
FIELD_STRENGTH = 0.2                               # T
FIELD_WIDTH = 0.02                                 # m, Gaussian standard deviation
MAGNET_OFFSET = 0.055                              # m, along x
DIVISIONS = 32
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
STRAIGHT_LABELS = ("Straight fibres",)
PERMEABILITY = 4 * np.pi * 1e-7                   # H/m, nonmagnetic conductor in free space
INDUCTIVE_DIVISIONS = DIVISIONS
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
def scene() -> tuple[Mesh, core.Response, core.Vector]:
    """The steady currents, heating and torque of all three material fields."""
    mesh = Mesh.concentric_disk(MATERIAL_RADII, DIVISIONS)
    radial = mesh.face_centers.normalized()                                 # Vector[F]
    directions = stack((radial, core.mv.xy | radial), axis=-1)               # [axes] Vector[F]
    principal = THICKNESS * np.stack((RADIAL_CONDUCTIVITY, CIRCUMFERENTIAL_CONDUCTIVITY), axis=-1)
    # Fibres occupy an annulus; the centre and rim conduct equally in both directions.
    distance = mesh.face_centers.norm()                                    # Scalar[F]
    fibre_band = as_scalar((distance >= FIBRE_INNER_RADIUS) & (distance <= FIBRE_OUTER_RADIUS)).field()
    mean_conductance = THICKNESS * CONDUCTIVITY
    local_principal = mean_conductance + fibre_band * (principal - mean_conductance)  # [cases, axes] Scalar[F]
    # Each dyad conducts along one direction; summing the weighted dyads gives the material law.
    conductivity = (directions * (directions | core.Vector) * local_principal).sum(axis=-1)  # [cases] Vector[F] <- Vector
    centre = core.mv.x * MAGNET_OFFSET                                      # [] Vector
    magnetic = core.field(mesh.edge_midpoints, centre, core.mv.xy,
                          FIELD_STRENGTH, FIELD_WIDTH)                      # Bivector[E]
    spin = ANGULAR_SPEED * core.mv.xy                                       # [] Bivector
    response = core.solve(mesh, magnetic, spin, conductivity)
    return mesh, response, centre


def spin_down() -> tuple[Mesh, Iterator[core.Motion], core.Vector]:
    """A disc with straight fibres, released to slow under its own magnetic drag."""
    mesh = Mesh.triangular_disk(RADIUS, DIVISIONS)
    directions = stack((core.mv.x, core.mv.y))                               # [axes] Vector
    principal = THICKNESS * np.array([ALONG_CONDUCTIVITY, ACROSS_CONDUCTIVITY])
    conductivity = (directions * (directions | core.Vector) * principal).sum(axis=-1)[None]  # [cases] Conductivity
    centre = core.mv.x * MAGNET_OFFSET                                      # [] Vector
    magnetic = core.field(mesh.edge_midpoints, centre, core.mv.xy,
                          FIELD_STRENGTH, FIELD_WIDTH)                      # Bivector[E]
    orientation = (core.mv.xy * (-INITIAL_ANGLE / 2)).exp()[None]             # [cases] Rotor
    spin = (core.mv.xy * INITIAL_SPEED)[None]                                # [cases] Bivector
    return mesh, core.braking(mesh, magnetic, core.mv.xy, conductivity,
                              INERTIA, orientation, spin, STEP, STEPS), centre


def materials_spin_down() -> tuple[Mesh, Iterator[core.Motion], core.Vector]:
    """The three rotationally invariant material patterns released at the same speed."""
    mesh, response, centre = scene()
    unit_response = response.scaled(core.mv.scalar([1 / ANGULAR_SPEED]))
    spin = core.mv.xy * INITIAL_SPEED                                       # [] Bivector
    frames = core.stationary_braking(unit_response, core.mv.xy, INERTIA, spin, STEP, STEPS)
    return mesh, frames, centre


def materials_heating() -> tuple[Mesh, Iterator[core.ThermalMotion], core.Vector]:
    """Joule heat retained by rotating material triangles as the three discs slow."""
    mesh = Mesh.concentric_disk(MATERIAL_RADII, DIVISIONS)
    radial = mesh.face_centers.normalized()                                # Vector[F]
    directions = stack((radial, core.mv.xy | radial), axis=-1)              # [axes] Vector[F]
    principal = THICKNESS * np.stack((RADIAL_CONDUCTIVITY, CIRCUMFERENTIAL_CONDUCTIVITY), axis=-1)
    # Fibres occupy an annulus; the centre and rim conduct equally in both directions.
    distance = mesh.face_centers.norm()                                    # Scalar[F]
    fibre_band = as_scalar((distance >= FIBRE_INNER_RADIUS) & (distance <= FIBRE_OUTER_RADIUS)).field()
    mean_conductance = THICKNESS * CONDUCTIVITY
    local_principal = mean_conductance + fibre_band * (principal - mean_conductance)  # [cases, axes] Scalar[F]
    conductivity = (directions * (directions | core.Vector) * local_principal).sum(axis=-1)  # [cases] Conductivity[F]
    centre = core.mv.x * MAGNET_OFFSET                                     # [] Vector
    orientation = core.mv.rotor().broadcast_to(conductivity.shape)          # [cases] Rotor
    spin = (core.mv.xy * INITIAL_SPEED).broadcast_to(orientation.shape)     # [cases] Bivector
    heat = (mesh.triangle_areas * 0).broadcast_to(conductivity.shape)       # [cases] Scalar[F]
    frames = core.material_braking(mesh, centre, core.mv.xy, FIELD_STRENGTH, FIELD_WIDTH,
                                    conductivity, INERTIA, orientation, spin, heat, STEP, STEPS)
    return mesh, frames, centre


def inductive_materials() -> tuple[Mesh, Iterator[core.InductiveMotion], core.Vector]:
    """Current build-up and magnetic braking, with inductive memory on the rotating discs."""
    mesh = Mesh.concentric_disk(MATERIAL_RADII, INDUCTIVE_DIVISIONS)
    radial = mesh.face_centers.normalized()                                # Vector[F]
    directions = stack((radial, core.mv.xy | radial), axis=-1)              # [axes] Vector[F]
    principal = THICKNESS * np.stack((RADIAL_CONDUCTIVITY, CIRCUMFERENTIAL_CONDUCTIVITY), axis=-1)
    # Fibres occupy an annulus; the centre and rim conduct equally in both directions.
    distance = mesh.face_centers.norm()                                    # Scalar[F]
    fibre_band = as_scalar((distance >= FIBRE_INNER_RADIUS) & (distance <= FIBRE_OUTER_RADIUS)).field()
    mean_conductance = THICKNESS * CONDUCTIVITY
    local_principal = mean_conductance + fibre_band * (principal - mean_conductance)  # [cases, axes] Scalar[F]
    conductivity = (directions * (directions | core.Vector) * local_principal).sum(axis=-1)  # [cases] Conductivity[F]
    modes = core.current_modes(mesh, conductivity, PERMEABILITY, CURRENT_MODES)
    centre = core.mv.x * MAGNET_OFFSET                                     # [] Vector
    orientation = core.mv.rotor().broadcast_to(conductivity.shape)          # [cases] Rotor
    spin = (core.mv.xy * INDUCTIVE_INITIAL_SPEED).broadcast_to(orientation.shape)  # [cases] Bivector
    heat = (mesh.triangle_areas * 0).broadcast_to(conductivity.shape)       # [cases] Scalar[F]
    amplitudes = core.mv.scalar([0]).broadcast_to(modes.decay.shape)        # [cases, modes] Scalar
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
    return mesh, frames, centre


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.animation import save_animation, save_figure
    from examples.electromagnetism.eddy_brake import render

    mesh, response, centre = scene()
    save_figure(render.draw(mesh, response, centre, FIELD_WIDTH, LABELS), "eddy_brake_materials")
    mesh, frames, centre = spin_down()
    fibres = (render.parallel_fibres(RADIUS, FIBRE_LINES, FIBRE_SAMPLES),)
    save_animation(render.animate(mesh, frames, centre, FIELD_WIDTH, STRAIGHT_LABELS, fibres),
                   "eddy_brake_spin_down", DURATION_MS)
    mesh, frames, centre = materials_spin_down()
    radial = render.radial_fibres(FIBRE_INNER_RADIUS, FIBRE_OUTER_RADIUS, FIBRE_LINES, FIBRE_SAMPLES)
    circular = render.circular_fibres(FIBRE_INNER_RADIUS, FIBRE_OUTER_RADIUS, FIBRE_LINES, FIBRE_SAMPLES)
    fibres = (radial[:0], radial, circular)
    save_animation(render.animate(mesh, frames, centre, FIELD_WIDTH, LABELS, fibres),
                   "eddy_brake_materials_spin_down", DURATION_MS)
    mesh, frames, centre = materials_heating()
    save_animation(render.animate_heat(mesh, frames, centre, FIELD_WIDTH, LABELS, fibres),
                   "eddy_brake_materials_heat", DURATION_MS)
    mesh, frames, centre = inductive_materials()
    states = list(frames)
    save_animation(render.animate_inductive(mesh, states, centre, FIELD_WIDTH, LABELS, fibres),
                   "eddy_brake_inductive_currents", DURATION_MS)
    save_animation(render.animate_heat(mesh, states, centre, FIELD_WIDTH, LABELS, fibres),
                   "eddy_brake_inductive_heat", DURATION_MS)


if __name__ == "__main__":
    main()
