"""Closed work cycles and overdamped waves from powered elastic bonds."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga.sparse import spdiag
from examples.mechanics.odd_elasticity import core

mv = core.mv
Scalar, Antivector, Vector, Shear = core.Scalar, core.Antivector, core.Vector, core.Shear
x, y = core.directions

# The bonds of the two work cycles: their stiffness, their activity, and the plane their sideways push
# turns in.
STIFFNESS = 1
ACTIVITY = 3
ACTUATION_PLANE = mv.xy                                                     # Bivector
N_CYCLE_STEPS = 72
# One bond: its rest length, and the radii of the ellipse its free end follows, along and across it.
REST_LENGTH = 1
PARALLEL_RADIUS = 0.25
TRANSVERSE_RADIUS = 0.18
# A small patch sheared around a circle of shear amplitudes.
SPACING = 1
SHEAR_AMPLITUDE = 0.14
N_PATCH_COLUMNS, N_PATCH_ROWS = 7, 7
# A wider lattice, released from a localized displacement: its bonds, the drag on every particle, the
# displacement's amplitude, width and wavenumber, and the time step.
WAVE_STIFFNESS = 0.2
WAVE_ACTIVITY = 4
DRAG = 1
WAVE_AMPLITUDE = 0.15
WAVE_WIDTH = 5
WAVE_NUMBER = 1.2
N_WAVE_COLUMNS, N_WAVE_ROWS = 41, 21
N_WAVE_FRAMES = 100
TIME_STEP = 0.06


# --- math -----------------------------------------------------------------------------
def turns(n_steps: int) -> Antivector:
    """A unit direction turned once around the actuation plane, `[steps + 1]`."""
    phases = np.linspace(0, 2 * np.pi, n_steps + 1)
    return (-ACTUATION_PLANE * phases / 2).exp() >> x                          # [steps + 1] Antivector


def bond_cycle(n_steps: int) -> tuple[Antivector, Antivector, Antivector, Iterator[tuple[Antivector, Vector, Scalar]]]:
    """An active bond delivers work as its endpoint follows an ellipse."""
    direction = x                                                             # Antivector
    anchor = 0 * direction                                                    # Antivector
    reference = REST_LENGTH * direction                                       # Antivector
    response = core.bond_response(direction, ACTUATION_PLANE, STIFFNESS, ACTIVITY)   # Vector <- Antivector
    across = direction | ACTUATION_PLANE                                      # Antivector
    # The ellipse stretches the unit circle by its two radii, along the bond and across it.
    ellipse = (PARALLEL_RADIUS * direction * (direction.dual_inverse() & Antivector)
               + TRANSVERSE_RADIUS * across * (across.dual_inverse() & Antivector))   # Antivector <- Antivector
    cycle = ellipse(turns(n_steps))                                           # [steps + 1] Antivector
    states = ((reference + displacement, force, work)
              for displacement, force, work in core.integrate_cycle(response, cycle))
    return anchor, reference, reference + cycle, states


def shear_cycle(n_steps: int) -> tuple[Antivector, np.ndarray, Iterator[tuple[Antivector, Vector, Scalar]]]:
    """Two shear deformations drive a patch around a closed cycle; work is per lattice cell."""
    reference, edges, incidence = core.triangular_patch(N_PATCH_COLUMNS, N_PATCH_ROWS, SPACING)

    # The two shear amplitudes stretch along the axes, or along the diagonals between them.
    horizontal_shear = x * (mv.x & Antivector) - y * (mv.y & Antivector)       # Antivector <- Antivector
    diagonal_shear = x * (mv.y & Antivector) + y * (mv.x & Antivector)         # Antivector <- Antivector
    strain = (mv.x & Shear) * horizontal_shear + (mv.y & Shear) * diagonal_shear   # Antivector <- (Shear, Antivector)
    # One bond of each of the three families in a lattice cell.
    bond_angles = np.arange(3) * np.pi / 3
    bond_directions = (-ACTUATION_PLANE * bond_angles / 2).exp() >> x          # [families] Antivector
    response = core.bond_response(bond_directions, ACTUATION_PLANE, STIFFNESS, ACTIVITY)   # [families] Vector <- Antivector
    bond_motion = strain(Shear, SPACING * bond_directions)                    # [families] Antivector <- Shear
    # The adjugate returns the bond forces to the shear amplitudes with their work kept.
    shear_response = bond_motion.adjugate()(response(bond_motion)).sum(axis=0)   # Vector <- Shear

    # Summing the same bond forces over the finite patch reveals its boundary loads.
    gradient = incidence * Antivector                                         # Antivector[bonds] <- Antivector[sites]
    edge_response = core.bond_response(gradient(reference).normalized(), ACTUATION_PLANE, STIFFNESS, ACTIVITY)   # (Vector <- Antivector)[bonds]
    patch_response = gradient.adjugate()(spdiag(edge_response)(gradient))     # Vector[sites] <- Antivector[sites]
    patch_motion = strain(Shear, reference)                                   # Antivector[sites] <- Shear
    patch_forces = patch_response(patch_motion)                               # Vector[sites] <- Shear
    cycle = SHEAR_AMPLITUDE * turns(n_steps)                                  # [steps + 1] Shear
    states = ((reference + patch_motion(shear), patch_forces(shear), work)
              for shear, _, work in core.integrate_cycle(shear_response, cycle))
    return reference, edges, states


def lattice_wave(n_frames: int, dt: float) -> tuple[Antivector, np.ndarray, Iterator[tuple[Antivector, Scalar]]]:
    """A localized displacement excites damped waves in a free triangular patch."""
    reference, edges, incidence = core.triangular_patch(N_WAVE_COLUMNS, N_WAVE_ROWS, SPACING)
    bond_directions = (incidence * reference).normalized()                    # Antivector[bonds]
    response = core.bond_response(bond_directions, ACTUATION_PLANE, WAVE_STIFFNESS, WAVE_ACTIVITY)   # (Vector <- Antivector)[bonds]
    drags = mv.scalar([[DRAG]] * N_WAVE_COLUMNS * N_WAVE_ROWS).field()         # Scalar[sites]
    evolution = core.evolution(incidence, response, drags)                    # Antivector[sites] <- Antivector[sites]
    envelope = (-reference.scalar_norm_squared() / (2 * WAVE_WIDTH**2)).exp()   # Scalar[sites]
    displacement = WAVE_AMPLITUDE * x * envelope * (WAVE_NUMBER * (mv.x & reference)).cos()   # Antivector[sites]
    return reference, edges, core.wave_states(reference, incidence, bond_directions, evolution, displacement, n_frames, dt)


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_animation
    from examples.mechanics.odd_elasticity import render

    duration_ms = 70
    force_scale = 0.55
    patch_force_scale = 1.3
    displacement_scale = 4
    extension_scale = 0.04
    tracked_index = (N_WAVE_ROWS // 2) * N_WAVE_COLUMNS + N_WAVE_COLUMNS // 2

    anchor, reference, loop, bond_states = bond_cycle(N_CYCLE_STEPS)
    bond_states = list(bond_states)
    save_animation(render.bond_frames(anchor, reference, loop, [state[:2] for state in bond_states], force_scale),
                   "odd_elasticity_bond", duration_ms)
    patch, edges, shear_states = shear_cycle(N_CYCLE_STEPS)
    shear_states = list(shear_states)
    save_animation(render.patch_frames(patch, edges, [state[:2] for state in shear_states], patch_force_scale),
                   "odd_elasticity_shear", duration_ms)
    save_animation(render.wave_frames(*lattice_wave(N_WAVE_FRAMES, TIME_STEP), displacement_scale,
                                      extension_scale, tracked_index), "odd_elasticity_wave", duration_ms)

    # checks
    # Around each loop the bonds deliver the activity times the area enclosed, oriented by the actuation
    # plane: for one bond, the ellipse its end follows; for the patch, three bonds per cell over the
    # circle of shear amplitudes.
    ellipse = loop - reference                                                # [steps + 1] Antivector
    ellipse_area = (ellipse[:-1] ^ ellipse[1:]).sum(axis=0) / 2                # Bivector
    shear_loop = SHEAR_AMPLITUDE * turns(N_CYCLE_STEPS)                        # [steps + 1] Shear
    shear_area = (shear_loop[:-1] ^ shear_loop[1:]).sum(axis=0) / 2            # Bivector
    bond_work = -ACTIVITY * ACTUATION_PLANE.scalar_product(ellipse_area)
    shear_work = -3 * ACTIVITY * ACTUATION_PLANE.scalar_product(shear_area)
    np.testing.assert_allclose(bond_states[-1][2].to_array(), bond_work.to_array(), atol=1e-12)
    np.testing.assert_allclose(shear_states[-1][2].to_array(), shear_work.to_array(), atol=1e-12)


if __name__ == "__main__":
    main()
