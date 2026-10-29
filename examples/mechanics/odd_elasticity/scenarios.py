"""Closed work cycles and overdamped waves from powered elastic bonds."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga.sparse import spdiag
from examples.mechanics.odd_elasticity import core

mv = core.mv
Scalar, Displacement, Force, Shear = core.Scalar, core.Displacement, core.Force, core.Shear
x, y = core.directions

# The bonds of the two work cycles: their stiffness, their activity, and the plane their sideways push
# turns in.
STIFFNESS = 1
ACTIVITY = 3
actuation_plane = mv.xy                                                       # Bivector+
N_CYCLE_STEPS = 126
# One bond: its rest length, and the radii of the ellipse its free end follows, along and across it.
REST_LENGTH = 1
PARALLEL_RADIUS = 0.25
TRANSVERSE_RADIUS = 0.18
# A small patch sheared around a circle of shear amplitudes.
SPACING = 1
SHEAR_AMPLITUDE = 0.14
N_PATCH_COLUMNS, N_PATCH_ROWS = 7, 7
# A wider lattice, released from a localized displacement: its bonds, the drag on every particle, the
# displacement's amplitude, width and wavenumber, and the frames, the steps per frame and the time step.
WAVE_STIFFNESS = 0.2
WAVE_ACTIVITY = 4
DRAG = 1
WAVE_AMPLITUDE = 0.15
WAVE_WIDTH = 2.5
WAVE_NUMBER = 1.2
N_WAVE_COLUMNS, N_WAVE_ROWS = 21, 11
N_WAVE_FRAMES = 180
N_WAVE_STEPS_PER_FRAME = 2
TIME_STEP = 1 / 60


# --- math -----------------------------------------------------------------------------
def turns(n_steps: int) -> Displacement:
    """A unit direction turned once around the actuation plane, `[steps + 1]`."""
    phases = np.linspace(0, 2 * np.pi, n_steps + 1)
    return (-actuation_plane * phases / 2).exp() >> x                         # [steps + 1] Displacement-


def bond_cycle(n_steps: int) -> tuple[Displacement, Displacement, Displacement, Iterator[tuple[Displacement, Force, Scalar]]]:
    """An active bond delivers work as its endpoint follows an ellipse."""
    direction = x                                                             # Displacement-
    anchor = 0 * direction                                                    # Displacement-
    reference = REST_LENGTH * direction                                       # Displacement-
    response = core.bond_response(direction, actuation_plane, STIFFNESS, ACTIVITY)   # Force+ <- Displacement-
    across = direction | actuation_plane                                      # Displacement-
    # The ellipse stretches the unit circle by its two radii, along the bond and across it.
    ellipse = (PARALLEL_RADIUS * direction * (direction.dual_inverse() & Displacement)
               + TRANSVERSE_RADIUS * across * (across.dual_inverse() & Displacement))   # Displacement- <- Displacement-
    cycle = ellipse(turns(n_steps))                                           # [steps + 1] Displacement-
    states = ((reference + displacement, force, work)
              for displacement, force, work in core.integrate_cycle(response, cycle))
    return anchor, reference, reference + cycle, states


def shear_cycle(n_steps: int) -> tuple[Displacement, np.ndarray, Iterator[tuple[Displacement, Force, Scalar]]]:
    """Two shear deformations drive a patch around a closed cycle; work is per lattice cell."""
    reference, edges, incidence = core.triangular_patch(N_PATCH_COLUMNS, N_PATCH_ROWS, SPACING)

    # The two shear amplitudes stretch along the axes, or along the diagonals between them.
    horizontal_shear = x * (mv.x & Displacement) - y * (mv.y & Displacement)  # Displacement- <- Displacement-
    diagonal_shear = x * (mv.y & Displacement) + y * (mv.x & Displacement)    # Displacement- <- Displacement-
    strain = (mv.x & Shear) * horizontal_shear + (mv.y & Shear) * diagonal_shear   # Displacement- <- (Shear-, Displacement-)
    # One bond of each of the three families in a lattice cell.
    bond_angles = np.arange(3) * np.pi / 3
    bond_directions = (-actuation_plane * bond_angles / 2).exp() >> x         # [families] Displacement-
    response = core.bond_response(bond_directions, actuation_plane, STIFFNESS, ACTIVITY)   # [families] Force+ <- Displacement-
    bond_motion = strain(Shear, SPACING * bond_directions)                    # [families] Displacement- <- Shear-
    # The adjugate returns the bond forces to the shear amplitudes with their work kept.
    shear_response = bond_motion.adjugate()(response(bond_motion)).sum(axis=0)   # Force+ <- Shear-

    # Summing the same bond forces over the finite patch reveals its boundary loads.
    gradient = incidence * Displacement                                       # (Displacement- <- Displacement-)[bonds <- sites]
    edge_response = core.bond_response(gradient(reference).normalized(), actuation_plane, STIFFNESS, ACTIVITY)   # (Force+ <- Displacement-)[bonds]
    patch_response = gradient.adjugate()(spdiag(edge_response)(gradient))     # (Force+ <- Displacement-)[sites <- sites]
    patch_motion = strain(Shear, reference)                                   # (Displacement- <- Shear-)[sites]
    patch_forces = patch_response(patch_motion)                               # (Force+ <- Shear-)[sites]
    cycle = SHEAR_AMPLITUDE * turns(n_steps)                                  # [steps + 1] Shear-
    states = ((reference + patch_motion(shear), patch_forces(shear), work)
              for shear, _, work in core.integrate_cycle(shear_response, cycle))
    return reference, edges, states


def lattice_wave(n_frames: int, n_steps_per_frame: int, dt: float) -> tuple[Displacement, np.ndarray, Iterator[tuple[Displacement, Scalar]]]:
    """A localized displacement excites damped waves in a free triangular patch."""
    reference, edges, incidence = core.triangular_patch(N_WAVE_COLUMNS, N_WAVE_ROWS, SPACING)
    bond_directions = (incidence * reference).normalized()                    # Displacement-[bonds]
    response = core.bond_response(bond_directions, actuation_plane, WAVE_STIFFNESS, WAVE_ACTIVITY)   # (Force+ <- Displacement-)[bonds]
    drags = mv.scalar([[DRAG]] * N_WAVE_COLUMNS * N_WAVE_ROWS).field()        # Scalar+[sites]
    evolution = core.evolution(incidence, response, drags)                    # (Velocity- <- Displacement-)[sites <- sites]
    envelope = (-reference.scalar_norm_squared() / (2 * WAVE_WIDTH**2)).exp()   # Scalar+[sites]
    displacement = WAVE_AMPLITUDE * x * envelope * (WAVE_NUMBER * (mv.x & reference)).cos()   # Displacement-[sites]
    return reference, edges, core.wave_states(reference, incidence, bond_directions, evolution, displacement, n_frames, n_steps_per_frame, dt)


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_animation
    from examples.mechanics.odd_elasticity import render

    duration_ms = 40
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
    save_animation(render.wave_frames(*lattice_wave(N_WAVE_FRAMES, N_WAVE_STEPS_PER_FRAME, TIME_STEP), displacement_scale,
                                      extension_scale, tracked_index), "odd_elasticity_wave", duration_ms)

    # checks
    # Around each loop the bonds deliver the activity times the area enclosed, oriented by the actuation
    # plane: for one bond, the ellipse its end follows; for the patch, three bonds per cell over the
    # circle of shear amplitudes.
    ellipse = loop - reference                                                # [steps + 1] Displacement-
    ellipse_area = (ellipse[:-1] ^ ellipse[1:]).sum(axis=0) / 2               # Bivector+
    shear_loop = SHEAR_AMPLITUDE * turns(N_CYCLE_STEPS)                       # [steps + 1] Shear-
    shear_area = (shear_loop[:-1] ^ shear_loop[1:]).sum(axis=0) / 2           # Bivector+
    bond_work = -ACTIVITY * actuation_plane.scalar_product(ellipse_area)
    shear_work = -3 * ACTIVITY * actuation_plane.scalar_product(shear_area)
    np.testing.assert_allclose(bond_states[-1][2].to_array(), bond_work.to_array(), atol=1e-12)
    np.testing.assert_allclose(shear_states[-1][2].to_array(), shear_work.to_array(), atol=1e-12)


if __name__ == "__main__":
    main()
