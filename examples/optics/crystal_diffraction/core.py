"""Spacetime deformations carry a crystal's atoms and its reciprocal diffraction peaks while
preserving wave phases.

Spacetime events are intrinsic antivectors and wavevectors are extrinsic vectors whose pairing
`k & x` measures the wave phase at an event. A planar stretch, shear, and rotation maps atom
events forward while leaving observer time unchanged, and solving through the map's adjugate
carries wavevectors forward without invoking the spacetime metric because
`deformation.adjugate()(k) & x == k & deformation(x)`. Circularly polarized incident light turns
in its wavefront plane as it advances, so each atom scatters the incident field rotated by half
its phase, and an observer at rest reads the scattered brightness from the field's energy density.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext
from numga.algebras import STA as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                                  # Scalar+
Wavevector = ga.gatype.vector()                                              # Vector+
Event = ga.gatype.antivector()                                               # Antivector-
Field = ga.gatype.bivector()                                                 # Bivector+
Deformation = ga.gatype((Event, Event))                                      # Antivector- <- Antivector-
# Circularly polarized light propagating along z turns its field in the xy wavefront plane.
wavefront_plane = mv.xy                                                      # [] Field+
# Rest-frame time direction pairing to +1 with the spatial slice `mv.t` under `t-` signature.
time = -mv.t.dual()                                                          # [] Event-
# Sandwiching `time` by a field gives twice its energy current, read against the observer's `mv.t` slice.
energy_density = 0.5 * (mv.t & (Field >> time))                              # [] Scalar+ <- (Field+, Field+)


# --- math -----------------------------------------------------------------------------
def layer_deformation(
    stretches: np.ndarray, shears: np.ndarray, angles: np.ndarray
) -> Deformation:
    """Planar stretches along x, shears from y into x, and xy rotations form spacetime event maps
    (`[...] Event- <- Event-`) that leave observer time fixed."""
    local_map = (
        Event
        + (stretches - 1) * mv.x.dual() * (mv.x & Event)
        + shears * mv.x.dual() * (mv.y & Event)
    )                                                                        # [...] Event- <- Event-
    # A positive angle rotates x toward y in the layer plane.
    rotors = (mv.xy * (-angles / 2)).exp()                                   # [...] Rotor+
    return rotors >> local_map                                               # [...] Event- <- Event-


def scattered_field(
    positions: Event, transfers: Wavevector, incident: Field
) -> Field:
    """Coherent scattering from equal atoms (`[..., n_atoms] Event-`) across `transfers`
    (`[..., n_rows, n_columns] Wavevector+`) sums the phase-turned `incident` wave into
    `[..., n_rows, n_columns] Field+`."""
    n_atoms = positions.shape[-1]
    # Elastic momentum transfers have no part along observer time, so event times drop out of the phase.
    phases = transfers[..., None] & positions[..., None, None, :]            # [..., n_atoms] Scalar+
    # Advancing a circularly polarized wave rotates its field in the wavefront plane with sense set by helicity.
    phase_rotors = (wavefront_plane * (phases / 2)).exp()                    # [..., n_atoms] Rotor+
    atom_waves = phase_rotors >> incident                                    # [..., n_atoms] Field+
    return atom_waves.sum(axis=-1) / n_atoms                                 # [...] Field+


# --- plumbing -------------------------------------------------------------------------
def square_lattice(
    n_shells: int, n_orders: int, spacing: float
) -> tuple[Event, Wavevector]:
    """A square layer of `(2 * n_shells + 1) ** 2` atom events at observer time zero (`[n_atoms] Event-`)
    pairs with `(2 * n_orders + 1) ** 2` reciprocal wavevectors (`[n_reflections] Wavevector+`)."""
    shell_indices = np.arange(-n_shells, n_shells + 1)
    order_indices = np.arange(-n_orders, n_orders + 1)
    atoms = (
        mv.x * shell_indices[None, :] + mv.y * shell_indices[:, None]
    ).reshape(-1).dual() * spacing
    lattice = (
        mv.x * order_indices[None, :] + mv.y * order_indices[:, None]
    ).reshape(-1) * (2 * np.pi / spacing)
    return atoms, lattice


def transfer_grid(half_width: float, n_pixels: int) -> Wavevector:
    """Uniformly spaced momentum transfers span `[-half_width, half_width]` across
    `[n_pixels, n_pixels] Wavevector+` in the crystal plane."""
    samples = np.linspace(-half_width, half_width, n_pixels)
    return mv.x * samples[None, :] + mv.y * samples[:, None]                 # [n_pixels, n_pixels] Wavevector+
