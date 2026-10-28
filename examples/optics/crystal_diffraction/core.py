"""A crystal's atoms and its diffraction peaks carried through a deformation, in the algebra of spacetime.

Events are antivectors and wavevectors vectors, and the phase of a wave at an event is their pairing, `k & x`.
The deformation, a map on events, the identity `Antivector` plus a stretch, a shear and a turn, moves the atoms
and leaves the observer's time alone; its adjugate is the map it induces on wavevectors,
`deformation.adjugate()(k) & x == k & deformation(x)`, so solving through it moves the wavevectors so that every
phase is kept, without the metric. The incident light is circularly polarized: as it advances, its field turns in the
plane of its wavefront, `mv.xy`, and each atom scatters it turned by the atom's phase, the rotor
`(mv.xy * (phases / 2)).exp()`, whose sense is the light's helicity. The brightness is the energy density
of the summed field that the observer measures, the observer's slice of space read against the energy current,
`0.5 * (mv.t & (field >> time))`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from numga import NumpyContext
from numga.algebras import STA as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                                  # Scalar+
# A wavevector: the slice whose pairing with an event is the phase there.
Vector = ga.gatype.vector()                                                  # Vector+
# Events are antivectors with intrinsic orientation.
Antivector = ga.gatype.antivector()                                          # Antivector-
Bivector = ga.gatype.bivector()                                              # Bivector+
Linear = ga.gatype((Antivector, Antivector))                                 # Antivector- <- Antivector-
# The plane of the incident light's wavefront, in which its circularly polarized field turns as it advances.
wavefront_plane = mv.xy                                                      # [] Bivector+
# The observer at rest with the crystal: its direction in time, which pairs to one with its slice of space
# `mv.t`; the dual carries the minus of t's square.
time = -mv.t.dual()                                                          # [] Antivector-
# A field's sandwich of the observer's direction in time is twice the energy current it carries, and the
# observer's slice of space reads the energy density from it.
energy_density = 0.5 * (mv.t & (Bivector >> time))                           # [] Scalar+ <- (Bivector+, Bivector+)


# --- math -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Crystal:
    positions: Antivector
    reciprocal: Vector

    def carried(self, deformation: Linear) -> Crystal:
        """The atoms moved, and the wavevectors moved so that every phase measured at those atoms is kept."""
        positions = deformation[..., None](self.positions)                  # [..., atoms] Antivector-
        # The adjugate pulls a wavevector back through the deformation; solving through it keeps every pairing.
        adjugate = deformation.adjugate()                                    # [...] Vector+ <- Vector+
        reciprocal = adjugate[..., None].solve(self.reciprocal)              # [..., reflections] Vector+
        return Crystal(positions, reciprocal)

    def scattered(self, transfers: Vector, incident: Bivector) -> Bivector:
        """The field equal atoms scatter to each transfer of a [rows, columns] grid, the incident wave turned by
        each atom's phase and summed; an in-phase peak carries the incident field.

        The transfer is the outgoing wavevector less the incoming one; no polarization factor is included.
        """
        # Light scattered without a change of frequency has a transfer with no part along the observer's time,
        # so the time of an atom's event does not enter its phase.
        phases = transfers[..., None] & self.positions[..., None, None, :]  # [..., rows, columns, atoms] Scalar+
        # The circularly polarized wave, turned in its wavefront by how far it has advanced: the sense of the turn
        # is its helicity.
        turns = (wavefront_plane * (phases / 2)).exp()                       # [..., rows, columns, atoms] Rotor+
        waves = turns >> incident                                            # [..., rows, columns, atoms] Bivector+
        return waves.sum(axis=-1) / self.positions.shape[-1]                 # [..., rows, columns] Bivector+


def deformation(stretch: np.ndarray, shear: np.ndarray, angle: np.ndarray) -> Linear:
    """The layer stretched along x, its horizontal rows slid along x and the whole turned in the plane xy; the
    observer's time stays where it is."""
    local = (Antivector + (stretch - 1) * mv.x.dual() * (mv.x & Antivector)
             + shear * mv.x.dual() * (mv.y & Antivector))                    # [...] Antivector- <- Antivector-
    # A positive angle turns x toward y.
    turns = (mv.xy * (-angle / 2)).exp()                                     # [...] Rotor+
    return turns >> local


# --- plumbing -------------------------------------------------------------------------
def square(shells: int, orders: int, spacing: float) -> Crystal:
    """A finite square layer of events at the observer's time zero, and its reciprocal lattice."""
    atoms = np.arange(-shells, shells + 1)
    reflections = np.arange(-orders, orders + 1)
    positions = (mv.x * atoms[None, :] + mv.y * atoms[:, None]).reshape(-1).dual() * spacing
    reciprocal = (mv.x * reflections[None, :] + mv.y * reflections[:, None]).reshape(-1) * (2 * np.pi / spacing)
    return Crystal(positions, reciprocal)


def transfer_grid(half_width: float, pixels: int) -> Vector:
    """A square section of momentum transfers in the plane of the layer, one per pixel."""
    samples = np.linspace(-half_width, half_width, pixels)
    return mv.x * samples[None, :] + mv.y * samples[:, None]
