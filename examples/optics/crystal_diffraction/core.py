"""A crystal's atoms and its diffraction peaks carried through a deformation.

A wavevector is a plane through the origin, and a phase its pairing with a position, a point, `k & x`.
The deformation, a map on points, the identity `Point` plus a stretch, a shear and a turn, moves the
atoms; its adjugate is the map it induces on planes, `deformation.adjugate()(k) & x == k & deformation(x)`,
so solving through it moves the wavevectors so that every phase is kept, without the metric. The
intensity sums each atom's wave, the exponential of the pseudoscalar times its phase; the hand the
pseudoscalar picks drops out of the intensity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from numga import NumpyContext
from numga.algebras import VGA3D as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
# A wavevector: the plane through the origin whose pairing with a point is the phase there.
Plane = ga.gatype.vector()
# Positions are points, antivectors with intrinsic orientation.
Point = ga.gatype.antivector()
Phasor = ga.gatype.scalar_pseudoscalar()
Linear = ga.gatype((Point, Point))                            # Point <- Point


# --- math -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Crystal:
    positions: Point
    reciprocal: Plane

    def carried(self, deformation: Linear) -> Crystal:
        """Move the atoms and preserve the phases measured at those atoms."""
        positions = deformation[..., None](self.positions)                  # [..., atoms] Point
        # The adjugate pulls a plane back through the deformation; solving through it keeps every pairing.
        adjugate = deformation.adjugate()                                    # [...] Plane <- Plane
        reciprocal = adjugate[..., None].solve(self.reciprocal)              # [..., reflections] Plane
        return Crystal(positions, reciprocal)

    def intensity(self, transfers: Plane) -> Scalar:
        """Coherent scattering from equal atoms, on a [rows, columns] transfer grid.

        The transfer is the outgoing wavevector less the incoming one. Intensities are
        normalized to one when every atom scatters in phase; no polarization factor is included.
        """
        phases = transfers[..., None] & self.positions[..., None, None, :]  # [..., rows, columns, atoms] Scalar
        # The pseudoscalar squares to minus one: its exponential carries the phase.
        waves = (mv.xyz * phases).exp()                                       # [..., rows, columns, atoms] Phasor
        amplitude = waves.sum(axis=-1) / self.positions.shape[-1]             # [..., rows, columns] Phasor
        return amplitude.scalar_norm_squared()                                # [..., rows, columns] Scalar


def deformation(stretch: np.ndarray, shear: np.ndarray, angle: np.ndarray) -> Linear:
    """Stretch along x, slide horizontal rows, and turn the whole layer."""
    local = Point + (stretch - 1) * mv.x.dual() * (mv.x & Point) + shear * mv.x.dual() * (mv.y & Point)
    return (mv.xy * (-angle / 2)).exp() >> local


# --- plumbing -------------------------------------------------------------------------
def square(shells: int, orders: int, spacing: float) -> Crystal:
    """A finite square layer and its reciprocal lattice, embedded in space."""
    atoms = np.arange(-shells, shells + 1)
    reflections = np.arange(-orders, orders + 1)
    positions = (mv.x * atoms[None, :] + mv.y * atoms[:, None]).reshape(-1).dual() * spacing
    reciprocal = (mv.x * reflections[None, :] + mv.y * reflections[:, None]).reshape(-1) * (2 * np.pi / spacing)
    return Crystal(positions, reciprocal)


def transfer_grid(half_width: float, pixels: int) -> Plane:
    samples = np.linspace(-half_width, half_width, pixels)
    return mv.x * samples[None, :] + mv.y * samples[:, None]
