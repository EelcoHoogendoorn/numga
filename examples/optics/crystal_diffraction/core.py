"""A crystal's atoms and its diffraction peaks carried through a deformation.

A phase is a pairing of a wavevector with a position, `k | x`. The deformation, a map on positions
built by leaving the position open, moves the atoms; its adjoint is defined by that pairing,
`deformation.adjoint()(k) | x == k | deformation(x)`, so solving through it moves the wavevectors so
that every phase is kept. Read as phase planes, the duals of the wavevectors, the same phase is the
incidence `planes & x`, carried across by the adjugate without the metric. The intensity sums each
atom's wave, the exponential of the pseudoscalar times its phase.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from numga import NumpyContext
from numga.algebras import VGA3D as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
Vector = ga.gatype.vector()
Bivector = ga.gatype.bivector()
Phasor = ga.gatype(ga.subspace.scalar() + ga.subspace.pseudoscalar())
Linear = ga.gatype((Vector, Vector))                          # Vector <- Vector


# --- math -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Crystal:
    positions: Vector
    reciprocal: Vector
    phase_planes: Bivector

    def carried(self, deformation: Linear) -> Crystal:
        """Move the atoms and preserve the phases measured at those atoms."""
        positions = deformation[..., None](self.positions)                  # [..., atoms] Vector
        # Moving the atoms and their phase measurements preserves every pairing.
        adjoint = deformation.adjoint()                                      # [...] Vector <- Vector
        adjugate = deformation.adjugate()                                    # [...] Bivector <- Bivector
        reciprocal = adjoint[..., None].solve(self.reciprocal)               # [..., reflections] Vector
        phase_planes = adjugate[..., None].solve(self.phase_planes)          # [..., reflections] Bivector
        return Crystal(positions, reciprocal, phase_planes)

    def intensity(self, transfers: Vector) -> Scalar:
        """Coherent scattering from equal atoms, on a [rows, columns] transfer grid.

        The transfer is the outgoing wavevector less the incoming one. Intensities are
        normalized to one when every atom scatters in phase; no polarization factor is included.
        """
        phases = transfers[..., None] | self.positions[..., None, None, :]  # [..., rows, columns, atoms] Scalar
        # The pseudoscalar squares to minus one: its exponential carries the phase.
        waves = (mv.xyz * phases).exp()                                       # [..., rows, columns, atoms] Phasor
        amplitude = waves.sum(axis=-1) / self.positions.shape[-1]             # [..., rows, columns] Phasor
        return amplitude.scalar_norm_squared()                                # [..., rows, columns] Scalar


def deformation(stretch: np.ndarray, shear: np.ndarray, angle: np.ndarray) -> Linear:
    """Stretch along x, slide horizontal rows, and turn the whole layer."""
    local = Vector + (stretch - 1) * mv.x * (mv.x | Vector) + shear * mv.x * (mv.y | Vector)
    return (mv.xy * (-angle / 2)).exp() >> local


# --- plumbing -------------------------------------------------------------------------
def square(shells: int, orders: int, spacing: float) -> Crystal:
    """A finite square layer and its reciprocal lattice, embedded in space."""
    atoms = np.arange(-shells, shells + 1)
    reflections = np.arange(-orders, orders + 1)
    positions = (mv.x * atoms[None, :] + mv.y * atoms[:, None]).reshape(-1) * spacing
    reciprocal = (mv.x * reflections[None, :] + mv.y * reflections[:, None]).reshape(-1) * (2 * np.pi / spacing)
    return Crystal(positions, reciprocal, reciprocal.dual())


def transfer_grid(half_width: float, pixels: int) -> Vector:
    samples = np.linspace(-half_width, half_width, pixels)
    return mv.x * samples[None, :] + mv.y * samples[:, None]
