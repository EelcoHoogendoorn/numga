"""Elastic waves in crystals: their speeds, their polarizations, and where they carry energy.

A crystal's stiffness is modelled as a map from three vectors to one: from the normal of a face, a
displacement, and the direction along which it varies, to the traction on that face. An isotropic solid
has two terms, a pressure along the normal from the dilation and a shear; a cubic crystal adds one more,
in which each of its axes reads all three arguments.

Binding a wave's heading as the normal and the direction gives the Christoffel map on displacements,
whose eigenpairs are the wave's density times squared speed and its polarization: one compressional wave
and two shear waves for every heading. Binding the polarization as the normal and the displacement, and
the heading as the direction, gives a vector, the flow of the wave's energy. In a crystal it swings away
from the heading, and where the energy of many headings converges on one, heat pulses focus into bright
caustics.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext
from numga.algebras import VGA3D as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
Vector = ga.gatype.vector()
# The traction on a face, from the face's normal, the displacement and its gradient direction.
Stiffness = ga.gatype((Vector, Vector, Vector, Vector))             # Vector <- (Vector, Vector, Vector)
# The crystal's cube axes, which its anisotropy singles out.
axes = mv.basis()                                                   # [3] Vector


# --- math ----------------------------------------------------------------------------------------
def stiffness(c11: np.ndarray, c12: np.ndarray, c44: np.ndarray) -> Stiffness:
    """The stiffness of cubic crystals from their elastic constants, batched over the constants.

    With normal n, displacement v and gradient h: the dilation `v | h` pushes along the normal, the
    shear pulls along v and h, `n | (v ^ h) == (n | v) * h - (n | h) * v` turning one into the other;
    along the crystal's axes the stiffness differs from isotropic by `c11 - c12 - 2 * c44`.
    """
    # The dilation is `n * (v | h)`, the shear `v * (n | h) + h * (n | v)`, and the cubic term has
    # each axis reading n, v and h.
    dilation = Vector * (Vector | Vector)                                        # Vector <- (Vector, Vector, Vector)
    shear = 2 * (Vector | Vector) * Vector - (Vector | (Vector ^ Vector))        # Vector <- (Vector, Vector, Vector)
    cubic = (axes * (axes | Vector) * (axes | Vector) * (axes | Vector)).sum()   # Vector <- (Vector, Vector, Vector)
    # The constants as scalars of the context, which carry their batch into the stiffness.
    c11, c12, c44 = (mv.scalar(np.asarray(value)[..., None]) for value in (c11, c12, c44))
    return c12 * dilation + c44 * shear + (c11 - c12 - 2 * c44) * cubic       # [...] Vector <- (Vector, Vector, Vector)


def waves(crystal: Stiffness, heading: Vector) -> tuple[Scalar, Vector]:
    """The waves travelling along each heading: density times squared speed, and polarization.

    The heading bound as the normal and the direction gives the Christoffel map on displacements;
    its eigenpairs are the three waves, slowest first.
    """
    return crystal(heading, Vector, heading).eigh()                  # [..., 3] Scalar, [..., 3] Vector


def energy_flow(crystal: Stiffness, heading: Vector, polarization: Vector, density: np.ndarray) -> Vector:
    """The velocity of each wave's energy: its group velocity.

    With the polarization bound as the normal and the displacement and the heading as the direction,
    the stiffness returns the flux of the wave's energy. Divided by density times phase speed
    it is the group velocity, whose component along the heading is the phase speed.
    """
    # The crystal, its density and the heading against each of the waves, and each wave's density
    # times squared speed.
    crystal, density, wave_heading = crystal[..., None], density[..., None], heading[..., None]   # [..., 1] each
    wave_modulus = polarization | crystal(wave_heading, polarization, wave_heading)   # [..., 3] Scalar
    return crystal(polarization, polarization, wave_heading) / (wave_modulus * density).square_root()
