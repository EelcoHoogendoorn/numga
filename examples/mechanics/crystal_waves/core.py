"""Elastic waves in crystals: their speeds, their polarizations, and where they carry energy.

A crystal's stiffness is modelled as a map from a face, a displacement and a wavevector, the plane
across the direction along which the displacement varies, to the traction on that face, a force:
`Stiffness = Vector <- (Vector, Antivector, Vector)`. Displacements are directions, intrinsic; faces,
wavevectors and forces read them, and are extrinsic. An isotropic solid has two terms, a pressure on the
face from the dilation, the wavevector's reading of the displacement, and a shear, where the metric
enters; a cubic crystal adds one more, in which each of its axes reads all three arguments.

Binding a wave's heading as the face and the wavevector gives the Christoffel map, from displacements to
forces; against the metric its eigenpairs are the wave's density times squared speed and its
polarization: one compressional wave and two shear waves for every heading. Binding the polarization as
the displacement and the heading as the wavevector, the traction read back on the polarization is the
power through each face, and the direction it reads is the flow of the wave's energy. In a crystal it swings away
from the heading, and where the energy of many headings converges on one, heat pulses focus into bright
caustics.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext
from numga.algebras import VGA3D as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
# Faces, wavevectors and forces read displacements: vectors, extrinsic. Displacements, polarizations and
# the flow of energy are directions, antivectors, intrinsic.
Vector = ga.gatype.vector()
Antivector = ga.gatype.antivector()
# The traction on a face, from the face, the displacement and its wavevector.
Stiffness = ga.gatype((Vector, Vector, Antivector, Vector))         # Vector+ <- (Vector+, Antivector-, Vector+)
# The crystal's cube axes, which its anisotropy singles out.
axes = mv.basis()                                                   # [3] Vector+


# --- math ----------------------------------------------------------------------------------------
def stiffness(c11: np.ndarray, c12: np.ndarray, c44: np.ndarray) -> Stiffness:
    """The stiffness of cubic crystals from their elastic constants, batched over the constants.

    With face n, displacement v and wavevector h: the dilation, the wavevector's reading of the
    displacement `h & v`, pushes on the face, and the shear pulls along the wavevector and along the
    plane across the displacement, `g(v) = v.dual_inverse()`, where the metric enters;
    `n | (g(v) ^ h) == (n & v) * h - (n | h) * g(v)` turns one into the other. Along the crystal's axes
    the stiffness differs from isotropic by `c11 - c12 - 2 * c44`.
    """
    # The dilation is `n * (h & v)`, the shear `g(v) * (n | h) + h * (n & v)`, and the cubic term has
    # each axis reading n, v and h, all three in the order of the stiffness's inputs.
    dilation = Vector * (Antivector & Vector)                                    # Vector+ <- (Vector+, Antivector-, Vector+)
    shear = 2 * (Vector & Antivector) * Vector - (Vector | (Antivector.dual_inverse() ^ Vector))   # Vector+ <- (Vector+, Antivector-, Vector+)
    cubic = (axes * (axes | Vector) * (axes & Antivector) * (axes | Vector)).sum()   # Vector+ <- (Vector+, Antivector-, Vector+)
    # The constants as scalars of the context, which carry their batch into the stiffness.
    c11, c12, c44 = (mv.scalar(np.asarray(value)[..., None]) for value in (c11, c12, c44))
    return c12 * dilation + c44 * shear + (c11 - c12 - 2 * c44) * cubic       # [...] Vector+ <- (Vector+, Antivector-, Vector+)


def waves(crystal: Stiffness, heading: Vector) -> tuple[Scalar, Antivector]:
    """The waves travelling along each heading: density times squared speed, and polarization.

    The heading bound as the face and the wavevector gives the Christoffel map, from displacements to
    the forces they cause; its form against the metric, the dot of the planes across the
    displacements, has the three waves as eigenpairs, slowest first.
    """
    christoffel = crystal(heading, Antivector, heading)              # [...] Vector+ <- Antivector-
    return (christoffel(Antivector) & Antivector).eigh(Antivector.dual().dot(Antivector.dual()))   # [..., 3] Scalar+, [..., 3] Antivector-


def energy_flow(crystal: Stiffness, heading: Vector, polarization: Antivector, density: np.ndarray) -> Antivector:
    """The velocity of each wave's energy: its group velocity.

    With the polarization bound as the displacement and the heading as the wavevector, the traction
    read back on the polarization is the power through each face, and the direction it reads is the
    flux of the wave's energy. Divided by density times phase speed it is the group velocity, whose
    reading by the heading is the phase speed.
    """
    # The crystal, its density and the heading against each of the waves, and each wave's density
    # times squared speed.
    crystal, density, wave_heading = crystal[..., None], density[..., None], heading[..., None]   # [..., 1] each
    wave_modulus = crystal(wave_heading, polarization, wave_heading) & polarization   # [..., 3] Scalar+
    # The power through each face, a form on faces; the trace pairs it with the directions the faces read.
    power = crystal(Vector, polarization, wave_heading) & polarization        # [..., 3] Scalar+ <- Vector+
    return (Antivector * power).trace(1, 2) / (wave_modulus * density).square_root()   # [..., 3] Antivector-
