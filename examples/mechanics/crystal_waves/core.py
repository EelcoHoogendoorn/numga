"""Elastic waves in crystals: speeds, polarizations, and energy focusing.

A crystal's stiffness is modelled as a map from a face, a displacement and a wavevector to the traction on that face (`Plane+ <- (Plane+, Direction-, Plane+)`). Displacements are intrinsic directions, whereas faces, wavevectors and forces are extrinsic planes that pair with them. An isotropic solid contributes a dilation term on the face and a metric shear term, while a cubic crystal adds an anisotropic term in which each cube axis reads all three arguments.

Binding a wave's heading as both the face and the wavevector forms the Christoffel map from displacements to forces; its eigenpairs against the displacement metric give density times squared speed and the polarization of one compressional wave and two shear waves. Binding the polarization as the displacement and the heading as the wavevector yields the power form across faces, whose trace gives the energy flux. Where the energy flux of many headings folds onto one direction, heat pulses focus into bright caustics.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext
from numga.algebras import VGA3D as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
# Faces, wavevectors and forces are extrinsic planes; displacements, polarizations and energy flux are intrinsic directions.
Plane = ga.gatype.vector()
Direction = ga.gatype.antivector()
# Traction on a face from the face, the displacement and its wavevector.
Stiffness = ga.gatype((Plane, Plane, Direction, Plane))             # Plane+ <- (Plane+, Direction-, Plane+)
# The crystal's cube axes singled out by cubic anisotropy.
axes = mv.basis()                                                   # [3] Plane+


# --- math ----------------------------------------------------------------------------------------
def stiffness(c11: np.ndarray, c12: np.ndarray, c44: np.ndarray) -> Stiffness:
    """Cubic crystal stiffness maps (`[...] Plane+ <- (Plane+, Direction-, Plane+)`) from elastic constants `c11`, `c12` and `c44`."""
    # Dilation pushes on the face by `wavevector & displacement`, shear couples through `Direction.dual_inverse()`, and each cube axis reads all three inputs.
    dilation = Plane * (Direction & Plane)                          # Plane+ <- (Plane+, Direction-, Plane+)
    shear = 2 * (Plane & Direction) * Plane - (Plane | (Direction.dual_inverse() ^ Plane))   # Plane+ <- (Plane+, Direction-, Plane+)
    cubic = (axes * (axes | Plane) * (axes & Direction) * (axes | Plane)).sum()   # Plane+ <- (Plane+, Direction-, Plane+)
    # Scalar extensors lift the elastic constants into the context and carry any material batch axis.
    c11, c12, c44 = (mv.scalar(np.asarray(value)[..., None]) for value in (c11, c12, c44))   # [...] Scalar+
    return c12 * dilation + c44 * shear + (c11 - c12 - 2 * c44) * cubic   # [...] Plane+ <- (Plane+, Direction-, Plane+)


def waves(crystal: Stiffness, heading: Plane) -> tuple[Scalar, Direction]:
    """Wave moduli (`[..., 3] Scalar+`, density times squared speed) and polarizations (`[..., 3] Direction-`) ordered slowest first along `heading`."""
    christoffel = crystal(heading, Direction, heading)              # [...] Plane+ <- Direction-
    return (christoffel(Direction) & Direction).eigh(Direction.dot(Direction))   # [..., 3] Scalar+, [..., 3] Direction-


def energy_flow(crystal: Stiffness, heading: Plane, polarization: Direction, density: np.ndarray) -> Direction:
    """Group velocities (`[..., 3] Direction-`) of the three wave modes from the face power form divided by density times phase speed."""
    # Broadcast the crystal, its density and the heading across the three wave modes at each heading.
    crystal, density, wave_heading = crystal[..., None], density[..., None], heading[..., None]   # [..., 1] Plane+ <- (Plane+, Direction-, Plane+), [..., 1], [..., 1] Plane+
    wave_modulus = crystal(wave_heading, polarization, wave_heading) & polarization   # [..., 3] Scalar+
    # Traction paired with polarization gives power through each face; tracing against `Direction` recovers the energy flux direction.
    power = crystal(Plane, polarization, wave_heading) & polarization   # [..., 3] Scalar+ <- Plane+
    return (Direction * power).trace(1, 2) / (wave_modulus * density).square_root()   # [..., 3] Direction-
