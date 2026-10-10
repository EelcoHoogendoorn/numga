"""Stretch and twist of balanced angle-ply strips under uniform tension.

The state holds axial strain as a scalar and thickness times twist per length in the yz plane.
Each ply measures that state's stretch along and across its fibres, and its shear. Pairing these
measurements with their stresses gives an energy form; summing through the thickness couples
stretch to twist. Uniform transverse strain relaxes until the transverse force vanishes.

This is a small-displacement, uniform-curvature model for balanced symmetric or antisymmetric
angle-ply stacks. It describes the interior response, without the boundary layers near grips.
"""

import numpy as np

from numga import NumpyContext, stack
from numga.extensor import Extensor
from numga.algebras import VGA3D as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
Vector = ga.gatype.vector()
# Both state components are dimensionless: axial strain and thickness times twist per length.
State = ga.gatype.from_blades("1 yz")
Measurement = ga.gatype((Scalar, State))                      # Scalar <- State
Energy = ga.gatype((Scalar, State, State))                    # Scalar <- (State, State)
Strain = ga.gatype((Vector, Vector, State))                   # Vector <- (Vector, State)
extension: Measurement = mv.scalar([1]).scalar_product(State)  # []
# The minus sign compensates for the negative square of the yz blade.
twist: Measurement = (-mv.yz).scalar_product(State)            # []


# --- math -----------------------------------------------------------------------------
def strain(height: Scalar, thickness: float) -> Strain:
    """The midsurface strain at a height through the thickness, as a map on directions with the
    stretch–twist state left open: the stretch along the strip, and the shear the twist brings at
    that height, opposite above and below the middle."""
    stretch = mv.x * (mv.x | Vector) * extension                                        # [] Vector <- (Vector, State)
    # Twist tilts the normal both ways in the plane; dividing by thickness gives angle per length.
    shear = -height / thickness * (mv.x * (mv.y | Vector) + mv.y * (mv.x | Vector)) * twist  # [...] Vector <- (Vector, State)
    return stretch + shear                                                              # [...] Vector <- (Vector, State)


def read(strain: Extensor, fibres: Vector) -> Extensor:
    """A strain map's readings along a ply's fibres, across them, and as their shear, stacked on a
    leading axis, with any further slots of the strain left open."""
    transverse: Vector = mv.xy | fibres
    return stack([fibres | strain(fibres), transverse | strain(transverse), 2 * (fibres | strain(transverse))])


def ply_stress(along: Extensor, across: Extensor, sliding: Extensor, longitudinal_modulus: float,
               transverse_modulus: float, poisson: float, shear_modulus: float) -> tuple[Extensor, Extensor, Extensor]:
    """A ply's stresses along and across its fibres and in shear, from its strain read the same
    ways, of any arity: the normal stresses coupled by Poisson contraction, shear on its own."""
    denominator = 1 - poisson**2 * transverse_modulus / longitudinal_modulus
    along_stress = (longitudinal_modulus * along + poisson * transverse_modulus * across) / denominator
    across_stress = transverse_modulus * (poisson * along + across) / denominator
    return along_stress, across_stress, shear_modulus * sliding


def response(fibres: Vector, heights: Vector, weights: np.ndarray, thickness: float,
             longitudinal_modulus: float, transverse_modulus: float, poisson: float,
             shear_modulus: float, tension: float) -> tuple[State, Scalar]:
    """Axial strain and twist, and relaxed width strain, from force per unit width.

    Fibres are [cases, plies, 1]; heights and integration weights are [plies, samples].
    Two Gauss samples per ply integrate the quadratic energy exactly.
    """
    moduli = (longitudinal_modulus, transverse_modulus, poisson, shear_modulus)
    # Each ply reads the strain along its fibres, across them, and as their shear.
    local: Strain = strain(heights | mv.z, thickness)                                  # [plies, samples]
    readings = read(local, fibres)                                                     # [3, cases, plies, samples] Measurement
    # A unit width strain, read the same ways.
    widening = mv.y * (mv.y | Vector)                                                  # [] Vector <- Vector
    width_readings = read(widening, fibres)                                            # [3, cases, plies, 1] Scalar
    stresses = stack(ply_stress(*readings, *moduli))                                   # [3, cases, plies, samples] Measurement
    width_stresses = stack(ply_stress(*width_readings, *moduli))                       # [3, cases, plies, 1] Scalar
    # Strain read against stress, summed over the three readings and through the thickness. Each
    # product of two readings leaves two State slots open: half energy(state, state) is the stored
    # energy per midsurface area.
    energy: Energy = (readings * stresses * weights).sum(axis=(0, -2, -1))             # [cases]
    # With the width held fixed, coupling(state) is the transverse force per unit length.
    coupling: Measurement = (width_readings * stresses * weights).sum(axis=(0, -2, -1))   # [cases]
    width_stiffness: Scalar = (width_readings * width_stresses * weights).sum(axis=(0, -2, -1))  # [cases]
    # Free edges let the width contract until its force vanishes, reducing the state's stiffness.
    relaxed: Energy = energy - coupling * coupling / width_stiffness                    # [cases]
    # Apply only axial force; the twist is free to settle through the coupling in the energy.
    state: State = relaxed.solve(tension * extension)                                  # [cases]
    width_strain: Scalar = -coupling(state) / width_stiffness                          # [cases]
    return state, width_strain


def deform(points: Vector, state: State, width_strain: Scalar, thickness: float) -> Vector:
    """The midsurface's uniform stretch and width contraction, with each cross-section turned about
    the strip's axis by the twist accumulated along the strip."""
    along: Scalar = points | mv.x                            # [length_samples, width_samples]
    across: Scalar = points | mv.y                           # [length_samples, width_samples]
    # Each state acts on the whole grid; the two trailing axes sample the midsurface.
    state, width_strain = state[..., None, None], width_strain[..., None, None]  # [..., 1, 1] State, Scalar
    # Twist turns each cross-section in the yz plane, by an angle growing along the strip.
    turn = (mv.yz * (-along * twist(state) / thickness / 2)).exp()  # [..., length_samples, width_samples] Rotor
    return (mv.x * along * (1 + extension(state))
            + (turn >> (mv.y * across * (1 + width_strain))))  # [..., length_samples, width_samples] Vector
