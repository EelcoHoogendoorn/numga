"""Extension-twist coupling stretches and twists balanced angle-ply strips under uniform tension.

The state combines axial strain as a scalar with thickness times twist per length in the yz plane.
Each ply reads that state's stretch along and across its fibres together with its shear; pairing
those readings with their stresses yields an energy form whose thickness sum couples stretch to
twist, while uniform transverse strain relaxes until the transverse force vanishes.

Small displacements and uniform curvature model the interior response of balanced symmetric or
antisymmetric angle-ply stacks away from grip boundary layers.
"""

from collections.abc import Iterator

import numpy as np

from numga import Algebra, NumpyContext, stack
from numga.algebras import VGA3D as ga
from numga.extensor import Extensor

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                                    # Scalar+
# Positions and directions in the strip are intrinsic, the duals of the planes x, y and z their unit steps.
Direction = ga.gatype.antivector()                                             # Antivector-
unit_x, unit_y, unit_z = mv.x.dual(), mv.y.dual(), mv.z.dual()                 # [] Direction-, Direction-, Direction-
# Both state components are dimensionless: axial strain and thickness times twist per length.
State = ga.gatype.even().derive.within(Algebra("y+z+"))                        # Rotor+
Plane = ga.gatype.vector()                                                     # Vector+
# Strain is a form on pairs of directions: as a duality map, it sends a direction to the plane that reads its stretch.
Strain = ga.gatype((Plane, Direction, State))                                  # Plane+ <- (Direction-, State+)
extension = mv.scalar([1]).scalar_product(State)                               # [] Scalar+ <- State+
# The minus sign compensates for the negative square of the yz blade.
twist = (-mv.yz).scalar_product(State)                                         # [] Scalar+ <- State+


# --- math -----------------------------------------------------------------------------
def height_strain(height: Scalar, thickness: float) -> Strain:
    """At height `height` (`[...] Scalar+`) through `thickness`, axial extension combines with a
    twist-induced shear of opposite sign above and below the midsurface (`[...] Plane+ <- (Direction-, State+)`)."""
    stretch = mv.x * (mv.x & Direction) * extension                            # [] Plane+ <- (Direction-, State+)
    # Twist tilts the normal both ways in the plane; dividing by thickness gives angle per length.
    shear = -height / thickness * (mv.x * (mv.y & Direction) + mv.y * (mv.x & Direction)) * twist  # [...] Plane+ <- (Direction-, State+)
    return stretch + shear                                                     # [...] Plane+ <- (Direction-, State+)


def ply_readings(strain_map: Extensor, fibres: Direction) -> Extensor:
    """Pairing `strain_map` along `fibres`, across them, and in shear stacks the three ply strain
    readings along a leading axis (`[3, ...]`)."""
    # A quarter turn in the strip's plane gives the in-plane direction perpendicular to the fibres.
    transverse = mv.xy.commutator(fibres)                                      # [...] Direction-
    return stack([fibres & strain_map(fibres), transverse & strain_map(transverse), 2 * (fibres & strain_map(transverse))])


def ply_stresses(longitudinal_strain: Extensor, transverse_strain: Extensor, shear_strain: Extensor,
                 longitudinal_modulus: float, transverse_modulus: float, poisson: float,
                 shear_modulus: float) -> tuple[Extensor, Extensor, Extensor]:
    """Orthotropic plane-stress relations couple normal stresses along and across the fibres by
    Poisson contraction while leaving shear stress proportional to `shear_strain`."""
    denominator = 1 - poisson**2 * transverse_modulus / longitudinal_modulus
    longitudinal_stress = (longitudinal_modulus * longitudinal_strain + poisson * transverse_modulus * transverse_strain) / denominator
    transverse_stress = transverse_modulus * (poisson * longitudinal_strain + transverse_strain) / denominator
    return longitudinal_stress, transverse_stress, shear_modulus * shear_strain


def tension_response(fibres: Direction, heights: Direction, weights: np.ndarray, thickness: float,
                     longitudinal_modulus: float, transverse_modulus: float, poisson: float,
                     shear_modulus: float, tension: float) -> tuple[State, Scalar]:
    """Integrating ply strain energy through `thickness` and relaxing the free width yields the
    extension-twist state (`[...] State+`) and transverse strain (`[...] Scalar+`) under axial `tension`.

    `fibres` has shape `[..., n_plies, 1]`, while `heights` and Gauss `weights` have shape
    `[n_plies, n_nodes]`; two Gauss nodes per ply integrate the quadratic strain energy exactly.
    """
    moduli = (longitudinal_modulus, transverse_modulus, poisson, shear_modulus)
    # Each ply reads the strain along its fibres, across them, and as their shear.
    strain_at_heights = height_strain(mv.z & heights, thickness)               # [...] Plane+ <- (Direction-, State+)
    readings = ply_readings(strain_at_heights, fibres)                         # [3, ...] Scalar+ <- State+
    # A unit width strain is read along the same ply axes.
    widening = mv.y * (mv.y & Direction)                                       # [] Plane+ <- Direction-
    width_readings = ply_readings(widening, fibres)                            # [3, ...] Scalar+
    stresses = stack(ply_stresses(*readings, *moduli))                         # [3, ...] Scalar+ <- State+
    width_stresses = stack(ply_stresses(*width_readings, *moduli))             # [3, ...] Scalar+
    # Strain read against stress and summed through the thickness forms the quadratic energy per midsurface area.
    energy = (readings * stresses * weights).sum(axis=(0, -2, -1))             # [...] Scalar+ <- (State+, State+)
    # With the width held fixed, coupling(state) reads the transverse force per unit length.
    coupling = (width_readings * stresses * weights).sum(axis=(0, -2, -1))     # [...] Scalar+ <- State+
    width_stiffness = (width_readings * width_stresses * weights).sum(axis=(0, -2, -1))  # [...] Scalar+
    # Free long edges let the width contract until transverse force vanishes, softening the state energy.
    free_width_energy = energy - coupling * coupling / width_stiffness         # [...] Scalar+ <- (State+, State+)
    # Axial tension drives extension alone; twist settles through the off-diagonal coupling in the energy.
    state = free_width_energy.solve(tension * extension)                       # [...] State+
    width_strain = -coupling(state) / width_stiffness                          # [...] Scalar+
    return state, width_strain


def midsurface_points(points: Direction, state: State, width_strain: Scalar, thickness: float) -> Direction:
    """Under `state` (`[...] State+`) and `width_strain` (`[...] Scalar+`), `points` (`[..., ...] Direction-`)
    stretch axially, contract across the width, and twist progressively along the strip (`[...] Direction-`)."""
    axial_coord = mv.x & points                                                # [...] Scalar+
    transverse_coord = mv.y & points                                           # [...] Scalar+
    # Each state acts on the whole grid; the two trailing axes sample the midsurface.
    state, width_strain = state[..., None, None], width_strain[..., None, None]  # [..., 1, 1] State+, Scalar+
    # Twist turns each cross-section in the yz plane by an angle proportional to axial position.
    section_rotors = (mv.yz * (-axial_coord * twist(state) / thickness / 2)).exp()  # [...] Rotor+
    cross_sections = section_rotors >> (unit_y * transverse_coord * (1 + width_strain))  # [...] Direction-
    return unit_x * axial_coord * (1 + extension(state)) + cross_sections      # [...] Direction-


def pull_frames(reference: Direction, placement: Direction, state: State, width_strain: Scalar,
                thickness: float, scales: np.ndarray) -> Iterator[Direction]:
    """Yields the displaced midsurface grids (`[...] Direction-`) as the load factor steps through `scales`."""
    for scale in scales:
        yield midsurface_points(reference, state * float(scale), width_strain * float(scale), thickness) + placement
