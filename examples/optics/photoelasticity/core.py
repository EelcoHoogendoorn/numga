"""Stress around a traction-free hole, read through polarized light.

The plate is thin, isotropic and infinite, with uniform stress far from its circular hole.
Its in-plane stress maps surface normals to traction. Light's polarization is a point on the
Poincaré sphere, a unit Stokes vector: linear polarizations on its equator, at twice their angle
in the plate, circular ones at its poles. The weak stress-optic law makes the stressed plate a
rotor of that sphere, about the state polarized along a principal stress direction, by the
retardance the principal stress difference builds through the thickness. A polarizer passes
the share of light its own state finds. Light travels normal to the plate; absorption,
reflection and ray bending are neglected.
"""

from numga import NumpyContext
from numga.algebras import VGA3D as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
# Vectors lying in the plate: positions, surface normals and tractions.
Planar = ga.gatype.from_blades("x y")
Stress = ga.gatype((Planar, Planar))                       # Planar <- Planar
# A unit Stokes vector on the Poincaré sphere, and a turn of that sphere.
Polarization = ga.gatype.vector()
Bivector = ga.gatype.bivector()
Retarder = ga.gatype.rotor()


# --- math -----------------------------------------------------------------------------
def kirsch(positions: Planar, radius: float, remote: Stress) -> Stress:
    """The exterior stress for a circular hole under any uniform symmetric remote stress."""
    distance_squared = positions.squared()                                        # [...] Scalar
    radial = positions.normalized()                                               # [...] Planar
    tangent = mv.xy | radial                                                      # [...] Planar
    radius_ratio = radius**2 / distance_squared                                   # [...] Scalar

    # Resolve the remote traction map in the local radial and tangential directions.
    mean = remote.trace() / 2                                                     # [...] Scalar
    anisotropy = (radial | remote(radial)) - mean                                 # [...] Scalar
    shear = radial | remote(tangent)                                              # [...] Scalar
    radial_stress = (mean * (1 - radius_ratio)
                     + anisotropy * (1 - 4 * radius_ratio + 3 * radius_ratio.squared()))  # [...] Scalar
    tangential_stress = (mean * (1 + radius_ratio)
                         - anisotropy * (1 + 3 * radius_ratio.squared()))         # [...] Scalar
    shear_stress = shear * (1 + 2 * radius_ratio - 3 * radius_ratio.squared())    # [...] Scalar

    # Leaving the surface normal open makes each dyad a traction map. At the rim,
    # radial traction vanishes; far away these three readings recover the remote map.
    return (radial_stress * radial * (radial | Planar)
            + tangential_stress * tangent * (tangent | Planar)
            + shear_stress * (radial * (tangent | Planar) + tangent * (radial | Planar)))  # [...] Planar <- Planar


def polarization(direction: Planar) -> Polarization:
    """Light polarized linearly along a unit direction in the plate, on the Poincaré sphere: at
    twice the direction's angle from x, which is x reflected in the direction."""
    return direction >> mv.x                                                      # [...] Polarization


def half_turn(stress: Stress, stress_phase: float) -> Bivector:
    """Half the plate's turn of the Poincaré sphere, with the retardance per stress `stress_phase`,
    2π times the stress-optic coefficient times the thickness over the wavelength: about the state
    polarized along a principal direction, by the retardance of the principal stress difference.
    The mean stress delays both principal directions alike and turns nothing."""
    principal, directions = stress.eigh()                                         # [..., modes] Scalar, Planar
    retardance = stress_phase * (principal[..., 1] - principal[..., 0])           # [...] Scalar
    return polarization(directions[..., 1]).dual() * (retardance / 2)             # [...] Bivector


def retarder(stress: Stress, stress_phase: float) -> Retarder:
    """The plate's turn of the Poincaré sphere."""
    return half_turn(stress, stress_phase).exp()                                  # [...] Retarder


def transmitted(state: Polarization, analyser: Polarization) -> Scalar:
    """The share of unit-intensity light in a state that a polarizer passing another lets through:
    Malus's law, one plus their inner product, halved."""
    return (1 + (analyser | state)) / 2                                           # [...] Scalar


def polariscope(turn: Bivector, incident: Polarization, analyser: Polarization) -> Scalar:
    """The share of light entering in one state that leaves through the analyser, after a plate
    whose half turn of the sphere is `turn`. Scaling the load scales the turn: the principal
    directions, and so the axis, stay put."""
    return transmitted(turn.exp() >> incident, analyser)                          # [...] Scalar
