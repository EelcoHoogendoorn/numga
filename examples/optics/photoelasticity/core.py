"""Stress around a traction-free hole, read through polarized light.

The plate is thin, isotropic and infinite, with uniform stress far from its circular hole. Its in-plane
stress maps the normal of a cut to the traction across it; positions, normals and tractions are
directions in the plate, antivectors, intrinsic like all geometry.

Light is done in spacetime. Its electric field is a bivector, extrinsic, through the observer's time,
with complex coefficients that carry its phase in time. Each axis of a polarizer, or principal direction
of the stress, sweeps a plane through the observer's time, and the field's component along the axis is
its pairing with that plane, `field & (axis & time)`. The weak stress-optic law delays the field along
each principal direction by a phase proportional to the stress along it: the plate is the sum over its
principal directions of their polarizers, each with its delay. Light travels normal to the plate;
absorption, reflection and ray bending are neglected.
"""

import numpy as np

from numga import NumpyContext
from numga.algebras import STA as ga

# Complex coefficients carry the light's phase in time.
mv = NumpyContext(ga, dtype=np.complex128).multivector
Scalar = ga.gatype.scalar()
# Directions in the plate: positions, the normals of cuts and the tractions across them.
Planar = ga.gatype.antivector().derive.containing("z t")
Stress = ga.gatype((Planar, Planar))                       # Planar <- Planar
# The electric field, and the optical components as maps on it.
Field = ga.gatype.bivector()
OpticalMap = ga.gatype((Field, Field))                     # Field <- Field

# The direction in time of an observer at rest with the plate, which pairs to one with `mv.t`; the dual
# carries the minus of t's square.
time = -mv.t.dual()                                        # [] Antivector
# An eighth and a quarter of a turn in the plate, from x towards y.
eighth_turn = (mv.xy * (-np.pi / 8)).exp()
quarter_turn = (mv.xy * (-np.pi / 4)).exp()


# --- math -----------------------------------------------------------------------------
def kirsch(positions: Planar, radius: float, remote: Stress) -> Stress:
    """The exterior stress for a circular hole under any uniform symmetric remote stress."""
    distance_squared = positions.squared()                                        # [...] Scalar
    radial = positions / distance_squared.square_root()                           # [...] Planar
    tangent = quarter_turn >> radial                                              # [...] Planar
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

    # Over `Planar`, the space of all normals, each dyad is a traction map. At the rim, radial traction
    # vanishes; far away these three readings recover the remote map.
    return (radial_stress * radial * (radial | Planar)
            + tangential_stress * tangent * (tangent | Planar)
            + shear_stress * (radial * (tangent | Planar) + tangent * (radial | Planar)))  # [...] Planar <- Planar


def field_along(axis: Planar) -> Field:
    """The unit field along an axis: the one that pairs to one with the plane the axis sweeps in time."""
    sweep = axis & time                                                           # [...] Antibivector
    field = sweep.dual_inverse()                                                  # [...] Field
    return field / (field & sweep)


def polarizer(axis: Planar) -> OpticalMap:
    """Transmit the field's component along a unit axis."""
    return field_along(axis) * (Field & (axis & time))                            # [...] Field <- Field


def quarter_wave(fast_axis: Planar) -> OpticalMap:
    """Delay the field across the fast axis by a quarter period."""
    return polarizer(fast_axis) + 1j * polarizer(quarter_turn >> fast_axis)       # [...] Field <- Field


def retarder(stress: Stress, stress_phase: float) -> OpticalMap:
    """The stressed plate: the field along each principal direction delayed by its principal stress
    times `stress_phase`, 2π times the stress-optic coefficient times the thickness over the wavelength.
    The mean stress delays both principal directions alike and changes no intensity."""
    principal, directions = stress.eigh()                                         # [..., modes] Scalar, Planar
    delays = (principal * (1j * stress_phase)).exp()                              # [..., modes] Scalar
    return (delays * polarizer(directions)).sum(axis=-1)                          # [...] Field <- Field


def intensity(field: Field, axis: Planar) -> Scalar:
    """The intensity a field puts through a polarizer along a unit axis, in units of a unit field's: its
    component along the axis, squared over both quadratures of its phase."""
    component = field & (axis & time)                                             # [...] Scalar
    return component.real().squared() + (-1j * component).real().squared()        # [...] Scalar


def linear_polariscope(plate: OpticalMap, axis: Planar) -> Scalar:
    """The share of light through a plate between crossed linear polarizers, the first along `axis`."""
    return intensity(plate(field_along(axis)), quarter_turn >> axis)              # [...] Scalar


def circular_polariscope(plate: OpticalMap, axis: Planar) -> Scalar:
    """The share of light through a plate between crossed circular polarizers: a linear polarizer along
    `axis` and a quarter-wave plate an eighth of a turn from it, the plate, and the same pair crossed. The
    two quarter-wave plates undo each other, so the view is dark without stress."""
    fast_axis = eighth_turn >> axis                                               # [...] Planar
    circular = quarter_wave(fast_axis)(field_along(axis))                         # [...] Field
    leaving = quarter_wave(quarter_turn >> fast_axis)(plate(circular))            # [...] Field
    return intensity(leaving, quarter_turn >> axis)                               # [...] Scalar
