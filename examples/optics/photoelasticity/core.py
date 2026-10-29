"""Stress around a traction-free hole, read through polarized light.

The plate is thin, isotropic and infinite, with uniform stress far from its circular hole. Its in-plane
stress maps a cut through the plate to the traction across it, `Stress = Traction <- Cut`: a cut is a
plane, extrinsic, which reads how far a direction crosses it, and positions and tractions are directions
in the plate, antivectors, intrinsic like all geometry. Each dyad weights a direction by a cut's reading
of another, `radial * (Cut & radial)`, with no metric; the metric enters where the cut across a direction
is meant, and in the principal stresses, the eigenvectors of the stress form `Cut & stress(Cut)` against
it.

Light is done in spacetime. Its electric field is a bivector, extrinsic, through the observer's time; each
coefficient holds the field's oscillation in time as a pair of amplitudes, the field now and a quarter period
later, so that a delay turns every pair alike. Each axis of a polarizer, or principal direction of the
stress, sweeps a plane through the observer's time, and the field's component along the axis is its
pairing with that plane, `field & (axis & time)`. The weak stress-optic law delays the field along each
principal direction by a share of the period proportional to the stress along it: the plate is the sum over
its principal directions of their polarizers, each with its delay. Light travels normal to the plate;
absorption, reflection and ray bending are neglected.
"""

import numpy as np

from numga import NumpyContext
from numga.algebras import STA as ga

# Each coefficient holds a pair of amplitudes, the field now and a quarter period later: complex numbers,
# on which a delay by an angle of the period, `np.exp(1j * angle)`, turns every pair alike.
mv = NumpyContext(ga, dtype=np.complex128).multivector
Scalar = ga.gatype.scalar()                                # Scalar+
# Directions in the plate: positions, and the tractions across cuts.
Planar = ga.gatype.antivector().derive.containing("z t")   # Antivector-
Traction = Planar                                          # Antivector-
# Cuts through the plate: planes, extrinsic, which read how far a direction crosses them.
Cut = ga.gatype.vector().derive.within("x y")              # Vector+
Stress = ga.gatype((Traction, Cut))                        # Traction- <- Cut+
# The electric field, and the optical components as maps on it.
Field = ga.gatype.bivector()                               # Bivector+
OpticalMap = ga.gatype((Field, Field))                     # Field+ <- Field+

# The direction in time of an observer at rest with the plate, which pairs to one with `mv.t`; the dual
# carries the minus of t's square.
time = -mv.t.dual()                                        # [] Antivector-
# An eighth and a quarter of a turn in the plate, from x towards y.
eighth_turn = (mv.xy * (-np.pi / 8)).exp()                 # [] Rotor+
quarter_turn = (mv.xy * (-np.pi / 4)).exp()                # [] Rotor+
# The field along a direction, the dual of the plane the direction sweeps in time: the vacuum's
# identification of a direction with the field along it, pairing to one with that plane for a unit direction.
field_along = (Planar & time).dual()                       # [] Field+ <- Planar-
# The field along one direction, weighted by the field's pairing with the plane another sweeps in time;
# bound to a unit axis twice, the polarizer that keeps the field's component along the axis.
polarizer = field_along * ((Planar & time) & Field)        # [] Field+ <- (Planar-, Planar-, Field+)


# --- math -----------------------------------------------------------------------------
def kirsch(positions: Planar, radius: float, remote: Stress) -> Stress:
    """The exterior stress for a circular hole under any uniform symmetric remote stress."""
    distance_squared = positions.squared()                                        # [...] Scalar+
    radial = positions / distance_squared.square_root()                           # [...] Planar-
    tangent = quarter_turn >> radial                                              # [...] Planar-
    radius_ratio = radius**2 / distance_squared                                   # [...] Scalar+

    # The cuts across the radial and tangential directions: where the metric enters.
    radial_cut, tangent_cut = radial.dual_inverse(), tangent.dual_inverse()       # [...] Cut+, Cut+
    # Resolve the remote traction map on those cuts; the mean is half the stress form's trace.
    mean = (Cut & remote(Cut)).contract() / 2                                     # [...] Scalar+
    anisotropy = (radial_cut & remote(radial_cut)) - mean                         # [...] Scalar+
    shear = radial_cut & remote(tangent_cut)                                      # [...] Scalar+
    radial_stress = (mean * (1 - radius_ratio)
                     + anisotropy * (1 - 4 * radius_ratio + 3 * radius_ratio.squared()))  # [...] Scalar+
    tangential_stress = (mean * (1 + radius_ratio)
                         - anisotropy * (1 + 3 * radius_ratio.squared()))         # [...] Scalar+
    shear_stress = shear * (1 + 2 * radius_ratio - 3 * radius_ratio.squared())    # [...] Scalar+

    # Each dyad weights a direction by a cut's reading of another. At the rim, radial traction vanishes;
    # far away these three readings recover the remote map.
    return (radial_stress * radial * (Cut & radial)
            + tangential_stress * tangent * (Cut & tangent)
            + shear_stress * (radial * (Cut & tangent) + tangent * (Cut & radial)))  # [...] Traction- <- Cut+


def principal_stresses(stress: Stress) -> tuple[Scalar, Planar]:
    """The principal stresses, from least to greatest, and their directions: the eigenvectors of the
    stress form against the metric are the principal cuts, and the directions are across them."""
    principal, cuts = (Cut & stress(Cut)).eigh(Cut | Cut)                        # [..., modes] Scalar+, Cut+
    return principal, cuts.dual()                                                 # [..., modes] Scalar+, Planar-


def retarder(stress: Stress, delay_per_stress: float) -> OpticalMap:
    """The stressed plate: the field along each principal direction delayed by its principal stress
    times `delay_per_stress`, the angle of the period a unit of stress delays the light by through the
    plate, 2π times the stress-optic coefficient times the thickness over the wavelength. The mean stress
    delays both principal directions alike and changes no intensity."""
    principal, directions = principal_stresses(stress)                            # [..., modes] Scalar+, Planar-
    delays = (principal * (1j * delay_per_stress)).exp()                          # [..., modes] Scalar+
    return (delays * polarizer(directions, directions)).sum(axis=-1)              # [...] Field+ <- Field+


def intensity(field: Field, axis: Planar) -> Scalar:
    """The intensity a field puts through a polarizer along a unit axis, in units of a unit field's: its
    component along the axis, squared and summed over both its amplitudes, now and a quarter period later."""
    component = field & (axis & time)                                             # [...] Scalar+
    return component.real().squared() + (-1j * component).real().squared()        # [...] Scalar+


def linear_polariscope(plate: OpticalMap, axis: Planar) -> Scalar:
    """The share of light through a plate between crossed linear polarizers, the first along `axis`."""
    return intensity(plate(field_along(axis)), quarter_turn >> axis)              # [...] Scalar+


def circular_polariscope(plate: OpticalMap, axis: Planar) -> Scalar:
    """The share of light through a plate between crossed circular polarizers: a linear polarizer along
    `axis` and a quarter-wave plate an eighth of a turn from it, the plate, and the same pair crossed. The
    two quarter-wave plates undo each other, so the view is dark without stress."""
    fast_axis = eighth_turn >> axis                                               # [...] Planar-
    slow_axis = quarter_turn >> fast_axis                                         # [...] Planar-
    # A quarter-wave plate delays the field along its slow axis by a quarter period; crossed, its fast and
    # slow axes trade places.
    entrance_plate = polarizer(fast_axis, fast_axis) + 1j * polarizer(slow_axis, slow_axis)   # [...] Field+ <- Field+
    exit_plate = polarizer(slow_axis, slow_axis) + 1j * polarizer(fast_axis, fast_axis)       # [...] Field+ <- Field+
    circular = entrance_plate(field_along(axis))                                  # [...] Field+
    leaving = exit_plate(plate(circular))                                         # [...] Field+
    return intensity(leaving, quarter_turn >> axis)                               # [...] Scalar+
