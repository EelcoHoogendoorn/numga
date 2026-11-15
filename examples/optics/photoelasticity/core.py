"""Polarized light transmitted through a loaded transparent plate reveals the stress concentration around a circular hole.

In-plane stress in a thin isotropic plate is modelled as a linear map `Traction- <- Cut+` from extrinsic cut planes to
intrinsic traction directions. Each dyad `radial * (Cut & radial)` weights a traction direction by how far a direction
crosses a cut plane without invoking a metric; the spatial metric enters when converting a direction to the cut plane
across it and when solving the generalized eigenproblem `(Cut & stress(Cut)).eigh(Cut | Cut)` for the principal
stresses and principal cuts.

Light propagation normal to the plate is modelled in spacetime with complex phasor coefficients holding the electric
field oscillation at zero and quarter-period phases. Each electric field is an extrinsic bivector paired against the
worldsheet plane `axis & time` of a polarizer or principal stress axis via `field & (axis & time)`. Under the weak
stress-optic law, each principal direction delays its field component proportionally to its principal stress.
"""

from __future__ import annotations

import numpy as np

from numga import Algebra, NumpyContext
from numga.algebras import STA as ga

# Complex coefficients hold the in-phase and quarter-period electric field amplitudes.
mv = NumpyContext(ga, dtype=np.complex128).multivector
Scalar = ga.gatype.scalar()                                # Scalar+
Cut = ga.gatype.vector().derive.within(Algebra("x+y+"))    # Vector+
Planar = Cut.derive.dual                                   # Antivector-
Traction = Planar                                          # Antivector-
Stress = ga.gatype((Traction, Cut))                        # Traction- <- Cut+
Field = ga.gatype.bivector()                               # Bivector+
OpticalMap = ga.gatype((Field, Field))                     # Field+ <- Field+

# Rest-frame time direction pairing to one with mv.t, and eighth/quarter turns in the plate.
time = -mv.t.dual()                                        # [] Antivector-
eighth_turn = (mv.xy * (-np.pi / 8)).exp()                 # [] Rotor+
quarter_turn = (mv.xy * (-np.pi / 4)).exp()                # [] Rotor+
# Electric field polarized along a planar direction, dual to the worldsheet plane swept in time.
axis_field = (Planar & time).dual()                        # [] Field+ <- Planar-
polarizer = axis_field * ((Planar & time) & Field)         # [] Field+ <- (Planar-, Planar-, Field+)


# --- math -----------------------------------------------------------------------------
def kirsch_stress(positions: Planar, radius: float, remote: Stress) -> Stress:
    """Kirsch's exterior solution maps `positions` (`[...] Planar-`) outside a traction-free hole of `radius` under
    `remote` load to the local stress (`[...] Traction- <- Cut+`)."""
    distance = positions.antinorm()                                               # [...] Scalar+
    radial = positions / distance                                                 # [...] Planar-
    tangent = quarter_turn >> radial                                              # [...] Planar-
    radius_ratio = (radius / distance).squared()                                  # [...] Scalar+

    # Cut planes orthogonal to the radial and tangential directions introduce the spatial metric.
    radial_cut, tangent_cut = radial.dual_inverse(), tangent.dual_inverse()       # [...] Cut+, Cut+
    # Resolve the remote stress on those cuts; half the stress form's trace gives the mean stress.
    mean = (Cut & remote(Cut)).contract() / 2                                     # [...] Scalar+
    anisotropy = (radial_cut & remote(radial_cut)) - mean                         # [...] Scalar+
    shear = radial_cut & remote(tangent_cut)                                      # [...] Scalar+
    radial_stress = (mean * (1 - radius_ratio)
                     + anisotropy * (1 - 4 * radius_ratio + 3 * radius_ratio.squared()))  # [...] Scalar+
    tangential_stress = (mean * (1 + radius_ratio)
                         - anisotropy * (1 + 3 * radius_ratio.squared()))         # [...] Scalar+
    shear_stress = shear * (1 + 2 * radius_ratio - 3 * radius_ratio.squared())    # [...] Scalar+

    # Metric-free dyads reconstruct the stress map with vanishing traction on the circular rim.
    return (radial_stress * radial * (Cut & radial)
            + tangential_stress * tangent * (Cut & tangent)
            + shear_stress * (radial * (Cut & tangent) + tangent * (Cut & radial)))  # [...] Traction- <- Cut+


def principal_stresses(stress: Stress) -> tuple[Scalar, Planar]:
    """Solving `(Cut & stress(Cut)).eigh(Cut | Cut)` yields the ascending `[..., 2] Scalar+` principal stresses and
    `[..., 2] Planar-` principal directions."""
    principal, cuts = (Cut & stress(Cut)).eigh(Cut | Cut)                         # [..., 2] Scalar+, Cut+
    return principal, cuts.dual()                                                 # [..., 2] Scalar+, Planar-


def retarder(stress: Stress, delay_per_stress: float) -> OpticalMap:
    """Delaying each principal axis projection by `delay_per_stress` times its principal stress gives the stressed
    plate's optical map (`[...] Field+ <- Field+`)."""
    principal, directions = principal_stresses(stress)                            # [..., 2] Scalar+, Planar-
    delays = (principal * (1j * delay_per_stress)).exp()                          # [..., 2] Scalar+
    return (delays * polarizer(directions, directions)).sum(axis=-1)              # [...] Field+ <- Field+


def intensity(field: Field, axis: Planar) -> Scalar:
    """Pairing `field` (`[...] Field+`) with the worldsheet `axis & time` and summing the squared real and
    quarter-period amplitudes gives the transmitted intensity (`[...] Scalar+`)."""
    component = field & (axis & time)                                             # [...] Scalar+
    return component.real().squared() + (-1j * component).real().squared()        # [...] Scalar+


def linear_polariscope(plate: OpticalMap, axis: Planar) -> Scalar:
    """Crossed linear polarizers with entrance along `axis` (`[...] Planar-`) transmit the intensity fraction
    (`[...] Scalar+`) passed by `plate`."""
    return intensity(plate(axis_field(axis)), quarter_turn >> axis)               # [...] Scalar+


def circular_polariscope(plate: OpticalMap, axis: Planar) -> Scalar:
    """Crossed quarter-wave plates at an eighth turn to crossed linear polarizers along `axis` (`[...] Planar-`)
    transmit the circular-polariscope intensity (`[...] Scalar+`) of `plate`. Without stress, the two quarter-wave
    plates cancel and the field is dark."""
    fast_axis = eighth_turn >> axis                                               # [...] Planar-
    slow_axis = quarter_turn >> fast_axis                                         # [...] Planar-
    # Crossed quarter-wave plates swap their fast and slow axes between entrance and exit.
    entrance_plate = polarizer(fast_axis, fast_axis) + 1j * polarizer(slow_axis, slow_axis)   # [...] Field+ <- Field+
    exit_plate = polarizer(slow_axis, slow_axis) + 1j * polarizer(fast_axis, fast_axis)       # [...] Field+ <- Field+
    incident_field = entrance_plate(axis_field(axis))                             # [...] Field+
    transmitted_field = exit_plate(plate(incident_field))                         # [...] Field+
    return intensity(transmitted_field, quarter_turn >> axis)                     # [...] Scalar+
