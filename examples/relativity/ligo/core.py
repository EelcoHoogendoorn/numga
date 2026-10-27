"""Gravitational-wave antenna patterns: how strongly an L-shaped interferometer such as LIGO responds to
a wave from each direction on the sky, in the geometric algebra of three-dimensional space.

A gravitational wave passing through stretches space along one direction across its path and
squeezes it along the other, by a strain: the change it makes to the metric, a map from offsets to
the planes that read them, `Strain = Plane <- Point`, symmetric and zero along the wave. For a wave
from a given direction, two strains span them all: plus, stretching along one axis across the wave as
much as it squeezes along the other, each offset's height above the planes facing east and north
weighting those planes, `east * (east & Point) - north * (north & Point)`, and cross, the same turned
by an eighth of a turn, `east * (north & Point) + north * (east & Point)`. Turning the axes across the
wave by an angle turns plus into cross after an eighth of a turn and back into itself after half a
turn: the strain is a spin-two field.

An arm along the unit direction `arm` lengthens by the fraction `(strain(arm) & arm) / 2`, the arm read
against the plane the strain hands it, and the interferometer reads the difference between its two
arms. The responses to a direction's plus and cross strains are the detector's antenna patterns. Their
root sum of squares does not depend on how the axes across the wave are turned: it is the detector's
sensitivity to that direction, largest straight overhead or underfoot, and zero for waves arriving in
the plane of the arms along the lines halfway between them. Only the picture of free masses moving
needs the metric: a mass moves perpendicular to the plane its strain hands it, `strain(mass).dual()`.

Places on Earth and directions on the sky share one frame: the rotor that turns the first axis to the
point at a latitude and longitude also turns the second axis to east and the third to north there. At
a detector's site it gives the level plane its arms lie in; for a direction on the sky it gives the
axes across a wave from there.

In the tensor notation of gravitational-wave astronomy the response reads as
$F = D^{ij} e_{ij}$, with the detector tensor $D = \\tfrac12(\\hat a \\otimes \\hat a - \\hat b \\otimes \\hat b)$
and $e$ the polarization tensor of the wave.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext
from numga.algebras import VGA3D as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                # Scalar+
# Offsets and directions are antivectors with intrinsic orientation.
Point = ga.gatype.antivector()                             # Antivector-
Plane = ga.gatype.vector()                                 # Vector+
Rotor = ga.gatype.rotor()                                  # Rotor+
# The change of the metric a wave carries: the plane that reads each offset's stretch.
Strain = ga.gatype((Plane, Point))                         # Plane+ <- Point-


# --- math -----------------------------------------------------------------------------
def place(latitudes: np.ndarray, longitudes: np.ndarray) -> Rotor:
    """The frame at each latitude and longitude, in radians: the rotor turning the first axis to that
    point on the unit sphere, the second to east and the third to north there; a turn from the equator
    towards the pole, then about the pole. The axes are fixed to the Earth: x through longitude zero on
    the equator, y through ninety degrees east on it, z through the north pole."""
    return (mv.xy * (-longitudes / 2)).exp() * (mv.xz * (-latitudes / 2)).exp()


def polarized(frames: Rotor, angles: np.ndarray) -> Rotor:
    """The frames turned about their first axis, the wave's direction, by the polarization angles."""
    return frames * (mv.yz * (-angles / 2)).exp()


def strains(frames: Rotor) -> tuple[Strain, Strain]:
    """The plus and cross strains of waves from the direction each frame's first axis points to, with
    its second and third axes across the wave."""
    # The planes facing east and north across the wave, each weighted by an offset's height above them.
    east, north = frames >> mv.y, frames >> mv.z           # [...] Plane+, Plane+
    plus = east * (east & Point) - north * (north & Point)   # [...] Plane+ <- Point-
    cross = east * (north & Point) + north * (east & Point)   # [...] Plane+ <- Point-
    return plus, cross


def arms(sites: Rotor, azimuths: np.ndarray) -> Point:
    """The arms of detectors at their sites `[...]`, from their azimuths `[..., 2]` in radians,
    clockwise from north: north turned towards east in each site's level plane."""
    east, north = sites[..., None] >> mv.y, sites[..., None] >> mv.z   # [..., 1] Plane+, Plane+
    return ((north ^ east) * (-azimuths / 2)).exp() >> north.dual()   # [..., 2] Point-


def response(arms: Point, strain: Strain) -> Scalar:
    """The interferometer's reading: the difference between its two arms' stretches, each arm
    lengthening by the fraction `(strain(arm) & arm) / 2`, the arm read against the plane the strain
    hands it."""
    stretches = (strain[..., None](arms) & arms) / 2       # [..., 2] Scalar+
    return stretches[..., 0] - stretches[..., 1]           # [...] Scalar+


def sensitivity(plus: Scalar, cross: Scalar) -> Scalar:
    """The root sum of squares of the responses to the plus and cross strains: the same for every turn
    of the axes across the wave."""
    return (plus.squared() + cross.squared()).square_root()
