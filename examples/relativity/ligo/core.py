"""Gravitational-wave antenna patterns compute how strongly an L-shaped interferometer such as LIGO responds to a wave from each direction on the sky in spacetime algebra.

A passing gravitational wave stretches proper distance along one transverse direction and squeezes it along the orthogonal direction via a spatial tidal strain `Strain = Plane+ <- Point-` orthogonal to both the observer's time axis and the wave's propagation direction. For each sky direction, two basis strains span all transverse traceless perturbations: plus (`east * (east & Point) - north * (north & Point)`) and cross (`east * (north & Point) + north * (east & Point)`). Rotating the transverse axes about the wave's propagation direction turns plus into cross after an eighth of a turn and back into itself after half a turn because gravitational radiation has spin two.

An interferometer arm along the unit spatial displacement `arm` (`arm.antinorm() == 1`) lengthens by the fractional stretch `(strain(arm) & arm) / 2`, and the differential readout subtracts the two arm stretches. The root sum of squares of the plus and cross responses forms the polarization-averaged directional sensitivity, peaking at unity straight overhead and underfoot while vanishing along the four bisectors in the detector's horizontal plane. Free test masses on a horizontal ring respond to an overhead wave through metric duality via `strain(mass).dual()`, tracing an ellipse that rotates with the polarization angle.

Places on Earth and arrival directions on the sky share one spatial rotor family: turning the first spatial axis to a given latitude and longitude simultaneously aligns the second and third spatial axes with local east and north, orienting both the horizontal interferometer arms at a site and the transverse polarization planes of an incoming wave.
"""

from __future__ import annotations

from collections.abc import Iterator
import numpy as np

from numga import NumpyContext
from numga.algebras import STA as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                # Scalar+
# Spatial displacements and sky directions lie orthogonal to `mv.t` as intrinsic 4-vectors.
Point = ga.gatype.antivector()                             # Antivector-
Plane = ga.gatype.vector()                                 # Vector+
Rotor = ga.gatype.rotor()                                  # Rotor+
# Transverse tidal metric perturbation mapping spatial displacements to spatial reading planes.
Strain = ga.gatype((Plane, Point))                         # Plane+ <- Point-


def earth_frame(latitudes: np.ndarray, longitudes: np.ndarray) -> Rotor:
    """Spatial rotors (`[...] Rotor+`) turn the first axis to each `(latitudes, longitudes)` direction in radians and align the second and third axes with local east and north."""
    latitude_turn = (mv.xz * (-latitudes / 2)).exp()                           # [...] Rotor+
    longitude_turn = (mv.xy * (-longitudes / 2)).exp()                         # [...] Rotor+
    return longitude_turn * latitude_turn


def polarized_frame(frames: Rotor, angles: np.ndarray) -> Rotor:
    """Spatial rotors (`[...] Rotor+`) rotate `frames` about their propagation axis by `angles` in radians."""
    polarization_turn = (mv.yz * (-angles / 2)).exp()                          # [...] Rotor+
    return frames * polarization_turn


def wave_strains(frames: Rotor) -> tuple[Strain, Strain]:
    """Plus and cross tidal strain maps (`[...] Plane+ <- Point-`) act across the transverse east and north planes of `frames`."""
    east, north = frames >> mv.y, frames >> mv.z                               # [...] Plane+, Plane+
    plus = east * (east & Point) - north * (north & Point)                     # [...] Plane+ <- Point-
    cross = east * (north & Point) + north * (east & Point)                    # [...] Plane+ <- Point-
    return plus, cross


def detector_arms(sites: Rotor, azimuths: np.ndarray) -> Point:
    """Unit spatial arm displacements (`[..., 2] Point-`) point along `azimuths` (`[..., 2]`, radians clockwise from north) in each site's horizontal plane."""
    east, north = sites[..., None] >> mv.y, sites[..., None] >> mv.z           # [..., 1] Plane+, Plane+
    azimuth_turn = ((north ^ east) * (-azimuths / 2)).exp()                    # [..., 2] Rotor+
    return azimuth_turn >> north.dual()                                        # [..., 2] Point-


def arm_response(arms: Point, strain: Strain) -> Scalar:
    """Differential arm stretch (`[...] Scalar+`) subtracts the two arm readings `(strain(arm) & arm) / 2`."""
    stretches = (strain[..., None](arms) & arms) / 2                           # [..., 2] Scalar+
    return stretches[..., 0] - stretches[..., 1]                               # [...] Scalar+


def directional_sensitivity(plus_response: Scalar, cross_response: Scalar) -> Scalar:
    """Root-sum-of-squares antenna sensitivity (`[...] Scalar+`) combines `plus_response` and `cross_response` invariantly under polarization turns."""
    return (plus_response.squared() + cross_response.squared()).square_root()


def polarization_states(
    site: Rotor,
    arms: Point,
    sky_frames: Rotor,
    angles: np.ndarray,
    rest_ring: Point,
    strain_scale: float,
) -> Iterator[tuple[Scalar, Point]]:
    """Yields `(sky_response, deformed_ring)` pairs (`[n_latitudes, n_longitudes] Scalar+`, `[n_masses] Point-`) as the wave polarization steps through `angles`."""
    for angle in angles:
        turned_sky = polarized_frame(sky_frames, np.asarray(angle))            # [n_latitudes, n_longitudes] Rotor+
        plus_sky, _ = wave_strains(turned_sky)                                 # [n_latitudes, n_longitudes] Plane+ <- Point-
        sky_response = arm_response(arms, plus_sky)                            # [n_latitudes, n_longitudes] Scalar+
        turned_overhead = polarized_frame(site, np.asarray(angle))             # [] Rotor+
        overhead_plus, _ = wave_strains(turned_overhead)                       # [] Plane+ <- Point-
        deformed_ring = rest_ring + strain_scale * overhead_plus(rest_ring).dual()   # [n_masses] Point-
        yield sky_response, deformed_ring
