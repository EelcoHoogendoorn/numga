"""Curvature of a gravitational wave: a map from spacetime areas to the planes they turn in, and the tides
it raises.

The curvature of a vacuum plane wave is modelled as a map from areas, the joins of two directions, to
planes, a sum of two dyads of the null planes `k ^ x` and `k ^ y` that contain the wave's direction `k`,
each paired with an area by incidence. Every plane it gives contains the wave's direction, `k ^ curvature`
is zero, which is what makes a wave in vacuum of type N. Binding an observer's velocity twice turns it into
the tidal map, from a separation to the tidal force on it, `observer & curvature(observer & separation)`,
read off as the Lorentz force reads a field. Paired with a second separation, it is the tidal form, whose
eigenvalues against the metric are the stretch and squeeze the observer measures; the metric also turns the
force into the relative acceleration.

The cross polarization is the plus pattern turned an eighth of a turn in the plane across the wave, and the
circular wave the plus pattern turning with half the carrier's phase. The same wave is a strain map, the
change it makes to the metric: from a bead's rest separation to the slice that reads its stretch, the type of
the tidal map, which is minus the strain's second time derivative. The metric turns the strain's reading into
the bead's displacement, and the strain's second derivative along the wave gives the curvature.

In the tensor notation of general relativity the curvature map reads as the Riemann tensor and the strain map
as half the metric perturbation.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_simpson

from numga import Extensor, NumpyContext
from numga.algebras import STA

mv = NumpyContext(STA).multivector
Scalar = STA.gatype.scalar()
# Wave vectors, and the planes fields and curvature turn in: vectors and bivectors, extrinsic.
Vector = STA.gatype.vector()
Bivector = STA.gatype.bivector()
# Events, velocities and separations, and the areas their joins span: antivectors and antibivectors.
Antivector = STA.gatype.antivector()
Antibivector = STA.gatype.antibivector()
Rotor = STA.gatype.rotor()

Curvature = STA.gatype((Bivector, Antibivector))    # Bivector <- Antibivector
Tidal = STA.gatype((Vector, Antivector))            # Vector <- Antivector
Strain = STA.gatype((Vector, Antivector))           # Vector <- Antivector

t, x, y, z = mv.t, mv.x, mv.y, mv.z
# One unit along the lab's time: the direction across the lab's slice of space `t`.
time_direction = -t.dual()                                   # [] Antivector
# A wave travelling along z at the speed of light, and its wave vector, the slice its dual gives that direction.
travel = time_direction + mv.z.dual()                        # [] Antivector
wave_vector = -travel.dual()                                 # [] Vector
# An eighth of a turn in the plane across the wave.
eighth_turn = (mv.xy * (-np.pi / 8)).exp()


# --- math -----------------------------------------------------------------------------
def plane_wave_curvature(k: Vector, a: Vector, b: Vector) -> Curvature:
    """Vacuum plane-wave curvature along the null wave vector k, polarized on the transverse pair (a, b).

    Each null plane `k ^ a` is weighted by its pairing with the area; opposite weights cancel the Ricci
    contraction, the curvature of vacuum.
    """
    na, nb = k ^ a, k ^ b                                                    # [] Bivector
    return na * (na & Antibivector) - nb * (nb & Antibivector)               # [] Bivector <- Antibivector


def polarizations() -> tuple[Curvature, Curvature]:
    """Unit plus and cross curvature maps for a wave travelling along +z: cross is plus turned an eighth of
    a turn in the plane across the wave, on its output and its input."""
    plus = plane_wave_curvature(wave_vector, x, y)                           # [] Bivector <- Antibivector
    return plus, eighth_turn >> plus(eighth_turn << Antibivector)


def tidal_map(curvature: Curvature, observer: Antivector) -> Tidal:
    """The observer's tidal map, from a separation to the tidal force on it.

    The observer and a separation span a ribbon of spacetime, the curvature turns it into a plane, and
    the observer reads the plane as a force, as a velocity reads a field.
    """
    return observer & curvature(observer & Antivector)                       # [...] Vector <- Antivector


def boosted_observers(rapidities: np.ndarray, direction: Antivector) -> Antivector:
    """Velocities of observers boosted along a spatial direction, one per rapidity."""
    # The plane of the boost is the join of the direction and time, which the dual turns into a generator.
    boosts = ((direction & time_direction).dual() * (rapidities / 2)).exp()  # [n] Rotor
    return boosts >> time_direction                                          # [n] Antivector


def strain_patterns() -> tuple[Strain, Strain]:
    """Unit plus and cross strain maps for a wave travelling along +z: the change the wave makes to the
    metric, from a separation to the slice that reads its stretch.

    The plus pattern stretches along x and squeezes along y, weighting the slices x and y by their readings
    of the separation; the cross pattern is the same map turned an eighth of a turn in the plane across the
    wave, as the curvature's is.
    """
    plus = x * (x & Antivector) - y * (y & Antivector)                       # [] Vector <- Antivector
    return plus, eighth_turn >> plus(eighth_turn << Antivector)


def polarized(plus: Extensor, cross: Extensor, profiles: np.ndarray) -> Extensor:
    """The plus and cross patterns, each weighted by its profile, `profiles` `[..., 2]`."""
    return profiles[..., 0] * plus + profiles[..., 1] * cross


def curvature_of_strain(k: Vector, second: Strain) -> Extensor:
    """Curvature as a map on pairs of directions, from the strain's second derivative along the wave.

    Applied to edges `a` and `b` it gives `(k & a) * (second(b) ^ k) - (second(a) ^ k) * (k & b)`; its two
    arguments are the plane's edges, in that order.
    """
    bent = second ^ k                                                        # [...] Bivector <- Antivector
    return (k & Antivector) * bent - bent * (k & Antivector)                 # [...] Bivector <- (Antivector, Antivector)


# --- plumbing -------------------------------------------------------------------------
def wave_packet(time: np.ndarray, duration: float, cycles: int, amplitude: float) -> tuple[np.ndarray, np.ndarray]:
    """The plus and cross strain profiles over time, and their second derivatives, `[times, 2]` each: a
    Gaussian about the middle of the window, a twelfth of it wide, times a carrier, the cross profile a
    quarter cycle behind the plus. At the window's ends the packet is about `exp(-18)` of its peak."""
    s = time - duration / 2
    sigma, rate = duration / 12, 2 * np.pi * cycles / duration
    # The two profiles as the real and imaginary parts of one complex exponential, quadratic in time.
    profile = amplitude * np.exp(-s**2 / (2 * sigma**2) + 1j * rate * s)
    second = ((-s / sigma**2 + 1j * rate) ** 2 - 1 / sigma**2) * profile
    return np.stack([profile.real, profile.imag], axis=-1), np.stack([second.real, second.imag], axis=-1)


def integrate_acceleration(time: np.ndarray, acceleration: Antivector) -> Antivector:
    """Integrate twice along the first batch axis, starting at rest."""
    velocity = cumulative_simpson(acceleration.kernel, x=time, axis=0, initial=0)
    displacement = cumulative_simpson(velocity, x=time, axis=0, initial=0)
    return acceleration.context.extensor(acceleration.gatype, displacement)
