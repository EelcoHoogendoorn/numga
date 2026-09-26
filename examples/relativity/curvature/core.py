"""A curvature map with only zero eigenvalues, and the tides it produces: core mathematics.

In a vacuum plane wave, curvature is a nonzero bivector-to-bivector extensor
whose composition with itself vanishes. Its image consists of null bivectors
containing the wave's propagation direction; that image lies in its kernel.
An indefinite metric allows a self-adjoint map to have this structure. All
six eigenvalues vanish, although the map and its physical effects do not.

Fix an observer. Wedge a neighbouring particle's separation with the observer's
velocity to form a spacetime ribbon, apply curvature, then let the resulting
bivector act on that velocity. Leaving separation open gives the tidal map,
from separation to relative acceleration. This observer binding is a composition of
different maps, not a similarity transformation. Its spatial eigenvalues can
therefore be positive, negative and zero even though curvature is nilpotent.

The full curvature is assembled from null-bivector dyads. The cross
polarization is the dual of plus, `-I * plus`: for a null curvature a duality
rotation by `(I * theta).exp()` turns the pattern by half that angle about the
wave axis, the mark of a spin-two field. A wave packet's phase is therefore the
pseudoscalar's exponential, and the circular case is one phasor times plus.
Geometry stays in GA, with coefficient work confined to the numerical
integration of the detector.

In gauge theory gravity the same wave is carried by a map on vectors, the
position gauge field, which differs from the identity by a strain map: each
bead's displacement is the strain map applied to its rest separation, with no
integration. Its second derivative along the wave, wedged with the wave vector
and weighted by the overlap with it, is the curvature as a map on pairs of
vectors, and it agrees with the dyad construction at every time and
polarization.

This is first-order geodesic deviation for a weak wave and a detector much
smaller than its wavelength. Acceleration acts on each bead's unperturbed
separation. A Gaussian strain packet is negligible at both ends of its window,
so the initially stationary ring returns to rest, to first order and to within
those tails.
Units set the speed of light to one.

In tensor index notation the curvature map reads as the Riemann tensor and
the strain map as half the metric perturbation.

References:
  Coley & Hervik, "Higher dimensional bivectors and classification of the Weyl
  operator", CQG 27 (2010) 015002, Sec. 4.5 and Appendix B (type N):
  https://arxiv.org/pdf/0909.1160#page=11
  Tong, General Relativity, Sec. 5.2.2, "Bobbing on the Waves":
  https://www.damtp.cam.ac.uk/user/tong/gr/grhtml/S5.html#S5.S2.SS2

This module never imports a plotting library.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_simpson

from numga import Extensor, NumpyContext, stack
from numga.algebras import STA

# ---------------------------------------------------------------------------
# 1. Spacetime Algebra Setup (STA: Algebra("t+x-y-z-"))
# ---------------------------------------------------------------------------
ctx = NumpyContext(STA)
mv = ctx.multivector

# Blade subspaces:
Scalar = STA.gatype.scalar()
# Grade 1: events, velocities, separations.
Vector = STA.gatype.vector()
# Grade 2: oriented spacetime areas, Lorentz generators.
Bivector = STA.gatype.bivector()
# Even subalgebra: rotations and boosts.
Rotor = STA.gatype.rotor()
# A scalar plus a pseudoscalar: an amplitude and a phase.
Phasor = STA.gatype(STA.subspace.scalar() + STA.subspace.pseudoscalar())

# Extensors (linear maps between blade subspaces), read output <- input. The curvature takes an
# area bivector to a curvature bivector, the tidal map a separation to a relative acceleration,
# and the strain a rest separation to a displacement.
Curvature = STA.gatype((Bivector, Bivector))        # Bivector <- Bivector
Tidal = STA.gatype((Vector, Vector))                # Vector <- Vector
Strain = STA.gatype((Vector, Vector))               # Vector <- Vector

# Canonical spacetime basis:
t, x, y, z = mv.vector(np.eye(4))
# The pseudoscalar squares to minus one and commutes with every bivector.
I = mv.txyz


# ---------------------------------------------------------------------------
# 2. Curvature and its observer binding
# ---------------------------------------------------------------------------
def plane_wave_curvature(k: Vector, a: Vector, b: Vector) -> Curvature:
    """Vacuum plane-wave curvature along the null direction k, polarized on the transverse pair (a, b).

    The null bivectors k ^ a and k ^ b are mutually orthogonal, so their dyads compose to zero
    and the map is nilpotent. Opposite weights cancel the Ricci contraction: this is vacuum
    curvature with every eigenvalue zero, although the map itself is nonzero.
    """
    na, nb = k.wedge(a), k.wedge(b)                                      # [] Bivector
    return na * (na | Bivector) - nb * (nb | Bivector)                    # [] Bivector <- Bivector


def polarizations() -> tuple[Curvature, Curvature]:
    """Unit plus and cross curvature maps for a wave travelling along +z.

    The cross polarization is the dual of plus: a quarter duality turn, `-I * plus`, turns the
    stretch and squeeze pattern of a null curvature by an eighth turn, 45 degrees about the axis.
    """
    plus = plane_wave_curvature(t + z, x, y)                             # [] Bivector <- Bivector
    return plus, -I * plus


def tidal_map(curvature: Curvature, observer: Vector) -> Tidal:
    """Bind the observer twice and leave the separation open: from separation to relative acceleration.

    observer ^ Vector sweeps a separation forward in time into a spacetime ribbon, the curvature
    maps that ribbon to a bivector, and the commutator with the observer reads it out as a
    vector. This composes different maps rather than conjugating one, which is why its
    eigenvalues can be nonzero even though the curvature's are all zero.
    """
    return curvature(observer.wedge(Vector)).commutator(observer)        # [] Vector <- Vector


def boosted_observers(rapidities: np.ndarray, direction: Vector) -> Vector:
    """Four-velocities of observers boosted along a spatial direction, one per rapidity."""
    boosts: Rotor = ((direction ^ t) * (rapidities / 2.0)).exp()         # [n] Rotor
    return boosts >> t                                                   # [n] Vector


# ---------------------------------------------------------------------------
# 3. The same wave as a strain map
# ---------------------------------------------------------------------------
def strain_patterns() -> tuple[Strain, Strain]:
    """Unit plus and cross strain maps for a wave travelling along +z.

    The plus pattern stretches along x and squeezes along y; the cross pattern is the same map
    turned by an eighth turn about the wave axis. The transverse plane `xy` squares to minus one and
    turns each output a quarter turn, which turns the stretch and squeeze pattern by an eighth: on
    the strain it makes the turn that the duality `-I` makes on the curvature.
    """
    plus = y * (y | Vector) - x * (x | Vector)                           # [] Vector <- Vector
    cross = mv.xy | plus                                                 # [] Vector <- Vector
    return plus, cross


def polarized_strain(plus: Strain, cross: Strain, profile: Phasor) -> Strain:
    """Weak-wave strain over (time, polarization): half the profile times the unit strain maps, as a
    map on separations.

    Applied to a bead's rest separation it gives that bead's displacement. With the profile's second
    derivative in place of the profile it gives the relative acceleration, the tidal map.

    `plus + I * cross` pairs the two patterns the way the phasor pairs its parts: the vector part of
    the phasor times it is the circular strain, and of the phasor's scalar or pseudoscalar part alone,
    the plus or the cross strain.

    In tensor index notation the strain reads as half the metric perturbation.
    """
    analytic = plus + I * cross                                          # [] Vector + Trivector <- Vector
    waves = (profile.select[0] * analytic, profile.select[4] * analytic, profile * analytic)   # [n_time] each
    return 0.5 * stack(waves, axis=1).select[1]                          # [n_time, n_polarizations] Vector <- Vector


def curvature_of_strain(k: Vector, second: Strain) -> Extensor:
    """Curvature as a map on pairs of vectors, from the strain's second derivative along the wave.

    Applied to edges `a` and `b` it gives
    `k.wedge(second(a)) * (k | b) - (k | a) * k.wedge(second(b))`; the two open vectors are the
    plane's edges, in that order. Binding one edge gives the same map as the dyad curvature applied
    to the wedge with that edge.
    """
    return k.wedge(second) * (k | Vector) - (k | Vector) * k.wedge(second)   # Bivector <- (Vector, Vector)


# ---------------------------------------------------------------------------
# 4. Wave packet and detector
# ---------------------------------------------------------------------------
def wave_packet(
    time: np.ndarray, duration: float, cycles: int, amplitude: float,
) -> tuple[Phasor, Phasor]:
    """The strain profile and its second derivative as phasors over time: a Gaussian about the middle
    of the window, a twelfth of it wide, times the carrier `(I * -phase).exp()`. Times plus, the
    scalar part weights the plus pattern and the pseudoscalar part the cross pattern, a quarter cycle
    behind.

    The packet is the exponential of one phasor quadratic in time, so its rate is the phasor
    `-s / sigma**2 - I * b` and its second derivative `(rate * rate - 1 / sigma**2) * profile`. At
    the window's ends it is about `exp(-18)` of its peak.
    """
    s = time - duration / 2
    sigma, b = duration / 12, 2 * np.pi * cycles / duration
    profile = (I * -(b * s)).exp() * (amplitude * np.exp(-s**2 / (2 * sigma**2)))   # [n_time] Phasor
    rate = -s / sigma**2 - I * b                                             # [n_time] Phasor
    return profile, (rate * rate - 1 / sigma**2) * profile                   # [n_time] Phasor each


def polarized_waves(plus: Curvature, second: Phasor) -> Curvature:
    """Weak-wave curvature over (time, polarization): plus, cross, and circular.

    The weak-wave curvature is minus half the profile's second derivative times the unit map. The
    circular wave is the phasor times plus; its scalar part alone is the plus wave, and its
    pseudoscalar part alone the cross wave, since cross is `-I * plus`.

    In tensor index notation this reads as the Riemann components with two time indices equal to
    minus half the second time derivative of the metric perturbation.
    """
    waves = (second.select[0] * plus, second.select[4] * plus, second * plus)   # [n_time] Bivector <- Bivector each
    return -0.5 * stack(waves, axis=1)                                  # [n_time, n_polarizations]


def detector_ring(count: int) -> Vector:
    """Unit reference separations in the plane transverse to the wave."""
    angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
    # x turned toward y by each angle; x and y square to minus one, which turns the sense.
    return (mv.xy * (angles / 2)).exp() >> mv.x                         # [count] Vector


def integrate_acceleration(time: np.ndarray, acceleration: Vector) -> Vector:
    """Numerical boundary: integrate twice along the first batch axis, starting at rest."""
    velocity = cumulative_simpson(acceleration.kernel, x=time, axis=0, initial=0)
    displacement = cumulative_simpson(velocity, x=time, axis=0, initial=0)
    return mv.vector(displacement)


def plane_patch(area: Bivector, edge: Vector) -> Vector:
    """Four corners of a simple area element, drawn from a unit spacelike edge in its plane."""
    other: Vector = edge.commutator(area) * 0.5
    edge = edge * 0.6
    return stack((-edge - other, -edge + other, edge + other, edge - other))  # [n_corners] Vector
