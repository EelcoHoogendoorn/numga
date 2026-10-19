"""Curvature of a gravitational wave: a nilpotent map on spacetime planes, and the tides it raises.

The curvature of a vacuum plane wave is modelled as a map from bivectors to bivectors, a sum of two dyads
of null planes that contain the wave's direction. It is nonzero, yet applied twice it gives zero, so all six
of its eigenvalues vanish. Binding an observer's velocity turns it into the tidal map, a map from separations
to relative accelerations, whose eigenvalues are the stretch and squeeze that observer measures: binding
composes different maps rather than conjugating one.

The cross polarization is the plus pattern turned an eighth of a turn in the plane across the wave, which for
a spin-two wave is a quarter duality turn, `-I * plus`, and a wave packet's phase is the pseudoscalar's
exponential. In gauge theory gravity the same wave is a strain map on vectors, whose value on
a bead's rest separation is its displacement, and whose second derivative along the wave gives the curvature.

In the tensor notation of general relativity the curvature map reads as the Riemann tensor and the strain map
as half the metric perturbation.
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
Phasor = STA.gatype.scalar_pseudoscalar()

# Extensors (linear maps between blade subspaces), read output <- input. The curvature takes an
# area bivector to a curvature bivector, the tidal map a separation to a relative acceleration,
# and the strain a rest separation to a displacement.
Curvature = STA.gatype((Bivector, Bivector))        # Bivector <- Bivector
Tidal = STA.gatype((Vector, Vector))                # Vector <- Vector
Strain = STA.gatype((Vector, Vector))               # Vector <- Vector

# Canonical spacetime basis:
t, x, y, z = mv.vector(np.eye(4))
# The pseudoscalar squares to minus one and commutes with every bivector: it turns a field by duality.
I = mv.txyz
# An eighth of a turn in the plane across a wave travelling along z.
eighth_turn = (mv.xy * (np.pi / 8)).exp()


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
    """Unit plus and cross curvature maps for a wave travelling along +z: cross is plus turned an eighth of
    a turn in the plane across the wave, on its output and its input. For this null curvature the turn is a
    quarter duality turn, `-I * plus`: the pattern turns by half the duality angle, as a spin-two field does.
    """
    plus = plane_wave_curvature(t + z, x, y)                             # [] Bivector <- Bivector
    return plus, eighth_turn >> plus(eighth_turn << Bivector)


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

    The plus pattern stretches along x and squeezes along y; the cross pattern is the same map turned an
    eighth of a turn in the plane across the wave, as the curvature's is.
    """
    plus = y * (y | Vector) - x * (x | Vector)                           # [] Vector <- Vector
    return plus, eighth_turn >> plus(eighth_turn << Vector)


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
    `k.wedge(second(a)) * (k | b) - (k | a) * k.wedge(second(b))`; its two arguments are the
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
