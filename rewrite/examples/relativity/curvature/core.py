"""A curvature map with only zero eigenvalues, and the tides it produces: core mathematics.

In a vacuum plane wave, curvature is a nonzero bivector-to-bivector extensor
whose composition with itself vanishes. Its image consists of null bivectors
containing the wave's propagation direction; that image lies in its kernel.
An indefinite metric allows a self-adjoint map to have this structure. All
six eigenvalues vanish, although the map and its physical effects do not.

Fix an observer. Wedge a neighbouring particle's separation with the observer's
velocity to form a spacetime ribbon, apply curvature, then let the resulting
bivector act on that velocity. Leaving separation open gives the tidal map:
separation -> relative acceleration. This observer binding is a composition of
different maps, not a similarity transformation. Its spatial eigenvalues can
therefore be positive, negative and zero even though curvature is nilpotent.

The full curvature is assembled from null-bivector dyads. A rotor constructs
the cross polarization from plus; quarter-cycle phase separation gives the
circular case. Geometry stays in GA, with coefficient work confined to the
waveform and the numerical integration of the detector.

This is first-order geodesic deviation for a weak wave and a detector much
smaller than its wavelength. Acceleration acts on each bead's unperturbed
separation. A smooth strain packet and its first derivative vanish at both
ends, so the initially stationary ring returns to rest to first order.
Units have c = 1.

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

from numga import NumpyContext, stack
from numga.algebras import STA

# ---------------------------------------------------------------------------
# 1. Spacetime Algebra Setup (STA: R_{1,3}, t+ x- y- z-)
# ---------------------------------------------------------------------------
ctx = NumpyContext(STA)
mv = ctx.multivector

# Blade subspaces:
Scalar = STA.gatype.scalar()
Vector = STA.gatype.vector()                        # Grade 1: events, velocities, separations
Bivector = STA.gatype.bivector()                    # Grade 2: oriented spacetime areas, Lorentz generators
Rotor = STA.gatype.rotor()                          # Even subalgebra: rotations and boosts

# Extensors (linear maps between blade subspaces), read output <- input:
Curvature = STA.gatype((Bivector, Bivector))        # curvature bivector <- area bivector
Tidal = STA.gatype((Vector, Vector))                # relative acceleration <- separation

# Canonical spacetime basis:
t, x, y, z = mv.vector(np.eye(4))


# ---------------------------------------------------------------------------
# 2. Curvature and its observer binding
# ---------------------------------------------------------------------------
def plane_wave_curvature(k: Vector, a: Vector, b: Vector) -> Curvature:
    """Vacuum plane-wave curvature along the null direction k, polarized on the transverse pair (a, b).

    The null bivectors k ^ a and k ^ b are mutually orthogonal, so their dyads compose to zero
    and the map is nilpotent. Opposite weights cancel the Ricci contraction: this is vacuum
    curvature with every eigenvalue zero, although the map itself is nonzero.
    """
    na, nb = k.wedge(a), k.wedge(b)                                      # [] Bivector (null planes)
    return na * (na | Bivector) - nb * (nb | Bivector)                    # [] Bivector <- Bivector


def polarizations() -> tuple[Curvature, Curvature]:
    """Unit plus and cross curvature maps for a wave travelling along +z.

    The cross polarization is the plus polarization conjugated by an eighth-turn rotor about
    the wave axis, which turns the stretch and squeeze pattern by 45 degrees.
    """
    plus = plane_wave_curvature(t + z, x, y)                             # [] Bivector <- Bivector
    eighth_turn = (mv.xy * (np.pi / 8)).exp()                            # [] Rotor
    cross = eighth_turn >> plus(eighth_turn << Bivector)                 # [] Bivector <- Bivector
    return plus, cross


def tidal_map(curvature: Curvature, observer: Vector) -> Tidal:
    """Bind the observer twice and leave the separation open: separation -> relative acceleration.

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
# 3. Wave packet and detector
# ---------------------------------------------------------------------------
def wave_packet(
    time: np.ndarray, duration: float, cycles: int, amplitude: float,
) -> tuple[Scalar, Scalar]:
    """Cosine/sine strain profiles and their second derivatives, as (time, phase) scalar batches.

    A sin^4 envelope has zero value and first two derivatives at its endpoints.
    Both profiles and their derivatives are zero outside the packet.
    """
    time = np.asarray(time, dtype=float)
    s = np.clip(time / duration, 0.0, 1.0)
    a, b = np.pi / duration, 2 * np.pi * cycles / duration
    sine, cosine = np.sin(np.pi * s), np.cos(np.pi * s)
    envelope = amplitude * sine**4
    first = amplitude * 4 * a * sine**3 * cosine
    second = amplitude * 4 * a**2 * (3 * sine**2 * cosine**2 - sine**4)
    phase = b * (time - duration / 2)
    carrier = np.stack((np.cos(phase), np.sin(phase)), axis=-1)
    derivative = b * np.stack((-np.sin(phase), np.cos(phase)), axis=-1)
    profile = envelope[:, None] * carrier
    acceleration = (second - b**2 * envelope)[:, None] * carrier + 2 * first[:, None] * derivative
    inside = ((time > 0) & (time < duration))[:, None]
    return mv.scalar((profile * inside)[..., None]), mv.scalar((acceleration * inside)[..., None])


def polarized_waves(plus: Curvature, cross: Curvature, second: Scalar) -> Curvature:
    """Weak-wave curvature over (time, polarization): plus, cross, and circular.

    The weak-wave curvature is R_{0i0j} = -1/2 h''_ij, so the strain's second derivative
    weights the unit maps. The circular case adds the cross polarization a quarter cycle behind.
    """
    cosine, sine = second[:, 0], second[:, 1]                            # [n_time] Scalar
    plus_wave, cross_wave = plus * cosine, cross * sine                  # [n_time] Bivector <- Bivector
    return -0.5 * stack((plus_wave, cross_wave, plus_wave + cross_wave), axis=1)  # [n_time, 3]


def detector_ring(count: int) -> Vector:
    """Unit reference separations in the plane transverse to the wave."""
    angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
    coordinates = np.zeros((count, 4))
    coordinates[:, 1:3] = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    return mv.vector(coordinates)                                        # [count] Vector


def integrate_acceleration(time: np.ndarray, acceleration: Vector) -> Vector:
    """Numerical boundary: integrate twice along the first batch axis, starting at rest."""
    velocity = cumulative_simpson(acceleration.kernel, x=time, axis=0, initial=0)
    displacement = cumulative_simpson(velocity, x=time, axis=0, initial=0)
    return mv.vector(displacement)


def plane_patch(area: Bivector, edge: Vector) -> Vector:
    """Four corners of a simple area element, drawn from a unit spacelike edge in its plane."""
    other: Vector = edge.commutator(area) * 0.5
    edge = edge * 0.6
    return stack((-edge - other, -edge + other, edge + other, edge - other))  # [4] Vector
