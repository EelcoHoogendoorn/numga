"""Curvature of a gravitational wave operates as a linear map from spacetime areas to bivector planes.

A vacuum plane wave builds its curvature from two dyads of the null planes that join its wave vector with a
transverse pair of spatial slices. Every output plane contains the wave vector, and opposite weights on the
two dyads cancel the Ricci trace. Binding an observer's velocity twice produces the tidal map from spatial
separations to tidal forces, and the same wave acts directly on separations as a transverse strain map whose
second derivative along the wave recovers the curvature.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext, concatenate
from numga.algebras import STA

mv = NumpyContext(STA).multivector
Scalar = STA.gatype.scalar()                        # Scalar+
# Wave vectors and the planes fields and curvature turn in are extrinsic vectors and bivectors.
Vector = STA.gatype.vector()                        # Vector+
Bivector = STA.gatype.bivector()                    # Bivector+
Trivector = STA.gatype.trivector()                  # Trivector+
# Events, velocities, separations, and the areas their joins span are intrinsic antivectors and antibivectors.
Antivector = STA.gatype.antivector()                # Antivector-
Antibivector = STA.gatype.antibivector()            # Antibivector-
Rotor = STA.gatype.rotor()                          # Rotor+

Curvature = STA.gatype((Bivector, Antibivector))    # Bivector+ <- Antibivector-
Tidal = STA.gatype((Vector, Antivector))            # Vector+ <- Antivector-
Strain = STA.gatype((Vector, Antivector))           # Vector+ <- Antivector-
# Curvature read on an ordered pair of plane edges.
CurvatureForm = STA.gatype((Bivector, Antivector, Antivector))   # Bivector+ <- (Antivector-, Antivector-)

# One unit along laboratory time, perpendicular to the spatial slice t.
time_direction = -mv.t.dual()                       # [] Antivector-
# Null propagation along +z and its dual wave vector.
travel = time_direction + mv.z.dual()               # [] Antivector-
wave_vector = -travel.dual()                        # [] Vector+
# Eighth-turn rotor in the transverse xy plane.
eighth_turn = (mv.xy * (-np.pi / 8)).exp()          # [] Rotor+


# --- math -----------------------------------------------------------------------------
def plane_wave_curvature(wave: Vector, first_slice: Vector, second_slice: Vector) -> Curvature:
    """Vacuum plane-wave curvature maps spacetime areas onto the null planes of `wave` with `first_slice` and `second_slice`."""
    first_plane = wave ^ first_slice                # [] Bivector+
    second_plane = wave ^ second_slice              # [] Bivector+
    return first_plane * (first_plane & Antibivector) - second_plane * (second_plane & Antibivector)   # [] Bivector+ <- Antibivector-


def polarization_curvatures() -> tuple[Curvature, Curvature]:
    """Unit plus and cross curvature maps propagate along +z and differ by an eighth-turn in the transverse plane."""
    plus = plane_wave_curvature(wave_vector, mv.x, mv.y)                # [] Bivector+ <- Antibivector-
    cross = eighth_turn >> plus(eighth_turn << Antibivector)            # [] Bivector+ <- Antibivector-
    return plus, cross


def tidal_map(curvature: Curvature, observer: Antivector) -> Tidal:
    """Double pairing of `curvature` with `observer` maps spatial separations to tidal force covectors."""
    return curvature(observer & Antivector) & observer                  # [...] Vector+ <- Antivector-


def boosted_observers(rapidities: np.ndarray, direction: Antivector) -> Antivector:
    """Lorentz boosts along `direction` carry the laboratory rest velocity to one four-velocity per rapidity."""
    # The join of the spatial direction with time spans the boost ribbon, whose dual generates the boost.
    boosts = ((direction & time_direction).dual() * (rapidities / 2)).exp()   # [...] Rotor+
    return boosts >> time_direction                                     # [...] Antivector-


def cumulative_integral(step: float, values: Antivector) -> Antivector:
    """Symmetric cubic quadrature accumulates the time integral of `values` along the first batch axis from zero."""
    zero = 0 * values[:1]                                               # [...] Antivector-
    if values.shape[0] < 4:
        steps = (step / 2) * (values[:-1] + values[1:])                 # [...] Antivector-
        return concatenate((zero, steps)).cumsum(axis=0)                # [...] Antivector-
    first_step = (step / 24) * (9 * values[0:1] + 19 * values[1:2] - 5 * values[2:3] + values[3:4])   # [...] Antivector-
    interior_steps = (step / 24) * (-values[:-3] + 13 * values[1:-2] + 13 * values[2:-1] - values[3:])   # [...] Antivector-
    last_step = (step / 24) * (values[-4:-3] - 5 * values[-3:-2] + 19 * values[-2:-1] + 9 * values[-1:])   # [...] Antivector-
    return concatenate((zero, first_step, interior_steps, last_step)).cumsum(axis=0)   # [...] Antivector-


def integrated_displacements(time: np.ndarray, acceleration: Antivector) -> Antivector:
    """Repeated cumulative quadrature integrates `acceleration` twice from rest along the uniform `time` axis."""
    if len(time) < 2:
        return 0 * acceleration                                         # [...] Antivector-
    step = float(time[1] - time[0])
    velocities = cumulative_integral(step, acceleration)                # [...] Antivector-
    return cumulative_integral(step, velocities)                        # [...] Antivector-


def strain_patterns() -> tuple[Strain, Strain]:
    """Unit plus and cross transverse strain maps send spatial separations to the covectors reading their stretch."""
    plus_strain = mv.x * (mv.x & Antivector) - mv.y * (mv.y & Antivector)   # [] Vector+ <- Antivector-
    cross_strain = eighth_turn >> plus_strain(eighth_turn << Antivector)    # [] Vector+ <- Antivector-
    return plus_strain, cross_strain


def curvature_of_strain(wave: Vector, second_strain: Strain) -> CurvatureForm:
    """Wedging `second_strain` with `wave` builds the linearized curvature map on ordered pairs of directions."""
    bent_slice = second_strain ^ wave                                   # [...] Bivector+ <- Antivector-
    return (wave & Antivector) * bent_slice - bent_slice * (wave & Antivector)   # [...] Bivector+ <- (Antivector-, Antivector-)


# --- plumbing -------------------------------------------------------------------------
def wave_packet(
    time: np.ndarray, duration: float, n_cycles: int, amplitude: float,
) -> tuple[np.ndarray, np.ndarray]:
    """A complex Gaussian carrier evaluates the plus and cross strain profiles (`[..., 2]`) and their second derivatives."""
    centered_time = time - duration / 2
    envelope_width = duration / 12
    carrier_rate = 2 * np.pi * n_cycles / duration
    # Both quadrature profiles and their second derivatives follow from one complex Gaussian carrier.
    complex_profile = amplitude * np.exp(-centered_time**2 / (2 * envelope_width**2) + 1j * carrier_rate * centered_time)
    complex_second = ((-centered_time / envelope_width**2 + 1j * carrier_rate) ** 2 - 1 / envelope_width**2) * complex_profile
    strains = np.stack([complex_profile.real, complex_profile.imag], axis=-1)
    second_derivatives = np.stack([complex_second.real, complex_second.imag], axis=-1)
    return strains, second_derivatives
