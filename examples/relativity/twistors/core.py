"""Conformal spacetime models twistors as an eight-component left ideal carrying null light rays, the Robinson congruence, and a linked vacuum electromagnetic pulse.

Three commuting projectors halve the conformal algebra three times to leave an eight-component subspace that left multiplication preserves, so every multivector acts on twistors. A spacetime event embedded as a null conformal antivector sends to zero the twistors of the light rays passing through it, while acting with each event on a fixed non-null twistor selects one light ray per event to form the Robinson congruence.

Because an event is intrinsic, its action reverses a twistor's orientation: a light ray's twistor is intrinsic, an event maps it to an extrinsic twistor, and acting with an event on an extrinsic twistor selects an intrinsic ray twistor, with the conformal-invariant pairing running between the two orientations.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import Algebra, NumpyContext

# Spacetime combined with one positive and one negative conformal direction.
spacetime = Algebra("x+y+z+t-")
ga = spacetime * Algebra("u+v-")
mv = NumpyContext(ga).multivector
exact = ga.exact.multivector
Scalar = ga.gatype.scalar()                                                  # Scalar+
Plane = ga.gatype.vector()                                                   # Plane+
Bivector = ga.gatype.bivector()                                              # Bivector+
Even = ga.gatype.even()                                                      # Even+
Full = ga.gatype.full()                                                      # Full+
Intrinsic = ga.gatype(Full * ga.gatype.antiscalar())                         # Full-
# Conformal points, light rays, spacetime events, spatial displacements, and ray twistors are intrinsic.
Point = ga.gatype.antivector()                                               # Point-
Ray = ga.gatype.antibivector()                                               # Ray-
Event = Plane.derive.within(spacetime).derive.dual                           # Event-
Spatial = Plane.derive.within(Algebra("x+y+z+")).derive.dual                 # Spatial-
Field = Bivector.derive.within(spacetime)                                    # Field+
ExtrinsicTwistor = Full.derive.within(Algebra("y+z+u+"))                     # ExtrinsicTwistor+
Twistor = Intrinsic.derive.within(Algebra("y+z+u+"))                         # Twistor-
# Handedness of the electromagnetic pulse and of the Hopf fibres carrying it.
Helicity = ga.gatype.antipseudoscalar()                                      # Helicity-

# Three commuting projectors halve the state space three times, leaving eight real components.
projector = (1 + exact.x) * (1 + exact.yt) * (1 + exact.zv) / 8              # [] Full+
# Multiplying by a twistor type embeds twistors in the full algebra; scaling and casting reads them back out.
embedding = Twistor * projector                                              # [] Full- <- Twistor-
extrinsic_embedding = ExtrinsicTwistor * projector                           # [] Full+ <- ExtrinsicTwistor+
readout = (8 * Intrinsic).cast(Twistor)                                      # [] Twistor- <- Full-
extrinsic_readout = (8 * Full).cast(ExtrinsicTwistor)                        # [] ExtrinsicTwistor+ <- Full+
# Left multiplication in the full algebra induces three orientation-typed actions on twistors.
representations = readout(Full * embedding)                                  # [] Twistor- <- (Full+, Twistor-)
incidence = extrinsic_readout(Intrinsic * embedding)                         # [] ExtrinsicTwistor+ <- (Full-, Twistor-)
selection = readout(Intrinsic * extrinsic_embedding)                         # [] Twistor- <- (Full-, ExtrinsicTwistor+)
# The conformal pseudoscalar's representation squares to minus the identity and commutes with every even action.
pseudoscalar = exact.xyztuv                                                  # [] Pseudoscalar+
volume = representations(pseudoscalar)                                       # [] Twistor- <- Twistor-

# Conformal-invariant pairing between an intrinsic twistor and an extrinsic twistor across the unit antiscalar.
antiscalar = exact.scalar([1]).dual()                                        # [] Antiscalar-
pairing = 8 * exact.yz.scalar_product(embedding.reverse() * antiscalar * extrinsic_embedding)  # [] Scalar+ <- (Twistor-, ExtrinsicTwistor+)
# Pairing a twistor against each ray's incidence action and contracting recovers its conformal light ray.
ray_readout = (Ray * pairing(Twistor, incidence(Ray.reverse(), Twistor))).contract(1, 3)  # [] Ray- <- (Twistor-, Twistor-)
robinson_twistor = 1 + exact.y - exact.zu + exact.yzu                        # [] ExtrinsicTwistor+
# Origin, point at infinity, weight-reading horizon plane, and unit future time displacement.
origin = (mv.v.dual() - mv.u.dual()) / 2                                     # [] Point-
infinity = mv.v.dual() + mv.u.dual()                                         # [] Point-
horizon = -(mv.v + mv.u)                                                     # [] Plane+
time = -mv.t.dual()                                                          # [] Event-
# Trivector spanning time and the conformal plane; joining a spatial patch across it gives a spatial displacement.
conformal_time = mv.tuv                                                      # [] Trivector+
# Spatial turning patch Dual to y and unit helicity of the pulse and its fibres.
symmetry_patch = conformal_time.inverse() | mv.y.dual()                      # [] Antibivector-
helicity = mv(Helicity, [1])                                                 # [] Helicity-
spacetime_pseudoscalar = exact.xyzt                                          # [] Quadvector+


# --- math -----------------------------------------------------------------------------
def conformal_point(event: Event) -> Point:
    """Embedding `event` (`[...] Event-`) on the conformal null cone produces a weight-one point (`[...] Point-`)."""
    return origin + event + event.squared() * infinity / 2                   # [...] Point-


def ray_twistor(first: Point, second: Point) -> Twistor:
    """Singular value decomposition of the two-point product's representation extracts the shared light ray's twistor (`[...] Twistor-`)."""
    family = representations(first * second)                                 # [...] Twistor- <- Twistor-
    states, _, _ = family.svd()                                              # [..., 8] Twistor-
    return states[..., 0]                                                    # [...] Twistor-


def ray_events(ray: Ray, times: Scalar) -> Event:
    """Intersecting `ray` (`[...] Ray-`) with constant-time hyperplanes recovers the spacetime events at `times` (`[...] Event-`)."""
    crossing = (mv.t - horizon * times) ^ ray                                # [...] Point-
    return crossing.cast(Event) / (horizon & crossing)                       # [...] Event-


def ray_direction(twistor: Twistor) -> Event:
    """Meeting `twistor`'s reconstructed light ray with the horizon plane gives its future null direction at unit time (`[...] Event-`)."""
    ray = ray_readout(twistor, twistor)                                      # [...] Ray-
    tangent = (horizon ^ ray).cast(Event)                                    # [...] Event-
    return tangent / (mv.t & tangent)                                        # [...] Event-


def robinson_direction(event: Event) -> Event:
    """Acting with `conformal_point(event)` on `robinson_twistor` selects the Robinson congruence's future null direction (`[...] Event-`)."""
    selected = selection(conformal_point(event), robinson_twistor)           # [...] Twistor-
    return ray_direction(selected)                                           # [...] Event-


def hopfion_field(event: Event) -> Field:
    """Bateman's spinor construction evaluates the source-free null electromagnetic field (`[...] Field+`) carried by the Robinson congruence."""
    elapsed = mv.t & event                                                   # [...] Scalar+
    # Spatial offset relative to a light flash running along y, contracted into a spatial patch.
    position = conformal_time.inverse() | (event - elapsed * (time + mv.y.dual()))  # [...] Antibivector-
    # Multiplying the spatial patch by helicity gives the spatial bivector turning polarization and propagation together.
    wave_spinor = 1 + position * helicity                                    # [...] Even+
    polarization = (mv.t + mv.y) ^ mv.x                                      # [] Field+
    turned_polarization = wave_spinor >> polarization                        # [...] Field+
    # Spacetime's pseudoscalar rotates electric into magnetic planes while the inverse supplies amplitude decay and phase.
    envelope = (1 + event.squared() + 2 * spacetime_pseudoscalar * elapsed).inverse()  # [...] Even+
    return turned_polarization * envelope.squared() * envelope               # [...] Field+


def propagated_lines(initial_lines: Spatial, frame_times: np.ndarray) -> Iterator[Event]:
    """Yields the linked field lines (`[...] Event-`) transported along their straight Robinson light rays at `frame_times`."""
    directions = robinson_direction(initial_lines)                           # [...] Event-
    for elapsed in frame_times:
        yield initial_lines + directions * elapsed                           # [...] Event-


# --- plumbing -------------------------------------------------------------------------
def hopf_fibres(polars: np.ndarray, n_circles: int, n_samples: int) -> Spatial:
    """Stereographic projection of Hopf orbits on nested tori samples linked circles (`[n_tori, n_circles, n_samples] Spatial-`) tangent to the Robinson directions at time zero."""
    n_tori = len(polars)
    azimuths = np.linspace(0, 2 * np.pi, n_circles, endpoint=False)          # [n_circles]
    phases = np.linspace(-np.pi, np.pi, n_samples, endpoint=False)           # [n_samples]
    azimuth_rotors = (mv.zx * (-azimuths[None, :] / 2)).exp()                # [1, n_circles] Even+
    polar_rotors = (mv.xy * (polars[:, None] / 2)).exp()                     # [n_tori, 1] Even+
    starts = azimuth_rotors * polar_rotors                                   # [n_tori, n_circles] Even+
    phase_rotors = (symmetry_patch * helicity * phases).exp()                # [n_samples] Even+
    orbit = starts[..., None] * phase_rotors                                 # [n_tori, n_circles, n_samples] Even+
    return (orbit.restrict[2] * helicity / (1 + orbit.select[0])) ^ conformal_time  # [n_tori, n_circles, n_samples] Spatial-
