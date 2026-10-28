"""Twistors and linked light: light rays carried by twistors, and a twisting family of rays filling spacetime.

A twistor is modelled as eight real components of the conformal algebra that left multiplication preserves,
so every multivector acts on twistors. An event, a null antivector, sends to zero the twistors of the light
rays through it; applying the event to a fixed non-null twistor picks one ray, and the resulting family is
the Robinson congruence.

An event is intrinsic, and its action turns a twistor's orientation: twistors come in two orientations that
events exchange, and every action is typed by the orientations it maps between. A ray's twistor is intrinsic,
the twistor an event selects it from extrinsic, and the pairing runs between the two.
"""

from __future__ import annotations

import numpy as np

from numga import Algebra, NumpyContext

ga = Algebra("x+y+z+t-u+v-")
mv = NumpyContext(ga).multivector
exact = ga.exact.multivector
Scalar = ga.gatype.scalar()
Vector = ga.gatype.vector()
Bivector = ga.gatype.bivector()
Full = ga.gatype.full()
Intrinsic = Full.derive.flipped_orientation
# Points, light rays and the twistors of rays are intrinsic, like all geometry.
Point = ga.gatype.antivector()                                               # Antivector-
Ray = ga.gatype.antibivector()                                               # Antibivector-
Event = Point.derive.containing("u v")                                       # Antivector-
Spatial = Point.derive.containing("t u v")                                   # Antivector-
Faraday = Bivector.derive.within("x y z t")                                  # Bivector+
ExtrinsicTwistor = Full.derive.within("y z u")                               # Full+
Twistor = ExtrinsicTwistor.derive.flipped_orientation                        # Full-
# The handedness of the pulse and of the fibres it is carried along.
Helicity = Scalar.derive.flipped_orientation                                 # Scalar-

# The three commuting factors each halve the state space, leaving eight real components.
projector = (1 + exact.x) * (1 + exact.yt) * (1 + exact.zv) / 8
# Multiplied by a twistor type, the projector embeds those twistors in the full algebra; scaled back, the full
# algebra reads them out.
embedding = Twistor * projector                                              # [] Full- <- Twistor-
extrinsic_embedding = ExtrinsicTwistor * projector                           # [] Full+ <- ExtrinsicTwistor+
readout = (8 * Intrinsic).cast(Twistor)                                      # [] Twistor- <- Full-
extrinsic_readout = (8 * Full).cast(ExtrinsicTwistor)                        # [] ExtrinsicTwistor+ <- Full+
# Embedding, multiplying on the left and reading out gives every multivector's action on twistors. An intrinsic
# multivector, such as an event, turns the twistor's orientation, so the actions are typed by the orientations
# they map between: a ray's twistor is intrinsic, and an event's action on an extrinsic twistor selects one.
representations = readout(Full * embedding)                                  # [] Twistor- <- (Full+, Twistor-)
incidence = extrinsic_readout(Intrinsic * embedding)                         # [] ExtrinsicTwistor+ <- (Full-, Twistor-)
selection = readout(Intrinsic * extrinsic_embedding)                         # [] Twistor- <- (Full-, ExtrinsicTwistor+)
# The conformal volume squares to minus one and commutes with every even multivector.
volume = representations(exact.xyztuv)                                       # [] Twistor- <- Twistor-

# A pairing of two twistors that conformal rotors preserve, across the unit intrinsic volume.
pairing = 8 * exact.yz.scalar_product(embedding.reverse() * exact.antiscalar([1]) * extrinsic_embedding)   # [] Scalar+ <- (Twistor-, ExtrinsicTwistor+)
# A twistor paired with each ray's action is a form on rays; contracting turns it into the ray itself.
# On a null twistor the result is its light ray, the join of its events.
ray_readout = (Ray * pairing(Twistor, incidence(Ray.reverse(), Twistor))).contract(1, 3)   # [] Ray- <- (Twistor-, Twistor-)
robinson_twistor = 1 + exact.y - exact.zu + exact.yzu                        # [] ExtrinsicTwistor+
# The origin and the point at infinity; the plane at infinity, which reads a point's weight; one unit along time.
origin = (mv.v.dual() - mv.u.dual()) / 2                                     # [] Point-
infinity = mv.v.dual() + mv.u.dual()                                         # [] Point-
horizon = -(mv.v + mv.u)                                                     # [] Vector+
time = -mv.t.dual()                                                          # [] Event-
# Time and the conformal pair: a patch of space joined across them is a direction in space.
across = mv.tuv
# y as a patch of space, the plane the pulse and its fibres turn in.
symmetry = across.inverse() | mv.y.dual()                                    # [] Bivector-
helicity = mv(Helicity, [1.0])                                               # [] Helicity-
spacetime_volume = exact.xyzt


# --- math -----------------------------------------------------------------------------
def point(event: Event) -> Point:
    """A spacetime event embedded as a null conformal antivector of weight one."""
    return origin + event + event.squared() * infinity / 2


def through(first: Point, second: Point) -> Twistor:
    """A twistor of the light ray through two distinct null-separated conformal events."""
    family = representations(first * second)                                 # [...] Twistor- <- Twistor-
    states, _, _ = family.svd()
    return states[..., 0]


def at_time(ray: Ray, times: Scalar) -> Event:
    """The events where a conformal light ray meets the given constant-time slices."""
    crossing = (mv.t - horizon * times) ^ ray
    return crossing.cast(Event) / (horizon & crossing)


def direction(twistor: Twistor) -> Event:
    """The future null direction of a ray, with its time component normalized to one."""
    ray = ray_readout(twistor, twistor)                                              # [...] Ray-
    # The ray meets the plane at infinity in its direction.
    tangent = (horizon ^ ray).cast(Event)                                    # [...] Event-
    return tangent / (mv.t & tangent)


def robinson(event: Event) -> Event:
    """The light ray selected at each event by the fixed non-null twistor robinson_twistor."""
    ray_twistor = selection(point(event), robinson_twistor)                  # [...] Twistor-
    return direction(ray_twistor)


def hopfion(event: Event) -> Faraday:
    """A source-free null electromagnetic field carried by the Robinson congruence."""
    elapsed = mv.t & event
    # The event's place in space relative to a flash of light running along y, as a patch of space: its
    # extent across time and the conformal pair undone.
    position = across.inverse() | (event - elapsed * (time + mv.y.dual()))      # [...] Bivector-
    # The spinor turns the polarization and its propagation direction together; the helicity turns the
    # patch into the plane it turns in.
    spinor = 1 + position * helicity
    polarization = (mv.t + mv.y) ^ mv.x
    # The spacetime volume rotates electric into magnetic polarization. Its inverse
    # supplies both the decaying amplitude and the local polarization angle.
    weight = (1 + event.squared() + 2 * spacetime_volume * elapsed).inverse()
    return (spinor >> polarization) * weight.squared() * weight


# --- plumbing -------------------------------------------------------------------------
def fibres(polars: np.ndarray, n_per_circle: int, n_fibre_samples: int) -> Spatial:
    """Linked circles tangent to the Robinson directions at time zero, sampled on nested tori."""
    azimuths = np.linspace(0, 2 * np.pi, n_per_circle, endpoint=False)
    phases = np.linspace(-np.pi, np.pi, n_fibre_samples + 1)
    starts = ((mv.zx * (-azimuths[None, :] / 2)).exp()
              * (mv.xy * (polars[:, None] / 2)).exp())                       # [polars, per circle] Even+
    orbit = starts[..., None] * (symmetry * helicity * phases).exp()         # [polars, per circle, fibre samples + 1] Even+
    # Stereographic projection: the turn's plane, by the helicity a patch of space, joined across time.
    return (orbit.restrict[2] * helicity / (1 + orbit.select[0])) ^ across
