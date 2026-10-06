"""Light rays carried by twistors, and a twisting family of rays filling spacetime.

A conformal event acts on eight real spinor components. Its kernel contains the twistors of the
light rays through that event. Applying the event to a fixed non-null twistor chooses one ray;
the resulting family is the Robinson congruence.
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
Event = ga.gatype.from_blades("x y z t")
Spatial = ga.gatype.from_blades("x y z")
Field = ga.gatype.from_blades("xy xz yz xt yt zt")
Twistor = ga.gatype.from_blades("1 y z u yz yu zu yzu")

# The three commuting factors each halve the state space, leaving eight real components.
PROJECTOR = (1 + exact.x) * (1 + exact.yt) * (1 + exact.zv) / 8
STATE_EMBEDDING = Twistor * PROJECTOR                                        # [] Full <- Twistor
STATE_READOUT = (8 * Full).cast(Twistor)                                     # [] Twistor <- Full
REPRESENTATIONS = STATE_READOUT(Full * STATE_EMBEDDING)                       # [] Twistor <- (Full, Twistor)
# The conformal volume squares to minus one and commutes with every even multivector.
VOLUME = REPRESENTATIONS(exact.xyztuv)                                       # [] Twistor <- Twistor
ORIGIN = (mv.v - mv.u) / 2                                                    # [] Vector
INFINITY = mv.v + mv.u                                                       # [] Vector

# A pairing of two twistors that conformal rotors preserve; with the volume applied to the first,
# a second one, also preserved.
PAIRING = 8 * exact.yz.scalar_product(STATE_EMBEDDING.reverse() * exact.xyztuv * STATE_EMBEDDING)

# Pair a twistor with the action of an open plane, and raise that plane's input slot.
# On a null twistor twice the result is the null plane of its light ray.
RAY = (Bivector * PAIRING(Twistor, VOLUME(REPRESENTATIONS((1 * Bivector).reverse(), Twistor)))).contract(1, 3)
ROBINSON = 1 + exact.y - exact.zu + exact.yzu                                 # [] Twistor
SPATIAL_VOLUME = exact.xyz


# --- math -----------------------------------------------------------------------------
def point(event: Event) -> Vector:
    """A spacetime event embedded as a null conformal vector of weight one."""
    return ORIGIN + event + event.squared() * INFINITY / 2


def through(first: Vector, second: Vector) -> Twistor:
    """A twistor of the light ray through two distinct null-separated conformal events."""
    family = REPRESENTATIONS(first * second)                                  # [...] Twistor <- Twistor
    states, _, _ = family.svd()
    return states[..., 0]


def at_time(plane: Bivector, times: Scalar) -> Event:
    """The events where a conformal light ray meets the given constant-time slices."""
    crossing = (mv.t - INFINITY * times) | plane
    return crossing.cast(Event) / -(INFINITY | crossing)


def direction(twistor: Twistor) -> Event:
    """The future null direction of a ray, with its time component normalized to one."""
    plane = RAY(twistor, twistor)                                             # [...] Bivector
    tangent = (INFINITY | plane).cast(Event)                                  # [...] Event
    return tangent / -(mv.t | tangent)


def robinson(event: Event) -> Event:
    """The light ray selected at each event by the fixed non-null twistor ROBINSON."""
    ray_twistor = REPRESENTATIONS(point(event), ROBINSON)                      # [...] Twistor
    return direction(ray_twistor)


def hopfion(event: Event) -> Field:
    """A source-free null electromagnetic field carried by the Robinson congruence."""
    position = -(event ^ mv.t) * mv.t
    time = -(event | mv.t)
    spinor = 1 + SPATIAL_VOLUME * (position + time * mv.y)
    # The spinor turns the polarization and its propagation direction together.
    polarization = (mv.t + mv.y) ^ mv.x
    # The spacetime volume rotates electric into magnetic polarization. Its inverse
    # supplies both the decaying amplitude and the local polarization angle.
    weight = (1 + event.squared() + 2 * mv.xyzt * (event | mv.t)).inverse()
    return (spinor >> polarization) * weight.squared() * weight


# --- plumbing -------------------------------------------------------------------------
def fibres(polars: np.ndarray, per_circle: int, fibre_samples: int) -> Spatial:
    """Linked circles tangent to the Robinson directions at time zero, sampled on nested tori."""
    azimuths = np.linspace(0, 2 * np.pi, per_circle, endpoint=False)
    phases = np.linspace(-np.pi, np.pi, fibre_samples + 1)
    starts = ((mv.zx * (-azimuths[None, :] / 2)).exp()
              * (mv.xy * (polars[:, None] / 2)).exp()).reshape((-1,))          # [curves] Even
    orbit = starts[:, None] * (SPATIAL_VOLUME * mv.y * phases).exp()          # [curves, fibre_samples + 1] Even
    return -SPATIAL_VOLUME * orbit.select[2] / (1 + orbit.select[0])
