"""Satellite navigation recovers a reception event and its uncertainty from transmission events.

Projective spacetime algebra describes events as intrinsic antivectors and readouts as
extrinsic vectors. A translated Lorentz correlation gives each signal's light cone.
Timing sensitivities contribute positive dyads to an information map; its inverse
describes uncertainty in both location and reception time.

Lengths are in Earth radii and times in Earth-radius light-travel times, so light speed is one.
"""

from __future__ import annotations

from numga import Algebra, NumpyContext, stack, where

ga = Algebra("x+y+z+t-w0")
mv = NumpyContext(ga).multivector
Event = ga.gatype.antivector()
Slice = ga.gatype.vector()
Scalar = ga.gatype.scalar()
Readout = ga.gatype.from_blades("x y z t")
Step = Readout.dual().gatype.derive.output
Cone = ga.gatype((Slice, Event))
Information = ga.gatype((Readout, Step))
Covariance = ga.gatype((Step, Readout))
Orbit = ga.gatype.bivector()

origin = mv.w.dual()
across = mv.x.dual()
time = -mv.t.dual()
# The Lorentz form, reading a displacement's interval: the homogeneous direction squares to zero, so the
# complement carries it.
spacetime_metric = Event.dual_inverse().cast(Readout)


def orbit_positions(planes: Orbit, phases: Scalar, radius: float,
                    angular_speed: float, times: Scalar) -> Event:
    """Circular satellite trajectories, evaluated at coordinate times in the local inertial frame."""
    rotation = (-planes * ((phases + angular_speed * times) / 2)).exp()
    return origin + (rotation >> across) * radius + time * times


def emitted_events(receiver: Event, planes: Orbit, phases: Scalar, radius: float,
                   angular_speed: float, n_steps: int) -> Event:
    """Retarded transmission events on the orbits whose signals reach the receiver together."""
    reception_time = mv.t & receiver
    emission_time = reception_time
    for _ in range(n_steps):
        satellites = orbit_positions(planes, phases, radius, angular_speed, emission_time)
        separation = receiver - satellites
        # The separation's length in the receiver's space: its interval, with the time it spans added back.
        distance = ((separation & spacetime_metric(separation)) + (mv.t & separation).squared()).square_root()
        emission_time = reception_time - distance
    return orbit_positions(planes, phases, radius, angular_speed, emission_time)


def light_cones(emitted: Event) -> Cone:
    """Homogeneous quadratic correlations whose vertices are the transmission events."""
    centred = Event - (emitted - origin) * (mv.w & Event)
    # Carry the interval form through the translation on both its arguments.
    return centred.adjugate()(spacetime_metric(centred))


def intersect_cones(cones: Cone) -> Event:
    """The two intersections of four light cones, in a finite coordinate-time chart."""
    # Two cones of one Lorentz form differ by a pair of planes, the plane at infinity and their radical plane: the
    # difference sends an event to the radical plane times its weight plus the plane at infinity times the radical
    # plane's reading of it. Read at the origin, the second term is half the difference's own reading of the origin.
    difference = cones[1:] - cones[0]
    at_origin = difference(origin)
    radicals = at_origin - mv.w * ((origin & at_origin) / 2)
    axis = radicals[0] ^ radicals[1] ^ radicals[2]

    # Two time slices locate points on their common line. Normalizing by the
    # weight also gives these intersections their intrinsic event orientation.
    base = axis ^ mv.t
    base = base / (mv.w & base)
    tip = axis ^ (mv.t - mv.w)
    tip = tip / (mv.w & tip)
    direction = tip - base

    # The remaining cone cuts the line in two points.
    quadratic = direction & cones[0](direction)
    linear = base & cones[0](direction)
    constant = base & cones[0](base)
    root = (linear.squared() - quadratic * constant).square_root()
    parameters = stack((-linear - root, -linear + root)) / quadratic
    return base + direction * parameters


def reception(candidates: Event, emitted: Event) -> Event:
    """Of the cones' two intersections, the one every signal reaches after it left."""
    after = (mv.t & (candidates[:, None] - emitted)) > 0                  # [candidates, signals]
    return where(after[0].all(), candidates[0], candidates[1])


def intervals(cones: Cone, emitted: Event, receiver: Event) -> tuple[Scalar, Readout, Scalar]:
    """Each signal's interval from its transmission to a reception event, `receiver & cone(receiver)`, which vanishes
    on the light cone; its readout of a small displacement of the reception, twice the cone's tangent plane there;
    and how far a timestamp's error moves it, twice the light-travel time."""
    tangent = cones(receiver[..., None])
    # The tangent plane's part across the plane at infinity reads no displacement.
    readout = 2 * tangent.cast(Readout)
    lever = 2 * (mv.t & (receiver[..., None] - emitted))
    return receiver[..., None] & tangent, readout, lever


def locate(emitted: Event, initial: Event, sigma: float,
           n_steps: int) -> tuple[Event, Information, Covariance]:
    """Fit the reception event to noisy timestamps, each to its light cone, and recover the local uncertainty map."""
    cones = light_cones(emitted)
    receiver = initial
    for _ in range(n_steps):
        interval, readout, lever = intervals(cones, emitted, receiver)
        # Each signal's readout contributes a positive cost for the displacements it can see, weighted by how far
        # its timestamp's uncertainty moves its interval.
        weight = 1 / (lever * sigma) ** 2
        information = (weight * readout * (readout & Step)).sum(axis=-1)
        mismatch = (weight * readout * interval).sum(axis=-1)
        receiver = receiver - information.solve(mismatch)

    _, readout, lever = intervals(cones, emitted, receiver)
    information = (readout * (readout & Step) / (lever * sigma) ** 2).sum(axis=-1)
    covariance = information.inverse()
    return receiver, information, covariance
