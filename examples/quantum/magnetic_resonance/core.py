"""Magnetic resonance: a spin in a magnetic field, driven by a radio-frequency field and relaxing, in the algebra of
spacetime.

The state of a spin, or of an ensemble of spins, is modelled as half the antiscalar, its trace, plus the spin. The
spin is intrinsic, as every angular momentum: the patch of the plane it turns in, that plane's dual, so that a spin
turning in the magnet's plane xy is `mv.xy.dual()`, in spacetime the patch zt through the lab's time. Its length is
one for a single pure spin and less for a mixture. The fields that turn it are extrinsic planes of turning, the
magnet's plane weighted by the detuning and the drive's plane by the drive strength, in the frame that turns with
the drive, and they turn the state by their commutator, `State.commutator(field)`.

Relaxation turns the part of the spin along the magnet's patch back toward that patch at the rate `1 / t1`, and lets
every other part the lab reads as spin decay at `1 / t2`. Which parts of a state are read as spin is the lab's to
say: those its reflection, `observer >> State`, leaves unchanged. Where the generator vanishes the state is steady;
since the generator keeps the trace and keeps the parts the lab cannot read, that state is a single solve with both
pinned.

The evolution over a short time is a map as well, the exponential of the generator to fourth order. Composed with
itself it spans two steps, then four; composed with the pulses, rotors in the drive's plane whose sandwiches are
maps too, it gives a whole pulse sequence as one map.

In the vector notation of the Bloch equations the spin reads as the Bloch vector, the field as the vector of
angular frequencies about which the Bloch vector turns by their cross product, and the relaxation as the decay of
the Bloch vector's transverse components at the rate one over T2 and of its component along the field toward one at
the rate one over T1.
"""

from __future__ import annotations

from collections.abc import Generator, Iterator

import numpy as np

from numga import NumpyContext
from numga.algebras import STA as ga

mv = NumpyContext(ga).multivector
Bivector = ga.gatype.bivector()                                     # Bivector+
# A spin: the patch of the plane it turns in, intrinsic.
Spin = Bivector.derive.flipped_orientation                          # Bivector-
# A spin's state: half the antiscalar, its trace, plus the spin.
State = ga.gatype.from_grades([2, 4]).derive.flipped_orientation    # Bivector- + Antiscalar-
Rates = ga.gatype((State, State))                                   # State- <- State-
# What becomes of each state over a stretch of time.
Evolution = ga.gatype((State, State))                               # State- <- State-
Rotor = ga.gatype.rotor()                                           # Rotor+
one = mv.scalar([1.0])                                              # [] Scalar+
antiscalar = mv.antiscalar([1.0])                                   # [] Antiscalar-
# The lab at rest: its slice of space, the plane the magnet's field turns spins in, and the plane the drive turns
# them in.
observer = mv.t                                                     # [] Vector+
magnet_plane = mv.xy                                                # [] Bivector+
drive_plane = mv.yz                                                 # [] Bivector+
# The spin part of a state: the state less its trace, which pairing with one reads.
planes = State - antiscalar * (one & State)                         # [] State- <- State-
# The parts the lab reads as spin, the patches through its time: those its reflection leaves unchanged, the
# observer's sandwich over its square being that reflection; and the parts it cannot read.
readable = (planes + (observer >> planes) / observer.squared()) / 2           # [] State- <- State-
unreadable = planes - readable                                                # [] State- <- State-
# The readable part along the magnet's patch: the part a half turn in the magnet's plane leaves unchanged.
along_magnet = (readable + (magnet_plane >> readable) / magnet_plane.scalar_norm_squared()) / 2   # [] State- <- State-
# What relaxes at the rate `1 / t1`: the part along the magnet's patch less the patch itself, scaled by the trace,
# where it relaxes to; and what decays at the rate `1 / t2`, every other readable part.
longitudinal = along_magnet - magnet_plane.dual() * (one & State)            # [] State- <- State-
transverse = readable - along_magnet                                          # [] State- <- State-


# --- math -----------------------------------------------------------------------------
def state(spin: Spin) -> State:
    """The state of the given spin."""
    return 0.5 * (antiscalar + spin)                                            # [...] State-


def generator(detuning: np.ndarray, drive: np.ndarray, t1: float, t2: float) -> Rates:
    """The rate of change of a state, in the frame that turns with the drive.

    The field turns the state in the magnet's plane at the detuning and in the drive's plane at the drive strength;
    the part along the magnet's patch relaxes toward equilibrium at the rate `1 / t1`, and the transverse part
    decays at the rate `1 / t2`.
    """
    field = magnet_plane * detuning + drive_plane * drive                       # [...] Bivector+
    # The field turns the state as the rotor `(field * (-dt / 2)).exp()` does, as the pulses turn it.
    turning = State.commutator(field)                                           # [...] State- <- State-
    relaxing = -longitudinal / t1 - transverse / t2                             # [] State- <- State-
    return turning + relaxing                                                   # [...] State- <- State-


def steady(rates: Rates) -> State:
    """The state that does not change.

    The generator keeps the trace and keeps the parts the lab cannot read among themselves, so it is singular.
    Adding the trace as a dyad on the antiscalar, and the unreadable parts as they are, pins both: the sum sends a
    state to its trace and unreadable parts plus its change, and solving that against half the antiscalar gives the
    state with trace one half, nothing unreadable, and no change.
    """
    return (rates + antiscalar * (one & State) + unreadable).solve(antiscalar * 0.5)   # [...] State-


def evolution(rates: Rates, dt: float) -> Evolution:
    """What becomes of each state over a time dt: the exponential of the generator times dt, to
    fourth order in dt, the powers of the generator by composition."""
    exponent = rates * dt                                                       # [...] State- <- State-
    term = total = State
    for order in range(1, 5):
        # The next term of the exponential series: the last one composed with the exponent, over the order.
        term = exponent(term) / order                                           # [...] State- <- State-
        total = total + term
    return total                                                                # [...] State- <- State-


def evolve(each_step: Evolution, rho: State, n_steps: int) -> Generator[State, None, State]:
    """The state before each of the given number of steps; returns the state after the last."""
    for _ in range(n_steps):
        yield rho                                                               # [...] State-
        rho = each_step(rho)
    return rho


def pulse(angle: float) -> Rotor:
    """A short strong pulse, as a rotor: it turns the spin in the drive's plane by the angle, in the same sense as
    the drive. It acts on a state by its sandwich."""
    return (drive_plane * (-angle / 2)).exp()                                   # [] Rotor+


def echo(each_step: Evolution, rho: State, n_before: int, n_after: int) -> Iterator[State]:
    """The states through a spin echo: tipped a quarter turn, left for the given number of steps, turned
    half a turn, and left again."""
    rho = yield from evolve(each_step, pulse(np.pi / 2) >> rho, n_before)
    yield from evolve(each_step, pulse(np.pi) >> rho, n_after)


def doublings(span: Evolution, n_doublings: int) -> Iterator[Evolution]:
    """The evolution over its own span, then over twice and four times that span, and so on: each
    composed with itself for the next."""
    for _ in range(n_doublings):
        yield span
        span = span(span)
