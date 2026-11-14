"""Magnetic resonance models a spin in a magnetic field, driven by a radio-frequency field and
relaxing, in the algebra of spacetime.

A spin state, or the state of an ensemble of spins, is modelled as half the antiscalar trace plus
the spin. Every angular momentum is intrinsic, represented by the patch of the plane it turns in,
that plane's dual, so a spin turning in the magnet's plane `xy` is `magnet_plane.dual()`, in
spacetime the patch `zt` through the lab's time. Its length is one for a pure spin and less for a
mixture. The fields that turn the spin are extrinsic planes of turning, weighted by the detuning in
the magnet's plane and by the drive strength in the drive's plane inside the rotating frame, and
they turn the state by their commutator `State.commutator(field)`.

Relaxation pulls the part of the spin along the magnet's patch back toward equilibrium at rate
`1 / t1` and decays every other readable spin component at rate `1 / t2`. The observer's reflection
`observer >> State` picks out the parts of a state that the lab reads as spin by leaving them
unchanged. Because the generator preserves both the trace and the unreadable parts, pinning both
turns the steady state where the generator vanishes into a single linear solve.

Short-time evolution is the fourth-order exponential series of the generator, a linear map on
states. Composing that map with itself doubles the waiting span at each step, and composing those
spans with rotor sandwiches for the radio-frequency pulses builds a whole pulse sequence as one map
from state to state.
"""

from __future__ import annotations

from collections.abc import Generator, Iterator

import numpy as np

from numga import NumpyContext
from numga.algebras import STA as ga

mv = NumpyContext(ga).multivector
Vector = ga.gatype.vector()                                         # Vector+
Bivector = ga.gatype.bivector()                                     # Bivector+
# A spin: the patch of the plane it turns in, intrinsic.
Spin = ga.gatype.antibivector()                                     # Antibivector-
# A spin's state: half the antiscalar, its trace, plus the spin.
State = ga.gatype(Spin + ga.gatype.antiscalar())                    # Antibivector- + Antiscalar-
Rates = ga.gatype((State, State))                                   # State- <- State-
# What becomes of each state over a stretch of time.
Evolution = ga.gatype((State, State))                               # State- <- State-
Rotor = ga.gatype.rotor()                                           # Rotor+
one = mv.scalar([1])                                                # [] Scalar+
antiscalar = one.dual()                                             # [] Antiscalar-
# The spin part of a state: the state less its trace, which pairing with one reads.
planes = State - antiscalar * (one & State)                         # [] State- <- State-
# The lab at rest: its slice of space, the plane the magnet's field turns spins in, and the plane the drive turns
# them in.
observer = mv.t                                                     # [] Vector+
magnet_plane = mv.xy                                                # [] Bivector+
drive_plane = mv.yz                                                 # [] Bivector+


# --- math -----------------------------------------------------------------------------
def spin_state(spin: Spin) -> State:
    """Forms the spin state `0.5 * (antiscalar + spin)` (`[...] State-`) of `spin`."""
    return 0.5 * (antiscalar + spin)                                            # [...] State-


def rate_map(detuning: np.ndarray, drive: np.ndarray, magnet_plane: Bivector, drive_plane: Bivector,
             observer: Vector, t1: float, t2: float) -> Rates:
    """Combines rotating-frame precession in `magnet_plane` and `drive_plane` with longitudinal `1 / t1`
    and transverse `1 / t2` relaxation seen by `observer` into one generator (`[...] State- <- State-`)."""
    field = magnet_plane * detuning + drive_plane * drive                       # [...] Bivector+
    # Precession under `field` acts as the infinitesimal rotor `(field * (-dt / 2)).exp()`.
    turning = State.commutator(field)                                           # [...] State- <- State-
    # Reflection across `observer` keeps the timelike patches that `observer` reads as spin.
    readable = (planes + (observer >> planes) / observer.squared()) / 2         # [] State- <- State-
    # A half turn in `magnet_plane` keeps the readable component along the magnet's patch.
    along_magnet = (readable + (magnet_plane >> readable) / magnet_plane.scalar_norm_squared()) / 2   # [] State- <- State-
    # Longitudinal spin relaxes toward `magnet_plane.dual()` scaled by the trace; transverse spin decays to zero.
    relaxing = -(along_magnet - magnet_plane.dual() * (one & State)) / t1 - (readable - along_magnet) / t2   # [] State- <- State-
    return turning + relaxing                                                   # [...] State- <- State-


def steady_state(rates: Rates, observer: Vector) -> State:
    """Solves for the stationary state (`[...] State-`) of `rates` by pinning the trace to one half and
    the spacelike planes unreadable to `observer` to zero."""
    unreadable = (planes - (observer >> planes) / observer.squared()) / 2       # [] State- <- State-
    return (rates + antiscalar * (one & State) + unreadable).solve(antiscalar * 0.5)   # [...] State-


def step_evolution(rates: Rates, dt: float) -> Evolution:
    """Approximates the short-time propagator over `dt` as the fourth-order composition series of
    `rates * dt` (`[...] State- <- State-`)."""
    exponent = rates * dt                                                       # [...] State- <- State-
    term = total = State
    for order in range(1, 5):
        # Each power in the exponential series composes the previous map with `exponent`.
        term = exponent(term) / order                                           # [...] State- <- State-
        total = total + term
    return total                                                                # [...] State- <- State-


def trajectory(step_map: Evolution, state: State, n_steps: int) -> Generator[State, None, State]:
    """Yields `state` (`[...] State-`) before each of `n_steps` applications of `step_map` and returns
    the state after the last step."""
    for _ in range(n_steps):
        yield state                                                             # [...] State-
        state = step_map(state)
    return state


def drive_pulse(drive_plane: Bivector, angle: float) -> Rotor:
    """Forms the rotor (`[] Rotor+`) of a resonant pulse turning spins in `drive_plane` by `angle`."""
    return (drive_plane * (-angle / 2)).exp()                                   # [] Rotor+


def echo_trajectory(step_map: Evolution, drive_plane: Bivector, state: State, n_before: int,
                    n_after: int) -> Iterator[State]:
    """Yields states (`[...] State-`) through a Hahn spin echo: a quarter-turn pulse in `drive_plane`,
    `n_before` free steps, a half-turn refocusing pulse, and `n_after` free steps."""
    tipped = drive_pulse(drive_plane, np.pi / 2) >> state                       # [...] State-
    state = yield from trajectory(step_map, tipped, n_before)
    turned = drive_pulse(drive_plane, np.pi) >> state                           # [...] State-
    yield from trajectory(step_map, turned, n_after)


def doubled_spans(span: Evolution, n_doublings: int) -> Iterator[Evolution]:
    """Yields `n_doublings` evolution maps (`[...] State- <- State-`), composing each span with itself
    to double the elapsed time."""
    for _ in range(n_doublings):
        yield span                                                              # [...] State- <- State-
        span = span(span)
