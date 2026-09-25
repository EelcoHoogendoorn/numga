"""Magnetic resonance: a spin in a magnetic field, driven by a radio-frequency field and relaxing,
in the Pauli algebra of three-dimensional space.

The state of a spin, or of an ensemble of spins, is `0.5 * (ONE + bloch)` for a vector `bloch`,
the Bloch vector: it points along the spin, and its length is one for a single pure spin and less
for a mixture. The pseudoscalar I squares to minus one and commutes with everything.

The state changes by a linear map, the generator. In the frame that turns with the drive, the
Hamiltonian is half the detuning along z plus half the drive strength along x, and the state turns
by -I times its commutator with the Hamiltonian. Relaxation enters through multivectors `process`,
each adding `(process >> rho) - 0.5 * (back * rho + rho * back)` to the change of
the state `rho`, with `back = process.reverse() * process`. One is the vector x times the state of
a spin against the field, which turns that part of the state over onto the field at the rate
`1 / t1`; one along z scrambles the phase. Both are built as maps by leaving the state open. Where
the generator vanishes the state is steady; since the generator keeps the scalar part, finding
that state is a linear equation with the scalar part pinned.

The evolution over a short time dt is a map as well, the exponential of the generator times dt to
fourth order in dt: a sum of powers of the generator. Composed with itself it spans two steps, then
four; composed with the pulses, whose sandwiches are maps too, it gives a whole pulse sequence as
one map.

In matrix notation the state reads as a 2x2 Hermitian density matrix, the reverse as the Hermitian
conjugate, the pseudoscalar as the imaginary unit, the generator as a 4x4 superoperator on the
flattened matrix, RAISE as the raising operator, and the generator's restriction to the Bloch
vector as the Bloch equations.
"""

from __future__ import annotations

from collections.abc import Generator, Iterator

import numpy as np

from numga import NumpyContext
from numga.algebras import VGA3D as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
Vector = ga.gatype.vector()
# A spin's state is its own reverse: a scalar plus a vector, `0.5 * (ONE + bloch)`.
State = ga.gatype.self_reverse()                   # 1 x y z
Rates = ga.gatype((State, State))              # State <- State
# What becomes of each state over a stretch of time.
Evolution = ga.gatype((State, State))              # State <- State
# A relaxation process is a vector plus a bivector, `v + I * w` for vectors v and w.
Process = ga.gatype(ga.subspace.vector() + ga.subspace.bivector())   # x y z xy xz yz
Rotor = ga.gatype.rotor()
# The pseudoscalar squares to minus one and commutes with everything.
I = mv.xyz                                         # [] Pseudoscalar
ONE = mv.scalar([1.0])
# x times the state of a spin against the field: turns that part of the state over onto the field.
RAISE = mv.x * (0.5 * (ONE - mv.z))                # [] Process


# --- math -----------------------------------------------------------------------------
def state(bloch: Vector) -> State:
    """The state with the given Bloch vector."""
    return 0.5 * (ONE + bloch)


def relaxation(process: Process) -> Rates:
    """The change of a state `rho` due to a relaxation `process`:
    `(process >> rho) - 0.5 * (back * rho + rho * back)`, with
    `back = process.reverse() * process`."""
    back = process.reverse().symmetric_reverse_product()   # [] State
    return (process >> State) - 0.5 * (back * State + State * back)


def generator(detuning: Scalar, drive: Scalar, t1: float, t2: float) -> Rates:
    """The rate of change of a state, in the frame that turns with the drive.

    The Hamiltonian turns the spin about the axis `mv.x * drive + mv.z * detuning` at that axis's
    length. Raising at the rate `1 / t1` relaxes the spin toward +z; together with it, dephasing
    along z at the rate `1 / t2 - 1 / (2 * t1)` makes the transverse part decay at `1 / t2`.
    """
    hamiltonian = 0.5 * (mv.z * detuning + mv.x * drive)
    turning = -I * (hamiltonian * State - State * hamiltonian)
    relaxing = (1 / t1) * relaxation(RAISE) + 0.5 * (1 / t2 - 1 / (2 * t1)) * relaxation(mv.z)
    return (turning + relaxing).cast(State)


def steady(rates: Rates) -> State:
    """The state that does not change.

    The generator keeps the scalar part, so its scalar output is always zero and it is singular.
    Adding the scalar part as a dyad on one pins it: the sum sends a state to its scalar part plus
    its change, and solving that against one half gives the state with scalar part one half whose
    change is zero.
    """
    scalar_part = ONE.scalar_product(State)                                  # Scalar <- State
    return (rates + ONE * scalar_part).solve(ONE * 0.5)


def evolution(rates: Rates, dt: float) -> Evolution:
    """What becomes of each state over a time dt: the exponential of the generator times dt, to
    fourth order in dt, the powers of the generator by composition."""
    small = rates * dt
    term = total = State
    for order in range(1, 5):
        term = small(term) / order
        total = total + term
    return total


def evolve(each_step: Evolution, rho: State, steps: int) -> Generator[State, None, State]:
    """The state before each of the given number of steps; returns the state after the last."""
    for _ in range(steps):
        yield rho
        rho = each_step(rho)
    return rho


def echo(each_step: Evolution, rho: State, before: int, after: int) -> Iterator[State]:
    """The states through a spin echo: tipped onto -y, left for the given number of steps, turned
    half a turn about x, and left again."""
    rho = yield from evolve(each_step, pulse(np.pi / 2) >> rho, before)
    yield from evolve(each_step, pulse(np.pi) >> rho, after)


def doublings(span: Evolution, count: int) -> Iterator[Evolution]:
    """The evolution over its own span, then over twice and four times that span, and so on: each
    composed with itself for the next."""
    for _ in range(count):
        yield span
        span = span(span)


def pulse(angle: float) -> Rotor:
    """A short strong pulse along x, as a rotor: it turns the Bloch vector about x by the angle,
    in the same sense as the drive. It acts on a state by its sandwich."""
    return (I * mv.x * (-angle / 2)).exp()


def bloch(rho: State) -> Vector:
    """The Bloch vector of a state: twice its vector part."""
    return 2 * rho.cast(Vector)
