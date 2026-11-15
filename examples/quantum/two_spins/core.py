"""Two spins entangled by exchange in the direct-sum algebra of two copies of spacetime.

One spin lives in a first spacetime copy with spatial directions x, y, z and time t, while the second
lives in a second copy with spatial directions X, Y, Z and time T. Separate copies make every element
of one spin commute with every element of the other. Each single-spin state is a spacelike even
multivector of its own copy, and the pair state is a product in the ideal of the correlator
`correlator = 0.5 * (1 - xy * XY)`. Within that ideal, right multiplication by either spin's xy plane
gives the same reference spin `reference_spin = correlator * xy`. Every state is normalized to unit
scalar part in its density `2 * state.symmetric_reverse_product()`.

The exchange interaction couples matching spatial planes across the two copies through
`coupling = yz * YZ + zx * ZX + xy * XY`. Because `coupling * coupling == 3 + 2 * coupling`, the
projectors `(1 + coupling) / 4` and `(3 - coupling) / 4` are idempotent and sum to one, splitting any
state into a singlet part with eigenvalue 3 and a triplet part with eigenvalue -1. Under exchange each
part turns on the right by its own rotor in the xy plane.

Each spin on its own shows the projection of `2 * (state >> reference_spin)` onto its own spatial
planes, with norm one for a product state and smaller norm when the pair is entangled. Spins remain
spatial bivectors here because the full-algebra complement spans both copies at once and an even
spatial reflection `x * X` transforms both copies together without pseudoscalars. Together the two
spins define a joint spin map from the second spin's planes to the first's, read from the grade-(2, 2)
part of the density so `first_plane.dot(joint(second_plane))` gives the expected product of outcomes.

Measuring the first spin with outcome +1 in a unit plane steers the second spin through the adjoint of
the joint spin map, `(second_plane + joint.adjoint()(plane)) / (1 + plane.dot(first_plane))`. Local
spatial rotors act on both sides of the joint spin map and preserve its singular values, which equal 1
and the concurrence repeated twice. From its two largest singular values follows the largest Bell
combination over all measurement planes, bounded above by `2 * sqrt(2)` because the square of the Bell
element equals four minus four times the product of the two local plane commutators.

A uniform magnetic field in the xy planes shifts the spin-aligned and spin-opposed triplets linearly
until the aligned triplet crosses the singlet at twice the exchange rate. A field difference between
the two copies mixes the singlet with the zero-spin triplet instead, where the time generator squares
to minus the squared generalized rate so the pair precesses on a singlet-triplet sphere.
"""

from __future__ import annotations

import numpy as np

from numga import Algebra, NumpyContext

# Two copies of spacetime, one per spin, and the observer's spatial subalgebra in each copy.
ga = Algebra("t-x+y+z+T-X+Y+Z+")
first_space = Algebra("x+y+z+")
second_space = Algebra("X+Y+Z+")
context = NumpyContext(ga)
mv = context.multivector
Scalar = ga.gatype.scalar()                                                    # Scalar+
Bivector = ga.gatype.bivector()                                                # Bivector+
FirstPlanes = ga.gatype.bivector().derive.within(first_space)                  # Bivector+
SecondPlanes = ga.gatype.bivector().derive.within(second_space)                # Bivector+
FirstSpinor = ga.gatype.even().derive.within(first_space)                      # Even+
SecondSpinor = ga.gatype.even().derive.within(second_space)                    # Even+
Spinor = ga.gatype(FirstSpinor * SecondSpinor)                                 # Even+
JointMap = ga.gatype((FirstPlanes, SecondPlanes))                              # FirstPlanes+ <- SecondPlanes+
# Correlator idempotent and reference spin shared on the right by both copies.
correlator = 0.5 * (1 - mv.xy * mv.XY)                                         # [] Spinor+
reference_spin = correlator * mv.xy                                            # [] Spinor+
# Exchange operator and its orthogonal singlet and triplet projectors.
coupling = mv.yz * mv.YZ + mv.zx * mv.ZX + mv.xy * mv.XY                       # [] Spinor+
singlet_part = 0.25 * (1 + coupling)                                           # [] Spinor+
triplet_part = 0.25 * (3 - coupling)                                           # [] Spinor+
# Uniform and differential magnetic field planes in the two xy planes.
field_plane = mv.xy + mv.XY                                                    # [] Bivector+
difference_plane = 0.5 * (mv.xy - mv.XY)                                       # [] Bivector+


# --- math -----------------------------------------------------------------------------
def spin_planes(state: Spinor) -> tuple[FirstPlanes, SecondPlanes]:
    """The pair spin `2 * (state >> reference_spin)` projects onto each spin's spatial planes (`[...] FirstPlanes+, SecondPlanes+`)."""
    pair_spin = 2 * (state >> reference_spin)                                  # [...] Bivector+
    return pair_spin.cast(FirstPlanes), pair_spin.cast(SecondPlanes)


def joint_map(state: Spinor) -> JointMap:
    """The grade-(2, 2) block of the state density defines the joint spin map (`[...] FirstPlanes+ <- SecondPlanes+`) between the two spins' planes."""
    density = 2 * state.symmetric_reverse_product()                            # [...] Spinor+
    return (density * SecondPlanes).cast(FirstPlanes)                          # [...] FirstPlanes+ <- SecondPlanes+


def bell_maximum(joint: JointMap) -> Scalar:
    """The two largest singular values of `joint` set the largest Bell combination (`[...] Scalar+`) over all measurement planes."""
    values = joint.svdvals()                                                   # [..., 3] Scalar+
    return 2 * (values[..., 0] ** 2 + values[..., 1] ** 2).square_root()       # [...] Scalar+


def exchange_state(state: Spinor, angle: np.ndarray) -> Spinor:
    """Exchange evolution over `angle` turns the singlet part forward by `3 * angle` and the triplet part backward by `angle` (`[...] Spinor+`)."""
    singlet_turn = (mv.xy * (3 * angle)).exp()                                 # [...] Rotor+
    triplet_turn = (mv.xy * -angle).exp()                                      # [...] Rotor+
    return singlet_part * state * singlet_turn + triplet_part * state * triplet_turn   # [...] Spinor+


def bell_element(
    first_plane: FirstPlanes,
    first_alt: FirstPlanes,
    second_plane: SecondPlanes,
    second_alt: SecondPlanes,
) -> Spinor:
    """Two measurement planes on each spin combine into the left-acting Bell multivector (`[...] Spinor+`)."""
    return -(first_plane * (second_plane + second_alt) + first_alt * (second_plane - second_alt))   # [...] Spinor+


def expectation(element: Spinor, state: Spinor) -> Scalar:
    """Sandwiching a left-acting multivector `element` from the reverse side gives its scalar expectation (`[...] Scalar+`) in `state`."""
    return 2 * (state << element).select[0]                                    # [...] Scalar+


def steered_spin(state: Spinor, plane: FirstPlanes) -> tuple[Scalar, SecondPlanes]:
    """Projecting the first spin onto outcome +1 in unit `plane` gives the outcome probability (`[...] Scalar+`) and the steered second spin plane (`[...] SecondPlanes+`)."""
    # Left multiplication by `-plane` and right multiplication by `reference_spin` gives the +1/-1 observable in `plane`.
    projected_state = 0.5 * (state - plane * state * reference_spin)           # [...] Spinor+
    probability = 2 * projected_state.scalar_norm_squared()                    # [...] Scalar+
    _, second_plane = spin_planes(projected_state / probability.square_root()) # [...] SecondPlanes+
    return probability, second_plane


def energy(state: Spinor, exchange_rate: np.ndarray, field: np.ndarray) -> Scalar:
    """Shifted exchange coupling and a uniform xy-plane magnetic field give the expected state energy (`[...] Scalar+`)."""
    first_plane, second_plane = spin_planes(state)                             # [...] FirstPlanes+, SecondPlanes+
    return exchange_rate * expectation(0.5 * (1 - coupling), state) - 0.5 * field * field_plane.dot(first_plane + second_plane)   # [...] Scalar+


def time_generator(state: Spinor, exchange_rate: np.ndarray, field: Bivector) -> Spinor:
    """Shifted exchange about `reference_spin` and spatial bivector `field` drive the infinitesimal time rate (`[...] Spinor+`) of `state`."""
    return -exchange_rate * (0.5 * (1 - coupling) * state) * reference_spin - field * state   # [...] Spinor+


def qubit_state(state: Spinor, exchange_rate: np.ndarray, difference: np.ndarray, times: np.ndarray) -> Spinor:
    """Harmonic precession under `exchange_rate` and field `difference` carries a zero-net-spin `state` to `times` (`[...] Spinor+`)."""
    rate = np.sqrt(exchange_rate**2 + difference**2)
    step_generator = time_generator(state, exchange_rate, difference * difference_plane)   # [...] Spinor+
    return np.cos(rate * times) * state + np.sin(rate * times) / rate * step_generator     # [...] Spinor+


def qubit_coordinates(state: Spinor) -> tuple[Scalar, Scalar, Scalar]:
    """Spin imbalance, relative phase, and singlet-triplet balance place `state` on the two-spin qubit sphere (`[...] Scalar+` each)."""
    first_plane, second_plane = spin_planes(state)                             # [...] FirstPlanes+, SecondPlanes+
    imbalance = 0.5 * (mv.xy.dot(first_plane) - mv.XY.dot(second_plane))       # [...] Scalar+
    phase_coordinate = expectation(-0.25 * (1 - coupling) * (mv.xy - mv.XY), state)   # [...] Scalar+
    singlet_balance = expectation(0.5 * (coupling - 1), state)                 # [...] Scalar+
    return imbalance, phase_coordinate, singlet_balance
