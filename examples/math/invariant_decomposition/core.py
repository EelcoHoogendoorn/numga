"""The invariant decomposition of a rotation of six-dimensional space into three independent turns.

A bivector of six dimensions generates a rotation. It splits into three simple bivectors that commute with
one another, so the rotation factors into three turns, one in each plane, at each plane's own rate. Two
methods recover the three planes.

The first method uses the wedge powers of the bivector, `pair = (bivector ^ bivector) / 2` and
`triple = (bivector ^ pair) / 3`. The squares of the three planes are the roots of a cubic whose
coefficients are the scalar parts of `bivector * ~bivector`, `pair * ~pair` and `triple * ~triple`, and at
each root `square` the plane is `(triple + square * bivector) * (pair + square).inverse()`. That inverse
acts on a scalar plus a quadvector, whose powers stay scalar plus quadvector and satisfy a quartic, giving
the inverse as a cubic polynomial.

The second method uses the acceleration of a direction turned by the rotation, the linear map
`Direction.commutator(bivector).commutator(bivector)`. Its eigenvalues are the same squares, each with
multiplicity two, and each pair of eigenvectors spans one plane. A unit direction `u` in a plane moves
with velocity `u.commutator(bivector)` within that plane alone, so the two span the plane's patch
`u & u.commutator(bivector)`, whose inverse dual is the plane itself.

Rigid motions in the projective algebra of six-dimensional space, with a seventh direction squaring to
zero, split in the same way into three commuting planes placed in space. Their directions are the spectral
planes of the Euclidean part, found on the directions at infinity. Two linear conditions fix each plane's
null placement: the placed plane commutes with the motion and stays simple. In six dimensions those three
placed planes account for the whole motion, whereas in odd dimension a translation along the fixed axis
remains.

Once the planes are known, the rotation over any elapsed time is the product of three plane rotors, each
`cos(angle / 2) + part * sin(angle / 2) / rate`, evaluated directly from the one-time decomposition without
a power series. A point turned by the rotation traces a tangle across any three spatial axes, while its
component in each invariant plane turns uniformly on a circle.

In matrix notation, the bivector reads as an antisymmetric six-by-six matrix, the invariant planes correspond
to the two-by-two blocks of its real Schur form, and the plane squares equal minus the squared rotation rates
from its conjugate imaginary eigenvalue pairs.
"""

from __future__ import annotations

from numga import Algebra, NumpyContext, stack
from numga.extensor import Extensor
from numga.gatype import GAType, ReverseProductOne, Versor
from examples.math.invariant_decomposition.numerical import cubic_roots

ga = Algebra("x+y+z+w+v+u+")
mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                                    # Scalar+
Direction = ga.gatype.antivector()                                             # Antivector-
Bivector = ga.gatype.bivector()                                                # Bivector+
Rotor = ga.gatype.rotor()                                                      # Rotor+
ScalarQuadvector = ga.gatype.from_grades([0, 4])                               # Even+
# The projective algebra of six-dimensional space: the same six directions, and t squaring to zero.
projective = Algebra("x+y+z+w+v+u+t0")
pmv = NumpyContext(projective).multivector
ProjectiveBivector = projective.gatype.bivector()                              # Bivector+
# The directions at infinity, the antivectors that involve t.
NullDirection = projective.gatype.antivector().derive.degenerate               # Antivector-
EuclideanBivector = ProjectiveBivector.derive.nondegenerate                    # Bivector+
NullBivector = ProjectiveBivector.derive.degenerate                            # Bivector+
Conditions = projective.gatype.from_grades([2, 4])                             # Even+


# --- math -----------------------------------------------------------------------------
def wedge_decomposition(bivector: Bivector) -> tuple[Scalar, Bivector]:
    """The wedge powers of `bivector` determine the squares (`[..., 3] Scalar+`) and commuting planes (`[..., 3] Bivector+`) of its rotation."""
    pair = (bivector ^ bivector) / 2                                           # [...] Antibivector+
    triple = (bivector ^ pair) / 3                                             # [...] Pseudoscalar+
    powers = stack([bivector, pair, triple], axis=-1)                          # [..., 3] Even+
    # The squares of the planes are the roots of the cubic with these coefficients after the leading one.
    squares = cubic_roots(powers.scalar_norm_squared())                        # [..., 3] Scalar+
    parts = (triple[..., None] + squares * bivector[..., None]) * quadvector_inverse(pair[..., None] + squares)   # [..., 3] Even+
    return squares, parts.cast(Bivector)                                       # [..., 3] Scalar+, [..., 3] Bivector+


def quadvector_inverse(value: ScalarQuadvector) -> ScalarQuadvector:
    """Newton's identities on the first four powers of `value` give its inverse (`[...] Even+`) as a cubic polynomial.

    Powers of a scalar plus a quadvector stay scalar plus quadvector and satisfy a quartic whose power sums
    are four times the scalar parts of the first four powers, so `value * (e3 - e2 * value + e1 * square - cube) == e4`.
    """
    square = value.squared()                                                   # [...] Even+
    # Powers of one element commute, so their product equals their anticommutator, staying scalar plus quadvector by type.
    cube = square.anticommutator(value)                                        # [...] Even+
    s1, s2, s3, s4 = (4 * power.select[0] for power in (value, square, cube, square.squared()))   # [...] Scalar+, Scalar+, Scalar+, Scalar+
    e1 = s1                                                                    # [...] Scalar+
    e2 = (e1 * s1 - s2) / 2                                                    # [...] Scalar+
    e3 = (e2 * s1 - e1 * s2 + s3) / 3                                          # [...] Scalar+
    e4 = (e3 * s1 - e2 * s2 + e1 * s3 - s4) / 4                                # [...] Scalar+
    return (e3 - e2 * value + e1 * square - cube) / e4                         # [...] Even+


def spectral_decomposition(bivector: Extensor, SpaceDirection: GAType) -> tuple[Extensor, Extensor]:
    """The directional acceleration map on `SpaceDirection` splits `bivector` into its plane squares (`[..., 3] Scalar+`) and commuting planes (`[..., 3] Bivector+`)."""
    # Turning a direction gives velocity `direction.commutator(bivector)` and a second commutator gives its acceleration.
    acceleration = SpaceDirection.commutator(bivector).commutator(bivector)    # [...] Antivector- <- Antivector-
    values, directions = acceleration.eigh()                                   # [..., 6] Scalar+, [..., 6] Antivector-
    # Eigenvalues come in ascending equal pairs, so every second eigenvector picks one direction per plane.
    plane_directions = directions[..., ::2]                                    # [..., 3] Antivector-
    # Each unit direction and its velocity within the plane join to form that plane's intrinsic patch.
    patches = plane_directions & plane_directions.commutator(bivector[..., None])   # [..., 3] Antibivector-
    return values[..., ::2], patches.dual_inverse()                            # [..., 3] Scalar+, [..., 3] Bivector+


def placed_planes(motion: ProjectiveBivector) -> ProjectiveBivector:
    """Commutation with `motion` and simplicity place the three Euclidean spectral planes in space (`[..., 3] Bivector+`)."""
    euclidean = motion.cast(EuclideanBivector)                                 # [...] Bivector+
    _, plane_directions = spectral_decomposition(euclidean, NullDirection)     # [..., 3] Bivector+
    # Commutation produces a bivector residual and simplicity produces a quadvector residual, solved together in least squares.
    conditions = (NullBivector.commutator(motion[..., None]) + (plane_directions ^ NullBivector)).cast(Conditions)   # [..., 3] Even+ <- Bivector+
    placement = conditions.lstsq(-plane_directions.commutator(motion[..., None]))   # [..., 3] Bivector+
    return plane_directions + placement                                        # [..., 3] Bivector+


def factored_rotor(squares: Scalar, parts: Bivector, times: Scalar) -> Rotor:
    """The three commuting plane rotors multiply into the total rotor (`[...] Rotor+`) at each elapsed time."""
    rates = (-squares).square_root()                                           # [3] Scalar+
    half_angles = times[..., None] * rates / 2                                 # [..., 3] Scalar+
    turns = (half_angles.cos() + parts * (half_angles.sin() / rates)).with_traits(ReverseProductOne, Versor)   # [..., 3] Rotor+
    return turns[..., 0] * turns[..., 1] * turns[..., 2]                       # [...] Rotor+
