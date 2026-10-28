"""The invariant decomposition of a rotation of six-dimensional space into three independent turns.

A bivector of six dimensions generates a rotation. It is the sum of three bivectors that are each a
single plane and commute with one another, so the rotation is the product of three turns, one in each
plane, at each plane's own rate. Two ways find the planes.

From the wedge powers of the bivector, `pair = (bivector ^ bivector) / 2` and
`triple = (bivector ^ pair) / 3`: the squares of the three planes are the roots of a cubic whose
coefficients are the scalar parts of `bivector * ~bivector`, `pair * ~pair` and `triple * ~triple`,
and at each root `square` the plane is `(triple + square * bivector) * (pair + square).inverse()`.
That inverse is of a scalar plus a quadvector, whose powers stay scalar plus quadvector: it satisfies
a quartic, so the inverse is a cubic in it.

From the acceleration of a direction turned by the rotation, the linear map
`Antivector.commutator(bivector).commutator(bivector)`: its eigenvalues are the same squares, each twice,
and each pair of eigenvectors spans one plane. A unit direction `u` in a plane moves with velocity
`u.commutator(bivector)` within that plane alone, so the two span the plane's patch,
`u.regressive(u.commutator(bivector))`, and the plane itself is that patch's dual.

The same holds for rigid motions, in the projective algebra of six-dimensional space, with a seventh
direction t that squares to zero: a motion's bivector is three commuting planes, now placed in space.
Their directions are the spectral planes of its Euclidean part, found from the directions at infinity.
Each plane's placement, its part that involves t, is fixed by two conditions linear in it: the plane
commutes with the bivector, and it stays simple. In six dimensions nothing is left over; in an odd
number a translation along the fixed axis would be.

Once the planes are known, the rotation over any time is the product of three plane rotors, each
`cos(angle / 2) + part * sin(angle / 2) / rate`: no exponential series, and the planes are found once
for all times. A point turned by the rotation traces a tangle in any three directions; seen in each
plane it turns on a circle, at that plane's rate.

In matrix notation the bivector reads as an antisymmetric six-by-six matrix, the planes as the
two-by-two blocks of its real Schur form, and the squares as minus the squares of the rotation rates,
whose eigenvalues are plus and minus i times the rates.
"""

from __future__ import annotations

from numga import Algebra, NumpyContext, stack
from numga.extensor import Extensor
from numga.gatype import GAType
from numga.gatype import ReverseProductOne, Versor
from examples.math.invariant_decomposition.numerical import cubic_roots

ga = Algebra("x+y+z+w+v+u+")
mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                                    # Scalar+
Antivector = ga.gatype.antivector()                                            # Antivector-
Bivector = ga.gatype.bivector()                                                # Bivector+
Rotor = ga.gatype.rotor()                                                      # Rotor+
ScalarQuadvector = ga.gatype.from_grades([0, 4])                               # Even+
# The projective algebra of six-dimensional space: the same six directions, and t squaring to zero.
projective = Algebra("x+y+z+w+v+u+t0")
pmv = NumpyContext(projective).multivector
ProjectiveBivector = projective.gatype.bivector()                              # Bivector+
# The directions at infinity, the antivectors that involve t.
NullAntivector = projective.gatype.antivector().derive.degenerate              # Antivector-
EuclideanBivector = ProjectiveBivector.derive.nondegenerate                    # Bivector+
NullBivector = ProjectiveBivector.derive.degenerate                            # Bivector+
Conditions = projective.gatype.from_grades([2, 4])                             # Even+


# --- math -----------------------------------------------------------------------------
def from_wedge_powers(bivector: Bivector) -> tuple[Scalar, Bivector]:
    """The squares of the three commuting planes of a bivector, and the planes, from its wedge powers."""
    pair = (bivector ^ bivector) / 2                                           # [...] Antibivector+
    triple = (bivector ^ pair) / 3                                             # [...] Pseudoscalar+
    powers = stack([bivector, pair, triple], axis=-1)                          # [..., 3] Even+
    # The squares of the planes are the roots of the cubic with these coefficients after the leading one.
    squares = cubic_roots(powers.scalar_norm_squared())                        # [..., 3] Scalar+
    parts = (triple[..., None] + squares * bivector[..., None]) * inverse(pair[..., None] + squares)   # [..., 3] Even+
    return squares, parts.cast(Bivector)                                       # [..., 3] Scalar+, [..., 3] Bivector+


def inverse(value: ScalarQuadvector) -> ScalarQuadvector:
    """The inverse of a scalar plus a quadvector of six dimensions, as a cubic in it.

    Its powers stay scalar plus quadvector, and it satisfies a quartic: four times the scalar parts
    of its first four powers are the power sums of the quartic's roots, from which Newton's
    identities give the coefficients. So `value * (e3 - e2 * value + e1 * square - cube) == e4`.
    """
    square = value.squared()                                                   # [...] Even+
    # Powers of one element commute, so their product is its symmetric part, scalar plus quadvector by type.
    cube = square.anticommutator(value)                                        # [...] Even+
    s1, s2, s3, s4 = (4 * power.select[0] for power in (value, square, cube, square.squared()))   # [...] Scalar+, Scalar+, Scalar+, Scalar+
    e1 = s1
    e2 = (e1 * s1 - s2) / 2
    e3 = (e2 * s1 - e1 * s2 + s3) / 3
    e4 = (e3 * s1 - e2 * s2 + e1 * s3 - s4) / 4
    return (e3 - e2 * value + e1 * square - cube) / e4                         # [...] Even+


def from_spectrum(bivector: Extensor, Direction: GAType) -> tuple[Extensor, Extensor]:
    """The squares of the three commuting planes of a bivector, and the planes, from the eigenvalues and
    eigenvectors of the acceleration of a direction turned by the rotation, a map on the given directions:
    the squares come each twice, and one direction of each pair gives its plane."""
    # A direction turned by the rotation moves with velocity `direction.commutator(bivector)`: this is its
    # acceleration.
    acceleration = Direction.commutator(bivector).commutator(bivector)         # [...] Antivector- <- Antivector-
    values, directions = acceleration.eigh()                                   # [..., 6] Scalar+, [..., 6] Antivector-
    # The eigenvalues come in equal pairs, in ascending order: one direction of each pair.
    spanning = directions[..., ::2]                                            # [..., 3] Antivector-
    # A unit direction and its velocity span the patch of its plane, whose dual is the plane.
    patches = spanning.regressive(spanning.commutator(bivector[..., None]))    # [..., 3] Antibivector-
    return values[..., ::2], patches.dual_inverse()                            # [..., 3] Scalar+, [..., 3] Bivector+


def placed(motion: ProjectiveBivector) -> ProjectiveBivector:
    """The commuting planes of a bivector of the projective algebra, placed in space. The planes' directions
    are the spectral planes of its Euclidean part. Each plane's placement is the null bivector that makes it
    commute with the motion and keeps it simple: both conditions are linear in the placement, one a bivector
    and the other a quadvector. Their sum is a map from placements to both conditions, whose least-squares
    solve satisfies them together."""
    euclidean = motion.cast(EuclideanBivector)                                 # [...] Bivector+
    _, turning = from_spectrum(euclidean, NullAntivector)                      # [..., 3] Bivector+
    conditions = (NullBivector.commutator(motion[..., None]) + (turning ^ NullBivector)).cast(Conditions)   # [..., 3] Even+ <- Bivector+
    placement = conditions.lstsq(-turning.commutator(motion[..., None]))       # [..., 3] Bivector+
    return turning + placement                                                 # [..., 3] Bivector+


def rotor(squares: Scalar, parts: Bivector, times: Scalar) -> Rotor:
    """The rotor the bivector generates over each time, as the product of its planes' rotors, each
    `cos(angle / 2) + part * sin(angle / 2) / rate` with the plane's rate the square root of minus its
    square: a unit rotor by construction."""
    rates = (-squares).square_root()                                           # [3] Scalar+
    half_angles = times[..., None] * rates / 2                                 # [times, 3] Scalar+
    turns = (half_angles.cos() + parts * (half_angles.sin() / rates)).with_traits(ReverseProductOne, Versor)   # [times, 3] Rotor+
    return turns[..., 0] * turns[..., 1] * turns[..., 2]                       # [times] Rotor+
