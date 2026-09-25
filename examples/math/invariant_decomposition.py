"""A rotation of six-dimensional space is three independent turns.

A bivector of six dimensions generates a rotation. It is the sum of three bivectors that are each a
single plane and commute with one another, so the rotation is the product of three turns, one in each
plane, at each plane's own rate. Two ways find the planes.

From the wedge powers of the bivector, `pair = (bivector ^ bivector) / 2` and
`triple = (bivector ^ pair) / 3`: the squares of the three planes are the roots of a cubic whose
coefficients are the scalar parts of `bivector * ~bivector`, `pair * ~pair` and `triple * ~triple`,
and at each root `square` the plane is `(triple + square * bivector) * (pair + square).inverse()`.
That inverse is of a scalar plus a quadvector, whose powers stay scalar plus quadvector: it satisfies
a quartic, so the inverse is a cubic in it.

From the map `(Vector | bivector) | bivector`, with the vector left open: its eigenvalues are the same
squares, each twice, and each pair of eigenvectors spans one plane. For a unit direction `u` in a plane,
`u | bivector` sees only that plane, so `u ^ (u | bivector)` is the plane itself.

The same holds for rigid motions, in the projective algebra of six-dimensional space, with a seventh
direction t that squares to zero: a motion's bivector is three commuting planes, now placed in space.
Their directions are the spectral planes of its Euclidean part. Each plane's placement, its part that
involves t, is fixed by two conditions linear in it: the plane commutes with the bivector, and it stays
simple. In six dimensions nothing is left over; in an odd number a translation along the fixed axis
would be.

Once the planes are known, the rotation over any time is the product of three plane rotors, each
`cos(angle / 2) + part * sin(angle / 2) / rate`: no exponential series, and the planes are found once
for all times. A point turned by the rotation traces a tangle in any three directions; seen in each
plane it turns on a circle, at that plane's rate.

In matrix notation the bivector reads as an antisymmetric six-by-six matrix, the planes as the
two-by-two blocks of its real Schur form, and the squares as minus the squares of the rotation rates,
whose eigenvalues are plus and minus i times the rates.

Run from the repository root with python -m examples.math.invariant_decomposition.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from numga import Algebra, NumpyContext, stack
from numga.gatype import GAType
from numga.gatype import ReverseProductOne, Versor
from examples.animation import capture

ga = Algebra("x+y+z+w+v+u+")
mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
Vector = ga.gatype.vector()
Bivector = ga.gatype.bivector()
Rotor = ga.gatype.rotor()
ScalarQuadvector = ga.gatype(ga.subspace.scalar() + ga.subspace.k_vector(4))
# The projective algebra of six-dimensional space: the same six directions, and t squaring to zero.
projective = Algebra("x+y+z+w+v+u+t0")
pmv = NumpyContext(projective).multivector
ProjectiveBivector = projective.gatype.bivector()
EuclideanVector = projective.gatype(projective.subspace.vector().nondegenerate())
EuclideanBivector = projective.gatype(projective.subspace.bivector().nondegenerate())
NullBivector = projective.gatype(projective.subspace.bivector().degenerate())
Conditions = projective.gatype(projective.subspace.bivector() + projective.subspace.k_vector(4))


# --- math -----------------------------------------------------------------------------
def from_wedge_powers(bivector: Bivector) -> tuple[Scalar, Bivector]:
    """The squares of the three commuting planes of a bivector, and the planes, from its wedge powers."""
    pair = (bivector ^ bivector) / 2                                           # [...] Quadvector
    triple = (bivector ^ pair) / 3                                             # [...] Pseudoscalar
    powers = stack([bivector, pair, triple], axis=-1)                          # [..., 3] Even
    # The squares of the planes are the roots of the cubic with these coefficients after the leading one.
    squares = cubic_roots(powers.scalar_product(powers.reverse()))            # [..., 3] Scalar
    parts = (triple[..., None] + squares * bivector[..., None]) * inverse(pair[..., None] + squares)   # [..., 3] Even
    return squares, parts.cast(Bivector)                                       # [..., 3] Scalar, [..., 3] Bivector


def inverse(value: ScalarQuadvector) -> ScalarQuadvector:
    """The inverse of a scalar plus a quadvector of six dimensions, as a cubic in it.

    Its powers stay scalar plus quadvector, and it satisfies a quartic: four times the scalar parts
    of its first four powers are the power sums of the quartic's roots, from which Newton's
    identities give the coefficients. So `value * (e3 - e2 * value + e1 * square - cube) == e4`.
    """
    square = value.squared()                                                   # [...] ScalarQuadvector
    cube = (square * value).cast(ScalarQuadvector)                             # [...] ScalarQuadvector
    sums = [4 * power.select[0] for power in (value, square, cube, square.squared())]   # [...] Scalar each
    e1 = sums[0]
    e2 = (e1 * sums[0] - sums[1]) / 2
    e3 = (e2 * sums[0] - e1 * sums[1] + sums[2]) / 3
    e4 = (e3 * sums[0] - e2 * sums[1] + e1 * sums[2] - sums[3]) / 4
    return (e3 - e2 * value + e1 * square - cube) / e4                         # [...] ScalarQuadvector


def from_spectrum(bivector: Bivector, Direction: GAType) -> tuple[Scalar, Bivector]:
    """The squares of the three commuting planes of a bivector of the given directions, and the planes,
    from the eigenvalues and eigenvectors of `Direction <- (Direction | bivector) | bivector`: the
    squares come each twice, and one direction of each pair gives its plane."""
    # A point turned by the rotation moves with velocity `point | bivector`: this is its acceleration.
    acceleration = (Direction | bivector) | bivector                           # [...] Direction <- Direction
    values, directions = acceleration.eigh()                                   # [..., 6] Scalar, [..., 6] Vector
    # The eigenvalues come in equal pairs, in ascending order: one direction of each pair.
    spanning = directions[..., ::2]                                            # [..., 3] Vector
    return values[..., ::2], spanning ^ (spanning | bivector[..., None])       # [..., 3] Scalar, [..., 3] Bivector


def placed(motion: ProjectiveBivector) -> tuple[ProjectiveBivector, ProjectiveBivector]:
    """The commuting planes of a bivector of the projective algebra, placed in space, and the part left
    over. The planes' directions are the spectral planes of its Euclidean part. Each plane's placement
    is the null bivector that makes it commute with the motion and keeps it simple: both conditions are
    linear in the placement, one a bivector and the other a quadvector, and a single least-squares
    solve with the placement left open meets them together."""
    euclidean = motion.cast(EuclideanBivector)                                 # [...] EuclideanBivector
    _, turning = from_spectrum(euclidean, EuclideanVector)                     # [..., 3] EuclideanBivector
    conditions = (NullBivector.commutator(motion[..., None]) + (turning ^ NullBivector)).cast(Conditions)   # [..., 3] Conditions <- NullBivector
    placement = conditions.lstsq(-turning.commutator(motion[..., None]))       # [..., 3] NullBivector
    planes = (turning + placement).cast(ProjectiveBivector)                    # [..., 3] ProjectiveBivector
    return planes, (motion - planes.sum(axis=-1)).cast(ProjectiveBivector)     # [..., 3], [...] ProjectiveBivector


def rotor(squares: Scalar, parts: Bivector, times: np.ndarray) -> Rotor:
    """The rotor the bivector generates over each time, as the product of its planes' rotors, each
    `cos(angle / 2) + part * sin(angle / 2) / rate` with the plane's rate the square root of minus its
    square: a unit rotor by construction."""
    cosines, sines = half_turns(squares, times)                                # [times, 3] Scalar each
    turns = (cosines + parts * sines).with_traits(ReverseProductOne, Versor)   # [times, 3] Rotor
    return turns[..., 0] * turns[..., 1] * turns[..., 2]                       # [times] Rotor


def orbit(squares: Scalar, parts: Bivector, start: Vector, times: np.ndarray) -> Vector:
    """The start turned by the rotation, for each time."""
    return rotor(squares, parts, times) >> start                               # [times] Vector


def projected(points: Vector, parts: Bivector) -> Vector:
    """The points projected into each plane: `(point | part) * part.inverse()`, a vector in the plane."""
    return ((points[None] | parts[:, None]) * parts[:, None].inverse()).cast(Vector)   # [3, ...] Vector


def main(frames: int, seed: int) -> tuple[Bivector, Vector, Vector]:
    rng = np.random.default_rng(seed)
    bivector = mv(Bivector, rng.normal(size=15))                               # [] Bivector
    start = mv(Vector, rng.normal(size=6)).normalized()                        # [] Vector
    squares, parts = from_spectrum(bivector, Vector)                           # [3] Scalar, [3] Bivector
    wedge_squares, wedge_parts = from_wedge_powers(bivector)                   # [3] Scalar, [3] Bivector
    times = np.linspace(0.0, 2 * np.pi / np.sqrt(-squares.to_array().max()), frames)
    points = orbit(squares, parts, start, times)                               # [times] Vector
    circles = projected(points, parts)                                         # [3, times] Vector
    motion = pmv(ProjectiveBivector, rng.normal(size=21))                      # [] ProjectiveBivector
    placed_planes, leftover = placed(motion)                                   # [3], [] ProjectiveBivector

    # --- checks
    # The planes sum to the bivector, commute, and each squares to a scalar; the two ways give the
    # same squares and the same planes. The product of the planes' rotors is the exponential of half
    # the bivector, the point keeps its length, and in each plane it stays on a circle.
    np.testing.assert_allclose((parts.sum(axis=0) - bivector).kernel, 0.0, atol=1e-11)
    np.testing.assert_allclose((parts[:, None] * parts[None] - parts[None] * parts[:, None]).kernel, 0.0, atol=1e-11)
    np.testing.assert_allclose((parts * parts - (parts * parts).select[0]).kernel, 0.0, atol=1e-10)
    np.testing.assert_allclose(wedge_squares.to_array(), squares.to_array(), atol=1e-11)
    np.testing.assert_allclose((wedge_parts - parts).kernel, 0.0, atol=1e-11)
    np.testing.assert_allclose((rotor(squares, parts, np.ones(1))[0] - (bivector / 2).exp()).kernel, 0.0, atol=1e-6)
    np.testing.assert_allclose((points | points).to_array(), 1.0, atol=1e-6)
    radii = (circles | circles).to_array()                                     # [3, times]
    np.testing.assert_allclose(radii - radii[:, :1], 0.0, atol=1e-6)
    # The placed planes of the motion are simple and commute with one another, and in six dimensions
    # they add up to the whole motion.
    np.testing.assert_allclose((placed_planes ^ placed_planes).kernel, 0.0, atol=1e-11)
    np.testing.assert_allclose((placed_planes[:, None] * placed_planes[None] - placed_planes[None] * placed_planes[:, None]).kernel, 0.0, atol=1e-11)
    np.testing.assert_allclose(leftover.kernel, 0.0, atol=1e-10)
    return parts, points, circles


# --- plumbing -------------------------------------------------------------------------
def half_turns(squares: Scalar, times: np.ndarray) -> tuple[Scalar, Scalar]:
    """For each time and plane: the cosine of half the plane's angle, and its sine over the plane's
    rate, the square root of minus its square."""
    rates = np.sqrt(-squares.to_array())                                       # [3]
    angles = times[:, None] * rates / 2                                        # [times, 3]
    return mv.scalar(np.cos(angles)[..., None]), mv.scalar((np.sin(angles) / rates)[..., None])


def cubic_roots(coefficients: Scalar) -> Scalar:
    """The roots of the cubics `x**3 + c[0] * x**2 + c[1] * x + c[2]` for the coefficients c along
    the last axis, ascending: the eigenvalues of their companion matrices; real here."""
    companion = np.zeros(coefficients.shape + (3,))
    companion[..., 0, :] = -coefficients.to_array()
    companion[..., [1, 2], [0, 1]] = 1.0
    return mv.scalar(np.sort(np.linalg.eigvals(companion).real, axis=-1)[..., None])   # [..., 3] Scalar


def components(vectors: Vector, names: str) -> np.ndarray:
    return vectors.cast(ga.subspace(names)).kernel


def plane_axes(circles: Vector, parts: Bivector) -> tuple[Vector, Vector]:
    """In each plane, the direction of the first projected point and the direction a quarter turn
    on from it."""
    first = circles[:, 0].normalized()                                         # [3] Vector
    return first, (first | parts).normalized()


def draw(points: Vector, circles: Vector, parts: Bivector, index: int) -> plt.Figure:
    """The orbit in the first three directions, and in each of the three planes, up to one moment."""
    figure = plt.figure(figsize=(12, 4.6))
    tangle = figure.add_axes((0.0, 0.0, 0.4, 1.0), projection="3d")
    path = components(points, "x y z")                                         # [times, 3]
    tangle.plot(*path.T, color="0.8", linewidth=0.8)
    tangle.plot(*path[: index + 1].T, color="#7d3c98", linewidth=1.5)
    tangle.scatter(*path[index], color="#7d3c98", s=30)
    tangle.set_axis_off()
    figure.text(0.2, 0.9, "x, y and z", ha="center", fontsize=12)
    across, along = plane_axes(circles, parts)
    flat = np.stack([(circles | across[:, None]).to_array(), (circles | along[:, None]).to_array()], axis=-1)   # [3, times, 2]
    for plane, colour in enumerate(("#c0392b", "#2e86c1", "#27ae60")):
        extent = np.abs(flat[plane]).max() * 1.2
        ax = figure.add_axes((0.42 + plane * 0.19, 0.2, 0.17, 0.6))
        ax.plot(*flat[plane].T, color="0.85", linewidth=0.8)
        ax.plot(*flat[plane, : index + 1].T, color=colour, linewidth=1.5)
        ax.scatter(*flat[plane, index], color=colour, s=30, zorder=3)
        ax.set(xlim=(-extent, extent), ylim=(-extent, extent), aspect="equal", xticks=[], yticks=[])
        ax.set_title(f"plane {plane + 1}")
    return figure


def animate(points: Vector, circles: Vector, parts: Bivector, frames: int) -> list[np.ndarray]:
    images = []
    for index in np.linspace(0, len(points) - 1, frames).astype(int):
        figure = draw(points, circles, parts, index)
        images.append(capture(figure))
        plt.close(figure)
    return images


if __name__ == "__main__":
    from examples.animation import save_animation, save_figure

    parts, points, circles = main(600, 0)
    save_figure(draw(points, circles, parts, len(points) - 1), "invariant_decomposition")
    save_animation(animate(points, circles, parts, 120), "invariant_decomposition", 50)
