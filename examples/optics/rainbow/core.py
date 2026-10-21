"""The rainbow: sunlight refracted into a raindrop, reflected inside it once or twice, and refracted
out again, in the geometric algebra of three-dimensional space.

Sunlight entering a spherical raindrop is bent at its surface, reflected from the back of the drop and
bent again on the way out, and comes back towards the sun turned by an angle that depends on how far
from the drop's centre it entered. That angle has an extreme: light entering near one particular
distance from the centre all leaves at nearly the same angle, and piles up there into a bright edge,
the bow, about 42 degrees from the point straight opposite the sun. Light reflected twice inside has
its own extreme near 51 degrees, the fainter secondary bow, and no light reflected once or twice leaves
between the two: Alexander's dark band. Water bends blue light slightly more than red, so each colour
has its bow at a slightly different angle.

Directions are modelled as antivectors, and so are the points of the drop's surface, as directions from
its centre. Half a turn about the surface's normal, `normal >> direction`, reverses the part of a
direction along the surface: `-(normal >> direction)` is the reflected direction, and the part along
the surface is half the difference. Snell's law keeps that part times the refractive index, which
holds both the law of sines and the light's staying in its plane of incidence; unit length fixes the
rest. Inside a sphere every chord meets the surface at the same angle at both ends, so each point
where the light meets the surface, and each chord, is the last one turned about the centre by the same
rotor: the light's path inside the drop is the powers of one turn, `turn >> vertex`.

In the vector notation of geometric optics, Snell's law reads as $n_1 \\sin\\theta_1 = n_2 \\sin\\theta_2$
with the rays and the normal in one plane, and the deflection of light entering at incidence $i$ and
refracted to $r$, after $k$ reflections inside, as $D = 2(i - r) + k(\\pi - 2r)$, stationary where
$\\cos^2 i = \\frac{n^2 - 1}{k(k + 2)}$.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import NumpyContext
from numga.algebras import VGA3D as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
# Directions, and the points of the drop's surface as directions from its centre, are antivectors,
# intrinsic; unit ones square to minus one.
Direction = ga.gatype.antivector()
Rotor = ga.gatype.rotor()


# --- math -----------------------------------------------------------------------------
def refract(direction: Direction, normal: Direction, ratio: np.ndarray) -> Direction:
    """The direction of light crossing a surface with unit `normal`, pointing into the medium it
    enters, which is `ratio` times as refractive as the one it leaves."""
    # Half a turn about the normal reverses the part of the direction along the surface; Snell's law
    # keeps that part times the refractive index.
    along = (direction - (normal >> direction)) / (2 * ratio)                 # [...] Direction
    # Unit length fixes the part along the normal.
    return along + normal * (1 - along.scalar_norm_squared()).square_root()   # [...] Direction


def entry(sunlight: Direction, offsets: Direction) -> Direction:
    """Where light along `sunlight`, passing the centre of the unit drop at each offset square to it,
    meets the drop: the point on its sunlit side, as a direction from the centre."""
    return offsets - sunlight * (1 - offsets.scalar_norm_squared()).square_root()   # [...] Direction


def bounce(vertex: Direction, chord: Direction) -> Rotor:
    """The turn about the drop's centre carrying a point where light meets the surface from inside, and
    the chord arriving there, to the next point and the next chord."""
    # The next point is this one reflected across the plane through the centre square to the chord.
    following = -(chord >> vertex)                                            # [...] Direction
    # The rotor taking one unit direction to another, in their plane.
    return (-(following * vertex)).square_root()                              # [...] Rotor


def reflectances(offsets: Direction, index: np.ndarray) -> tuple[Scalar, Scalar]:
    """The fraction of light the surface reflects, polarized across and along the plane of incidence:
    Fresnel's coefficients, the same where the light enters as wherever it meets the surface again."""
    # The cosines of the angles of incidence and refraction; the offset is the sine of incidence.
    incidence = (1 - offsets.scalar_norm_squared()).square_root()             # [...] Scalar
    refraction = (1 - offsets.scalar_norm_squared() / index**2).square_root() # [...] Scalar
    across = ((incidence - index * refraction) / (incidence + index * refraction)).squared()
    along = ((refraction - index * incidence) / (refraction + index * incidence)).squared()
    return across, along


def paths(sunlight: Direction, offsets: Direction, index: np.ndarray, n_reflections: int) -> Iterator[tuple[Direction, Direction, Scalar]]:
    """Per number of reflections inside the drop, from none up to `n_reflections`: the point where the
    light leaves, the direction it leaves in, and the fraction of unpolarized sunlight leaving there."""
    vertex = entry(sunlight, offsets)                                         # [...] Direction
    chord = refract(sunlight, -vertex, index)                                 # [...] Direction
    # Every chord of a sphere meets it at the same angle at both ends: each point and chord inside is
    # the last one turned by the same rotor.
    turn = bounce(vertex, chord)                                              # [...] Rotor
    across, along = reflectances(offsets, index)
    # The light transmitted into the drop and out of it again, in each polarization.
    passed_across, passed_along = (1 - across).squared(), (1 - along).squared()
    for _ in range(n_reflections + 1):
        vertex = turn >> vertex                                               # [...] Direction
        yield vertex, refract(chord, vertex, 1 / index), (passed_across + passed_along) / 2
        chord = turn >> chord                                                 # [...] Direction
        passed_across, passed_along = passed_across * across, passed_along * along
