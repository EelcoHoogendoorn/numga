"""Spin groups of every signature up to six dimensions sit inside one nine-dimensional algebra.

Six directions square to plus one (`x`, `y`, `z`, `w`, `v`, `u`) and three square to minus one
(`t`, `s`, `r`). Choosing `p` of the positive directions and `q` of the negative directions picks
out signature `(p, q)`: the bivectors of the chosen directions generate its rotors, forming the spin
group, and their even products form its spinors. Three maps built from types alone separate the
signatures.

The Lie algebra's invariant form traces what the double commutator of two bivectors does to a third,
and equals `2 * (n - 2)` times the algebra's own inner product `Bivector | Bivector`.

Sandwiching the bivectors with the product of the chosen positive directions defines an involution
that negates the `p * q` planes mixing a positive with a negative direction and fixes the
`p * (p - 1) // 2 + q * (q - 1) // 2` planes within either sign. Fixed planes square to minus one
and generate rotations; negated planes square to plus one and generate boosts, so a spin group is
compact when every plane generates a rotation.

In even dimensions the pseudoscalar of the chosen directions commutes with every spinor. When it
squares to plus one, `(1 + pseudoscalar) / 2` and `(1 - pseudoscalar) / 2` split the spinors into
two invariant halves; when it squares to minus one, multiplying by it turns every pair of spinor
components through a quarter turn. In four dimensions the pseudoscalar also maps bivectors to
bivectors, splitting the Lie algebra itself in signature `(4, 0)` into two commuting halves whose
exponentials turn every plane through one angle and trace the Hopf fibration and its mirror image on
the three-sphere.
"""

from __future__ import annotations

import numpy as np

from numga import Algebra, NumpyContext
from numga.extensor import Extensor
from numga.gatype import GAType

ga = Algebra("x+y+z+w+v+u+t-s-r-")
context = NumpyContext(ga)
mv = context.multivector
positive_names, negative_names = "xyzwvu", "tsr"

# Points of the four Euclidean directions x y z w are the antivectors containing the five remaining directions.
Euclidean = ga.gatype.vector().derive.within(Algebra("x+y+z+w+")).derive.dual   # Antivector-
Antitrivector = ga.gatype.antitrivector()                                       # Antitrivector-

# The four-dimensional Euclidean pseudoscalar commutes with bivectors and squares to one, splitting each
# generator into isoclinic halves that turn every plane along with or against its orthogonal complement.
euclidean_pseudoscalar = mv.xyzw                                                # [] Quadvector+
plus = (1 + euclidean_pseudoscalar) / 2                                         # [] Even+
minus = (1 - euclidean_pseudoscalar) / 2                                        # [] Even+


# --- math -----------------------------------------------------------------------------
def invariant_form(Bivector: GAType) -> Extensor:
    """The trace of the double commutator on `Bivector` defines the Lie algebra's invariant bilinear
    form (`[] Scalar+ <- (Bivector+, Bivector+)`)."""
    return Bivector.commutator(Bivector.commutator(Bivector)).trace(0, 3)       # [] Scalar+ <- (Bivector+, Bivector+)


def isoclinic_factors(generator: Extensor) -> tuple[Extensor, Extensor]:
    """Projecting a four-dimensional bivector `generator` onto `plus` and `minus` and exponentiating
    gives its two commuting isoclinic rotor factors (`[] Rotor+, Rotor+`)."""
    left_factor = (plus * generator).exp()                                      # [] Rotor+
    right_factor = (minus * generator).exp()                                    # [] Rotor+
    return left_factor, right_factor


def stereographic(points: Euclidean) -> Euclidean:
    """Stereographic projection from the pole at `-w` maps `points` (`[...] Antivector-`) on the
    three-sphere into the three-dimensional space of `x`, `y`, and `z` (`[...] Antivector-`)."""
    w_coordinate = mv.w & points                                                # [...] Scalar+
    return (points - mv.w.dual() * w_coordinate) / (1 + w_coordinate)           # [...] Antivector-


def isoclinic_orbits(n_azimuths: int, heights: np.ndarray, n_steps: int
                     ) -> tuple[Euclidean, Euclidean, Euclidean]:
    """Stereographic projections of the two opposite isoclinic circle families (`[n_heights,
    n_azimuths, n_steps + 1] Antivector-`) and of the `(2, 3)` torus-knot orbits (`[n_heights, 1,
    n_steps + 1] Antivector-`) on the three-sphere."""
    n_heights = len(heights)
    # Turning xy along with its orthogonal plane zw and against it gives the two isoclinic generators.
    along, against = plus * mv.xy, minus * mv.xy                                # [] Bivector+, Bivector+
    combined = 2 * mv.xy + 3 * mv.zw                                            # [] Bivector+
    # Starting points turn from +x toward +z by each height angle and around the xy plane by each azimuth.
    azimuths = np.linspace(0, 2 * np.pi, n_azimuths, endpoint=False)
    azimuth_turns = (mv.xy * (-azimuths / 2)).exp()                             # [n_azimuths] Rotor+
    height_turns = (mv.xz * (-heights[:, None] / 2)).exp()                      # [n_heights, 1] Rotor+
    starts = (azimuth_turns * height_turns) >> mv.x.dual()                      # [n_heights, n_azimuths] Antivector-
    # Because the sandwich product turns planes by twice the rotor angle, pi completes one full turn.
    turn_angles = np.linspace(0, np.pi, n_steps + 1)
    along_rotors = (2 * along * turn_angles).exp()                              # [n_steps + 1] Rotor+
    against_rotors = (2 * against * turn_angles).exp()                          # [n_steps + 1] Rotor+
    combined_rotors = (combined * turn_angles).exp()                            # [n_steps + 1] Rotor+
    left_orbits = stereographic(along_rotors >> starts[..., None])              # [n_heights, n_azimuths, n_steps + 1] Antivector-
    right_orbits = stereographic(against_rotors >> starts[..., None])           # [n_heights, n_azimuths, n_steps + 1] Antivector-
    knotted_orbits = stereographic(combined_rotors >> starts[:, :1, None])      # [n_heights, 1, n_steps + 1] Antivector-
    return left_orbits, right_orbits, knotted_orbits


def linking(first: Euclidean, second: Euclidean) -> Antitrivector:
    """Gauss's linking integral of two closed polygonal loops (`[n_vertices] Antivector-`) in
    three-dimensional space yields an integer multiple of the unit volume (`[] Antitrivector-`)
    whose sign reverses under reflection."""
    n_segments = first.shape[0] - 1
    first_steps = first[1:] - first[:-1]                                        # [n_segments] Antivector-
    second_steps = second[1:] - second[:-1]                                     # [n_segments] Antivector-
    first_midpoints = 0.5 * (first[1:] + first[:-1])                            # [n_segments] Antivector-
    second_midpoints = 0.5 * (second[1:] + second[:-1])                         # [n_segments] Antivector-
    separations = first_midpoints[:, None] - second_midpoints[None, :]          # [n_segments, n_segments] Antivector-
    # Joining the separation vector with both segment displacements measures the signed spanned volume.
    volumes = separations & first_steps[:, None] & second_steps[None, :]        # [n_segments, n_segments] Antitrivector-
    lengths = separations.antinorm()                                            # [n_segments, n_segments] Scalar+
    return (volumes / lengths ** 3).sum() / (4 * np.pi)                         # [] Antitrivector-


# --- plumbing -------------------------------------------------------------------------
def signature(p: int, q: int) -> tuple[GAType, GAType, Extensor, Extensor]:
    """Choosing the first `p` positive and `q` negative directions yields their `Bivector` and `Even`
    types alongside the product of the positive directions and the signature's pseudoscalar."""
    names = positive_names[:p] + negative_names[:q]
    directions = Algebra.from_pqr(p, q, 0, names)
    Bivector = ga.gatype.bivector().derive.within(directions)
    Even = ga.gatype.even().derive.within(directions)
    positives = getattr(mv, positive_names[:p])
    pseudoscalar = getattr(mv, names)
    return Bivector, Even, positives, pseudoscalar
