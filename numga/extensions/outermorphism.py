"""The outermorphism: a map between vectors and antivectors, extended to k-fold products.

A map t on vectors extends to bivectors by t(a ^ b) = t(a) ^ t(b), and on to every grade;
a map T on antivectors extends by the join, T(p & q) = T(p) & T(q). A correlation, from
antivectors to vectors or back, extends through the product of each side: a polarity C from
points to planes carries the join of two points to the meet of their planes,
C(p & q) = C(p) ^ C(q). Each side multiplies by the exterior product of its own space, `^` for
vectors and `&` for antivectors. No metric enters. On the top grade of an endomorphism the
extension is the determinant, and composition carries over grade by grade. This is the extension operator of A. M. Moya, V. V. Fernández and W. A. Rodrigues Jr.,
"Extensors in Geometric Algebras", arXiv:math/0501558.
"""

from __future__ import annotations

from functools import lru_cache, reduce
from itertools import combinations
from operator import and_, xor
from typing import Any, Callable

import numpy as np

from numga.extensor import Extensor, stack
from numga.gatype import GAType, Orientation
from numga.subspace import SubSpace


@lru_cache(maxsize=None)
def _exterior(space: SubSpace, target: SubSpace) -> tuple[Callable[[Any, Any], Any], GAType, int]:
    """The exterior product of the space that reaches the target grade, its unit, and the number of
    factors in the target grade. Vectors multiply by the wedge, whose unit is the scalar; antivectors
    by the join, whose unit is the pseudoscalar oriented within itself. When a space is both, in two
    dimensions, the product with more factors in the target grade is taken."""
    if len(_grades(target)) != 1:
        raise TypeError(f"an outermorphism acts on one grade, not on {target}")
    algebra = space.algebra
    n = algebra.dimension
    (requested,) = _grades(target)
    inputs = _grades(space)
    volume = algebra.gatype.oriented((algebra.subspace.pseudoscalar(),), (Orientation.intrinsic,))
    exteriors = [(xor, algebra.gatype.scalar(), requested)] * (inputs == {1}) + \
                [(and_, volume, n - requested)] * (inputs == {n - 1})
    return max(exteriors, key=lambda exterior: exterior[2])


def _grades(space) -> set[int]:
    return {space.algebra.grade(mask) for mask in space.masks}


@lru_cache(maxsize=None)
def _image_product(space: SubSpace, images: SubSpace, product: Callable[[Any, Any], Any], correlation: bool
                   ) -> tuple[Callable[[Any, Any], Any], GAType]:
    """The exterior product the images multiply by, and its unit: the product of the images' own space,
    `^` for vectors and `&` for antivectors. Where the two sides are of one grade, in two dimensions, the
    orientations decide: the inputs' own product for a map within one kind, the other for a correlation,
    whose extrinsic side meets and whose intrinsic side joins."""
    algebra = space.algebra
    wedge = (xor, algebra.gatype.scalar())
    join = (and_, algebra.gatype.oriented((algebra.subspace.pseudoscalar(),), (Orientation.intrinsic,)))
    if _grades(images) == _grades(space):
        return wedge if (product is xor) != correlation else join
    return wedge if _grades(images) == {1} else join


@Extensor.outermorphism.register(lambda t: t.is_exterior_map)
def outermorphism(value: Extensor, grade: GAType) -> Extensor:
    """The map on `grade` whose value on a product of input elements is the product of their images.

    A vector map extends through `^` and raises grade: t.outermorphism(Bivector)(a ^ b) ==
    t(a) ^ t(b). An antivector map extends through `&` and lowers it: in PGA3D
    T.outermorphism(Line)(p & q) == T(p) & T(q). A correlation extends through each side's own
    product: from points to planes, C.outermorphism(Line)(p & q) == C(p) ^ C(q). On the
    pseudoscalar of a vector map, or the scalar of an antivector map, it multiplies by det(t). When
    a space is both vectors and antivectors, in two dimensions, the product with more factors in
    `grade` is taken, and the images multiply by the same.
    """
    space, target = value.axes[1], grade.output_subspace
    product, unit_type, count = _exterior(space, target)
    correlation = value.gatype.orientation is not value.gatype.input_orientations[0]
    image_product, image_unit_type = _image_product(space, value.axes[0], product, correlation)
    mv = value.context.multivector
    unit = mv(unit_type, [1.0])
    image_unit = mv(image_unit_type, [1.0])

    basis = mv(value.gatype.derive.axis(1), np.eye(len(space.masks)))  # [m] the input basis
    images = value[..., None](basis)                                    # [..., m] its images
    tuples = list(combinations(range(len(space.masks)), count))
    sources = stack([reduce(product, [basis[i] for i in I], unit) for I in tuples])
    if not sources.output_subspace.same_support(target):
        raise TypeError(f"products of {count} elements of {space} span {sources.output_subspace}, not {target}")
    if sources.gatype.orientation is not grade.orientation:
        raise TypeError(f"products of {count} elements of {space} are {sources.gatype.orientation.value}, not {grade.orientation.value}")
    targets = stack([reduce(image_product, [images[..., i] for i in I], image_unit.broadcast_to(value.shape)) for I in tuples], axis=-1)
    # each source blade's coefficient in an element of the requested grade, read with its complement
    readings = (grade & sources.dual()) / (sources & sources.dual())    # [k] Scalar <- grade
    return (targets * readings).sum(axis=-1)
