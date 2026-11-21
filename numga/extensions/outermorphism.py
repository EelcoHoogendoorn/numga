"""Linear maps between grade-`1` vectors and grade-`(n - 1)` antivectors extend across all
homogeneous grades as metric-free outermorphisms (following A. M. Moya, V. V. Fernández, and
W. A. Rodrigues Jr., *Extensors in Geometric Algebras*, arXiv:math/0501558). Each side multiplies by
the exterior product native to its own space, the wedge `^` on vectors and the regressive join `&`
on antivectors: a vector map `t` extends upward in grade via `t(a ^ b) == t(a) ^ t(b)`, an
antivector map `T` extends downward via `T(p & q) == T(p) & T(q)`, and a correlation `C` between
antivectors and vectors exchanges products across the arrow so `C(p & q) == C(p) ^ C(q)`.

Because only `^`, `&`, and complementary blade readings enter, outermorphisms depend on no metric,
compose grade by grade (`t(s).outermorphism(G) == t.outermorphism(G)(s.outermorphism(G))`), and act
on the top exterior power (the pseudoscalar for vector maps, or the scalar for antivector maps) as
multiplication by `det`. In two dimensions, where grade `1` coincides with grade `n - 1`, the input
side selects the exterior product with more factors in the target grade, while relative orientation
distinguishes correlations from maps within one kind. Outermorphisms and the determinant are
introduced in `docs/tutorial/extensors/14-outermorphisms-and-the-determinant.md`.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache, reduce
from itertools import combinations
from operator import and_, xor

import numpy as np

from numga.extensor import Extensor, stack
from numga.gatype import GAType, Orientation
from numga.subspace import SubSpace
from numga.units import outermorphism_unit


@lru_cache(maxsize=None)
def _exterior(space: SubSpace, target: SubSpace) -> tuple[Callable, GAType, int]:
    if len(_grades(target)) != 1:
        raise TypeError(f"an outermorphism acts on one grade, not on {target}")
    algebra = space.algebra
    n = algebra.dimension
    (requested,) = _grades(target)
    inputs = _grades(space)
    volume = algebra.gatype.oriented((algebra.subspace.pseudoscalar(),), (Orientation.intrinsic,))
    # In 2D (where 1 == n - 1), pick whichever exterior product takes more factors to reach target.
    exteriors = [(xor, algebra.gatype.scalar(), requested)] * (inputs == {1}) + \
                [(and_, volume, n - requested)] * (inputs == {n - 1})
    return max(exteriors, key=lambda exterior: exterior[2])


def _grades(space: SubSpace) -> set[int]:
    return {space.algebra.grade(mask) for mask in space.masks}


@lru_cache(maxsize=None)
def _image_product(space: SubSpace, images: SubSpace, product: Callable, correlation: bool
                   ) -> tuple[Callable, GAType]:
    algebra = space.algebra
    wedge = (xor, algebra.gatype.scalar())
    join = (and_, algebra.gatype.oriented((algebra.subspace.pseudoscalar(),), (Orientation.intrinsic,)))
    # In 2D, correlations flip between wedge and join whereas maps within one kind keep the input product.
    if _grades(images) == _grades(space):
        return wedge if (product is xor) != correlation else join
    return wedge if _grades(images) == {1} else join


@Extensor.outermorphism.register(lambda t: t.is_exterior_map, unit=outermorphism_unit)
def outermorphism(operator: Extensor, target: GAType) -> Extensor:
    """Extending a vector map, antivector map, or correlation to `target` sends each `^` or `&`
    product of input elements to the corresponding `^` or `&` product of their images."""
    space, target_subspace = operator.axes[1], target.output_subspace
    product, unit_type, count = _exterior(space, target_subspace)
    correlation = operator.gatype.orientation is not operator.gatype.input_orientations[0]
    image_product, image_unit_type = _image_product(space, operator.axes[0], product, correlation)
    mv = operator.context.multivector
    unit = mv(unit_type, [1.0])
    image_unit = mv(image_unit_type, [1.0])

    basis = mv(operator.gatype.derive.axis(1), np.eye(len(space.masks)))  # [m]
    images = operator[..., None](basis)                                   # [..., m]
    tuples = list(combinations(range(len(space.masks)), count))
    sources = stack([reduce(product, [basis[i] for i in I], unit) for I in tuples])
    if not sources.output_subspace.same_support(target_subspace):
        raise TypeError(f"products of {count} elements of {space} span {sources.output_subspace}, not {target_subspace}")
    if sources.gatype.orientation is not target.orientation:
        raise TypeError(f"products of {count} elements of {space} are {sources.gatype.orientation.value}, not {target.orientation.value}")
    targets = stack([reduce(image_product, [images[..., i] for i in I], image_unit.broadcast_to(operator.shape)) for I in tuples], axis=-1)
    # Read each source blade's coefficient in target by pairing against its dual complement.
    readings = (target & sources.dual()) / (sources & sources.dual())     # [k] Scalar <- target
    return (targets * readings).sum(axis=-1)
