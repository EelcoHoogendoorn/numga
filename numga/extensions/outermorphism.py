"""The outermorphism: a map on vectors or on antivectors, extended to k-fold products.

A map t on vectors extends to bivectors by t(a ^ b) = t(a) ^ t(b), and on to every grade;
a map T on antivectors extends by the join, T(p & q) = T(p) & T(q). Both are the extension
of a linear map along the exterior product of its own space, which is the product that
multiplies that space's elements: `^` for vectors, `&` for antivectors. No metric enters.
On the top grade the extension is the determinant, and composition carries over grade by
grade. This is the extension operator of A. M. Moya, V. V. Fernández and W. A. Rodrigues Jr.,
"Extensors in Geometric Algebras", arXiv:math/0501558.
"""

from __future__ import annotations

from functools import reduce
from itertools import combinations
from operator import and_, xor

import numpy as np

from numga.extensor import Extensor, stack
from numga.gatype import GAType


def _grades(space) -> set[int]:
    return {space.algebra.grade(mask) for mask in space.masks}


def _exterior_map(t: GAType) -> bool:
    """A map from vectors to vectors, or from antivectors to antivectors."""
    if t.arity != 1:
        return False
    n = t.algebra.dimension
    inputs, outputs = _grades(t.input_subspaces[0]), _grades(t.output_subspace)
    return inputs == outputs and inputs in ({1}, {n - 1})


@Extensor.outermorphism.register(_exterior_map)
def outermorphism(value: Extensor, grade: GAType) -> Extensor:
    """The map on `grade` whose value on a product of input elements is the product of their images.

    A vector map extends through `^` and raises grade: t.outermorphism(Bivector)(a ^ b) ==
    t(a) ^ t(b). An antivector map extends through `&` and lowers it: in PGA3D
    T.outermorphism(Line)(p & q) == T(p) & T(q). On the pseudoscalar of a vector map, or the
    scalar of an antivector map, it multiplies by det(t). When a space is both vectors and
    antivectors, in two dimensions, the product with more factors in `grade` is taken.
    """
    algebra, space = value.algebra, value.axes[1]
    mv = value.context.multivector
    n = algebra.dimension
    target = grade.output_subspace
    if len(_grades(target)) != 1:
        raise TypeError(f"an outermorphism acts on one grade, not on {target}")
    (requested,) = _grades(target)
    inputs = _grades(space)
    # (product, its unit, number of factors in the requested grade) for each exterior product of the space
    exteriors = [(xor, mv.scalar([1.0]), requested)] * (inputs == {1}) + \
                [(and_, mv(algebra.subspace.pseudoscalar(), [1.0]), n - requested)] * (inputs == {n - 1})
    product, unit, count = max(exteriors, key=lambda exterior: exterior[2])

    basis = mv(space, np.eye(len(space.masks)))                         # [m] the input basis
    images = value[..., None](basis)                                    # [..., m] its images
    tuples = list(combinations(range(len(space.masks)), count))
    sources = stack([reduce(product, [basis[i] for i in I], unit) for I in tuples])
    if not sources.output_subspace.same_support(target):
        raise TypeError(f"products of {count} elements of {space} span {sources.output_subspace}, not {target}")
    targets = stack([reduce(product, [images[..., i] for i in I], unit.broadcast_to(value.shape)) for I in tuples], axis=-1)
    # each source blade's coefficient in an element of the requested grade, read with its complement
    readings = (grade & sources.dual()) / (sources & sources.dual())    # [k] Scalar <- grade
    return (targets * readings).sum(axis=-1)
