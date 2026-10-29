"""The values of a subalgebra in an algebra that contains its generators.

An algebra contains another when every generator of the smaller one is among its own, by name and with the same
square: the algebra of space `x+y+z+` lies in its projective algebra `w0x+y+z+`. Its elements then have a place in
the larger algebra, and which place follows from their kind. An extrinsic element, built from planes by products and
meets, keeps its blades: a plane through the origin stays that plane, a rotor that rotor. An intrinsic element, built
from points by joins, is measured against the volume: it keeps its complement, the extrinsic element that completes
it to the unit volume, so a point of the algebra of space, an offset from its origin, becomes the projective point at
infinity of that offset, a direction, and the unit volume of space the unit volume of the projective algebra.

Both rules send each blade to one blade with a sign, so the map is a signed reordering of coefficients, fixed per
pair of types and applied without arithmetic beyond the signs.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from numga.algebra.bitops import mask_from_indices, permutation_sign
from numga.gatype import EMPTY_TRAITS, GAType, GATypeError, Orientation
from numga.operator.kernel import SymbolicKernel

if TYPE_CHECKING:
    from numga.algebra import Algebra
    from numga.backend import Context
    from numga.extensor import Extensor
    from numga.gatype import GATypeFactory


class Inclusion(NamedTuple):
    """The type of a subalgebra's values in the larger algebra, and for each axis, output first, where each of its
    coefficients comes from in the source's layout and with which sign."""
    gatype: GAType
    sources: tuple[tuple[int, ...], ...]
    signs: tuple[tuple[int, ...], ...]


def _included_blade(source: Algebra, target: Algebra, mask: int) -> tuple[int, int]:
    """The target blade of a source blade's generators, and the sign of their source order in the target's."""
    positions = [target.basis_names.index(source.basis_names[generator])
                 for generator in range(source.dimension) if mask >> generator & 1]
    return mask_from_indices(positions), permutation_sign(positions)


def _image(source: Algebra, target: Algebra, mask: int, orientation: Orientation) -> tuple[int, int]:
    """The target blade a canonical source blade goes to, and its sign: the same blade for an extrinsic element; for
    an intrinsic one the volume complement of the inclusion of the blade it completes to the source's unit volume."""
    if orientation is Orientation.extrinsic:
        return _included_blade(source, target, mask)
    completed, completed_sign = source.operator._volume_complement_inverse(mask)
    included, included_sign = _included_blade(source, target, completed)
    image, image_sign = target.operator._volume_complement(included)
    return image, completed_sign * included_sign * image_sign


@lru_cache(maxsize=None)
def inclusion(source: GAType, target: GATypeFactory) -> Inclusion:
    """How the values of a source type sit in the target algebra, cached per pair."""
    algebra, larger = source.algebra, target.algebra
    for name, square in zip(algebra.basis_names, algebra.signature):
        if name not in larger.basis_names or larger.signature[larger.basis_names.index(name)] != square:
            raise GATypeError(f"{algebra} is no subalgebra of {larger}: its generator {name} is not one of {larger}'s "
                              f"with the same square")
    axes, sources, signs = [], [], []
    for space, orientation in zip(source.subspaces, source.orientations):
        images = [_image(algebra, larger, mask, orientation) for mask in space.masks]
        axis = larger.subspace.from_masks(mask for mask, _ in images)
        position = {mask: index for index, mask in enumerate(axis.masks)}
        # Coefficient i of the source's layout lands at position[mask], with the two layouts' blade signs.
        order = sorted(range(len(images)), key=lambda i: position[images[i][0]])
        axes.append(axis)
        sources.append(tuple(order))
        signs.append(tuple(images[i][1] * space.signs[i] * axis.signs[position[images[i][0]]] for i in order))
    # The inclusion of extrinsic values preserves the geometric product, and with it the facts a value's type states.
    keeps_traits = not source.arity and source.orientation is Orientation.extrinsic
    gatype = target.oriented(axes, source.orientations, source.traits if keeps_traits else EMPTY_TRAITS, source.fields)
    return Inclusion(gatype, tuple(sources), tuple(signs))


def from_subalgebra(context: Context, value: Extensor) -> Extensor:
    """The value of a subalgebra's element in the algebra of `context`, as `MultivectorFactory.from_subalgebra`."""
    from numga.extensor import Extensor

    plan = inclusion(value.gatype, context.algebra.gatype)
    if context.is_exact and not value.context.is_exact:
        raise TypeError("an exact context takes only exact values from a subalgebra")
    if context.is_exact:
        xp, kernel = np, value.kernel.values
    elif value.context.is_exact:
        xp, kernel = context.xp, context.materialize(value.kernel)
    else:
        xp, kernel = context.xp, context.prepare_kernel(value.kernel)
    count = len(plan.sources)
    for axis, (sources, signs) in enumerate(zip(plan.sources, plan.signs)):
        # Each coefficient axis, counted from the end past the batch and site axes, reordered and signed.
        signs = xp.reshape(xp.asarray(signs, dtype=kernel.dtype), (len(signs),) + (1,) * (count - 1 - axis))
        kernel = xp.take(kernel, xp.asarray(sources, dtype=int), axis=axis - count) * signs
    if context.is_exact:
        kernel = SymbolicKernel(kernel)
    return Extensor._from_prepared_kernel(context, plan.gatype, kernel)
