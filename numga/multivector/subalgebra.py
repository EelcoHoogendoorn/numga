"""An algebra embeds as a subalgebra of a larger ambient algebra whenever every generator used by
an extensor's type appears in the ambient algebra with the same name and metric square, such as
Euclidean space `x+y+z+` inside projective space `w0x+y+z+` or spatial slices inside spacetime
`x+y+z+t-`. How each structural axis embeds depends on its orientation. Extrinsic axes represent
planes, meets, and rotors built by exterior and geometric products, so each canonical blade keeps
its generator names unchanged. Intrinsic axes represent points and joins measured against the
volume element, so each blade keeps its volume complement; a Euclidean point `x.dual()` (an offset
from the origin) maps to the projective ideal direction `x.dual()` at infinity, while the Euclidean
unit volume maps to the projective unit volume.

Because both rules send each source basis blade to a single target basis blade up to sign,
including any value, linear map, or multilinear form across all its structural axes reduces to a
signed index gather per axis with no floating-point arithmetic beyond sign flips. `inclusion`
computes and caches this axis-by-axis gather plan once per `(source_gatype, target_gatype_factory)`
pair, preserving certified traits on extrinsic nullary values whose geometric products are
preserved by generator inclusion. Subalgebra inclusion across contexts is also summarized in
`docs/internals.md`.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, NamedTuple

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
    """Cached per-axis index gathers `sources` and `signs` map coefficients of a subalgebra
    extensor onto `gatype` in the target algebra."""

    gatype: GAType
    sources: tuple
    signs: tuple


def _included_blade(source: Algebra, target: Algebra, mask: int) -> tuple[int, int]:
    positions = [target.basis_names.index(source.basis_names[generator])
                 for generator in range(source.dimension) if mask >> generator & 1]
    return mask_from_indices(positions), permutation_sign(positions)


def _image(source: Algebra, target: Algebra, mask: int, orientation: Orientation) -> tuple[int, int]:
    if orientation is Orientation.extrinsic:
        return _included_blade(source, target, mask)
    # Intrinsic blades embed by transporting their volume complements as extrinsic blades.
    completed, completed_sign = source.operator._volume_complement_inverse(mask)
    included, included_sign = _included_blade(source, target, completed)
    image, image_sign = target.operator._volume_complement(included)
    return image, completed_sign * included_sign * image_sign


@lru_cache(maxsize=None)
def inclusion(source: GAType, target: GATypeFactory) -> Inclusion:
    """Each structural axis of `source` maps to its extrinsic or complement-preserving counterpart
    in `target`, caching the gather indices and signs."""
    algebra, larger = source.algebra, target.algebra
    # Intrinsic axes depend on the generators of their volume complements rather than their raw masks.
    generators = 0
    for space, orientation in zip(source.subspaces, source.orientations):
        for mask in space.masks:
            if orientation is Orientation.intrinsic:
                mask, _ = algebra.operator._volume_complement_inverse(mask)
            generators |= mask
    for generator, (name, square) in enumerate(zip(algebra.basis_names, algebra.signature)):
        if generators >> generator & 1 and (
                name not in larger.basis_names or larger.signature[larger.basis_names.index(name)] != square):
            raise GATypeError(f"{algebra} is no subalgebra of {larger}: its generator {name} is not one of {larger}'s "
                              f"with the same square")
    axes, sources, signs = [], [], []
    for space, orientation in zip(source.subspaces, source.orientations):
        images = [_image(algebra, larger, mask, orientation) for mask in space.masks]
        axis = larger.subspace.from_masks(mask for mask, _ in images)
        position = {mask: index for index, mask in enumerate(axis.masks)}
        # Coefficient i of the source layout lands at position[mask] with the product of both layouts' signs.
        order = sorted(range(len(images)), key=lambda i: position[images[i][0]])
        axes.append(axis)
        sources.append(tuple(order))
        signs.append(tuple(images[i][1] * space.signs[i] * axis.signs[position[images[i][0]]] for i in order))
    # Extrinsic nullary inclusions preserve geometric products and therefore keep certified traits.
    keeps_traits = not source.arity and source.orientation is Orientation.extrinsic
    gatype = target.oriented(axes, source.orientations, source.traits if keeps_traits else EMPTY_TRAITS, source.fields)
    return Inclusion(gatype, tuple(sources), tuple(signs))


def include_from_algebra(context: Context, extensor: Extensor) -> Extensor:
    """Reordering and signing each structural axis according to `inclusion` embeds `extensor` from
    a subalgebra into `context`."""
    from numga.extensor import Extensor

    plan = inclusion(extensor.gatype, context.algebra.gatype)
    if context.is_exact and not extensor.context.is_exact:
        raise TypeError("an exact context takes only exact values from a subalgebra")
    if context.is_exact:
        xp, kernel = np, extensor.kernel.values
    elif extensor.context.is_exact:
        xp, kernel = context.xp, context.materialize(extensor.kernel)
    else:
        xp, kernel = context.xp, context.prepare_kernel(extensor.kernel)
    count = len(plan.sources)
    for axis, (sources, signs) in enumerate(zip(plan.sources, plan.signs)):
        # Structural axes sit after batch and site axes, so negative indices address them directly.
        signs = xp.reshape(xp.asarray(signs, dtype=kernel.dtype), (len(signs),) + (1,) * (count - 1 - axis))
        kernel = xp.take(kernel, xp.asarray(sources, dtype=int), axis=axis - count) * signs
    if context.is_exact:
        kernel = SymbolicKernel(kernel)
    return Extensor._from_prepared_kernel(context, plan.gatype, kernel)
