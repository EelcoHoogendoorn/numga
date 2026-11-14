"""A field axis ranges over discrete sites alongside its blade subspace. Binding assigns a site
label to every field axis and unifies paired axes: binding a field into a field-valued input pairs
their sites and sums over them, binding a field into a blade-only input acts site by site, and
composing maps carries the argument map's input sites into the result. Before blade-level binding
runs, each participant aligns its site axes into a shared trailing batch frame with size-one axes
wherever a site label is absent.

When concrete operands sum over paired sites, materializing an intermediate array with one entry per
pair of sites would square memory traffic. Summed field bindings instead contract batch, site, and
structural blade axes together across pairwise `einsum` steps, squeezing size-one axes and summing
each contracted site axis as soon as the last operand carrying it enters so applying a dense field
map reduces directly to a matrix product.

An extensor without site axes represents the same algebraic element at every site. Adding a
site-free or output-only field map to a `(sites <- sites)` map promotes it onto the site diagonal
via `on_diagonal()` so `(A + B)(f) == A(f) + B(f)`. Blade-level extensions wrapped with
`site_by_site` align the site axes of their leading operands onto a single trailing batch axis and
restore `.field()` on results matching that site count. The user-facing model of fields, coupled
maps, derivatives, and sparse storage is developed in `docs/fields.md`.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache, partial
from math import prod
from string import ascii_letters
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

import numga
from numga.backend.context import binding_context
from numga.backend.dense import transform_output
from numga.binding import AxisTransformKind, BindingPlan, TypeRules, nullary_identity_groups

if TYPE_CHECKING:
    from numga.gatype import GAType

    from .extensor import Extensor


def bind_fields(target: Extensor, operands: dict[int, Extensor]) -> Extensor:
    """Binding operands by zero-based input index aligns shared site axes and contracts summed site
    pairs."""
    axes = tuple(sorted(operands))
    plan = _plan(target.gatype, tuple((axis, operands[axis].gatype) for axis in axes))
    lowered = {}

    def lower(extensor: Extensor, positions: tuple[int, ...]) -> Extensor:
        # Caching by object identity and site placement preserves shared-operand identity groups.
        key = (id(extensor), positions)
        if key not in lowered:
            lowered[key] = _aligned(extensor, positions, plan.sizes)
        return lowered[key]

    plain_target = lower(target, plan.target)
    plain_operands = {axis: lower(operands[axis], positions) for axis, positions in zip(axes, plan.operands)}
    if plan.n_summed and not target.context.is_exact:
        result = _summed(plain_target, plain_operands, plan.n_summed)
        return type(result)._from_prepared_kernel(result.context, result.gatype.derive.with_fields(plan.fields), result.kernel)
    if len(plain_operands) == target.arity:
        result = plain_target(*(plain_operands[axis] for axis in range(target.arity)))
    else:
        result = plain_target.bind(plain_operands)

    kernel, gatype = result.kernel, result.gatype
    if plan.n_summed:
        ndim = result.ndim
        kernel = result.context.xp.sum(kernel, axis=tuple(range(ndim - plan.n_summed, ndim)))
        gatype = gatype.derive.structural
    return type(result)._from_prepared_kernel(result.context, gatype.derive.with_fields(plan.fields), kernel)


def _summed(target: Extensor, operands: dict[int, Extensor], n_summed: int) -> Extensor:
    context = binding_context(target.context, tuple(operand.context for operand in operands.values()))
    normalized = tuple(sorted(operands.items()))
    aligned = tuple(context.lower(extensor) for extensor in (target, *(operand for _, operand in normalized)))
    result_type, contract = _contraction(
        context.xp, target.gatype, tuple((axis, operand.gatype) for axis, operand in normalized),
        nullary_identity_groups(normalized), tuple(extensor._kernel.shape for extensor in aligned),
        tuple(extensor.ndim for extensor in aligned), n_summed,
    )
    return type(target)._from_prepared_kernel(context, result_type, contract(tuple(extensor._kernel for extensor in aligned)))


@lru_cache(maxsize=None)
def _contraction(
    xp, target: GAType, operands: tuple, groups: tuple,
    shapes: tuple, batches: tuple[int, ...], n_summed: int,
) -> tuple[GAType, Callable]:
    plan = BindingPlan.from_types(target, operands, groups)
    restrict = plan.result_subspaces[0] is not target.output_subspace
    indices = xp.asarray(plan.output_indices, dtype=int) if restrict else None

    def prepare(kernels: tuple) -> tuple:
        kernel = xp.take(kernels[0], indices, axis=batches[0]) if restrict else kernels[0]
        return (kernel,) + tuple(
            transform_output(xp, operand, binding.operand_gatype, binding.transform)
            for operand, binding in zip(kernels[1:], plan.bindings)
        )
    # Output restriction and input basis conversions each replace the first structural axis length.
    lengths = [len(plan.output_indices) if restrict else None] + [
        None if binding.transform.kind is AxisTransformKind.EXACT else len(binding.transform.target)
        for binding in plan.bindings
    ]
    shapes = tuple(
        shape if length is None else shape[:batch] + (length,) + shape[batch + 1:]
        for shape, batch, length in zip(shapes, batches, lengths)
    )
    ndim = max(batches)
    # Batch axes align from the right under leading labels; each bound operand shares its target input label.
    labels = iter(range(ndim, len(ascii_letters)))
    target_labels = [next(labels) for _ in range(1 + plan.target_gatype.arity)]
    structural = [target_labels]
    output = [target_labels[0]]
    bound = {binding.axis: binding for binding in plan.bindings}
    for axis in range(plan.target_gatype.arity):
        binding = bound.get(axis)
        if binding is None:
            output.append(target_labels[axis + 1])
            continue
        inputs = [next(labels) for _ in range(binding.operand_gatype.arity)]
        structural.append([target_labels[axis + 1]] + inputs)
        output.extend(inputs)
    # Squeezing size-1 axes exposes true extents so pairwise contractions dispatch to matrix products.
    sizes = {}
    squeezed = []
    for shape, batch, axes in zip(shapes, batches, structural):
        kept = [(label, size) for label, size in zip(list(range(ndim - batch, ndim)) + axes, shape) if size != 1]
        sizes.update(kept)
        squeezed.append((tuple(size for _, size in kept), [label for label, _ in kept]))
    result_labels = list(range(ndim - n_summed)) + output
    present = [label for label in result_labels if label in sizes]
    # Operands contract from last to first, summing each label once neither the result nor any remaining operand needs it.
    labels, steps = squeezed[0][1], []
    order = list(range(len(squeezed) - 1, 0, -1))
    for step, index in enumerate(order):
        operand_labels = squeezed[index][1]
        needed = set(present).union(*(squeezed[later][1] for later in order[step + 1:]))
        joined = [label for label in dict.fromkeys(labels + operand_labels) if label in needed]
        expression = "".join(ascii_letters[label] for label in labels) + ","
        expression += "".join(ascii_letters[label] for label in operand_labels) + "->"
        expression += "".join(ascii_letters[label] for label in joined)
        work = prod(sizes[label] for label in set(labels + operand_labels))
        # NumPy's planned contraction reaches BLAS; on small arrays its planning costs more than it saves.
        steps.append((index, partial(np.einsum, expression, optimize=work > 4096) if xp is np else partial(xp.einsum, expression)))
        labels = joined
    permutation = tuple(labels.index(label) for label in present)
    shape = tuple(sizes.get(label, 1) for label in result_labels)

    def contract(kernels: tuple):
        arrays = [xp.reshape(array, kept) for array, (kept, _) in zip(prepare(kernels), squeezed)]
        result = arrays[0]
        for index, apply in steps:
            result = apply(result, arrays[index])
        return xp.reshape(xp.transpose(result, permutation), shape)
    return TypeRules.bind(plan).derive.structural, contract


class _Plan(NamedTuple):
    target: tuple[int, ...]
    operands: tuple
    sizes: tuple[int, ...]
    n_summed: int
    fields: tuple


@lru_cache(maxsize=None)
def _plan(target: GAType, operands: tuple) -> _Plan:
    labels = _Labels()
    target_sites = labels.of(target)
    bound = dict(operands)
    operand_sites = []
    # Output sites and all operands acted on site by site share a single site label.
    output = [target_sites[0]] if 0 in target_sites else []
    inputs = []
    for axis in range(1, target.arity + 1):
        operand = bound.get(axis - 1)
        if operand is None:
            inputs.append(target_sites.get(axis))
            continue
        sites = labels.of(operand)
        operand_sites.append(sites)
        own_inputs = [sites.get(inner) for inner in range(1, operand.arity + 1)]
        if axis in target_sites:
            if 0 in sites:
                labels.join(target_sites[axis], sites[0])
            elif operand.arity == 1 and own_inputs[0] is None:
                # A blade-only unary map acts site by site, passing the target axis's sites to its own input.
                own_inputs = [target_sites[axis]]
            else:
                raise TypeError(
                    f"axis {axis - 1} ranges over {labels.sizes[target_sites[axis]]} sites; "
                    f"a {operand.signature} does not"
                )
        elif 0 in sites:
            output.append(sites[0])
        inputs.extend(own_inputs)
    for label in output[1:]:
        labels.join(output[0], label)

    kept = [labels.find(label) for label in output[:1]] + [labels.find(label) for label in inputs if label is not None]
    if len(set(kept)) != len(kept):
        raise TypeError("a binding that gives two output axes the same sites needs their diagonal")
    every = [*target_sites.values()] + [label for sites in operand_sites for label in sites.values()]
    summed = sorted({labels.find(label) for label in every} - set(kept))
    frame = {label: position for position, label in enumerate(kept + summed)}

    def positions(sites: dict[int, int]) -> tuple[int, ...]:
        return tuple(frame[labels.find(sites[axis])] for axis in sorted(sites))

    output_axes = ([0] if output else []) + [axis + 1 for axis, label in enumerate(inputs) if label is not None]
    return _Plan(
        positions(target_sites), tuple(positions(sites) for sites in operand_sites),
        tuple(labels.sizes[label] for label in kept + summed), len(summed),
        tuple((axis, labels.sizes[label]) for axis, label in zip(output_axes, kept)),
    )


def _aligned(extensor: Extensor, positions: tuple[int, ...], sizes: tuple[int, ...]) -> Extensor:
    gatype = extensor.gatype.derive.plain
    if not positions and (extensor.context.is_exact or not extensor.ndim):
        return extensor if gatype is extensor.gatype else type(extensor)._from_prepared_kernel(extensor.context, gatype, extensor.kernel)
    xp = extensor.context.xp
    batch, coefficients = extensor.ndim, len(extensor.axes)
    kernel = extensor.kernel
    order = sorted(range(len(positions)), key=positions.__getitem__)
    if order != list(range(len(positions))):
        kernel = xp.transpose(
            kernel,
            tuple(range(batch)) + tuple(batch + site for site in order)
            + tuple(range(batch + len(positions), batch + len(positions) + coefficients)),
        )
    present = set(positions)
    frame = tuple(size if position in present else 1 for position, size in enumerate(sizes))
    shape = extensor.shape + frame + kernel.shape[kernel.ndim - coefficients:]
    if shape != kernel.shape:
        kernel = xp.reshape(kernel, shape)
    return type(extensor)._from_prepared_kernel(extensor.context, gatype, kernel)


class _Labels:
    def __init__(self) -> None:
        self.parent = []
        self.sizes = []

    def of(self, gatype: GAType) -> dict[int, int]:
        sites = {}
        for axis, count in gatype.fields:
            sites[axis] = len(self.parent)
            self.parent.append(len(self.parent))
            self.sizes.append(count)
        return sites

    def find(self, label: int) -> int:
        while self.parent[label] != label:
            label = self.parent[label]
        return label

    def join(self, first: int, second: int) -> None:
        first, second = self.find(first), self.find(second)
        if self.sizes[first] != self.sizes[second]:
            raise ValueError(f"cannot pair {self.sizes[first]} sites with {self.sizes[second]}")
        self.parent[second] = first


def add_fields(left: Extensor, right: Extensor) -> Extensor:
    """Field addition aligns shared site axes and promotes site-free or output-only maps onto the
    site diagonal when paired with a `(sites <- sites)` map, preserving `(A + B)(f) == A(f) + B(f)`."""
    left, right = _on_diagonal_of(left, right), _on_diagonal_of(right, left)
    sites = dict(left.gatype.fields)
    for axis, count in right.gatype.fields:
        if sites.setdefault(axis, count) != count:
            raise ValueError(f"cannot add {sites[axis]} sites to {count} in axis {axis}")
    fields = tuple(sorted(sites.items()))
    position = {axis: index for index, (axis, _) in enumerate(fields)}
    sizes = tuple(count for _, count in fields)
    result = (
        _aligned(left, tuple(position[axis] for axis, _ in left.gatype.fields), sizes)
        + _aligned(right, tuple(position[axis] for axis, _ in right.gatype.fields), sizes)
    )
    return type(result)._from_prepared_kernel(result.context, result.gatype.derive.with_fields(fields), result.kernel)


def _on_diagonal_of(extensor: Extensor, other: Extensor) -> Extensor:
    sites, own = dict(other.gatype.fields), dict(extensor.gatype.fields)
    missing = [axis for axis in sites if axis and axis not in own]
    if not missing:
        return extensor
    if missing != [1] or sites.get(0) != sites[1] or [axis for axis in own if axis] or extensor.arity < 1:
        raise TypeError(f"{extensor.gatype.signature} has no reading on the sites of {other.gatype.signature}")
    if 0 not in own:
        # Broadcast a site-free map across all output sites before taking its site diagonal.
        extensor = other.context.lower(extensor)
        extensor = extensor[..., None].broadcast_to(extensor.shape + (sites[0],)).field()
    return extensor.on_diagonal()


def site_by_site(implementation: Callable, n_operands: int) -> Callable:
    """Leading field operands align onto one trailing site axis before blade-level evaluation, and
    outputs retaining that axis return as fields."""
    def run(*arguments, **kwargs):
        operands = arguments[:n_operands]
        counts = {sites for operand in operands for _, sites in operand.gatype.fields}
        if len(counts) != 1:
            raise ValueError(f"fields acted on site by site need one number of sites; got {sorted(counts)}")
        (count,) = counts
        lowered = tuple(_aligned(operand, (0,) * bool(operand.gatype.fields), (count,)) for operand in operands)

        def refield(result):
            if isinstance(result, tuple):
                return tuple(refield(item) for item in result)
            if isinstance(result, numga.extensor.Extensor) and result.ndim and result.shape[-1] == count:
                return result.field()
            return result

        return refield(implementation(*lowered, *arguments[n_operands:], **kwargs))

    return run
