"""Binding over field axes.

A field axis ranges over sites as well as blades. Binding treats every site axis as a batch axis
with a label: a field bound into a field axis pairs its sites with the axis's and sums over them,
a field bound into a axis over blades alone is acted on site by site, and composition preserves
the sites of the argument map's inputs. The labels are aligned as trailing batch axes, and the
ordinary bind runs over them. Where sites are summed, the values contract by their labels instead,
each summed label within the contraction of the operand that carries it, so that no term per pair
of sites is ever held and a dense field map applies as a matrix product.
"""

from __future__ import annotations

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
    """Bind operands, by input from 0, into a target where either side has field axes."""
    axes = tuple(sorted(operands))
    plan = _plan(target.gatype, tuple((axis, operands[axis].gatype) for axis in axes))
    lowered: dict[tuple[int, tuple[int, ...]], Extensor] = {}

    def lower(value: Extensor, positions: tuple[int, ...]) -> Extensor:
        # One lowered view per value and placement keeps the evidence that two axes hold one value.
        key = (id(value), positions)
        if key not in lowered:
            lowered[key] = _aligned(value, positions, plan.sizes)
        return lowered[key]

    plain_target = lower(target, plan.target)
    plain_operands = {axis: lower(operands[axis], positions) for axis, positions in zip(axes, plan.operands)}
    if plan.summed and not target.context.is_exact:
        result = _summed(plain_target, plain_operands, plan.summed)
        return type(result)._from_prepared_kernel(result.context, result.gatype.derive.with_fields(plan.fields), result.kernel)
    if len(plain_operands) == target.arity:
        result = plain_target(*(plain_operands[axis] for axis in range(target.arity)))
    else:
        result = plain_target.bind(plain_operands)

    kernel, gatype = result.kernel, result.gatype
    if plan.summed:
        ndim = result.ndim
        kernel = result.context.xp.sum(kernel, axis=tuple(range(ndim - plan.summed, ndim)))
        gatype = gatype.derive.structural
    return type(result)._from_prepared_kernel(result.context, gatype.derive.with_fields(plan.fields), kernel)


def _summed(target: Extensor, operands: dict[int, Extensor], summed: int) -> Extensor:
    """The bind of aligned values over blades alone, its last `summed` batch axes, the paired sites,
    summed within the contraction of the values, so that a field map's product with a field never
    holds a term per pair of sites."""
    context = binding_context(target.context, tuple(operand.context for operand in operands.values()))
    normalized = tuple(sorted(operands.items()))
    values = tuple(context.lower(value) for value in (target, *(operand for _, operand in normalized)))
    result_type, contract = _contraction(
        context.xp, target.gatype, tuple((axis, operand.gatype) for axis, operand in normalized),
        nullary_identity_groups(normalized), tuple(value._kernel.shape for value in values),
        tuple(value.ndim for value in values), summed,
    )
    return type(target)._from_prepared_kernel(context, result_type, contract(tuple(value._kernel for value in values)))


@lru_cache(maxsize=None)
def _contraction(xp, target: GAType, operands: tuple[tuple[int, GAType], ...], groups: tuple[tuple[int, ...], ...],
                 shapes: tuple[tuple[int, ...], ...], batches: tuple[int, ...], summed: int):
    """The type of a bind and the contraction of the target's kernel with its operands' kernels, of
    these shapes with these counts of batch axes, the last `summed` batch axes summed: planned once
    per signature."""
    plan = BindingPlan.from_types(target, operands, groups)
    restrict = plan.result_subspaces[0] is not target.output_subspace
    indices = xp.asarray(plan.output_indices, dtype=int) if restrict else None

    def prepare(kernels):
        kernel = xp.take(kernels[0], indices, axis=batches[0]) if restrict else kernels[0]
        return (kernel,) + tuple(
            transform_output(xp, operand, binding.operand_gatype, binding.transform)
            for operand, binding in zip(kernels[1:], plan.bindings)
        )
    # The restriction and the conversions change the length of one structural axis each.
    lengths = [len(plan.output_indices) if restrict else None] + [
        None if binding.transform.kind is AxisTransformKind.EXACT else len(binding.transform.target)
        for binding in plan.bindings
    ]
    shapes = tuple(
        shape if length is None else shape[:batch] + (length,) + shape[batch + 1:]
        for shape, batch, length in zip(shapes, batches, lengths)
    )
    ndim = max(batches)
    # Batch axes align from the right and take the first labels; the structural axes follow, an
    # operand's output under the label of the input it is bound into.
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
    # Every axis of length one is squeezed from its value, so that each contraction meets only true
    # extents and can run as a matrix product.
    sizes: dict[int, int] = {}
    squeezed = []
    for shape, batch, axes in zip(shapes, batches, structural):
        kept = [(label, size) for label, size in zip(list(range(ndim - batch, ndim)) + axes, shape) if size != 1]
        sizes.update(kept)
        squeezed.append((tuple(size for _, size in kept), [label for label, _ in kept]))
    result_labels = list(range(ndim - summed)) + output
    present = [label for label in result_labels if label in sizes]
    # The operands contract into the target one at a time, from the last axis to the first, each
    # label summed as soon as no later operand and not the result carries it.
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

    def contract(kernels):
        arrays = [xp.reshape(array, kept) for array, (kept, _) in zip(prepare(kernels), squeezed)]
        result = arrays[0]
        for index, apply in steps:
            result = apply(result, arrays[index])
        return xp.reshape(xp.transpose(result, permutation), shape)
    return TypeRules.bind(plan).derive.structural, contract


class _Plan(NamedTuple):
    """Where each value's site axes go among the trailing batch axes of a binding, by label."""

    target: tuple[int, ...]
    operands: tuple[tuple[int, ...], ...]
    sizes: tuple[int, ...]
    summed: int
    fields: tuple[tuple[int, int], ...]


@lru_cache(maxsize=None)
def _plan(target: GAType, operands: tuple[tuple[int, GAType], ...]) -> _Plan:
    labels = _Labels()
    target_sites = labels.of(target)
    bound = dict(operands)
    operand_sites: list[dict[int, int]] = []
    # The sites of the output, and of every field acted on site by site, are one set of sites.
    output = [target_sites[0]] if 0 in target_sites else []
    inputs: list[int | None] = []
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
                # A map on blades alone acts site by site: the axis's sites pass on to its input.
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


def _aligned(value: Extensor, positions: tuple[int, ...], sizes: tuple[int, ...]) -> Extensor:
    """The value over blades alone, its site axes moved to their places among the trailing batch
    axes of the frame and the frame's other places of length one."""
    gatype = value.gatype.derive.plain
    if not positions and (value.context.is_exact or not value.ndim):
        return value if gatype is value.gatype else type(value)._from_prepared_kernel(value.context, gatype, value.kernel)
    xp = value.context.xp
    batch, coefficients = value.ndim, len(value.axes)
    kernel = value.kernel
    order = sorted(range(len(positions)), key=positions.__getitem__)
    if order != list(range(len(positions))):
        kernel = xp.transpose(
            kernel,
            tuple(range(batch)) + tuple(batch + site for site in order)
            + tuple(range(batch + len(positions), batch + len(positions) + coefficients)),
        )
    present = set(positions)
    frame = tuple(size if position in present else 1 for position, size in enumerate(sizes))
    shape = value.shape + frame + kernel.shape[kernel.ndim - coefficients:]
    if shape != kernel.shape:
        kernel = xp.reshape(kernel, shape)
    return type(value)._from_prepared_kernel(value.context, gatype, kernel)


class _Labels:
    """Site axes by label, with the labels that are paired joined into one."""

    def __init__(self) -> None:
        self.parent: list[int] = []
        self.sizes: list[int] = []

    def of(self, gatype: GAType) -> dict[int, int]:
        """A fresh label for every field axis of a type."""
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
    """The sum where either side has field axes. A value or map without sites is the same at every
    site, as it is under application: added to a map from a field to a field over the same sites, a map
    acting at every site is that map on the site diagonal, so that `(A + B)(f) == A(f) + B(f)`."""
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


def _on_diagonal_of(value: Extensor, other: Extensor) -> Extensor:
    """A map acting at every site, read on the site diagonal where the other's input ranges over the
    sites of its output and the map's input does not."""
    sites, own = dict(other.gatype.fields), dict(value.gatype.fields)
    missing = [axis for axis in sites if axis and axis not in own]
    if not missing:
        return value
    if missing != [1] or sites.get(0) != sites[1] or [axis for axis in own if axis] or value.arity < 1:
        raise TypeError(f"{value.gatype.signature} has no reading on the sites of {other.gatype.signature}")
    if 0 not in own:
        # The same map at every site.
        value = other.context.lower(value)
        value = value[..., None].broadcast_to(value.shape + (sites[0],)).field()
    return value.on_diagonal()


def site_by_site(implementation, operand_count: int):
    """An implementation over blades, run at every site of fields of values: each field's sites read
    as one trailing batch axis, shared by all of them, and a result whose last batch axis holds them
    read as a field again."""
    def run(*arguments, **kwargs):
        operands = arguments[:operand_count]
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

        return refield(implementation(*lowered, *arguments[operand_count:], **kwargs))

    return run
