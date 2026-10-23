"""Binding over field axes.

A field axis ranges over sites as well as blades. Binding treats every site axis as a batch axis
with a label: a field bound into a field axis pairs its sites with the axis's and sums over them,
a field bound into a axis over blades alone is acted on site by site, and composition preserves
the sites of the argument map's inputs. The labels are aligned as trailing batch axes, the ordinary
bind runs over them, and the summed ones are reduced.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, NamedTuple

import numga

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
