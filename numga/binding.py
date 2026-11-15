"""Binding feeds operands into selected input axes of a target extensor to evaluate linear maps,
partially evaluate multilinear forms, or compose operators. Each bind is planned from `GAType`
metadata alone without touching runtime coefficient arrays. For each bound input slot,
`AxisTransform` compares the operand's output `SubSpace` against the target's required input
`SubSpace`. Equal layouts (`EXACT`), blade reorderings or sign flips (`RELAYOUT`), and zero-padded
embeddings into a wider support (`EMBED`) are lossless and accepted implicitly during binding,
whereas support projections (`PROJECT`), partially overlapping supports (`REFRAME`), and
cross-algebra pairs (`INCOMPATIBLE`) are rejected unless requested through an explicit `cast`.

`BindingPlan` combines these per-axis transforms, verifies that every operand's output orientation
matches the target input orientation, and splices the remaining input axes in input-axis order. An
unbound target input stays in place, a nullary operand consumes its slot, and a positive-arity
operand splices its own input subspaces and orientations into the slot it fills
(`result_input_splices`). Because `from_types` caches plans by static type signature and never
retains array values, repeated calls with the same operand types share one plan across numerical
backends and traced executions.

When the same nullary `Extensor` object enters multiple input slots simultaneously (such as a versor
`v` bound into both outer slots of a ternary sandwich `T(v, x, v)`), `nullary_identity_groups`
records those shared-argument groups on the plan. `bind_subspaces` and `TypeRules.bind` then
propagate certified output-support restrictions (narrowing `execution_gatype` and `output_indices`
before contraction) together with algebraic traits onto the resulting `GAType`. Binding plans,
lossless coordinate conversions, and repeated-argument trait propagation are detailed in
`docs/internals.md` and `docs/tutorial/extensors/02-binding.md`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from functools import lru_cache
from typing import Mapping, Protocol

from numga.gatype import GAType, GATypeError, Orientation
from numga.gatype.propagation import bind_subspaces, bind_traits
from numga.subspace import SubSpace


class HasGAType(Protocol):
    gatype: GAType


class AxisTransformKind(str, Enum):
    EXACT = "exact"
    RELAYOUT = "relayout"
    EMBED = "embed"
    PROJECT = "project"
    REFRAME = "reframe"
    INCOMPATIBLE = "incompatible"


@dataclass(frozen=True, slots=True)
class AxisTransform:
    """For each target blade slot `j`, `target_from_source[j]` stores the matching source blade
    index or `None` for a structural zero, while `coordinate_matrix` folds in relative blade signs."""

    source: SubSpace
    target: SubSpace
    kind: AxisTransformKind
    target_from_source: tuple[int | None, ...]

    @classmethod
    def plan(cls, source: SubSpace, target: SubSpace) -> AxisTransform:
        """Comparing `source` and `target` blade supports classifies and indexes the coordinate
        mapping from `source` into `target`."""
        return _plan_axis_transform(source, target)

    @property
    def is_compatible(self) -> bool:
        return self.kind is not AxisTransformKind.INCOMPATIBLE

    @property
    def is_lossless(self) -> bool:
        return self.kind in {
            AxisTransformKind.EXACT,
            AxisTransformKind.RELAYOUT,
            AxisTransformKind.EMBED,
        }

    @property
    def is_implicit_bind_compatible(self) -> bool:
        return self.is_lossless

    @property
    def coordinate_matrix(self) -> tuple:
        """The target-by-source integer matrix folds relative blade orientation signs into the
        coordinate permutation, embedding, or projection."""
        return _axis_transform_coordinate_matrix(self)


@lru_cache(maxsize=None)
def _plan_axis_transform(source: SubSpace, target: SubSpace) -> AxisTransform:
    if source.algebra is not target.algebra:
        return AxisTransform(source, target, AxisTransformKind.INCOMPATIBLE, ())

    source_index = {mask: index for index, mask in enumerate(source.masks)}
    target_from_source = tuple(source_index.get(mask) for mask in target.masks)

    if source == target:
        kind = AxisTransformKind.EXACT
    elif source.same_support(target):
        kind = AxisTransformKind.RELAYOUT
    elif source.support_is_subset_of(target):
        kind = AxisTransformKind.EMBED
    elif target.support_is_subset_of(source):
        kind = AxisTransformKind.PROJECT
    else:
        kind = AxisTransformKind.REFRAME
    return AxisTransform(source, target, kind, target_from_source)


@lru_cache(maxsize=None)
def _axis_transform_coordinate_matrix(transform: AxisTransform) -> tuple:
    return tuple(
        tuple(
            int(source_index == candidate)
            * transform.source.signs[candidate]
            * transform.target.signs[target_index]
            if source_index is not None
            else 0
            for candidate in range(len(transform.source))
        )
        for target_index, source_index in enumerate(transform.target_from_source)
    )


@dataclass(frozen=True, slots=True)
class BoundAxis:
    """Each bound entry pairs an input axis index of the target `GAType` with the bound operand
    `GAType` and its `AxisTransform`."""

    axis: int
    operand_gatype: GAType
    transform: AxisTransform


@dataclass(frozen=True, slots=True)
class BindingPlan:
    """Static type metadata fixes the per-axis coordinate transforms, spliced result axes, and
    shared-operand equality groups of an atomic bind."""

    target_gatype: GAType
    bindings: tuple[BoundAxis, ...]
    result_subspaces: tuple[SubSpace, ...]
    result_orientations: tuple[Orientation, ...]
    equality_groups: tuple

    @classmethod
    def build(
        cls,
        target_gatype: GAType,
        operands_by_axis: Mapping[int, HasGAType],
    ) -> BindingPlan:
        ordered = tuple(sorted(operands_by_axis.items()))
        return cls.from_types(
            target_gatype,
            tuple((axis, operand.gatype) for axis, operand in ordered),
            nullary_identity_groups(ordered),
        )

    @classmethod
    def from_types(
        cls,
        target_gatype: GAType,
        operands_by_axis: tuple,
        equality_groups: tuple = (),
    ) -> BindingPlan:
        return _binding_plan_from_types(target_gatype, operands_by_axis, equality_groups)

    @property
    def execution_gatype(self) -> GAType:
        """Certified output-support restrictions narrow the target output axis before contraction
        while leaving its input axes intact."""
        return self.target_gatype.algebra.gatype.oriented(
            (self.result_subspaces[0],) + self.target_gatype.input_subspaces,
            self.target_gatype.orientations,
        )

    @property
    def output_indices(self) -> tuple[int, ...]:
        """Indices into `target_gatype.output_subspace` select the output blades retained in
        `result_subspaces[0]`."""
        indices = {
            mask: index for index, mask in enumerate(self.target_gatype.output_subspace.masks)
        }
        return tuple(indices[mask] for mask in self.result_subspaces[0].masks)

    @property
    def axes(self) -> tuple[int, ...]:
        return tuple(binding.axis for binding in self.bindings)

    @property
    def operand_gatypes(self) -> tuple[GAType, ...]:
        return tuple(binding.operand_gatype for binding in self.bindings)

    @property
    def result_input_splices(self) -> tuple:
        """Each target input axis maps to the range of result input axes it produces. Unbound inputs
        keep one axis, nullary operands contribute none, and positive-arity operands splice their
        own inputs in place independently of backend contraction order."""
        bindings = {binding.axis: binding for binding in self.bindings}
        splices = []
        next_output_axis = 0
        for axis in range(self.target_gatype.arity):
            binding = bindings.get(axis)
            width = 1 if binding is None else binding.operand_gatype.arity
            splice = tuple(
                range(next_output_axis, next_output_axis + width)
            )
            splices.append(splice)
            next_output_axis += width
        return tuple(splices)

    @property
    def type_signature(self) -> tuple[object, ...]:
        """Combining the target type, axis transforms, input splices, and equality groups forms a
        value-free cache key."""
        return (
            self.target_gatype,
            tuple(
                (binding.axis, binding.operand_gatype, binding.transform)
                for binding in self.bindings
            ),
            self.result_subspaces,
            self.result_input_splices,
            self.equality_groups,
        )


@lru_cache(maxsize=None)
def _binding_plan_from_types(
    target_gatype: GAType,
    operands_by_axis: tuple,
    equality_groups: tuple = (),
) -> BindingPlan:
    entries = []

    for axis, operand in operands_by_axis:
        if axis < 0 or axis >= target_gatype.arity:
            raise IndexError(f"input {axis} is out of range for arity {target_gatype.arity}")
        required = target_gatype.input_subspaces[axis]
        actual = operand.output_subspace
        transform = AxisTransform.plan(actual, required)
        if not transform.is_implicit_bind_compatible:
            raise GATypeError(
                "cannot bind output axis "
                f"{actual!r} into input {axis} requiring {required!r}: "
                f"conversion is {transform.kind.value!r}"
            )
        if operand.orientation is not target_gatype.input_orientations[axis]:
            raise GATypeError(
                f"cannot bind an {operand.orientation.value} {actual.type_name} into input {axis}, "
                f"which takes an {target_gatype.input_orientations[axis].value} orientation"
            )
        entries.append(BoundAxis(axis, operand, transform))

    by_axis = dict(operands_by_axis)
    result_axes = [target_gatype.output_subspace]
    result_orientations = [target_gatype.orientation]
    for axis, (target_axis, target_orientation) in enumerate(
        zip(target_gatype.input_subspaces, target_gatype.input_orientations)
    ):
        operand = by_axis.get(axis)
        if operand is None:
            result_axes.append(target_axis)
            result_orientations.append(target_orientation)
        else:
            result_axes.extend(operand.input_subspaces)
            result_orientations.extend(operand.input_orientations)

    plan = BindingPlan(
        target_gatype=target_gatype,
        bindings=tuple(entries),
        result_subspaces=tuple(result_axes),
        result_orientations=tuple(result_orientations),
        equality_groups=equality_groups,
    )
    subspaces = bind_subspaces(plan)
    return (
        replace(plan, result_subspaces=subspaces)
        if subspaces != plan.result_subspaces else plan
    )


class TypeRules:
    """Cached result-type inference combines a `BindingPlan`'s spliced subspaces, orientations, and
    propagated traits into the output `GAType`."""

    @staticmethod
    @lru_cache(maxsize=None)
    def bind(plan: BindingPlan) -> GAType:
        return plan.target_gatype.algebra.gatype.oriented(
            plan.result_subspaces,
            plan.result_orientations,
            bind_traits(plan),
        )


def normalize_bind_arguments(arity: int, operands: tuple) -> dict[int, HasGAType]:
    """Normalizing positional arguments or an explicit `{axis: operand}` mapping produces an
    axis-indexed dictionary."""
    if len(operands) == 1 and isinstance(operands[0], Mapping):
        return dict(operands[0])
    if len(operands) > arity:
        raise ValueError(f"received {len(operands)} operands for arity-{arity} extensor")
    return dict(enumerate(operands))


def nullary_identity_groups(operands_by_axis: tuple) -> tuple:
    """Grouping input axes that receive the identical nullary operand object records
    repeated-argument identities for `BindingPlan`."""
    axes_by_identity = {}
    for axis, operand in operands_by_axis:
        if operand.gatype.arity == 0:
            axes_by_identity.setdefault(id(operand), []).append(axis)
    return tuple(
        tuple(axes)
        for axes in axes_by_identity.values()
        if len(axes) > 1
    )
