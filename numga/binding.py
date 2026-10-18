"""Value-free planning for Extensor binding."""

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
    """Executable coordinate relationship from one output axis to another.

    ``target_from_source[j]`` is the source coordinate copied into target
    coordinate ``j``, or ``None`` when that target coordinate is a structural
    zero. Orientation changes are folded into the cached coordinate matrix.
    """

    source: SubSpace
    target: SubSpace
    kind: AxisTransformKind
    target_from_source: tuple[int | None, ...]

    @classmethod
    def plan(cls, source: SubSpace, target: SubSpace) -> "AxisTransform":
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
    def coordinate_matrix(self) -> tuple[tuple[int, ...], ...]:
        """Dense target-by-source transform in exact integer coordinates.

        Fold orientation signs into the permutation/projection/embedding.
        """
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
def _axis_transform_coordinate_matrix(
    transform: AxisTransform,
) -> tuple[tuple[int, ...], ...]:
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
class BoundSlot:
    """One value-free binding entry, indexed in the original target."""

    slot: int
    operand_gatype: GAType
    transform: AxisTransform


@dataclass(frozen=True, slots=True)
class BindingPlan:
    """Deterministic, value-free metadata for an atomic bind."""

    target_gatype: GAType
    bindings: tuple[BoundSlot, ...]
    result_subspaces: tuple[SubSpace, ...]
    result_orientations: tuple[Orientation, ...]
    equality_groups: tuple[tuple[int, ...], ...]

    @classmethod
    def build(
        cls,
        target_gatype: GAType,
        operands_by_slot: Mapping[int, HasGAType],
    ) -> "BindingPlan":
        normalized = tuple(sorted(operands_by_slot.items()))
        return cls.from_types(
            target_gatype,
            tuple((slot, operand.gatype) for slot, operand in normalized),
            nullary_identity_groups(normalized),
        )

    @classmethod
    def from_types(
        cls,
        target_gatype: GAType,
        operands_by_slot: tuple[tuple[int, GAType], ...],
        equality_groups: tuple[tuple[int, ...], ...] = (),
    ) -> "BindingPlan":
        return _binding_plan_from_types(target_gatype, operands_by_slot, equality_groups)

    @property
    def execution_gatype(self) -> GAType:
        """Target axes after any certified output restriction."""

        return self.target_gatype.algebra.gatype.oriented(
            (self.result_subspaces[0],) + self.target_gatype.input_subspaces,
            self.target_gatype.orientations,
        )

    @property
    def output_indices(self) -> tuple[int, ...]:
        """Target rows retained by a certified output-support restriction."""

        indices = {
            mask: index for index, mask in enumerate(self.target_gatype.output_subspace.masks)
        }
        return tuple(indices[mask] for mask in self.result_subspaces[0].masks)

    @property
    def slots(self) -> tuple[int, ...]:
        return tuple(binding.slot for binding in self.bindings)

    @property
    def operand_gatypes(self) -> tuple[GAType, ...]:
        return tuple(binding.operand_gatype for binding in self.bindings)

    @property
    def result_input_splices(self) -> tuple[tuple[int, ...], ...]:
        """Result slots contributed by each original target input slot.

        An unbound slot contributes one slot, a nullary operand contributes
        none, and a positive-arity operand contributes its inputs in place.
        The mapping is computed in structural order, independently of the
        backend's contraction order.
        """

        bindings = {binding.slot: binding for binding in self.bindings}
        splices: list[tuple[int, ...]] = []
        next_result_slot = 0
        for slot in range(self.target_gatype.arity):
            binding = bindings.get(slot)
            width = 1 if binding is None else binding.operand_gatype.arity
            splice = tuple(
                range(next_result_slot, next_result_slot + width)
            )
            splices.append(splice)
            next_result_slot += width
        return tuple(splices)

    @property
    def type_signature(self) -> tuple[object, ...]:
        """Backend-free key suitable for result-type inference caches."""

        return (
            self.target_gatype,
            tuple(
                (binding.slot, binding.operand_gatype, binding.transform)
                for binding in self.bindings
            ),
            self.result_subspaces,
            self.result_input_splices,
            self.equality_groups,
        )


@lru_cache(maxsize=None)
def _binding_plan_from_types(
    target_gatype: GAType,
    operands_by_slot: tuple[tuple[int, GAType], ...],
    equality_groups: tuple[tuple[int, ...], ...] = (),
) -> BindingPlan:
    """Plan once per static signature, never retaining coefficient arrays."""

    entries: list[BoundSlot] = []

    for slot, operand in operands_by_slot:
        if slot < 0 or slot >= target_gatype.arity:
            raise IndexError(f"input slot {slot} is out of range for arity {target_gatype.arity}")
        required = target_gatype.input_subspaces[slot]
        actual = operand.output_subspace
        transform = AxisTransform.plan(actual, required)
        if not transform.is_implicit_bind_compatible:
            raise GATypeError(
                "cannot bind output axis "
                f"{actual!r} into input slot {slot} requiring {required!r}: "
                f"conversion is {transform.kind.value!r}"
            )
        if operand.orientation is not target_gatype.input_orientations[slot]:
            raise GATypeError(
                f"cannot bind an {operand.orientation.value} {actual.type_name} into input slot {slot}, "
                f"which takes an {target_gatype.input_orientations[slot].value} orientation"
            )
        entries.append(BoundSlot(slot, operand, transform))

    by_slot = dict(operands_by_slot)
    result_axes: list[SubSpace] = [target_gatype.output_subspace]
    result_orientations: list[Orientation] = [target_gatype.orientation]
    for slot, (target_axis, target_orientation) in enumerate(
        zip(target_gatype.input_subspaces, target_gatype.input_orientations)
    ):
        operand = by_slot.get(slot)
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
    """Structural first-epoch type inference hooks.

    Trait-aware implementations can replace these rules without changing
    contraction code.
    """

    @staticmethod
    @lru_cache(maxsize=None)
    def bind(plan: BindingPlan) -> GAType:
        return plan.target_gatype.algebra.gatype.oriented(
            plan.result_subspaces,
            plan.result_orientations,
            bind_traits(plan),
        )


def normalize_bind_arguments(
    arity: int,
    args: tuple[HasGAType | Mapping[int, HasGAType], ...],
) -> dict[int, HasGAType]:
    """Normalize ``bind(mapping)`` or ``bind(arg0, ...)`` syntax."""

    if len(args) == 1 and isinstance(args[0], Mapping):
        return dict(args[0])
    if len(args) > arity:
        raise ValueError(f"received {len(args)} operands for arity-{arity} extensor")
    return dict(enumerate(args))


def nullary_identity_groups(
    normalized: tuple[tuple[int, HasGAType], ...],
) -> tuple[tuple[int, ...], ...]:
    slots_by_identity: dict[int, list[int]] = {}
    for slot, operand in normalized:
        if operand.gatype.arity == 0:
            slots_by_identity.setdefault(id(operand), []).append(slot)
    return tuple(
        tuple(slots)
        for slots in slots_by_identity.values()
        if len(slots) > 1
    )
