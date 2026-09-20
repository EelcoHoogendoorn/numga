"""Value-free planning for Extensor binding."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from functools import lru_cache
from typing import Mapping, Protocol

from numga.gatype import GAType
from numga.gatype.propagation import bind_subspaces, bind_traits, operation_traits
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
    @lru_cache(maxsize=None)
    def plan(cls, source: SubSpace, target: SubSpace) -> "AxisTransform":
        if source.algebra is not target.algebra:
            return cls(source, target, AxisTransformKind.INCOMPATIBLE, ())

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
        return cls(source, target, kind, target_from_source)

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
    @lru_cache(maxsize=None)
    def coordinate_matrix(self) -> tuple[tuple[int, ...], ...]:
        """Dense target-by-source transform in exact integer coordinates.

        Fold orientation signs into the permutation/projection/embedding.
        """

        return tuple(
            tuple(
                int(source_index == candidate) * self.source.signs[candidate] * self.target.signs[target_index]
                if source_index is not None
                else 0
                for candidate in range(len(self.source))
            )
            for target_index, source_index in enumerate(self.target_from_source)
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
    @lru_cache(maxsize=None)
    def from_types(
        cls,
        target_gatype: GAType,
        operands_by_slot: tuple[tuple[int, GAType], ...],
        equality_groups: tuple[tuple[int, ...], ...] = (),
    ) -> "BindingPlan":
        """Plan once per static signature, never retaining coefficient arrays."""

        entries: list[BoundSlot] = []

        for slot, operand in operands_by_slot:
            if slot < 0 or slot >= target_gatype.arity:
                raise IndexError(f"input slot {slot} is out of range for arity {target_gatype.arity}")
            required = target_gatype.input_subspaces[slot]
            actual = operand.output_subspace
            transform = AxisTransform.plan(actual, required)
            if not transform.is_implicit_bind_compatible:
                raise ValueError(
                    "cannot bind output axis "
                    f"{actual!r} into input slot {slot} requiring {required!r}: "
                    f"conversion is {transform.kind.value!r}"
                )
            entries.append(BoundSlot(slot, operand, transform))

        by_slot = dict(operands_by_slot)
        result_axes: list[SubSpace] = [target_gatype.output_subspace]
        for slot, target_axis in enumerate(target_gatype.input_subspaces):
            operand = by_slot.get(slot)
            if operand is None:
                result_axes.append(target_axis)
            else:
                result_axes.extend(operand.input_subspaces)

        plan = cls(
            target_gatype=target_gatype,
            bindings=tuple(entries),
            result_subspaces=tuple(result_axes),
            equality_groups=equality_groups,
        )
        subspaces = bind_subspaces(plan)
        return (
            replace(plan, result_subspaces=subspaces)
            if subspaces != plan.result_subspaces else plan
        )

    @property
    @lru_cache(maxsize=None)
    def execution_gatype(self) -> GAType:
        """Target axes after any certified output restriction."""

        return self.target_gatype.algebra.gatype(
            (self.result_subspaces[0],) + self.target_gatype.input_subspaces,
        )

    @property
    @lru_cache(maxsize=None)
    def output_indices(self) -> tuple[int, ...]:
        """Target rows retained by a certified output-support restriction."""

        indices = {
            mask: index for index, mask in enumerate(self.target_gatype.output_subspace.masks)
        }
        return tuple(indices[mask] for mask in self.result_subspaces[0].masks)

    @property
    @lru_cache(maxsize=None)
    def slots(self) -> tuple[int, ...]:
        return tuple(binding.slot for binding in self.bindings)

    @property
    @lru_cache(maxsize=None)
    def operand_gatypes(self) -> tuple[GAType, ...]:
        return tuple(binding.operand_gatype for binding in self.bindings)

    @property
    @lru_cache(maxsize=None)
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


class TypeRules:
    """Structural first-epoch type inference hooks.

    Trait-aware implementations can replace these rules without changing
    contraction code.
    """

    @staticmethod
    @lru_cache(maxsize=None)
    def bind(plan: BindingPlan) -> GAType:
        return plan.target_gatype.algebra.gatype(
            plan.result_subspaces,
            bind_traits(plan),
        )

    @staticmethod
    @lru_cache(maxsize=None)
    def operation(
        operation: str,
        operand_gatypes: tuple[GAType, ...],
        result_subspaces: tuple[SubSpace, ...],
    ) -> GAType:
        if operand_gatypes:
            algebra = operand_gatypes[0].algebra
        elif result_subspaces:
            algebra = result_subspaces[0].algebra
        else:
            raise ValueError("type inference requires an operand or result axis")

        traits = operation_traits(
            operation,
            operand_gatypes,
            result_subspaces,
        )
        return algebra.gatype(result_subspaces, traits)

    @staticmethod
    def collection(
        operation: str,
        _parameters: object,
        operand_gatypes: tuple[GAType, ...],
    ) -> GAType:
        if not operand_gatypes:
            raise ValueError("collection inference requires at least one operand")
        first = operand_gatypes[0]
        if operation in {"stack", "concatenate"}:
            if all(gatype == first for gatype in operand_gatypes[1:]):
                return first
            output = first.output_subspace
            common_traits = set(first.effective_traits.closure)
            for gatype in operand_gatypes[1:]:
                if gatype.input_subspaces != first.input_subspaces:
                    raise ValueError("collection inference requires equal input axes")
                output = output.union(gatype.output_subspace)
                common_traits.intersection_update(gatype.effective_traits.closure)
            # Whole-value facts survive zero embedding of multivectors. Map
            # properties such as orthogonality need their original codomain.
            if first.arity and any(g.output_subspace != output for g in operand_gatypes):
                common_traits.clear()
            return first.algebra.gatype((output,) + first.input_subspaces, common_traits)
        if any(
            gatype.subspaces != first.subspaces
            for gatype in operand_gatypes[1:]
        ):
            raise ValueError("collection inference requires equal structural axes")

        if operation in {
            "index",
            "reshape",
            "broadcast_to",
        }:
            if any(gatype != first for gatype in operand_gatypes[1:]):
                raise ValueError(
                    "trait-preserving collection inference requires equal GATypes"
                )
            return first

        if operation in {"sum", "mean", "set"}:
            return first.algebra.gatype(first.subspaces)

        raise NotImplementedError(
            f"no collection type rule is defined for {operation!r}"
        )

    @staticmethod
    def transform(
        gatype: GAType,
        transforms_by_axis: Mapping[int, AxisTransform],
    ) -> GAType:
        axes = list(gatype.subspaces)
        for axis, transform in transforms_by_axis.items():
            if axis < 0 or axis >= len(axes):
                raise IndexError(f"structural axis {axis} is out of range")
            if axes[axis] != transform.source:
                raise ValueError("axis transform source does not match GAType axis")
            if not transform.is_compatible:
                raise ValueError("cannot transform a GAType across incompatible algebras")
            if transform.target.algebra is not gatype.algebra:
                raise ValueError("axis transform target belongs to another algebra")
            axes[axis] = transform.target
        return gatype.algebra.gatype(tuple(axes))


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
