"""Small native factory for exact output-first operation Extensors."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any, Callable, Iterable

import numpy as np

from numga.algebra.self_product import grade_transform_sign, symmetric_product_terms
from numga.binding import AxisTransform, TypeRules
from numga.extensor import Extensor
from numga.gatype import GAType
from numga.gatype.traits import Identity
from numga.operator.kernel import SymbolicKernel
from numga.subspace import SubSpace

if TYPE_CHECKING:
    from numga.algebra import Algebra


GradeRule = Callable[[int, int, int], bool]
BasisRule = Callable[[int, int], tuple[int, int]]
OperandType = SubSpace | GAType


class OperatorFactory:
    """Construct and cache exact operation Extensors for one algebra."""

    __slots__ = ("algebra", "subspaces")

    def __init__(self, algebra: Algebra) -> None:
        self.algebra = algebra
        self.subspaces = algebra.subspace

    def build(
        self, axes: Iterable[SubSpace], coefficients: Any,
    ) -> Extensor:
        axes = tuple(axes)
        if not axes:
            raise ValueError("an Extensor requires at least an output axis")
        if any(axis.algebra is not self.algebra for axis in axes):
            raise ValueError("every Extensor axis must belong to this factory's algebra")
        return Extensor(
            self.algebra.exact,
            self.algebra.gatype(axes),
            SymbolicKernel(coefficients),
        )

    @lru_cache(maxsize=None)
    def unit(self, space: SubSpace) -> Extensor:
        """The exact scalar identity projected onto a coefficient layout."""

        return self.algebra.exact.multivector(space)

    @lru_cache(maxsize=None)
    def identity(self, space: SubSpace) -> Extensor:
        self._require_space(space)
        return self.build((space, space), SymbolicKernel.identity(len(space))).with_traits(Identity)

    @lru_cache(maxsize=None)
    def _grade_transform(self, space: SubSpace, transform: str) -> Extensor:
        coefficients = np.zeros((len(space), len(space)), dtype=np.int8)
        for index, mask in enumerate(space.masks):
            coefficients[index, index] = grade_transform_sign(self.algebra, transform, mask)
        return self.build((space, space), coefficients)

    def reverse(self, space: SubSpace) -> Extensor:
        return self._grade_transform(space, "reverse")

    def clifford_conjugate(self, space: SubSpace) -> Extensor:
        """Reverse blades and negate odd grades, without conjugating scalars."""

        return self._grade_transform(space, "clifford_conjugate")

    def involute(self, space: SubSpace) -> Extensor:
        return self._grade_transform(space, "involute")

    def scalar_negation(self, space: SubSpace) -> Extensor:
        return self._grade_transform(space, "scalar_negation")

    def pseudoscalar_negation(self, space: SubSpace) -> Extensor:
        return self._grade_transform(space, "pseudoscalar_negation")

    @lru_cache(maxsize=None)
    def _symmetric_product(
        self, gatype: GAType, transform: str, *, scalar_only: bool = False,
    ) -> Extensor:
        """Combine symmetric terms on the result carrier proved by the GAType."""

        space = gatype.output_subspace
        output = (
            self.subspaces.scalar() if scalar_only
            else gatype._self_product(transform).output_subspace
        )
        indices = {mask: index for index, mask in enumerate(output.masks)}
        # A cross term splits evenly over its two symmetric entries: doubled here, halved exactly
        # below, since the two orders of a basis product are equal or opposite.
        coefficients = np.zeros((len(output), len(space), len(space)), dtype=np.int8)
        for mask, left, right, coefficient in symmetric_product_terms(space, transform):
            if mask not in indices:
                continue
            coefficient *= space.signs[left] * space.signs[right] * output.signs[indices[mask]]
            if left == right:
                coefficients[indices[mask], left, right] = 2 * coefficient
            else:
                coefficients[indices[mask], left, right] = coefficient
                coefficients[indices[mask], right, left] = coefficient
        return self.build((output, space, space), SymbolicKernel(coefficients).halved())

    def squared(self, operand: OperandType) -> Extensor:
        return self._symmetric_product(self._operand_gatype(operand), "identity")

    def symmetric_reverse_product(self, operand: OperandType) -> Extensor:
        return self._symmetric_product(self._operand_gatype(operand), "reverse")

    def scalar_norm_squared(self, operand: OperandType) -> Extensor:
        """The scalar part of ``x * reverse(x)``, with only that part built."""

        return self._symmetric_product(self._operand_gatype(operand), "reverse", scalar_only=True)

    def symmetric_conjugate_product(self, operand: OperandType) -> Extensor:
        return self._symmetric_product(self._operand_gatype(operand), "clifford_conjugate")

    def symmetric_scalar_negation_product(self, operand: OperandType) -> Extensor:
        return self._symmetric_product(self._operand_gatype(operand), "scalar_negation")

    def study_norm_squared(self, operand: OperandType) -> Extensor:
        """Scalar self-product for the generalized Study norm."""

        return self._symmetric_product(
            self._operand_gatype(operand), "scalar_negation", scalar_only=True,
        )

    def symmetric_pseudoscalar_negation_product(self, operand: OperandType) -> Extensor:
        return self._symmetric_product(self._operand_gatype(operand), "pseudoscalar_negation")

    def symmetric_involute_product(self, operand: OperandType) -> Extensor:
        return self._symmetric_product(self._operand_gatype(operand), "involute")

    @lru_cache(maxsize=None)
    def cast(self, source: SubSpace, target: SubSpace) -> Extensor:
        """Explicit signed coordinate conversion, including projection and zero fill."""

        self._require_space(source)
        self._require_space(target)
        transform = AxisTransform.plan(source, target)
        if not transform.is_compatible:
            raise ValueError("cannot cast between SubSpaces from different algebras")
        coefficients = SymbolicKernel(
            transform.coordinate_matrix, shape=(len(target), len(source)),
        )
        result = self.build((target, source), coefficients)
        return result.with_traits(Identity) if transform.is_lossless else result

    select = cast

    @lru_cache(maxsize=None)
    def restrict(self, source: SubSpace, target: SubSpace) -> Extensor:
        """Project onto the intersection of the two static blade supports."""

        return self.select(source, source.intersection(target))

    def geometric_product(self, left: OperandType, right: OperandType) -> Extensor:
        """Build a product selected from the operands' complete GATypes."""

        return self._geometric_product(
            self._operand_gatype(left),
            self._operand_gatype(right),
        )

    @lru_cache(maxsize=None)
    def _geometric_product(self, left: GAType, right: GAType) -> Extensor:
        return self._product(
            "geometric_product",
            left,
            right,
            lambda _left, _right, _output: True,
        )

    product = geometric_product

    def scalar_product(self, left: OperandType, right: OperandType) -> Extensor:
        return self._grade_product(self._operand_gatype(left), self._operand_gatype(right), 0)

    def bivector_product(self, left: OperandType, right: OperandType) -> Extensor:
        return self._grade_product(self._operand_gatype(left), self._operand_gatype(right), 2)

    def trivector_product(self, left: OperandType, right: OperandType) -> Extensor:
        return self._grade_product(self._operand_gatype(left), self._operand_gatype(right), 3)

    def inner(self, left: OperandType, right: OperandType) -> Extensor:
        """The inner product: the grade |r - s| part of the product of grades r and s."""
        return self._inner(self._operand_gatype(left), self._operand_gatype(right))

    @lru_cache(maxsize=None)
    def _inner(self, left: GAType, right: GAType) -> Extensor:
        return self._product(
            "inner", left, right,
            lambda l, r, output: output == abs(np.subtract(l, r, dtype=np.int16)),
        )

    def left_contraction(self, left: OperandType, right: OperandType) -> Extensor:
        """The left contraction: the grade s - r part of the product of grades r and s."""
        return self._left_contraction(self._operand_gatype(left), self._operand_gatype(right))

    @lru_cache(maxsize=None)
    def _left_contraction(self, left: GAType, right: GAType) -> Extensor:
        return self._product(
            "left_contraction", left, right,
            lambda l, r, output: output == np.subtract(r, l, dtype=np.int16),
        )

    def right_contraction(self, left: OperandType, right: OperandType) -> Extensor:
        """The right contraction: the grade r - s part of the product of grades r and s."""
        return self._right_contraction(self._operand_gatype(left), self._operand_gatype(right))

    @lru_cache(maxsize=None)
    def _right_contraction(self, left: GAType, right: GAType) -> Extensor:
        return self._product(
            "right_contraction", left, right,
            lambda l, r, output: output == np.subtract(l, r, dtype=np.int16),
        )

    def left_interior(self, left: OperandType, right: OperandType) -> Extensor:
        """The left interior product: the anti-wedge of the left complement of left with right."""
        return self._left_interior(self._operand_gatype(left), self._operand_gatype(right))

    @lru_cache(maxsize=None)
    def _left_interior(self, left: GAType, right: GAType) -> Extensor:
        def basis_rule(left_mask: int, right_mask: int) -> tuple[int, int]:
            complement, sign = self._left_complement(left_mask)
            output, coefficient = self._anti_wedge_rule(complement, right_mask)
            return output, sign * coefficient

        return self._bilinear("left_interior", left, right, basis_rule)

    def right_interior(self, left: OperandType, right: OperandType) -> Extensor:
        """The right interior product: the anti-wedge of left with the right complement of right."""
        return self._right_interior(self._operand_gatype(left), self._operand_gatype(right))

    @lru_cache(maxsize=None)
    def _right_interior(self, left: GAType, right: GAType) -> Extensor:
        def basis_rule(left_mask: int, right_mask: int) -> tuple[int, int]:
            complement, sign = self._right_complement(right_mask)
            output, coefficient = self._anti_wedge_rule(left_mask, complement)
            return output, sign * coefficient

        return self._bilinear("right_interior", left, right, basis_rule)

    def _swap_sign(self, left_mask: int, right_mask: int) -> int:
        """The sign of reordering the generators of left_mask * right_mask, ignoring the metric."""
        swaps = sum((left_mask >> (bit + 1)).bit_count() for bit in range(right_mask.bit_length()) if right_mask >> bit & 1)
        return -1 if swaps % 2 else 1

    def _negative_sign(self, mask: int) -> int:
        return -1 if (mask & self.algebra.negative_mask).bit_count() % 2 else 1

    def _right_complement(self, mask: int) -> tuple[int, int]:
        """The right complement, mask * I with the metric's negative signs: the blade that follows mask."""
        pseudoscalar = self.algebra.pseudoscalar_mask
        return mask ^ pseudoscalar, self._swap_sign(mask, pseudoscalar) * self._negative_sign(mask)

    def _left_complement(self, mask: int) -> tuple[int, int]:
        """The left complement, I * mask with the metric's negative signs: the blade that precedes mask."""
        pseudoscalar = self.algebra.pseudoscalar_mask
        return mask ^ pseudoscalar, self._swap_sign(pseudoscalar, mask) * self._negative_sign(mask)

    def _anti_wedge_rule(self, left_mask: int, right_mask: int) -> tuple[int, int]:
        """The anti-wedge of two basis blades: the left complement of the wedge of right complements."""
        left_complement, left_sign = self._right_complement(left_mask)
        right_complement, right_sign = self._right_complement(right_mask)
        if left_complement & right_complement:
            return 0, 0
        wedge = left_complement | right_complement
        output, output_sign = self._left_complement(wedge)
        return output, left_sign * right_sign * self._swap_sign(left_complement, right_complement) * output_sign

    @lru_cache(maxsize=None)
    def _grade_product(self, left: GAType, right: GAType, grade: int) -> Extensor:
        """Construct only the requested output grade, retaining sparse support."""

        return self._product(
            "grade_product", left, right,
            lambda _left, _right, output: output == grade,
        )

    def wedge(self, left: OperandType, right: OperandType) -> Extensor:
        return self._wedge(
            self._operand_gatype(left),
            self._operand_gatype(right),
        )

    @lru_cache(maxsize=None)
    def _wedge(self, left: GAType, right: GAType) -> Extensor:
        return self._product(
            "wedge",
            left,
            right,
            lambda left_grade, right_grade, output_grade: (
                left_grade + right_grade == output_grade
            ),
        )

    exterior_product = wedge

    def commutator(self, left: OperandType, right: OperandType) -> Extensor:
        """Build ``(left * right - right * left) / 2`` exactly."""

        return self._commutator(
            self._operand_gatype(left),
            self._operand_gatype(right),
        )

    @lru_cache(maxsize=None)
    def _commutator(self, left: GAType, right: GAType) -> Extensor:
        def basis_rule(left_mask: int, right_mask: int) -> tuple[int, int]:
            forward = self.algebra.geometric_product(left_mask, right_mask)
            reverse = self.algebra.geometric_product(right_mask, left_mask)
            if forward.blade != reverse.blade:
                raise AssertionError("basis products must have the same XOR blade")
            # Basis blades commute or anticommute: the half difference is 0 or the product itself.
            return forward.blade, (forward.coefficient - reverse.coefficient) // 2

        return self._bilinear("commutator", left, right, basis_rule)

    def anticommutator(self, left: OperandType, right: OperandType) -> Extensor:
        """Build ``(left * right + right * left) / 2`` exactly."""

        return self._anticommutator(
            self._operand_gatype(left),
            self._operand_gatype(right),
        )

    @lru_cache(maxsize=None)
    def _anticommutator(self, left: GAType, right: GAType) -> Extensor:
        def basis_rule(left_mask: int, right_mask: int) -> tuple[int, int]:
            forward = self.algebra.geometric_product(left_mask, right_mask)
            reverse = self.algebra.geometric_product(right_mask, left_mask)
            if forward.blade != reverse.blade:
                raise AssertionError("basis products must have the same XOR blade")
            # Basis blades commute or anticommute: the half sum is the product itself or 0.
            return forward.blade, (forward.coefficient + reverse.coefficient) // 2

        return self._bilinear("anticommutator", left, right, basis_rule)

    def regressive(self, left: OperandType, right: OperandType) -> Extensor:
        """Build ``dual_inverse(dual(left) ^ dual(right))`` exactly."""

        return self._regressive(
            self._operand_gatype(left),
            self._operand_gatype(right),
        )

    @lru_cache(maxsize=None)
    def _regressive(self, left: GAType, right: GAType) -> Extensor:
        def basis_rule(left_mask: int, right_mask: int) -> tuple[int, int]:
            left_dual, left_sign = self._right_hodge(left_mask)
            right_dual, right_sign = self._right_hodge(right_mask)
            if left_dual & right_dual:
                return 0, 0

            wedge = self.algebra.geometric_product(left_dual, right_dual)
            output, inverse_sign = self._right_hodge_inverse(wedge.blade)
            return (
                output,
                left_sign * right_sign * wedge.coefficient * inverse_sign,
            )

        return self._bilinear("regressive", left, right, basis_rule)

    @lru_cache(maxsize=None)
    def cross(self, vector: SubSpace) -> Extensor:
        """The oriented Euclidean 3-vector cross product."""

        self._require_space(vector)
        if self.algebra.dimension != 3 or self.algebra.signature != (1, 1, 1):
            raise ValueError("cross is defined here only for Euclidean 3-space")
        if not vector.same_support(self.subspaces.vector()):
            raise ValueError("cross requires the algebra's full vector SubSpace")

        coefficients = np.zeros((3, 3, 3), dtype=np.int8)
        for output in range(3):
            for left in range(3):
                for right in range(3):
                    if len({output, left, right}) != 3:
                        continue
                    masks = (vector.masks[output], vector.masks[left], vector.masks[right])
                    inversions = sum(
                        first > second
                        for position, first in enumerate(masks)
                        for second in masks[position + 1 :]
                    )
                    coefficients[output, left, right] = (
                        (-1 if inversions % 2 else 1)
                        * vector.signs[output] * vector.signs[left] * vector.signs[right]
                    )
        return self.build((vector, vector, vector), coefficients)

    def sandwich(
        self,
        sandwicher: OperandType,
        passenger: OperandType,
        output: SubSpace | None = None,
    ) -> Extensor:
        """Polarize ``sandwicher * passenger * reverse(sandwicher)``.

        The two sandwicher slots remain distinct. Passing the same nullary
        Extensor into slots 0 and 2 is an atomic diagonal bind at runtime.
        Exact symmetrization removes terms that cancel for repeated sandwichers.
        ``output`` requests an explicit cast.
        """

        if output is not None:
            self._require_space(output)
        return self._sandwich(
            self._operand_gatype(sandwicher),
            self._operand_gatype(passenger),
            output,
        )

    @lru_cache(maxsize=None)
    def _sandwich(
        self,
        sandwicher: GAType,
        passenger: GAType,
        output: SubSpace | None,
    ) -> Extensor:
        result = self.full_sandwich(
            sandwicher.output_subspace,
            passenger.output_subspace,
            output,
        )
        kernel = result.kernel
        # The two sandwicher slots hold the same versor: symmetrize them. For basis blades the two
        # orders are equal or opposite, so the average is exact.
        kernel = (kernel + kernel.transpose((0, 3, 2, 1))).halved()
        result = self.build(result.axes, kernel).squeeze_output()
        if output is not None:
            # A requested projection is not an unrestricted sandwich action.
            return result
        gatype = TypeRules.operation(
            "sandwich",
            (sandwicher, passenger, sandwicher),
            result.axes,
        )
        if gatype is result.gatype:
            return result
        return Extensor(result.context, gatype, result.kernel)

    def reverse_sandwich(
        self,
        sandwicher: OperandType,
        passenger: OperandType,
        output: SubSpace | None = None,
    ) -> Extensor:
        """Polarize ``reverse(sandwicher) * passenger * sandwicher`` as one operator: the sandwich
        with the carrier's reverse folded into both sandwicher slots."""

        if output is not None:
            self._require_space(output)
        return self._reverse_sandwich(
            self._operand_gatype(sandwicher),
            self._operand_gatype(passenger),
            output,
        )

    @lru_cache(maxsize=None)
    def _reverse_sandwich(
        self,
        sandwicher: GAType,
        passenger: GAType,
        output: SubSpace | None,
    ) -> Extensor:
        forward = self._sandwich(sandwicher, passenger, output)
        reverse = self.reverse(sandwicher.output_subspace)
        # The reverse is diagonal on the carrier: folding it into both slots keeps every axis, the
        # grade cancellations and the sandwich's type. A versor's reverse is a versor, so the
        # sandwich's laws hold for it too.
        folded = forward.bind({0: reverse, 2: reverse})
        return Extensor(folded.context, forward.gatype, folded.kernel)

    @lru_cache(maxsize=None)
    def full_sandwich(
        self,
        sandwicher: SubSpace,
        passenger: SubSpace,
        output: SubSpace | None = None,
    ) -> Extensor:
        """Unrestricted sandwich tensor, optionally cast to ``output``."""

        self._require_space(sandwicher)
        self._require_space(passenger)
        if output is not None:
            self._require_space(output)

        left_product = self.geometric_product(sandwicher, passenger)
        right_product = self.geometric_product(
            left_product.output_subspace, sandwicher
        )
        unrestricted = right_product.bind(
            {
                0: left_product,
                1: self.reverse(sandwicher),
            }
        )
        if output is None or unrestricted.output_subspace == output:
            return unrestricted
        return self.cast(unrestricted.output_subspace, output).bind({0: unrestricted})

    def _product(
        self,
        operation: str,
        left_type: GAType,
        right_type: GAType,
        grade_rule: GradeRule,
    ) -> Extensor:
        left = left_type.output_subspace
        right = right_type.output_subspace
        self._require_space(left)
        self._require_space(right)
        table = self.algebra.geometric_product_table(left.masks, right.masks)

        l_grades = self.algebra.grade(np.asarray(left.masks, dtype=self.algebra.blade_dtype))[:, None]
        r_grades = self.algebra.grade(np.asarray(right.masks, dtype=self.algebra.blade_dtype))[None, :]
        o_grades = self.algebra.grade(table.blades)
        coeffs = table.coefficients

        try:
            valid = (coeffs != 0) & np.asarray(grade_rule(l_grades, r_grades, o_grades), dtype=bool)
        except TypeError:
            valid = np.zeros(coeffs.shape, dtype=bool)
            for i, lm in enumerate(left.masks):
                for j, rm in enumerate(right.masks):
                    if coeffs[i, j]:
                        valid[i, j] = grade_rule(
                            int(l_grades[i, 0]), int(r_grades[0, j]), int(o_grades[i, j])
                        )

        output_masks = np.unique(table.blades[valid])
        output = self.subspaces.from_masks(output_masks)
        output_signs = np.asarray(output.signs, dtype=np.int8)
        l_signs = np.asarray(left.signs, dtype=np.int8)
        r_signs = np.asarray(right.signs, dtype=np.int8)

        signed_coeffs = coeffs * l_signs[:, None] * r_signs[None, :]
        coefficients = np.zeros((len(output), len(left), len(right)), dtype=np.int8)
        for k, mask in enumerate(output.masks):
            match = valid & (table.blades == mask)
            coefficients[k, match] = signed_coeffs[match] * output_signs[k]

        gatype = TypeRules.operation(
            operation,
            (left_type, right_type),
            (output, left, right),
        )
        return Extensor(self.algebra.exact, gatype, SymbolicKernel(coefficients))

    def _bilinear(
        self,
        operation: str,
        left_type: GAType,
        right_type: GAType,
        basis_rule: BasisRule,
    ) -> Extensor:
        """Build a sparse-by-construction exact bilinear operation."""

        left = left_type.output_subspace
        right = right_type.output_subspace
        self._require_space(left)
        self._require_space(right)

        terms: list[tuple[int, int, int, int]] = []
        output_masks: set[int] = set()
        for left_index, left_mask in enumerate(left.masks):
            for right_index, right_mask in enumerate(right.masks):
                output_mask, coefficient = basis_rule(left_mask, right_mask)
                if coefficient:
                    output_masks.add(output_mask)
                    terms.append(
                        (output_mask, left_index, right_index, coefficient)
                    )

        output = self.subspaces.from_masks(output_masks)
        output_index = {mask: index for index, mask in enumerate(output.masks)}
        coefficients = np.zeros((len(output), len(left), len(right)), dtype=np.int8)
        for output_mask, left_index, right_index, coefficient in terms:
            coefficients[
                output_index[output_mask], left_index, right_index
            ] += (
                coefficient * left.signs[left_index] * right.signs[right_index]
                * output.signs[output_index[output_mask]]
            )

        gatype = TypeRules.operation(
            operation,
            (left_type, right_type),
            (output, left, right),
        )
        return Extensor(self.algebra.exact, gatype, SymbolicKernel(coefficients))

    @lru_cache(maxsize=None)
    def dual(self, space: SubSpace) -> Extensor:
        """Right-Hodge dual, expressed on the algebra's default output layout."""

        return self._dual(space, inverse=False)

    @lru_cache(maxsize=None)
    def dual_inverse(self, space: SubSpace) -> Extensor:
        return self._dual(space, inverse=True)

    def _dual(self, space: SubSpace, *, inverse: bool) -> Extensor:
        rule = self._right_hodge_inverse if inverse else self._right_hodge
        terms = tuple(rule(mask) for mask in space.masks)
        output = self.subspaces.from_masks(mask for mask, _ in terms)
        indices = {mask: index for index, mask in enumerate(output.masks)}
        coefficients = np.zeros((len(output), len(space)), dtype=np.int8)
        for column, (mask, sign) in enumerate(terms):
            row = indices[mask]
            coefficients[row, column] = sign * space.signs[column] * output.signs[row]
        return self.build((output, space), coefficients)

    def _right_hodge(self, mask: int) -> tuple[int, int]:
        """Return the complement mask and right-Hodge orientation sign."""

        complement = self.algebra.complement(mask)
        orientation = self.algebra.geometric_product(mask, complement).coefficient
        negative_sign = (
            -1
            if (mask & self.algebra.negative_mask).bit_count() % 2
            else 1
        )
        return complement, orientation * negative_sign

    def _right_hodge_inverse(self, mask: int) -> tuple[int, int]:
        """Return the inverse image of one right-Hodge basis blade."""

        complement = self.algebra.complement(mask)
        _, sign = self._right_hodge(complement)
        return complement, sign

    def _operand_gatype(self, operand: OperandType) -> GAType:
        if isinstance(operand, SubSpace):
            return self.algebra.gatype(operand)
        if not isinstance(operand, GAType) or operand.algebra is not self.algebra:
            raise ValueError("operation operand type belongs to another algebra")
        return operand

    def _require_space(self, space: SubSpace) -> None:
        if not isinstance(space, SubSpace) or space.algebra is not self.algebra:
            raise ValueError("SubSpace belongs to another algebra")
