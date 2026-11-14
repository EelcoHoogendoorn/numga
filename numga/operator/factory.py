"""Every primitive algebraic operation in `numga` (identities, grade involutions, layout casts,
Hodge duals, bilinear products, and versor sandwiches) originates as an exact, unbatched `Extensor`
in `algebra.exact` backed by an integer `SymbolicKernel`. Axis `0` holds the output `SubSpace` and
axes `1..arity` hold the operand `SubSpace` layouts in argument order, with relative blade signs
from both input and output layouts folded directly into the integer kernel entries. Caching each
operator by `GAType` signature on the `Algebra` lets backends lower or stage each exact tensor once
without recomputing basis-blade sign tables.

Bilinear products retain only the output basis blades with non-zero table entries for the given
input supports, keeping exact operators sparse in support before runtime values exist. Each builder
pairs its subspaces with an `OperationRule` from `numga.gatype.rules` (or explicit traits such as
`Identity` and `BladeProduct`) so orientation propagation, grade parity, versor closure, and
self-product identities stay attached to the operator's `GAType`. Operators act on each operand's
output blade layout while stripping field markers, so maps compose through their output axes and
field operands broadcast site by site during binding.

Metric products consult the Clifford signature, whereas `wedge` and `regressive` depend only on
basis-blade incidence and ordering. `regressive` measures volume completion against the intrinsic
unit pseudoscalar without any metric sign, flipping orientation once so an extrinsic covector and
an intrinsic vector pair to an extrinsic scalar. Conversely, `dot` extends the metric across grades
within a single orientation. On extrinsic operands `dot` takes the scalar part of
`left * right.reverse()`, while on intrinsic operands (points, directions, and their joins) it
composes that pairing through `dual` (`left.dual().dot(right.dual())`) into one kernel on the
operands' native layouts, rejecting mixed orientations with `GATypeError`.

Self-products (`squared`, `scalar_norm_squared`, `study_norm_squared`, and symmetric involution
products) symmetrize cross-blade cancellations in `int8` before runtime so off-diagonal blades that
cancel identically under `x * f(x)` never enter the output support. Similarly, `transform` builds a
ternary operator over `(versor, passenger, versor)`, symmetrizes the two versor slots with exact
integer halving, and applies the passenger's grade-involution and intrinsic reflection sign raised
to the versor's grade parity, rejecting mixed-parity versor types whose reflection count is
indeterminate. Sign, product, dual, and norm conventions are set out in `docs/conventions.md`, and
exact table construction in `docs/internals.md` (sections 2 and 4).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

from numga.algebra.self_product import grade_transform_signs, symmetric_product_table
from numga.binding import AxisTransform
from numga.extensor import Extensor
from numga.gatype import GAType, Orientation, UnorientedSubspace, rules
from numga.gatype.errors import GATypeError
from numga.gatype.rules import OperationRule
from numga.gatype.traits import EMPTY_TRAITS, BladeProduct, Identity, TraitSet
from numga.operator.kernel import SymbolicKernel
from numga.subspace import SubSpace

if TYPE_CHECKING:
    from numga.algebra import Algebra


class OperatorFactory:
    """Cached builders construct exact output-first operation extensors in one `Algebra`'s symbolic
    context."""

    __slots__ = ("algebra", "subspaces")

    def __init__(self, algebra: Algebra) -> None:
        self.algebra = algebra
        self.subspaces = algebra.subspace

    def build(
        self, axes: Iterable[SubSpace], orientations: Iterable[Orientation], coefficients: object,
        traits: TraitSet = EMPTY_TRAITS,
    ) -> Extensor:
        """Constructing from output-first `SubSpace` axes, orientations, and exact coefficients
        produces an unbatched symbolic `Extensor`."""
        axes = tuple(axes)
        if not axes:
            raise ValueError("an Extensor requires at least an output axis")
        if any(axis.algebra is not self.algebra for axis in axes):
            raise ValueError("every Extensor axis must belong to this factory's algebra")
        return Extensor(
            self.algebra.exact,
            self.algebra.gatype.oriented(axes, orientations, traits),
            SymbolicKernel(coefficients),
        )

    @lru_cache(maxsize=None)
    def unit(self, gatype: GAType) -> Extensor:
        """Projecting the multiplicative unit `1` onto `gatype` produces an exact multivector in
        `gatype`'s blade layout."""
        return self.algebra.exact.multivector(gatype)

    @lru_cache(maxsize=None)
    def identity(self, gatype: GAType) -> Extensor:
        """An identity map on `gatype`'s subspace preserves both blade layout and orientation while
        carrying the `Identity` trait."""
        gatype = self._nullary(gatype)
        space = gatype.output_subspace
        orientations = (gatype.orientation, gatype.orientation)
        return self.build((space, space), orientations, SymbolicKernel.identity(len(space))).with_traits(Identity)

    @lru_cache(maxsize=None)
    def _grade_transform(self, gatype: GAType, transform: str) -> Extensor:
        gatype = self._nullary(gatype)
        space = gatype.output_subspace
        coefficients = np.diag(grade_transform_signs(self.algebra, transform, space.masks))
        return self.build((space, space), (gatype.orientation, gatype.orientation), coefficients)

    def reverse(self, gatype: GAType) -> Extensor:
        """Blade reversal negates grades `2` and `3` modulo `4` by reversing generator order in
        each blade."""
        return self._grade_transform(gatype, "reverse")

    def clifford_conjugate(self, gatype: GAType) -> Extensor:
        """Clifford conjugation composes blade reversal with grade involution, negating grades `1`
        and `2` modulo `4` without conjugating scalars."""
        return self._grade_transform(gatype, "clifford_conjugate")

    def involute(self, gatype: GAType) -> Extensor:
        """Grade involution negates odd-grade blades and leaves even-grade blades unchanged."""
        return self._grade_transform(gatype, "involute")

    def scalar_negation(self, gatype: GAType) -> Extensor:
        """Study conjugation preserves scalar blades and negates every non-scalar blade."""
        return self._grade_transform(gatype, "scalar_negation")

    def pseudoscalar_negation(self, gatype: GAType) -> Extensor:
        """Scalar-pseudoscalar conjugation preserves scalar and pseudoscalar blades while negating
        every intermediate grade."""
        return self._grade_transform(gatype, "pseudoscalar_negation")

    @lru_cache(maxsize=None)
    def _symmetric_product(
        self, gatype: GAType, transform: str, *, scalar_only: bool = False,
    ) -> Extensor:
        space = gatype.output_subspace
        output = (
            self.subspaces.scalar() if scalar_only
            else gatype.derive.self_product(transform).output_subspace
        )
        blades, table = symmetric_product_table(space, transform)
        # Every pair's coefficient goes to the row of its output blade, in both layouts' blade orientations;
        # a blade the output does not hold, such as non-scalar terms of a scalar part, is left out.
        rows = np.full(1 << self.algebra.dimension, -1)
        rows[list(output.masks)] = np.arange(len(output))
        left, right = np.nonzero((rows[blades] >= 0) & (table != 0))
        signs, output_signs = np.asarray(space.signs), np.asarray(output.signs)
        row = rows[blades[left, right]]
        coefficients = np.zeros((len(output), len(space), len(space)), dtype=np.int8)
        coefficients[row, left, right] = table[left, right] * signs[left] * signs[right] * output_signs[row]
        # Pairing a value against itself supplies equal orientations on both sides, giving an extrinsic output.
        orientations = (Orientation.extrinsic, gatype.orientation, gatype.orientation)
        return self.build((output, space, space), orientations, coefficients)

    def squared(self, operand: GAType) -> Extensor:
        """Symmetrized self-multiplication `x * x` cancels anticommuting cross-blade pairs in the
        static support."""
        return self._symmetric_product(self._operand_gatype(operand), "identity")

    def symmetric_reverse_product(self, operand: GAType) -> Extensor:
        """Symmetrized reverse multiplication `x * x.reverse()` retains only the grades certified
        by `operand`'s traits and blade support."""
        return self._symmetric_product(self._operand_gatype(operand), "reverse")

    def scalar_norm_squared(self, operand: GAType) -> Extensor:
        """Building only the grade-zero row of `x * x.reverse()` yields the extrinsic scalar
        norm-squared form."""
        return self._symmetric_product(self._operand_gatype(operand), "reverse", scalar_only=True)

    def symmetric_conjugate_product(self, operand: GAType) -> Extensor:
        """Symmetrized Clifford-conjugate multiplication `x * x.clifford_conjugate()` cancels
        vanishing cross-blade pairs in the static support."""
        return self._symmetric_product(self._operand_gatype(operand), "clifford_conjugate")

    def symmetric_scalar_negation_product(self, operand: GAType) -> Extensor:
        """Symmetrized Study-conjugate multiplication `x * x.scalar_negation()` cancels vanishing
        cross-blade pairs in the static support."""
        return self._symmetric_product(self._operand_gatype(operand), "scalar_negation")

    def study_norm_squared(self, operand: GAType) -> Extensor:
        """Building only the grade-zero row of `x * x.scalar_negation()` yields the generalized
        Study norm-squared form."""
        return self._symmetric_product(
            self._operand_gatype(operand), "scalar_negation", scalar_only=True,
        )

    def symmetric_pseudoscalar_negation_product(self, operand: GAType) -> Extensor:
        """Symmetrized multiplication `x * x.pseudoscalar_negation()` cancels vanishing cross-blade
        pairs in the static support."""
        return self._symmetric_product(self._operand_gatype(operand), "pseudoscalar_negation")

    def symmetric_involute_product(self, operand: GAType) -> Extensor:
        """Symmetrized grade-involution multiplication `x * x.involute()` cancels vanishing
        cross-blade pairs in the static support."""
        return self._symmetric_product(self._operand_gatype(operand), "involute")

    @lru_cache(maxsize=None)
    def cast(self, source: GAType, target: SubSpace) -> Extensor:
        """Signed coordinate conversion reorders, projects, or zero-pads `source` into `target`
        while preserving orientation, attaching `Identity` whenever the conversion is lossless."""
        source = self._nullary(source)
        source_space = source.output_subspace
        self._require_space(target)
        transform = AxisTransform.plan(source_space, target)
        if not transform.is_compatible:
            raise ValueError("cannot cast between SubSpaces from different algebras")
        coefficients = SymbolicKernel(
            transform.coordinate_matrix, shape=(len(target), len(source_space)),
        )
        result = self.build((target, source_space), (source.orientation, source.orientation), coefficients)
        return result.with_traits(Identity) if transform.is_lossless else result

    select = cast

    @lru_cache(maxsize=None)
    def restrict(self, source: GAType, target: SubSpace) -> Extensor:
        """Intersecting `source`'s support with `target` projects away absent blades without
        zero-padding missing target blades."""
        source = self._nullary(source)
        return self.select(source, source.output_subspace.intersection(target))

    def geometric_product(self, left: GAType, right: GAType) -> Extensor:
        """Full Clifford multiplication retains every non-zero output blade of `left` and `right`
        and propagates traits through `rules.GEOMETRIC`."""
        return self._geometric_product(
            self._operand_gatype(left),
            self._operand_gatype(right),
        )

    @lru_cache(maxsize=None)
    def _geometric_product(self, left: GAType, right: GAType) -> Extensor:
        return self._product(
            rules.GEOMETRIC,
            left,
            right,
            lambda _left_grade, _right_grade, _output_grade: True,
        )

    product = geometric_product

    def scalar_product(self, left: GAType, right: GAType) -> Extensor:
        """Grade-zero projection builds only the scalar row of the geometric product of `left` and
        `right`."""
        return self._grade_product(self._operand_gatype(left), self._operand_gatype(right), 0)

    def dot(self, left: GAType, right: GAType) -> Extensor:
        """Metric pairing extends across every grade according to the operands' shared orientation.
        On two extrinsic operands it takes the scalar part of `left * right.reverse()`, giving the
        determinant of vector-factor dot products within a grade (positive on every blade in a
        Euclidean signature, zero between distinct grades, and equal to `scalar_norm_squared` on a
        value with itself). On two intrinsic operands (points, directions, and their joins) it
        evaluates `left.dual().dot(right.dual())` using the metric dual in a non-degenerate algebra
        and the complement in a degenerate one. In a non-degenerate algebra the two readings differ
        by the sign of the antiscalar's own reverse product, agreeing in space and opposing in
        spacetime so a spatial direction's dot with itself equals its squared length. In a
        projective algebra the reverse product of a point measures weight whereas `dot` measures
        bulk (weight times distance from the origin), and on a direction `dot` gives its squared
        length. Intrinsic and extrinsic operands carry no metric between them and pair instead
        through the regressive product."""
        left, right = self._operand_gatype(left), self._operand_gatype(right)
        if left.orientation is not right.orientation:
            raise GATypeError(
                f"dot takes operands of one orientation; got {left.orientation.name} and {right.orientation.name}: "
                "an extrinsic value and an intrinsic one pair by the regressive product, extrinsic & intrinsic"
            )
        if left.orientation is Orientation.extrinsic:
            return self._dot(left, right)
        return self._dual_dot(left, right)

    @lru_cache(maxsize=None)
    def _dot(self, left: GAType, right: GAType) -> Extensor:
        product = self._grade_product(left, right, 0)
        # Reversing every blade of the right operand multiplies the scalar-product columns by their reverse signs.
        reverses = grade_transform_signs(self.algebra, "reverse", right.output_subspace.masks)
        return Extensor(self.algebra.exact, product.gatype, SymbolicKernel(product._kernel._values * reverses))

    @lru_cache(maxsize=None)
    def _dual_dot(self, left: GAType, right: GAType) -> Extensor:
        left_dual, right_dual = self.dual(left.derive.output), self.dual(right.derive.output)
        # Because duals are signed blade permutations, composing dot with both duals yields a single
        # kernel on the operands' native layouts while preserving any map input axes.
        composed = self._dot(left_dual.gatype.derive.output, right_dual.gatype.derive.output).bind(
            {0: left_dual, 1: right_dual},
        )
        gatype = rules.MULTIPLIED.result((left, right), composed.axes)
        return Extensor(self.algebra.exact, gatype, composed.kernel)

    def bivector_product(self, left: GAType, right: GAType) -> Extensor:
        """Grade-two projection builds only the bivector rows of the geometric product of `left`
        and `right`."""
        return self._grade_product(self._operand_gatype(left), self._operand_gatype(right), 2)

    def trivector_product(self, left: GAType, right: GAType) -> Extensor:
        """Grade-three projection builds only the trivector rows of the geometric product of `left`
        and `right`."""
        return self._grade_product(self._operand_gatype(left), self._operand_gatype(right), 3)

    def inner(self, left: GAType, right: GAType) -> Extensor:
        """Symmetric inner multiplication retains only the grade-`|r - s|` terms of blades of
        grades `r` and `s`."""
        return self._inner(self._operand_gatype(left), self._operand_gatype(right))

    @lru_cache(maxsize=None)
    def _inner(self, left: GAType, right: GAType) -> Extensor:
        return self._product(
            rules.MULTIPLIED, left, right,
            lambda left_grade, right_grade, output_grade: (
                output_grade == abs(np.subtract(left_grade, right_grade, dtype=np.int16))
            ),
        )

    def left_contraction(self, left: GAType, right: GAType) -> Extensor:
        """Left contraction retains only the grade-`s - r` terms of blades of grades `r` and `s`,
        vanishing whenever `r > s`."""
        return self._left_contraction(self._operand_gatype(left), self._operand_gatype(right))

    @lru_cache(maxsize=None)
    def _left_contraction(self, left: GAType, right: GAType) -> Extensor:
        return self._product(
            rules.MULTIPLIED, left, right,
            lambda left_grade, right_grade, output_grade: (
                output_grade == np.subtract(right_grade, left_grade, dtype=np.int16)
            ),
        )

    def right_contraction(self, left: GAType, right: GAType) -> Extensor:
        """Right contraction retains only the grade-`r - s` terms of blades of grades `r` and `s`,
        vanishing whenever `s > r`."""
        return self._right_contraction(self._operand_gatype(left), self._operand_gatype(right))

    @lru_cache(maxsize=None)
    def _right_contraction(self, left: GAType, right: GAType) -> Extensor:
        return self._product(
            rules.MULTIPLIED, left, right,
            lambda left_grade, right_grade, output_grade: (
                output_grade == np.subtract(left_grade, right_grade, dtype=np.int16)
            ),
        )

    def left_interior(self, left: GAType, right: GAType) -> Extensor:
        """Left interior multiplication takes the anti-wedge of the metric left complement of
        `left` with `right`."""
        return self._left_interior(self._operand_gatype(left), self._operand_gatype(right))

    @lru_cache(maxsize=None)
    def _left_interior(self, left: GAType, right: GAType) -> Extensor:
        def basis_rule(left_mask: int, right_mask: int) -> tuple[int, int]:
            complement, sign = self._left_complement(left_mask)
            output, coefficient = self._anti_wedge_rule(complement, right_mask)
            return output, sign * coefficient

        return self._bilinear(rules.MULTIPLIED, left, right, basis_rule)

    def right_interior(self, left: GAType, right: GAType) -> Extensor:
        """Right interior multiplication takes the anti-wedge of `left` with the metric right
        complement of `right`."""
        return self._right_interior(self._operand_gatype(left), self._operand_gatype(right))

    @lru_cache(maxsize=None)
    def _right_interior(self, left: GAType, right: GAType) -> Extensor:
        def basis_rule(left_mask: int, right_mask: int) -> tuple[int, int]:
            complement, sign = self._right_complement(right_mask)
            output, coefficient = self._anti_wedge_rule(left_mask, complement)
            return output, sign * coefficient

        return self._bilinear(rules.MULTIPLIED, left, right, basis_rule)

    def _swap_sign(self, left_mask: int, right_mask: int) -> int:
        swaps = sum((left_mask >> (bit + 1)).bit_count() for bit in range(right_mask.bit_length()) if right_mask >> bit & 1)
        return -1 if swaps % 2 else 1

    def _negative_sign(self, mask: int) -> int:
        return -1 if (mask & self.algebra.negative_mask).bit_count() % 2 else 1

    def _right_complement(self, mask: int) -> tuple[int, int]:
        # Right complement including negative metric signs, ordered as mask followed by the pseudoscalar.
        pseudoscalar = self.algebra.pseudoscalar_mask
        return mask ^ pseudoscalar, self._swap_sign(mask, pseudoscalar) * self._negative_sign(mask)

    def _left_complement(self, mask: int) -> tuple[int, int]:
        # Left complement including negative metric signs, ordered as the pseudoscalar followed by mask.
        pseudoscalar = self.algebra.pseudoscalar_mask
        return mask ^ pseudoscalar, self._swap_sign(pseudoscalar, mask) * self._negative_sign(mask)

    def _volume_complement(self, mask: int) -> tuple[int, int]:
        # Metric-free right volume complement satisfying mask ^ complement == pseudoscalar.
        pseudoscalar = self.algebra.pseudoscalar_mask
        return mask ^ pseudoscalar, self._swap_sign(mask, mask ^ pseudoscalar)

    def _volume_complement_inverse(self, mask: int) -> tuple[int, int]:
        # Metric-free left volume complement satisfying complement ^ mask == pseudoscalar.
        pseudoscalar = self.algebra.pseudoscalar_mask
        return mask ^ pseudoscalar, self._swap_sign(mask ^ pseudoscalar, mask)

    def _anti_wedge_rule(self, left_mask: int, right_mask: int) -> tuple[int, int]:
        left_complement, left_sign = self._right_complement(left_mask)
        right_complement, right_sign = self._right_complement(right_mask)
        if left_complement & right_complement:
            return 0, 0
        wedge = left_complement | right_complement
        output, output_sign = self._left_complement(wedge)
        return output, left_sign * right_sign * self._swap_sign(left_complement, right_complement) * output_sign

    @lru_cache(maxsize=None)
    def _grade_product(self, left: GAType, right: GAType, grade: int) -> Extensor:
        return self._product(
            rules.MULTIPLIED, left, right,
            lambda _left_grade, _right_grade, output_grade: output_grade == grade,
        )

    def wedge(self, left: GAType, right: GAType) -> Extensor:
        """Exterior multiplication retains only the grade-`r + s` terms of blades of grades `r` and
        `s` without consulting the metric."""
        return self._wedge(
            self._operand_gatype(left),
            self._operand_gatype(right),
        )

    @lru_cache(maxsize=None)
    def _wedge(self, left: GAType, right: GAType) -> Extensor:
        return self._product(
            rules.WEDGE,
            left,
            right,
            lambda left_grade, right_grade, output_grade: (
                left_grade + right_grade == output_grade
            ),
        )

    exterior_product = wedge

    def commutator(self, left: GAType, right: GAType) -> Extensor:
        """Exact integer halving of `left * right - right * left` retains only anticommuting
        basis-blade pairs."""
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
            # Basis blades either commute or anticommute, so half their difference is zero or the product itself.
            return forward.blade, (forward.coefficient - reverse.coefficient) // 2

        return self._bilinear(rules.MULTIPLIED, left, right, basis_rule)

    def anticommutator(self, left: GAType, right: GAType) -> Extensor:
        """Exact integer halving of `left * right + right * left` retains only commuting
        basis-blade pairs."""
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
            # Basis blades either commute or anticommute, so half their sum is the product itself or zero.
            return forward.blade, (forward.coefficient + reverse.coefficient) // 2

        return self._bilinear(rules.MULTIPLIED, left, right, basis_rule)

    def regressive(self, left: GAType, right: GAType) -> Extensor:
        """Complementary blades satisfy `left ^ right == (left & right) * antiscalar`, measuring
        their spanned volume in units of the intrinsic unit pseudoscalar `antiscalar`; at general
        grades the regressive product extracts the subspace shared by `left` and `right` scaled by
        the volume they span together. No metric enters, and dividing out the intrinsic unit volume
        flips orientation once."""
        return self._regressive(
            self._operand_gatype(left),
            self._operand_gatype(right),
        )

    @lru_cache(maxsize=None)
    def _regressive(self, left: GAType, right: GAType) -> Extensor:
        def basis_rule(left_mask: int, right_mask: int) -> tuple[int, int]:
            # Each blade's volume complement holds the generators it lacks, so wedging those complements
            # and taking the inverse complement isolates the generators the two blades share.
            left_complement, left_sign = self._volume_complement(left_mask)
            right_complement, right_sign = self._volume_complement(right_mask)
            if left_complement & right_complement:
                return 0, 0
            swap_sign = self._swap_sign(left_complement, right_complement)
            output, output_sign = self._volume_complement_inverse(left_complement | right_complement)
            return output, left_sign * right_sign * swap_sign * output_sign

        return self._bilinear(rules.REGRESSIVE, left, right, basis_rule)

    @lru_cache(maxsize=None)
    def cross(self, vector_space: SubSpace) -> Extensor:
        """In Euclidean 3-space, the Hodge dual of the wedge of two extrinsic vectors produces an
        intrinsic vector."""
        self._require_space(vector_space)
        if self.algebra.dimension != 3 or self.algebra.signature != (1, 1, 1):
            raise ValueError("cross is defined here only for Euclidean 3-space")
        if not vector_space.same_support(self.subspaces.vector()):
            raise ValueError("cross requires the algebra's full vector SubSpace")

        coefficients = np.zeros((3, 3, 3), dtype=np.int8)
        for output in range(3):
            for left in range(3):
                for right in range(3):
                    if len({output, left, right}) != 3:
                        continue
                    masks = (vector_space.masks[output], vector_space.masks[left], vector_space.masks[right])
                    inversions = sum(
                        first > second
                        for position, first in enumerate(masks)
                        for second in masks[position + 1 :]
                    )
                    coefficients[output, left, right] = (
                        (-1 if inversions % 2 else 1)
                        * vector_space.signs[output] * vector_space.signs[left] * vector_space.signs[right]
                    )
        # Taking the dual of an extrinsic wedge flips the output orientation to intrinsic.
        return self.build(
            (vector_space, vector_space, vector_space),
            (Orientation.intrinsic, Orientation.extrinsic, Orientation.extrinsic),
            coefficients,
        )

    def transform(self, versor: GAType, passenger: GAType) -> Extensor:
        """Reflecting an extrinsic passenger `x` in a unit vector `m` gives
        `m * x.involute() * m.inverse()` and negates that image when `x` is intrinsic; a versor
        factored into `k` vectors composes `k` reflections, so both grade involution and the
        intrinsic sign enter `k` times and cancel for even versors. Replacing the inverse with
        `versor.reverse()` matches the inverse on unit versors while keeping the ternary operator
        polynomial, and mixed-parity versor types are rejected because `k mod 2` is indeterminate."""
        return self._transform(self._operand_gatype(versor), self._operand_gatype(passenger))

    @lru_cache(maxsize=None)
    def _transform(self, versor: GAType, passenger: GAType) -> Extensor:
        parities = set(self.algebra.grade(np.asarray(versor.output_subspace.masks, dtype=self.algebra.blade_dtype)) % 2)
        if len(parities) > 1:
            raise GATypeError(f"a sandwich needs an operand of one parity, even or odd, and {versor} holds both: "
                              "cast it to the parity it has, or write the product out")
        parity = int(1 in parities)
        result = self.full_sandwich(versor.derive.structural, passenger.derive.structural)
        kernel = result.kernel
        # Because axes 1 and 3 receive the same versor, averaging the two axis orders is exact on basis blades.
        kernel = (kernel + kernel.transpose((0, 3, 2, 1))).halved()
        result = self.build(result.axes, result.gatype.orientations, kernel).squeeze_output()
        # Odd-parity versors apply an odd number of reflections, multiplying by grade involution and the intrinsic sign.
        intrinsic = -1 if passenger.orientation is Orientation.intrinsic else 1
        involutes = grade_transform_signs(self.algebra, "involute", passenger.output_subspace.masks)
        signs = SymbolicKernel(np.diag((involutes * intrinsic) ** parity).astype(np.int8))
        passenger_space = passenger.output_subspace
        reflected = self.build((passenger_space, passenger_space), (passenger.orientation,) * 2, signs)
        result = result.bind({1: reflected})
        gatype = rules.TRANSFORM.result((versor, passenger, versor), result.axes)
        if gatype is result.gatype:
            return result
        return Extensor(result.context, gatype, result.kernel)

    def reverse_transform(self, versor: GAType, passenger: GAType) -> Extensor:
        """Folding `versor.reverse()` into both outer axes of `transform` gives the action of the
        reversed versor, which inverts `transform` on unit versors."""
        return self._reverse_transform(self._operand_gatype(versor), self._operand_gatype(passenger))

    @lru_cache(maxsize=None)
    def _reverse_transform(self, versor: GAType, passenger: GAType) -> Extensor:
        forward = self._transform(versor, passenger)
        reverse = self.reverse(versor.derive.structural)
        # Because reversal is diagonal on the space, folding it into both versor axes preserves grade
        # cancellations and versor traits.
        folded = forward.bind({0: reverse, 2: reverse})
        return Extensor(folded.context, forward.gatype, folded.kernel)

    @lru_cache(maxsize=None)
    def full_sandwich(self, sandwicher: GAType, passenger: GAType) -> Extensor:
        """Ternary multiplication `left * passenger * right.reverse()` builds the unsymmetrized
        sandwich tensor before parity or diagonal reductions."""
        left_product = self.geometric_product(sandwicher, passenger)
        right_product = self.geometric_product(left_product.gatype.derive.output, sandwicher)
        return right_product.bind({0: left_product, 1: self.reverse(sandwicher)})

    def _product(
        self,
        rule: OperationRule,
        left_type: GAType,
        right_type: GAType,
        grade_rule: Callable,
    ) -> Extensor:
        left = left_type.output_subspace
        right = right_type.output_subspace
        self._require_space(left)
        self._require_space(right)
        table = self.algebra.geometric_product_table(left.masks, right.masks)

        left_grades = self.algebra.grade(np.asarray(left.masks, dtype=self.algebra.blade_dtype))[:, None]
        right_grades = self.algebra.grade(np.asarray(right.masks, dtype=self.algebra.blade_dtype))[None, :]
        output_grades = self.algebra.grade(table.blades)
        coeffs = table.coefficients

        try:
            valid = (coeffs != 0) & np.asarray(grade_rule(left_grades, right_grades, output_grades), dtype=bool)
        except TypeError:
            valid = np.zeros(coeffs.shape, dtype=bool)
            for i in range(len(left.masks)):
                for j in range(len(right.masks)):
                    if coeffs[i, j]:
                        valid[i, j] = grade_rule(
                            int(left_grades[i, 0]), int(right_grades[0, j]), int(output_grades[i, j])
                        )

        output_masks = np.unique(table.blades[valid])
        output = self.subspaces.from_masks(output_masks)
        output_signs = np.asarray(output.signs, dtype=np.int8)
        left_signs = np.asarray(left.signs, dtype=np.int8)
        right_signs = np.asarray(right.signs, dtype=np.int8)

        signed_coeffs = coeffs * left_signs[:, None] * right_signs[None, :]
        coefficients = np.zeros((len(output), len(left), len(right)), dtype=np.int8)
        for row, mask in enumerate(output.masks):
            match = valid & (table.blades == mask)
            coefficients[row, match] = signed_coeffs[match] * output_signs[row]

        gatype = rule.result((left_type, right_type), (output, left, right))
        return Extensor(self.algebra.exact, gatype, SymbolicKernel(coefficients))

    def _bilinear(
        self,
        rule: OperationRule,
        left_type: GAType,
        right_type: GAType,
        basis_rule: Callable,
    ) -> Extensor:
        left = left_type.output_subspace
        right = right_type.output_subspace
        self._require_space(left)
        self._require_space(right)

        terms = []
        output_masks = set()
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

        gatype = rule.result((left_type, right_type), (output, left, right))
        return Extensor(self.algebra.exact, gatype, SymbolicKernel(coefficients))

    @lru_cache(maxsize=None)
    def dual(self, gatype: GAType) -> Extensor:
        """Right-Hodge duality maps each blade to its signed complement in the algebra's canonical
        output layout and flips orientation between intrinsic and extrinsic."""
        return self._dual(gatype, inverse=False)

    @lru_cache(maxsize=None)
    def dual_inverse(self, gatype: GAType) -> Extensor:
        """Inverting `dual` restores each complement blade with the forward sign of its preimage so
        `dual_inverse(dual(x)) == x` holds in every signature, flipping orientation back."""
        return self._dual(gatype, inverse=True)

    def _dual(self, gatype: GAType, *, inverse: bool) -> Extensor:
        gatype = self._nullary(gatype)
        space = gatype.output_subspace
        rule = self._right_hodge_inverse if inverse else self._right_hodge
        terms = tuple(rule(mask) for mask in space.masks)
        output = self.subspaces.from_masks(mask for mask, _ in terms)
        indices = {mask: index for index, mask in enumerate(output.masks)}
        coefficients = np.zeros((len(output), len(space)), dtype=np.int8)
        for column, (mask, sign) in enumerate(terms):
            row = indices[mask]
            coefficients[row, column] = sign * space.signs[column] * output.signs[row]
        # The complement of a blade is a blade.
        return self.build((output, space), (gatype.orientation.flipped, gatype.orientation), coefficients,
                          TraitSet((BladeProduct((0,)),)))

    def _right_hodge(self, mask: int) -> tuple[int, int]:
        complement = self.algebra.complement(mask)
        orientation = self.algebra.geometric_product(mask, complement).coefficient
        negative_sign = (
            -1
            if (mask & self.algebra.negative_mask).bit_count() % 2
            else 1
        )
        return complement, orientation * negative_sign

    def _right_hodge_inverse(self, mask: int) -> tuple[int, int]:
        complement = self.algebra.complement(mask)
        _, sign = self._right_hodge(complement)
        return complement, sign

    def _operand_gatype(self, operand: GAType) -> GAType:
        if isinstance(operand, SubSpace):
            raise UnorientedSubspace()
        if not isinstance(operand, GAType) or operand.algebra is not self.algebra:
            raise ValueError("operation operand type belongs to another algebra")
        # Operators act on blades; field operands broadcast site by site when bound.
        return operand.derive.plain

    def _nullary(self, gatype: GAType) -> GAType:
        if isinstance(gatype, SubSpace):
            raise UnorientedSubspace()
        if not isinstance(gatype, GAType) or gatype.algebra is not self.algebra or gatype.arity:
            raise TypeError("an operator is built from a nullary GAType of this algebra")
        return gatype

    def _require_space(self, space: SubSpace) -> None:
        if not isinstance(space, SubSpace) or space.algebra is not self.algebra:
            raise ValueError("SubSpace belongs to another algebra")
