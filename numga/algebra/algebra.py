"""A finite-dimensional geometric algebra with up to 64 orthogonal generators maps each canonical
basis blade to a non-negative integer bit mask whose `i`-th bit records generator `i` in increasing
index order. Because the metric is diagonal with entries in `(+1, -1, 0)`, the geometric product of
any two canonical basis blades produces a single output mask `left ^ right` scaled by an integer
sign in `(-1, 0, +1)` combining pairwise generator swaps with negative and null metric contractions.

`Algebra` accepts an `AlgebraDescription`, a compact interleaved string such as `"x+y+z+w0"`, or a
`(p, q, r)` count tuple; per-generator metric sequences go through `Algebra.from_signature` so a
three-dimensional metric tuple `(1, 1, 1)` cannot collide with signature counts. Instances are
weakly interned by `(description, subspace_factory, gatype_factory, multivector_factory)` so
repeated constructions of `Algebra("x+y+")` return the same `Algebra` object and their extensors
interoperate, whereas stock algebras with custom layouts or value constructors such as `PGA3D`
remain distinct from plain `Algebra("w0x+y+z+")`.

Each `Algebra` owns the per-algebra flyweight namespaces that are independent of a numerical array
backend: `subspace` for signed blade layouts, `gatype` for whole-extensor types and static trait
deduction, `operator` for exact bilinear kernels, and `exact` for symbolic integer-array execution,
while storing the `multivector_factory` callable bound by each numerical `Context`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from operator import index
from types import NotImplementedType
from typing import TYPE_CHECKING, NamedTuple
import weakref

import numpy as np

from .bitops import (
    bit_count,
    mask_from_indices,
    parity_to_sign,
    permutation_sign,
    unsigned_dtype,
)
from .description import AlgebraDescription

if TYPE_CHECKING:
    from numga.backend.context import Context
    from numga.backend.exact import ExactContext
    from numga.multivector.factory import MultivectorFactory
    from numga.gatype.factory import GATypeFactory
    from numga.operator.factory import OperatorFactory
    from numga.subspace.factory import SubSpaceFactory


class BladeProduct(NamedTuple):
    """Multiplying two canonical basis blades yields an integer `coefficient` in `(-1, 0, +1)` and
    a canonical output `blade` mask."""

    coefficient: int
    blade: int

    @property
    def mask(self) -> int:
        return self.blade


class OrientedBlade(NamedTuple):
    """Parsing a spelled exterior blade yields its canonical `blade` mask and permutation `sign` in
    `(-1, +1)`."""

    blade: int
    sign: int

    @property
    def mask(self) -> int:
        return self.blade


@dataclass(frozen=True, slots=True, eq=False)
class ProductTable:
    """Pairwise geometric products across two ordered blade sequences store immutable output
    `blades` and integer `coefficients` matrices."""

    blades: np.ndarray
    coefficients: np.ndarray

    def __post_init__(self) -> None:
        blades = np.asarray(self.blades)
        coefficients = np.asarray(self.coefficients, dtype=np.int8)
        if blades.shape != coefficients.shape or blades.ndim != 2:
            raise ValueError("product-table arrays must be equally shaped matrices")

        # Arrays backed by immutable bytes cannot be made writeable again by a caller,
        # unlike an owning ndarray with only its writeable flag cleared.
        frozen_blades = np.frombuffer(blades.tobytes(), dtype=blades.dtype).reshape(
            blades.shape
        )
        frozen_coefficients = np.frombuffer(
            coefficients.tobytes(), dtype=np.int8
        ).reshape(coefficients.shape)
        object.__setattr__(self, "blades", frozen_blades)
        object.__setattr__(self, "coefficients", frozen_coefficients)

    @property
    def shape(self) -> tuple[int, int]:
        return self.blades.shape


def _describe(specification: AlgebraDescription | str | tuple[int, int, int]) -> AlgebraDescription:
    if isinstance(specification, AlgebraDescription):
        description = specification
    elif isinstance(specification, str):
        description = AlgebraDescription.parse(specification)
    elif isinstance(specification, tuple) and len(specification) == 3:
        description = AlgebraDescription.from_pqr(*specification)
    else:
        raise TypeError(
            "Algebra expects an AlgebraDescription, compact string, or (p, q, r) tuple"
        )
    if description.dimension > 64:
        raise ValueError("the bit-mask implementation supports at most 64 generators")
    return description


# Weakly interning algebras by description and factory callables lets repeated constructions share one Algebra instance.
_interned = weakref.WeakValueDictionary()


@dataclass(frozen=True, slots=True, init=False, weakref_slot=True)
class Algebra:
    """An immutable geometric algebra represents basis blades as canonical integer bit masks and
    interns instances by description and factory callables."""

    description: AlgebraDescription
    _subspace_factory: SubSpaceFactory = field(compare=False, repr=False)
    _gatype_factory: GATypeFactory = field(compare=False, repr=False)
    _exact_context: ExactContext = field(compare=False, repr=False)
    _operator_factory: OperatorFactory = field(compare=False, repr=False)
    _multivector_factory: Callable = field(compare=False, repr=False)

    def __new__(
        cls,
        specification: AlgebraDescription | str | tuple[int, int, int],
        *,
        subspace_factory: Callable | None = None,
        gatype_factory: Callable | None = None,
        multivector_factory: Callable | None = None,
    ) -> Algebra:
        key = (_describe(specification), subspace_factory, gatype_factory, multivector_factory)
        try:
            return _interned[key]
        except KeyError:
            instance = object.__new__(cls)
            _interned[key] = instance
            return instance

    def __init__(
        self,
        specification: AlgebraDescription | str | tuple[int, int, int],
        *,
        subspace_factory: Callable | None = None,
        gatype_factory: Callable | None = None,
        multivector_factory: Callable | None = None,
    ) -> None:
        # Returning an already-interned instance from __new__ still invokes __init__.
        try:
            self.description
            return
        except AttributeError:
            pass
        object.__setattr__(self, "description", _describe(specification))

        # Owning the construction namespaces ties their flyweight pools to the algebra's lifetime.
        from numga.backend.exact import ExactContext
        from numga.gatype.factory import GATypeFactory
        from numga.operator.factory import OperatorFactory
        from numga.subspace.factory import SubSpaceFactory

        object.__setattr__(self, "_subspace_factory", (subspace_factory or SubSpaceFactory)(self))
        object.__setattr__(self, "_gatype_factory", (gatype_factory or GATypeFactory)(self))
        # Storing the multivector factory callable lets every Context specialize value constructors.
        from numga.multivector.factory import MultivectorFactory
        object.__setattr__(self, "_multivector_factory", multivector_factory or MultivectorFactory)
        object.__setattr__(self, "_exact_context", ExactContext(self))
        object.__setattr__(self, "_operator_factory", OperatorFactory(self))

    @classmethod
    def from_description(
        cls,
        description: AlgebraDescription,
        *,
        subspace_factory: Callable | None = None,
        gatype_factory: Callable | None = None,
        multivector_factory: Callable | None = None,
    ) -> Algebra:
        return cls(
            description,
            subspace_factory=subspace_factory,
            gatype_factory=gatype_factory,
            multivector_factory=multivector_factory,
        )

    @classmethod
    def from_signature(
        cls,
        signature: Iterable[int],
        basis_names: Iterable[str] | None = None,
        *,
        subspace_factory: Callable | None = None,
        gatype_factory: Callable | None = None,
        multivector_factory: Callable | None = None,
    ) -> Algebra:
        """Per-generator metric squares in `(-1, 0, +1)` build an algebra without colliding with
        `(p, q, r)` count tuples."""
        return cls(
            AlgebraDescription.from_signature(signature, basis_names),
            subspace_factory=subspace_factory,
            gatype_factory=gatype_factory,
            multivector_factory=multivector_factory,
        )

    @classmethod
    def from_pqr(
        cls,
        p: int,
        q: int,
        r: int,
        basis_names: Iterable[str] | None = None,
        *,
        subspace_factory: Callable | None = None,
        gatype_factory: Callable | None = None,
        multivector_factory: Callable | None = None,
    ) -> Algebra:
        """Signature counts `(p, q, r)` order `+1` generators before `-1` and `0`."""
        return cls(
            AlgebraDescription.from_pqr(p, q, r, basis_names),
            subspace_factory=subspace_factory,
            gatype_factory=gatype_factory,
            multivector_factory=multivector_factory,
        )

    @property
    def dimension(self) -> int:
        return self.description.dimension

    @property
    def blade_count(self) -> int:
        return 1 << self.dimension

    @property
    def basis_names(self) -> tuple[str, ...]:
        return self.description.basis_names

    @property
    def signature(self) -> tuple[int, ...]:
        return self.description.signature

    @property
    def pqr(self) -> tuple[int, int, int]:
        return self.description.pqr

    @property
    def subspace(self) -> SubSpaceFactory:
        """Canonical `SubSpace` supports and signed blade layouts are pooled per algebra."""
        return self._subspace_factory

    @property
    def gatype(self) -> GATypeFactory:
        """Whole-extensor `GAType` instances and static trait derivations are pooled per algebra."""
        return self._gatype_factory

    @property
    def operator(self) -> OperatorFactory:
        """Exact bilinear and unary `SymbolicKernel` tables are cached per algebra."""
        return self._operator_factory

    @property
    def exact(self) -> ExactContext:
        """Symbolic integer-array execution runs in an algebra-owned `ExactContext`."""
        return self._exact_context

    @property
    def blade_dtype(self) -> np.dtype:
        return unsigned_dtype(self.dimension)

    @property
    def blade_masks(self) -> range:
        """Canonical integer blade masks span `range(blade_count)` lazily without allocating an
        array."""
        return range(self.blade_count)

    @property
    def basis_vector_masks(self) -> tuple[int, ...]:
        return tuple(1 << generator for generator in range(self.dimension))

    @property
    def pseudoscalar_mask(self) -> int:
        return self.blade_count - 1

    @property
    def positive_mask(self) -> int:
        return mask_from_indices(
            generator for generator, sign in enumerate(self.signature) if sign == 1
        )

    @property
    def negative_mask(self) -> int:
        return mask_from_indices(
            generator for generator, sign in enumerate(self.signature) if sign == -1
        )

    @property
    def degenerate_mask(self) -> int:
        return mask_from_indices(
            generator for generator, sign in enumerate(self.signature) if sign == 0
        )

    @property
    def negatives(self) -> int:
        return self.negative_mask

    @property
    def positives(self) -> int:
        return self.positive_mask

    @property
    def zeros(self) -> int:
        return self.degenerate_mask

    @property
    def blade_nbytes(self) -> int:
        return self.blade_dtype.itemsize

    @property
    def pseudo_scalar_squared(self) -> int:
        return self.pseudoscalar_squared

    @property
    def n_dimensions(self) -> int:
        return self.dimension

    @property
    def n_blades(self) -> int:
        return self.blade_count

    @property
    def n_grades(self) -> int:
        return self.dimension + 1

    def __len__(self) -> int:
        return self.blade_count

    @property
    def pseudoscalar_squared(self) -> int:
        return self.geometric_product(
            self.pseudoscalar_mask, self.pseudoscalar_mask
        ).coefficient

    def _mask(self, blade: int) -> int:
        try:
            mask = index(blade)
        except TypeError as error:
            raise TypeError("a blade mask must be an integer") from error
        if mask < 0 or mask >= self.blade_count:
            raise ValueError(
                f"blade mask {mask} is outside this {self.dimension}-dimensional algebra"
            )
        return mask

    def _masks_to_array(self, blades: Iterable[int] | np.ndarray) -> np.ndarray:
        if isinstance(blades, np.ndarray):
            arr = blades
        elif isinstance(blades, (list, tuple, range)):
            if not blades:
                return np.empty(0, dtype=self.blade_dtype)
            arr = np.asarray(blades)
        else:
            blades_tuple = tuple(blades)
            if not blades_tuple:
                return np.empty(0, dtype=self.blade_dtype)
            arr = np.asarray(blades_tuple)

        if arr.dtype.kind not in ("i", "u"):
            raise TypeError("a blade mask must be an integer")
        if np.any((arr < 0) | (arr >= self.blade_count)):
            raise ValueError(
                f"blade mask is outside this {self.dimension}-dimensional algebra"
            )
        return arr.astype(self.blade_dtype)

    def bit_dot(
        self, left: int | np.ndarray, right: int | np.ndarray
    ) -> int | np.ndarray:
        """Bitwise overlap counts how many basis generators appear in both `left` and `right`."""
        if isinstance(left, (int, np.integer)) and isinstance(right, (int, np.integer)):
            return (int(left) & int(right)).bit_count()
        return bit_count(np.bitwise_and(left, right))

    def grade(self, blade: int | Iterable[int] | np.ndarray) -> int | np.ndarray:
        """Set bits in a canonical blade mask count the basis generators in its wedge product."""
        if isinstance(blade, (int, np.integer)):
            return self._mask(blade).bit_count()
        return bit_count(self._masks_to_array(blade))

    def complement(self, blade: int | Iterable[int] | np.ndarray) -> int | np.ndarray:
        """XORing with `pseudoscalar_mask` toggles every generator bit to give the unsigned
        complement mask."""
        if isinstance(blade, (int, np.integer)):
            return self._mask(blade) ^ self.pseudoscalar_mask
        return np.bitwise_xor(self._masks_to_array(blade), self.pseudoscalar_mask)

    def involute_sign(self, blade: int | Iterable[int] | np.ndarray) -> int | np.ndarray:
        """Grade involution multiplies grade `k` by `(-1)**k`."""
        grade = self.grade(blade)
        return parity_to_sign(grade)

    involute = involute_sign

    def reverse_sign(self, blade: int | Iterable[int] | np.ndarray) -> int | np.ndarray:
        """Reversing generator order in a `k`-blade multiplies by `(-1)**(k * (k - 1) // 2)`."""
        grade = self.grade(blade)
        if isinstance(grade, (int, np.integer)):
            return parity_to_sign(int(grade) * (int(grade) - 1) // 2)
        return parity_to_sign((grade * (grade - 1)) // 2)

    reverse = reverse_sign

    def conjugate_sign(self, blade: int | Iterable[int] | np.ndarray) -> int | np.ndarray:
        """Clifford conjugation composes grade involution with reversion, multiplying grade `k` by
        `(-1)**(k * (k + 1) // 2)`."""
        grade = self.grade(blade)
        if isinstance(grade, (int, np.integer)):
            return parity_to_sign(int(grade) * (int(grade) + 1) // 2)
        return parity_to_sign((grade * (grade + 1)) // 2)

    conjugate = conjugate_sign

    def cayley(
        self, left: np.ndarray, right: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Pairwise XOR gives the output blade masks of `left * right`, paired with the adjacent
        generator swap counts needed to restore canonical increasing generator order."""
        left = np.asarray(left, dtype=self.blade_dtype)
        right = np.asarray(right, dtype=self.blade_dtype)
        cayley = np.bitwise_xor(left, right)

        if self.dimension > 1:
            shifted = right
            swaps = 0
            for _ in range(self.dimension - 1):
                shifted = np.left_shift(shifted, 1)
                swaps = swaps + self.bit_dot(left, shifted)
            swaps = np.asarray(swaps, dtype=np.int8)
        else:
            swaps = np.zeros(np.broadcast_shapes(left.shape, right.shape), dtype=np.int8)

        return cayley, swaps

    def product(
        self, left: Iterable[int] | np.ndarray, right: Iterable[int] | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluating the outer geometric product across two blade sequences yields
        `[len(left), len(right)]` arrays of canonical output masks and `int8` signs in `(-1, 0, +1)`
        combining reordering swaps with metric contractions."""
        arr_left = self._masks_to_array(left)
        arr_right = self._masks_to_array(right)

        # Generators present on both sides contract under the diagonal metric.
        doubles = np.bitwise_and.outer(arr_left, arr_right)
        negatives = self.bit_dot(doubles, self.negative_mask)
        zeros = self.bit_dot(doubles, self.degenerate_mask)

        cayley, swaps = self.cayley(arr_left[:, None], arr_right[None, :])
        signs = parity_to_sign(negatives + swaps) * (zeros == 0)
        return cayley, signs.astype(np.int8)

    def parse_blade(self, blade: str | Iterable[str]) -> OrientedBlade:
        """Parsing an exterior blade spelling yields its canonical generator mask and permutation
        sign. Strings accept `"1"` or `""` for the scalar, a single generator name, caret-delimited
        tokens such as `"e0^e1"`, or concatenated single-character names such as `"yx"`, while
        string iterables resolve multi-character generator names directly."""
        if isinstance(blade, str):
            if blade in ("", "1"):
                tokens = ()
            elif "^" in blade:
                tokens = tuple(blade.split("^"))
                if any(not token for token in tokens):
                    raise ValueError("a caret-delimited blade contains an empty token")
            elif blade in self.basis_names:
                tokens = (blade,)
            elif all(len(name) == 1 for name in self.basis_names):
                tokens = tuple(blade)
            else:
                raise ValueError(
                    "multi-character basis names require caret-delimited or tuple spelling"
                )
        else:
            tokens = tuple(blade)

        positions = []
        for token in tokens:
            if not isinstance(token, str):
                raise TypeError("blade tokens must be strings")
            try:
                positions.append(self.basis_names.index(token))
            except ValueError as error:
                raise ValueError(f"unknown basis generator {token!r}") from error
        if len(set(positions)) != len(positions):
            raise ValueError("a basis blade cannot repeat a generator")

        return OrientedBlade(
            mask_from_indices(positions),
            permutation_sign(positions),
        )

    def blade_name(self, blade: int) -> str:
        """Canonical blade names list generators in increasing index order, joined with `"^"` when
        any generator name has multiple characters, with mask `0` named `"1"`."""
        mask = self._mask(blade)
        if mask == 0:
            return "1"
        names = tuple(
            name
            for generator, name in enumerate(self.basis_names)
            if mask & (1 << generator)
        )
        separator = "" if all(len(name) == 1 for name in self.basis_names) else "^"
        return separator.join(names)

    def oriented_blade_name(self, blade: int, sign: int) -> str:
        """Negative orientation swaps the first two generators (`"zx"` for `-"xz"`), using a leading
        `"-"` only on scalars and vectors so spellings round-trip through `subspace.from_blades`."""
        name = self.blade_name(blade)
        if sign > 0:
            return name
        separator = "" if all(len(name) == 1 for name in self.basis_names) else "^"
        names = name.split(separator) if separator else list(name)
        if len(names) < 2 or name == "1":
            return "-" + name
        return separator.join([names[1], names[0], *names[2:]])

    def geometric_product(self, left: int, right: int) -> BladeProduct:
        """Two canonical basis blades multiply to a single output mask `left ^ right` and integer
        coefficient in `(-1, 0, +1)`; contractions on null generators return coefficient `0` with
        the deterministic XOR mask."""
        left_mask = self._mask(left)
        right_mask = self._mask(right)
        output = left_mask ^ right_mask

        doubles = left_mask & right_mask
        if doubles & self.degenerate_mask:
            return BladeProduct(0, output)

        negatives = (doubles & self.negative_mask).bit_count()
        swaps = 0
        shifted = right_mask
        for _ in range(self.dimension - 1):
            shifted <<= 1
            swaps += (left_mask & shifted).bit_count()

        return BladeProduct(parity_to_sign(negatives + swaps), output)

    def geometric_product_table(
        self,
        left_blades: Iterable[int],
        right_blades: Iterable[int],
    ) -> ProductTable:
        """Ordered left and right blade supports determine an immutable `ProductTable` of canonical
        output masks and integer signs."""
        blades, coefficients = self.product(left_blades, right_blades)
        return ProductTable(blades, coefficients)

    def __mul__(self, other: object) -> Algebra | NotImplementedType:
        """Multiplying two algebras with disjoint generator names constructs their orthogonal
        direct-sum algebra."""
        if not isinstance(other, Algebra):
            return NotImplemented
        return Algebra(self.description * other.description)

    def __repr__(self) -> str:
        try:
            specification = self.description.to_compact_string()
        except ValueError:
            return f"Algebra({self.description!r})"
        return f"Algebra({specification!r})"
