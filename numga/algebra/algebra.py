"""Canonical bit-blade algebra for an orthogonal diagonal metric."""

from __future__ import annotations

from dataclasses import dataclass, field
from operator import index
from typing import TYPE_CHECKING, Callable, Iterable, NamedTuple, Sequence

import numpy as np

from .bitops import (
    bit_count,
    mask_from_indices,
    parity_sign,
    parity_to_sign,
    permutation_sign,
    unsigned_dtype,
)
from .description import AlgebraDescription

if TYPE_CHECKING:
    from numga.backend.exact import ExactContext
    from numga.gatype.factory import GATypeFactory
    from numga.operator.factory import OperatorFactory
    from numga.subspace.factory import SubSpaceFactory


class BladeProduct(NamedTuple):
    """One term of a canonical basis-blade geometric product."""

    coefficient: int
    blade: int

    @property
    def mask(self) -> int:
        return self.blade


class OrientedBlade(NamedTuple):
    """A spelled blade reduced to a canonical mask and orientation."""

    blade: int
    sign: int

    @property
    def mask(self) -> int:
        return self.blade


@dataclass(frozen=True, slots=True, eq=False)
class ProductTable:
    """Dense blade and coefficient tables for two ordered mask sequences."""

    blades: np.ndarray
    coefficients: np.ndarray

    def __post_init__(self) -> None:
        blades = np.asarray(self.blades)
        coefficients = np.asarray(self.coefficients, dtype=np.int8)
        if blades.shape != coefficients.shape or blades.ndim != 2:
            raise ValueError("product-table arrays must be equally shaped matrices")

        # Arrays backed by immutable bytes cannot be made writeable again by a
        # consumer, unlike an owning ndarray with only its flag cleared.
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


@dataclass(frozen=True, slots=True, init=False)
class Algebra:
    """An immutable geometric algebra with canonical integer bit blades.

    ``Algebra(spec)`` accepts an :class:`AlgebraDescription`, compact notation
    such as ``"x+y+z+w0"``, or a ``(p, q, r)`` tuple. A signature tuple is
    intentionally not accepted by this shorthand; use :meth:`from_signature`
    when the tuple contains metric entries rather than signature counts.
    """

    description: AlgebraDescription
    _subspace_factory: SubSpaceFactory = field(compare=False, repr=False)
    _gatype_factory: GATypeFactory = field(compare=False, repr=False)
    _exact_context: ExactContext = field(compare=False, repr=False)
    _operator_factory: OperatorFactory = field(compare=False, repr=False)

    def __init__(
        self,
        spec: AlgebraDescription | str | tuple[int, int, int],
        *,
        subspace_factory: Callable[[Algebra], SubSpaceFactory] | None = None,
        gatype_factory: Callable[[Algebra], GATypeFactory] | None = None,
    ) -> None:
        if isinstance(spec, AlgebraDescription):
            description = spec
        elif isinstance(spec, str):
            description = AlgebraDescription.parse(spec)
        elif isinstance(spec, tuple) and len(spec) == 3:
            description = AlgebraDescription.from_pqr(*spec)
        else:
            raise TypeError(
                "Algebra expects an AlgebraDescription, compact string, or (p, q, r) tuple"
            )
        if description.dimension > 64:
            raise ValueError("the bit-mask implementation supports at most 64 generators")
        object.__setattr__(self, "description", description)

        # Construction namespaces are owned by the algebra, so their flyweight
        # pools have the same natural lifetime. This is separate from the
        # deliberately type-level registry of Extensor extension methods.
        from numga.backend.exact import ExactContext
        from numga.gatype.factory import GATypeFactory
        from numga.operator.factory import OperatorFactory
        from numga.subspace.factory import SubSpaceFactory

        object.__setattr__(self, "_subspace_factory", (subspace_factory or SubSpaceFactory)(self))
        object.__setattr__(self, "_gatype_factory", (gatype_factory or GATypeFactory)(self))
        object.__setattr__(self, "_exact_context", ExactContext(self))
        object.__setattr__(self, "_operator_factory", OperatorFactory(self))

    @classmethod
    def from_description(
        cls,
        description: AlgebraDescription,
        *,
        subspace_factory: Callable[[Algebra], SubSpaceFactory] | None = None,
        gatype_factory: Callable[[Algebra], GATypeFactory] | None = None,
    ) -> "Algebra":
        return cls(
            description,
            subspace_factory=subspace_factory,
            gatype_factory=gatype_factory,
        )

    @classmethod
    def from_signature(
        cls,
        signature: str | Iterable[int],
        basis_names: Sequence[str] | None = None,
        *,
        subspace_factory: Callable[[Algebra], SubSpaceFactory] | None = None,
        gatype_factory: Callable[[Algebra], GATypeFactory] | None = None,
    ) -> "Algebra":
        return cls(
            AlgebraDescription.from_signature(signature, basis_names),
            subspace_factory=subspace_factory,
            gatype_factory=gatype_factory,
        )

    @classmethod
    def from_pqr(
        cls,
        p: int,
        q: int,
        r: int,
        basis_names: Sequence[str] | None = None,
        *,
        subspace_factory: Callable[[Algebra], SubSpaceFactory] | None = None,
        gatype_factory: Callable[[Algebra], GATypeFactory] | None = None,
    ) -> "Algebra":
        return cls(
            AlgebraDescription.from_pqr(p, q, r, basis_names),
            subspace_factory=subspace_factory,
            gatype_factory=gatype_factory,
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
        """Canonical SubSpace construction namespace for this algebra."""

        return self._subspace_factory

    @property
    def gatype(self) -> GATypeFactory:
        """Canonical whole-extensor type factory for this algebra."""

        return self._gatype_factory

    @property
    def operator(self) -> OperatorFactory:
        """Canonical exact operation factory for this algebra."""

        return self._operator_factory

    @property
    def exact(self) -> ExactContext:
        """The algebra-owned exact Extensor execution context."""

        return self._exact_context

    @property
    def blade_dtype(self) -> np.dtype:
        return unsigned_dtype(self.dimension)

    @property
    def blade_masks(self) -> range:
        """All canonical blade masks, without eagerly allocating them."""

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
        self, a: int | np.ndarray, b: int | np.ndarray
    ) -> int | np.ndarray:
        """Dot-product between bit-blades, counting their shared high bits."""
        if isinstance(a, (int, np.integer)) and isinstance(b, (int, np.integer)):
            return (int(a) & int(b)).bit_count()
        return bit_count(np.bitwise_and(a, b))

    def grade(self, blade: int | Sequence[int] | np.ndarray) -> int | np.ndarray:
        """Grade of each blade; number of basis vectors present in each blade."""
        if isinstance(blade, (int, np.integer)):
            return self._mask(blade).bit_count()
        return bit_count(self._masks_to_array(blade))

    def complement(self, blade: int | Sequence[int] | np.ndarray) -> int | np.ndarray:
        """Complement of a blade or set of blades."""
        if isinstance(blade, (int, np.integer)):
            return self._mask(blade) ^ self.pseudoscalar_mask
        return np.bitwise_xor(self._masks_to_array(blade), self.pseudoscalar_mask)

    def involute_sign(self, blade: int | Sequence[int] | np.ndarray) -> int | np.ndarray:
        grade = self.grade(blade)
        return parity_to_sign(grade)

    involute = involute_sign

    def reverse_sign(self, blade: int | Sequence[int] | np.ndarray) -> int | np.ndarray:
        grade = self.grade(blade)
        if isinstance(grade, (int, np.integer)):
            return parity_to_sign(int(grade) * (int(grade) - 1) // 2)
        return parity_to_sign((grade * (grade - 1)) // 2)

    reverse = reverse_sign

    def conjugate_sign(self, blade: int | Sequence[int] | np.ndarray) -> int | np.ndarray:
        grade = self.grade(blade)
        if isinstance(grade, (int, np.integer)):
            return parity_to_sign(int(grade) * (int(grade) + 1) // 2)
        return parity_to_sign((grade * (grade + 1)) // 2)

    conjugate = conjugate_sign

    def cayley(
        self, a: np.ndarray, b: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Cayley table of the geometric product a * b.

        Parameters
        ----------
        a: np.ndarray, [...], blade_dtype
            Basis blades encoded as integer bit patterns.
        b: np.ndarray, [...], blade_dtype
            Basis blades encoded as integer bit patterns.

        Returns
        -------
        cayley: np.ndarray, [...], blade_dtype
            Bit patterns of the blades that the product maps to.
        swaps: np.ndarray, [...], np.int8
            Minimal number of pairwise swaps required to reorder basis vectors back
            to canonical sorted order.
        """
        a = np.asarray(a, dtype=self.blade_dtype)
        b = np.asarray(b, dtype=self.blade_dtype)
        cayley = np.bitwise_xor(a, b)

        if self.dimension > 1:
            shifted = b
            swaps = 0
            for _ in range(self.dimension - 1):
                shifted = np.left_shift(shifted, 1)
                swaps = swaps + self.bit_dot(a, shifted)
            swaps = np.asarray(swaps, dtype=np.int8)
        else:
            swaps = np.zeros(np.broadcast_shapes(a.shape, b.shape), dtype=np.int8)

        return cayley, swaps

    def product(
        self, a: Iterable[int] | np.ndarray, b: Iterable[int] | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Construct geometric product between elements of a and b.

        Parameters
        ----------
        a: np.ndarray, [A], blade_dtype
            Basis blades encoded as integer bit patterns.
        b: np.ndarray, [B], blade_dtype
            Basis blades encoded as integer bit patterns.

        Returns
        -------
        cayley: np.ndarray, [A, B], blade_dtype
            Bit patterns of the blades that the product maps to.
        signs: np.ndarray, [A, B], np.int8
            Signs / coefficients of the product.
        """
        arr_a = self._masks_to_array(a)
        arr_b = self._masks_to_array(b)

        # Basis vectors occurring on both sides, eliminated by the metric
        doubles = np.bitwise_and.outer(arr_a, arr_b)
        negatives = self.bit_dot(doubles, self.negative_mask)
        zeros = self.bit_dot(doubles, self.degenerate_mask)

        cayley, swaps = self.cayley(arr_a[:, None], arr_b[None, :])
        signs = parity_to_sign(negatives + swaps) * (zeros == 0)
        return cayley, signs.astype(np.int8)

    def parse_blade(self, blade: str | Iterable[str]) -> OrientedBlade:
        """Normalize a spelled exterior blade to canonical mask and sign.

        A string may be ``"1"`` for the scalar, one exact generator name,
        caret-delimited tokens such as ``"e0^e1"``, or concatenated one-character
        generator names such as ``"yx"``. An iterable of names is always
        unambiguous and is preferred for multi-character generators.
        """

        if isinstance(blade, str):
            if blade in ("", "1"):
                tokens: tuple[str, ...] = ()
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

        positions: list[int] = []
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
        """Format a canonical mask in increasing generator order."""

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
        """Name an oriented blade as the subspace parser reads it: a negative orientation swaps
        the first two generators, zx for -xz, and only a scalar or a vector takes a minus sign."""

        name = self.blade_name(blade)
        if sign > 0:
            return name
        separator = "" if all(len(name) == 1 for name in self.basis_names) else "^"
        names = name.split(separator) if separator else list(name)
        if len(names) < 2 or name == "1":
            return "-" + name
        return separator.join([names[1], names[0], *names[2:]])

    def geometric_product(self, left: int, right: int) -> BladeProduct:
        """Multiply two canonical basis blades.

        The result remains a single blade because the supported metric is
        diagonal. A degenerate contraction returns coefficient zero and the
        deterministic XOR output mask; callers discard zero terms.
        """
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
        """Build the dense canonical product table for two ordered supports."""
        blades, coefficients = self.product(left_blades, right_blades)
        return ProductTable(blades, coefficients)

    def __mul__(self, other: object) -> "Algebra":
        if not isinstance(other, Algebra):
            return NotImplemented
        return Algebra(self.description * other.description)

    def __repr__(self) -> str:
        try:
            specification = self.description.to_compact_string()
        except ValueError:
            return f"Algebra({self.description!r})"
        return f"Algebra({specification!r})"
