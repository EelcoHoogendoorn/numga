"""`SparseExtensor` couples discrete field sites through typed `Extensor` cells. A field is an
`Extensor` whose output axis ranges over sites (`Vector[vertices]`), created from a trailing batch
axis via `.field()`, while any leading batch axes hold independent fields. In a `SparseExtensor`,
`cells[..., coupling]` couples input site `columns[coupling]` to output site `rows[coupling]`, and
leading axes of `cells` hold separate maps sharing one sparsity pattern.

Algebraic products and map evaluations act through the stored cells and sum across shared sites.
Multiplying `S` on the right by a field (`S * f`, `S ^ f`, `S | f`, `S & f`) evaluates that product
between each cell and the field element at its input site and sums into each output site, whereas
multiplying on the left (`f * S`) pairs field elements with output sites and sums into input sites.
Multiplying two sparse maps chains cell products across shared intermediate sites, multiplying by a
`GAType` (`S * Even`) converts multivector cells into linear map cells, and calling a map-celled
`SparseExtensor` (`S(field)` or `S(T)`) applies or composes cells while preserving any input axes
carried by a map-valued field.

Operations that reverse multiplication or composition order also transpose `(rows, columns)`:
`reverse` and `clifford_conjugate` transpose multivector-celled maps so `~(S * f) == ~f * ~S`, while
`adjugate` and `adjoint` transpose unary map-celled operators across regressive pairings and metric
scalar products. Operations that act on cell outputs alone (`involute`, grade and subspace
projections, and `reverse` or `clifford_conjugate` on map cells) preserve the coupling direction.

Diagonal promotion mirrors dense field algebra where a map acts as an operator. Calling a field of
maps `F` on a sparse map (`F(S) == spdiag(F)(S)`) composes site by site across output rows, and
adding a field or site-free value or map to a square sparse map places it on the site diagonal so
`(S + F)(f) == S(f) + F(f)`. In products, as operands passed into `S(...)`, and in linear solvers
(`solve`, `lstsq`, `eigh`, assembled over element blocks via SciPy on the NumPy backend), diagonal
embedding is requested explicitly through `spdiag(F)`. Whenever an algebra-free `Sparse` map meets
an `Extensor`, `GAType`, or `SparseExtensor`, `typed` promotes its numeric weights to extrinsic
`Scalar` cells of that algebra. Sparse field maps, `spdiag`, and sparse solvers are developed in
`docs/fields.md` ("Sparse storage").
"""

from __future__ import annotations

from collections.abc import Callable
from functools import singledispatch
from numbers import Number
from typing import TYPE_CHECKING, Any, overload

import numpy as np

from numga import expression
from numga.extension import ExtensionMethod
from numga.extensor import Extensor, concatenate, operands
from numga.gatype import GAType

from .numeric import Sparse, chain, summed_runs, typed

if TYPE_CHECKING:
    from numga.backend.context import Context
    from numga.subspace import SubSpace


class SparseExtensor(Sparse):
    """Typed `Extensor` cells couple `shape[1]` input sites to `shape[0]` output sites, caching
    per-batch LU factorizations in `factors` on the first `solve`."""

    __slots__ = ("factors",)
    apply = ExtensionMethod("apply", takes="a sparse map")
    reverse = ExtensionMethod("reverse", takes="a sparse extensor")
    clifford_conjugate = ExtensionMethod("clifford_conjugate", takes="a sparse extensor")
    adjugate = ExtensionMethod("adjugate", takes="a sparse map")
    adjoint = ExtensionMethod("adjoint", takes="a sparse map")
    solve = ExtensionMethod("solve", takes="a sparse map whose cells map between spaces of equal dimension")
    lstsq = ExtensionMethod("lstsq", takes="a sparse map")
    eigh = ExtensionMethod("eigh", takes="a sparse map whose cells map a space to itself, or to its dual")

    def __init__(self, cells: Extensor, rows: np.ndarray, columns: np.ndarray, shape: tuple[int, int]) -> None:
        super().__init__(cells, rows, columns, shape)
        self.factors = {}

    def _scattered(self, cells: Extensor, index: np.ndarray, size: int) -> Extensor:
        return _scatter(cells, index, size)

    @classmethod
    def from_diagonal(cls, field: Extensor) -> SparseExtensor:
        """Each site element of `field` forms the `(i, i)` coupling cell of a square diagonal map."""
        cells = _sites(field)
        index = np.arange(cells.shape[-1])
        return cls(cells, index, index, (len(index), len(index)))

    def diagonal(self) -> Extensor:
        """Entries with `rows == columns` form a field over `shape[0]` sites, with zeros on uncoupled rows."""
        return super().diagonal().field()

    def to_dense(self) -> Extensor:
        """Scattering `cells` into a zero array of shape `[..., *shape, *structural_shape]` builds
        the dense two-field `Extensor` `(Out[rows] <- In[columns])`."""
        cell_type, cells = self.gatype, self.cells
        context = cells.context
        # Functional `.at[...].set` updates work on traced JAX arrays as well as NumPy arrays.
        zeros = context.xp.zeros(cells.shape[:-1] + self.shape + cells.structural_shape, dtype=cells.dtype)
        dense = context.extensor(cell_type, zeros).at[..., self.rows, self.columns].set(cells)
        return context.extensor(cell_type, dense.kernel).field(0, 1)

    @property
    def gatype(self) -> GAType:
        """Extension dispatch inspects the `GAType` of `cells`, while `shape` tracks the output and input site counts."""
        return self.cells.gatype

    # --- products --------------------------------------------------------------------------
    @overload
    def __mul__(self, other: SparseExtensor | Sparse) -> SparseExtensor: ...
    @overload
    def __mul__(self, other: Extensor) -> Extensor: ...
    @overload
    def __mul__(self, other: GAType) -> SparseExtensor: ...
    @overload
    def __mul__(self, other: float) -> SparseExtensor: ...

    def __mul__(self, other: Any) -> SparseExtensor | Extensor:
        """Numbers scale every cell, fields contract against input sites into an output field, and
        sparse maps chain cell geometric products across shared sites."""
        return _right(other, self, expression.geometric_product)

    def __rmul__(self, other: Any) -> SparseExtensor | Extensor:
        return _left(other, self, expression.geometric_product)

    def __xor__(self, other: Any) -> SparseExtensor | Extensor:
        return _right(other, self, expression.wedge)

    def __rxor__(self, other: Any) -> SparseExtensor | Extensor:
        return _left(other, self, expression.wedge)

    def __or__(self, other: Any) -> SparseExtensor | Extensor:
        return _right(other, self, expression.inner)

    def __ror__(self, other: Any) -> SparseExtensor | Extensor:
        return _left(other, self, expression.inner)

    def __and__(self, other: Any) -> SparseExtensor | Extensor:
        return _right(other, self, expression.regressive)

    def __rand__(self, other: Any) -> SparseExtensor | Extensor:
        return _left(other, self, expression.regressive)

    # --- cellwise operations preserving coupling direction ---------------------------------
    def __invert__(self) -> SparseExtensor:
        return self.reverse()

    def involute(self) -> SparseExtensor:
        """Applying grade involution to each cell preserves the site couplings."""
        return _cellwise(self, self.cells.involute())

    def select_grade(self, grade: int) -> SparseExtensor:
        """Selecting `grade` on each cell zero-pads absent blades of that grade while preserving site couplings."""
        return _cellwise(self, self.cells.select_grade(grade))

    def restrict_grade(self, grade: int) -> SparseExtensor:
        """Restricting each cell to `grade` intersects with existing cell support while preserving site couplings."""
        return _cellwise(self, self.cells.restrict_grade(grade))

    def select_subspace(self, subspace: SubSpace) -> SparseExtensor:
        """Projecting each cell onto `subspace` zero-pads absent target blades while preserving site couplings."""
        return _cellwise(self, self.cells.select_subspace(subspace))

    def restrict_subspace(self, subspace: SubSpace) -> SparseExtensor:
        """Intersecting each cell with `subspace` keeps only supported target blades while preserving site couplings."""
        return _cellwise(self, self.cells.restrict_subspace(subspace))

    # --- unary map evaluation --------------------------------------------------------------
    @overload
    def __call__(self, operand: SparseExtensor | Sparse) -> SparseExtensor: ...
    @overload
    def __call__(self, operand: Extensor) -> Extensor: ...

    def __call__(self, operand: Any) -> SparseExtensor | Extensor:
        """Evaluating unary map cells applies them to input field elements or composes them with
        another sparse map across shared sites."""
        return self.apply(operand)

    def _applied_by(self, operator: Extensor) -> SparseExtensor:
        return _diagonal(operator, self.shape[0], _context(operator))(self)

    # --- sums ------------------------------------------------------------------------------
    def __add__(self, other: SparseExtensor | Sparse | Extensor | GAType) -> SparseExtensor:
        """Adding sparse maps of equal `shape` merges their couplings, while fields and site-free
        values or maps add along the diagonal of a square map so `(S + F)(f) == S(f) + F(f)`."""
        other = _summand(other, self)
        if self.shape != other.shape:
            raise ValueError(f"sparse shapes {self.shape} and {other.shape} differ")
        batch = np.broadcast_shapes(self.cells.shape[:-1], other.cells.shape[:-1])
        return SparseExtensor(
            concatenate([self.cells.broadcast_to(batch + self.cells.shape[-1:]),
                         other.cells.broadcast_to(batch + other.cells.shape[-1:])], axis=-1),
            np.concatenate([self.rows, other.rows]),
            np.concatenate([self.columns, other.columns]),
            self.shape,
        )

    def __radd__(self, other: Sparse | Extensor | GAType) -> SparseExtensor:
        # Delegate reflected addition here so `Sparse + SparseExtensor` promotes the numeric map.
        return self + other


@singledispatch
def _summand(other: Extensor | GAType, sparse_map: SparseExtensor) -> SparseExtensor:
    return _on_diagonal(sparse_map, other)


@_summand.register
def _summand_sparse(other: SparseExtensor, sparse_map: SparseExtensor) -> SparseExtensor:
    return other


@_summand.register
def _summand_numbers(other: Sparse, sparse_map: SparseExtensor) -> SparseExtensor:
    return typed(sparse_map, other)


def _on_diagonal(sparse_map: SparseExtensor, other: Extensor | GAType) -> SparseExtensor:
    if sparse_map.shape[0] != sparse_map.shape[1]:
        raise TypeError(f"a value has no diagonal in a sparse map of shape {sparse_map.shape}")
    promoted = operands.linear(sparse_map.cells.context, other)
    if promoted is None:
        raise TypeError(f"cannot add {type(other).__name__} to a sparse extensor")
    return _diagonal(promoted, sparse_map.shape[0], sparse_map.cells.context)


def _diagonal(extensor: Extensor, n_sites: int, context: Context) -> SparseExtensor:
    if not extensor.gatype.fields:
        extensor = context.lower(extensor)
        extensor = extensor[..., None].broadcast_to(extensor.shape + (n_sites,)).field()
    return SparseExtensor.from_diagonal(extensor)


# Construct a square diagonal SparseExtensor whose `(i, i)` cell holds site `i` of a field.
spdiag = SparseExtensor.from_diagonal


# --- promotion of numeric Sparse maps ------------------------------------------------------
def _context(other: Extensor | GAType) -> Context:
    if isinstance(other, GAType) or other.context.is_exact:
        from numga.backend.numpy import NumpyContext
        return NumpyContext(other.algebra)
    return other.context


def _scalars(numeric_map: Sparse, context: Context) -> SparseExtensor:
    return SparseExtensor(operands.batch(context, numeric_map.cells), numeric_map.rows, numeric_map.columns, numeric_map.shape)


@typed.register(Extensor)
@typed.register(GAType)
def _typed_beside_value(operand: Extensor | GAType, numeric_map: Sparse) -> SparseExtensor:
    return _scalars(numeric_map, _context(operand))


@typed.register
def _typed_beside_sparse(operand: SparseExtensor, numeric_map: Sparse) -> SparseExtensor:
    return _scalars(numeric_map, operand.cells.context)


# --- products and applications, by operand -------------------------------------------------
def _operand(sparse_map: SparseExtensor, other: object) -> object:
    return operands.batch(sparse_map.cells.context, other) if operands.kind(other) is operands.Kind.ARRAY else other


def _scaled(sparse_map: SparseExtensor, number: Number) -> SparseExtensor:
    return _cellwise(sparse_map, sparse_map.cells * number)


@singledispatch
def _right(other: Any, sparse_map: SparseExtensor, product: Callable) -> SparseExtensor | Extensor:
    return _cellwise(sparse_map, product(sparse_map.cells, _operand(sparse_map, other)))


@_right.register
def _right_number(other: Number, sparse_map: SparseExtensor, product: Callable) -> SparseExtensor:
    return _scaled(sparse_map, other)


@_right.register
def _right_field(other: Extensor, sparse_map: SparseExtensor, product: Callable) -> Extensor:
    reads = _reads(sparse_map) if product is expression.geometric_product else None
    if reads is not None:
        # Pure 0/1 selection maps index the field directly so unit-versor and subtype traits survive.
        sites = _sites(other)
        batch = np.broadcast_shapes(reads.shape[:-1], sites.shape[:-1])
        reads = np.broadcast_to(reads, batch + reads.shape[-1:])
        sites = sites.broadcast_to(batch + sites.shape[-1:])
        return sites[np.indices(reads.shape, sparse=True)[:-1] + (reads,)].field()
    return _scatter(_cell_product(product, sparse_map.cells, _sites(other)[..., sparse_map.columns]), sparse_map.rows, sparse_map.shape[0]).field()


def _reads(sparse_map: SparseExtensor) -> np.ndarray | None:
    # Scalar 0/1 maps with a single 1 per output row act as pure index gathers; JAX tracers fall back to multiplication.
    if not sparse_map.cells.gatype.is_scalar:
        return None
    try:
        kernel = np.asarray(sparse_map.cells.kernel)
    except TypeError:
        return None
    ones = kernel[..., 0] == 1                                                # [..., couplings]
    if not np.all(ones | (kernel[..., 0] == 0)):
        return None
    *maps, couplings = np.nonzero(ones)
    rows = sparse_map.rows[couplings]
    counts = np.zeros(ones.shape[:-1] + (sparse_map.shape[0],), dtype=int)
    np.add.at(counts, (*maps, rows), 1)
    if not np.all(counts == 1):
        return None
    reads = np.empty(counts.shape, dtype=int)
    reads[(*maps, rows)] = sparse_map.columns[couplings]
    return reads


@_right.register
def _right_sparse(other: SparseExtensor, sparse_map: SparseExtensor, product: Callable) -> SparseExtensor:
    left, right = chain(sparse_map, other)
    cells = _cell_product(product, sparse_map.cells[..., left], other.cells[..., right])
    return SparseExtensor(cells, sparse_map.rows[left], other.columns[right], (sparse_map.shape[0], other.shape[1]))


@_right.register
def _right_numbers(other: Sparse, sparse_map: SparseExtensor, product: Callable) -> SparseExtensor:
    return _right_sparse(typed(sparse_map, other), sparse_map, product)


@singledispatch
def _left(other: Any, sparse_map: SparseExtensor, product: Callable) -> SparseExtensor | Extensor:
    if operands.kind(other) is operands.Kind.ARRAY:
        return _cellwise(sparse_map, product(_operand(sparse_map, other), sparse_map.cells))
    return _transposed(sparse_map, product(other, sparse_map.cells))


@_left.register
def _left_number(other: Number, sparse_map: SparseExtensor, product: Callable) -> SparseExtensor:
    return _scaled(sparse_map, other)


@_left.register
def _left_numbers(other: Sparse, sparse_map: SparseExtensor, product: Callable) -> SparseExtensor:
    return _right_sparse(sparse_map, typed(sparse_map, other), product)


@_left.register
def _left_field(other: Extensor, sparse_map: SparseExtensor, product: Callable) -> Extensor:
    return _scatter(_cell_product(product, _sites(other)[..., sparse_map.rows], sparse_map.cells), sparse_map.columns, sparse_map.shape[1]).field()


@SparseExtensor.apply.register(lambda cell_type: cell_type.arity == 1)
def apply(operator: SparseExtensor, operand: object) -> SparseExtensor | Extensor:
    """Unary map cells evaluate on input-site field elements and sum by output site, or compose
    across shared sites when `operand` is sparse."""
    return _application(operand, operator)


@singledispatch
def _application(operand: object, operator: SparseExtensor) -> SparseExtensor | Extensor:
    return _application_field(operands.batch(operator.cells.context, operand).field(), operator)


@_application.register
def _application_field(operand: Extensor, operator: SparseExtensor) -> Extensor:
    return _scatter(operator.cells(_sites(operand)[..., operator.columns]), operator.rows, operator.shape[0]).field()


@_application.register
def _application_sparse(operand: SparseExtensor, operator: SparseExtensor) -> SparseExtensor:
    left, right = chain(operator, operand)
    return SparseExtensor(
        operator.cells[..., left](operand.cells[..., right]),
        operator.rows[left], operand.columns[right], (operator.shape[0], operand.shape[1]),
    )


@_application.register
def _application_numbers(operand: Sparse, operator: SparseExtensor) -> SparseExtensor:
    return operator * operand


# --- order-reversing involutions and transposes --------------------------------------------
def _cellwise(sparse_map: SparseExtensor, cells: Extensor) -> SparseExtensor:
    return SparseExtensor(cells, sparse_map.rows, sparse_map.columns, sparse_map.shape)


def _transposed(sparse_map: SparseExtensor, cells: Extensor) -> SparseExtensor:
    return SparseExtensor(cells, sparse_map.columns, sparse_map.rows, sparse_map.shape[::-1])


@SparseExtensor.reverse.register(lambda cell_type: cell_type.arity == 0)
def reverse(sparse_map: SparseExtensor) -> SparseExtensor:
    """Reversing multivector cells also transposes `(rows, columns)` so `~(S * T) == ~T * ~S` holds across site couplings."""
    return _transposed(sparse_map, sparse_map.cells.reverse())


@SparseExtensor.reverse.register(lambda cell_type: cell_type.arity > 0)
def reverse_maps(operator: SparseExtensor) -> SparseExtensor:
    """Reversing map cells reverses each cell's output without transposing site couplings, so `(~S)(f) == ~(S(f))`."""
    return _cellwise(operator, operator.cells.reverse())


@SparseExtensor.clifford_conjugate.register(lambda cell_type: cell_type.arity == 0)
def clifford_conjugate(sparse_map: SparseExtensor) -> SparseExtensor:
    """Clifford-conjugating multivector cells also transposes `(rows, columns)` so conjugation reverses product order across site couplings."""
    return _transposed(sparse_map, sparse_map.cells.clifford_conjugate())


@SparseExtensor.clifford_conjugate.register(lambda cell_type: cell_type.arity > 0)
def clifford_conjugate_maps(operator: SparseExtensor) -> SparseExtensor:
    """Clifford-conjugating map cells conjugates each cell's output while keeping site couplings in place."""
    return _cellwise(operator, operator.cells.clifford_conjugate())


@SparseExtensor.adjugate.register(lambda cell_type: cell_type.arity == 1)
def adjugate(operator: SparseExtensor) -> SparseExtensor:
    """Taking each map cell's metric-free `adjugate` and transposing `(rows, columns)` preserves
    regressive pairings summed over sites: `(S.adjugate()(c) & x).sites.sum() == (c & S(x)).sites.sum()`
    between intrinsic spaces."""
    return _transposed(operator, operator.cells.adjugate())


@SparseExtensor.adjoint.register(lambda cell_type: cell_type.arity == 1)
def adjoint(operator: SparseExtensor) -> SparseExtensor:
    """Taking each map cell's metric `adjoint` and transposing `(rows, columns)` preserves scalar
    products summed over sites: `S.adjoint()(b).scalar_product(x).sites.sum() == b.scalar_product(S(x)).sites.sum()`."""
    return _transposed(operator, operator.cells.adjoint())


# --- linear algebra of map cells on the NumPy backend --------------------------------------
@SparseExtensor.solve.register(lambda cell_type, rhs: cell_type.maps_equal_dimensions, fields=True)
def solve(operator: SparseExtensor, rhs: Extensor) -> Extensor:
    """Sparse LU factorization solves `operator(x) == rhs` for each leading batch index of `operator`,
    caching the SciPy `splu` factors on `operator.factors` so subsequent solves only substitute."""
    from scipy.sparse.linalg import splu

    def substituted(case: tuple[int, ...], blocks: np.ndarray, right: np.ndarray) -> np.ndarray:
        if case not in operator.factors:
            operator.factors[case] = splu(_matrix(operator, blocks))
        return operator.factors[case].solve(np.ascontiguousarray(right.T)).reshape(-1, len(right)).T

    return _per_map(operator, rhs, substituted)


@SparseExtensor.lstsq.register(lambda cell_type, rhs: cell_type.arity == 1, fields=True)
def lstsq(operator: SparseExtensor, rhs: Extensor) -> Extensor:
    """Iterative `lsqr` computes the minimum-coordinate-norm field `x` minimizing `operator(x) - rhs`
    in blade coefficients, leaving null-space gauges such as pure translations at zero."""
    from scipy.sparse.linalg import lsqr

    def least(case: tuple[int, ...], blocks: np.ndarray, right: np.ndarray) -> np.ndarray:
        system = _matrix(operator, blocks)
        return np.stack([lsqr(system, column, atol=0.0, btol=0.0,
                              iter_lim=10 * operator.shape[1] * len(operator.cells.axes[1]))[0] for column in right])

    return _per_map(operator, rhs, least)


@SparseExtensor.eigh.register(lambda cell_type, metric: cell_type.is_endomorphism or cell_type.is_correlation)
def eigh(operator: SparseExtensor, metric: SparseExtensor, n_modes: int) -> tuple[Extensor, Extensor]:
    """Shift-invert `eigsh` near zero computes the `n_modes` smallest-magnitude generalized
    eigenpairs of `operator(x) == eigenvalue * metric(x)` with `metric`-orthonormal eigenfields,
    shifting slightly below zero relative to the pencil scale so positive-semidefinite operators
    with null spaces still factorize."""
    from scipy.sparse.linalg import eigsh

    batch = np.broadcast_shapes(operator.cells.shape[:-1], metric.cells.shape[:-1])
    values, vectors = [], []
    for case in np.ndindex(batch):
        matrix, mass = _matrix(operator, _blocks(operator, batch)[case]), _matrix(metric, _blocks(metric, batch)[case])
        scale = np.abs(matrix.diagonal()).max() / np.abs(mass.diagonal()).max()
        found, modes = eigsh(matrix, k=n_modes, M=mass, sigma=-np.sqrt(np.finfo(matrix.dtype).eps) * scale)
        values.append(found)
        vectors.append(modes.T)
    fields = _field(operator, np.reshape(vectors, batch + (n_modes, -1)))
    return fields.context.multivector.scalar(np.reshape(values, batch + (n_modes, 1))), fields


def _per_map(operator: SparseExtensor, rhs: Extensor, solver: Callable) -> Extensor:
    rhs = _sites(rhs)
    batch = np.broadcast_shapes(operator.cells.shape[:-1], rhs.shape[:-1])
    maps = (1,) * (len(batch) - (operator.cells.ndim - 1)) + operator.cells.shape[:-1]
    right = np.broadcast_to(np.asarray(rhs.kernel), batch + rhs.kernel.shape[rhs.ndim - 1:])
    right = right.reshape(batch + (-1,))
    solved = np.empty(batch + (operator.shape[1] * len(operator.cells.axes[1]),), dtype=right.dtype)
    blocks = _blocks(operator, maps)
    for case in np.ndindex(maps):
        sides = tuple(slice(None) if size == 1 else index for index, size in zip(case, maps))
        block = right[sides]
        right_sides = block.reshape(int(np.prod(block.shape[:-1])), block.shape[-1])
        own = case[len(maps) - (operator.cells.ndim - 1):]
        solved[sides] = solver(own, blocks[case], right_sides).reshape(block.shape[:-1] + (-1,))
    return _field(operator, solved)


def _blocks(operator: SparseExtensor, batch: tuple[int, ...]) -> np.ndarray:
    blocks = np.asarray(operator.cells.kernel)
    return np.broadcast_to(blocks, batch + blocks.shape[-3:])


def _matrix(operator: SparseExtensor, blocks: np.ndarray) -> Any:
    from scipy.sparse import coo_matrix

    _, height, width = blocks.shape
    rows = (operator.rows[:, None, None] * height + np.arange(height)[:, None]).repeat(width, axis=2)
    columns = (operator.columns[:, None, None] * width + np.arange(width)[None, :]).repeat(height, axis=1)
    return coo_matrix(
        (blocks.ravel(), (rows.ravel(), columns.ravel())), shape=(operator.shape[0] * height, operator.shape[1] * width),
    ).tocsc()


def _field(operator: SparseExtensor, coefficients: np.ndarray) -> Extensor:
    subspace = operator.cells.axes[1]
    return operator.cells.context.extensor(
        operator.cells.gatype.derive.axis(1), coefficients.reshape(coefficients.shape[:-1] + (-1, len(subspace))),
    ).field()


def _cell_product(product: Callable, left: Extensor, right: Extensor) -> Extensor:
    # Scalar geometric products scale cell kernels directly when the output subspaces already match.
    for scalars, cells in ((left, right), (right, left)):
        if product is expression.geometric_product and scalars.gatype.is_scalar and cells.shape[-1]:
            typed_pair = product(left[..., :1], right[..., :1])
            if typed_pair.gatype.subspaces == cells.gatype.subspaces:
                factors = scalars.kernel[..., 0]
                factors = factors.reshape(factors.shape + (1,) * len(cells.structural_shape))
                return cells.context.extensor(typed_pair.gatype, cells.kernel * factors)
    return product(left, right)


def _scatter(cells: Extensor, index: np.ndarray, size: int) -> Extensor:
    # NumPy uses sorted pairwise `reduceat` for long-run accuracy; other backends use functional `.at[...].add`.
    kernel = cells.kernel
    if not isinstance(kernel, np.ndarray):
        zeros = cells.context.xp.zeros(cells.shape[:-1] + (size,) + cells.structural_shape, dtype=cells.dtype)
        return cells.context.extensor(cells.gatype, zeros).at[..., index].add(cells)
    return cells.context.extensor(cells.gatype.derive.structural, summed_runs(kernel, index, size, cells.ndim - 1))


def _sites(field: Extensor) -> Extensor:
    if [axis for axis, _ in field.gatype.fields] != [0]:
        raise TypeError(f"a sparse field map acts on fields over its output's sites; {field.gatype.signature} is not one")
    return field.batch()
