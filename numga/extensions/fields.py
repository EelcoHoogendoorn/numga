"""Linear systems over fields: a field map, or a bilinear form over fields, solved as one coupled
system over every pair of a site and a blade."""

from __future__ import annotations

from math import prod

import numpy as np

from numga.extensor import Extensor
from numga.gatype import GAType

from numga.extensions._linalg import _result, _scalars


def _is_field_map_system(value: GAType, rhs: GAType) -> bool:
    """A map whose output takes the right-hand side's output, either side ranging over sites."""
    return value.arity == 1 and rhs.output_subspace.support_is_subset_of(value.output_subspace)


def _is_field_form_system(value: GAType, rhs: GAType) -> bool:
    """A scalar form of two slots against a scalar-valued extensor ending in the form's last slot."""
    scalar = value.algebra.subspace.scalar()
    return (
        value.arity == 2
        and value.output_subspace.same_support(scalar)
        and rhs.arity >= 1
        and rhs.output_subspace.same_support(scalar)
        and 0 not in dict(rhs.fields)
        and rhs.subspaces[-1].same_support(value.subspaces[2])
        and dict(rhs.fields).get(rhs.arity) == dict(value.fields).get(2)
    )


@Extensor.solve.register(_is_field_map_system, fields=True)
def solve_field_map(value: Extensor, rhs: Extensor) -> Extensor:
    """Solve A(x) == rhs for x, coupled over the sites of A's slots: A.solve(A(y)) == y. The open
    slots of rhs are kept as open slots of the solution."""
    return _solve(value, 0, 1, _on_output_sites(value, rhs), 0, _exactly(value))


@Extensor.lstsq.register(_is_field_map_system, fields=True)
def lstsq_field_map(value: Extensor, rhs: Extensor, *, rcond: float = 1e-15) -> Extensor:
    """The x of least coefficient norm minimizing the sum of squared coefficients of A(x) - rhs over
    every site, discarding singular values at most rcond times the largest; a singular system's gauge,
    such as an anchored camera's step, is left at zero."""
    return _solve(value, 0, 1, _on_output_sites(value, rhs), 0, _least_squares(value, rcond))


@Extensor.pinv.register(lambda value: value.arity == 1, fields=True)
def pinv_field_map(value: Extensor, *, rcond: float = 1e-15) -> Extensor:
    """The Moore-Penrose inverse over every pair of a site and a blade, from the output's sites to the
    input's."""
    identity = value.context.lower(value.algebra.operator.identity(value.output_subspace))   # X <- X
    sites = dict(value.gatype.fields).get(0)
    if sites:
        identity = identity[..., None].broadcast_to((sites,)).field().on_diagonal()           # X[n] <- X[n]
    return lstsq_field_map(value, identity, rcond=rcond)


@Extensor.solve.register(_is_field_form_system, fields=True)
def solve_field_form(value: Extensor, rhs: Extensor) -> Extensor:
    """Solve F(x, y) == rhs(..., y) for all y, x in the form's first slot, coupled over the sites of
    both slots: F.solve(F(x)) == x. For a Hessian and a gradient, the Newton step up to sign."""
    return _solve(value, 2, 1, rhs, rhs.arity, _exactly(value))


@Extensor.lstsq.register(_is_field_form_system, fields=True)
def lstsq_field_form(value: Extensor, rhs: Extensor, *, rcond: float = 1e-15) -> Extensor:
    """Least-squares version of the field form solve, with lstsq's cutoff on singular values."""
    return _solve(value, 2, 1, rhs, rhs.arity, _least_squares(value, rcond))


def _exactly(value: Extensor):
    def solve(matrix, columns):
        if matrix.shape[-1] != matrix.shape[-2]:
            raise TypeError(f"{value.gatype.signature} is not square over its sites and blades")
        return value.context.xp.linalg.solve(matrix, columns)
    return solve


def _least_squares(value: Extensor, rcond: float):
    return lambda matrix, columns: value.context.xp.linalg.pinv(matrix, rcond) @ columns


def _on_output_sites(value: Extensor, rhs: Extensor) -> Extensor:
    """The right-hand side of a field map's system, over the sites of the map's output: a value without
    sites is the same at every site."""
    rhs = rhs.cast(value.output_subspace)
    sites = dict(value.gatype.fields).get(0)
    if not rhs.gatype.fields and sites:
        rhs = rhs[..., None].broadcast_to(rhs.shape + (sites,)).field()
    if dict(rhs.gatype.fields).get(0) != sites:
        raise TypeError(f"{rhs.gatype.signature} does not range over the sites of {value.gatype.signature}")
    return rhs


def _solve(value: Extensor, equations: int, unknowns: int, rhs: Extensor, given: int, solver) -> Extensor:
    """Solve for the unknown slot of value, pairing its equation slot with the given slot of rhs, by a
    solver of the blocks read as one matrix. The solution has the unknown slot as its output and the
    other slots of rhs as inputs."""
    xp = value.context.xp
    batch = np.broadcast_shapes(value.shape, rhs.shape)
    value, rhs = value.broadcast_to(batch), rhs.broadcast_to(batch)
    rows = _slot_axes(value, equations)
    columns = _slot_axes(value, unknowns)
    # A form's scalar output is one blade long and folds into the matrix.
    rest = tuple(axis for axis in range(len(batch), value.kernel.ndim) if axis not in rows + columns)
    matrix = xp.transpose(value.kernel, _batch_axes(batch) + rows + columns + rest)
    height = prod(value.kernel.shape[axis] for axis in rows)
    width = prod(value.kernel.shape[axis] for axis in columns)
    matrix = xp.reshape(matrix, batch + (height, width))

    # The other slots of rhs: its scalar output for a form, its open inputs either way.
    kept = [slot for slot in range(rhs.arity + 1) if slot != given]
    sites = [axis for slot in kept for axis in _slot_axes(rhs, slot)[:-1]]
    blades = [_slot_axes(rhs, slot)[-1] for slot in kept]
    given_axes = _slot_axes(rhs, given)
    columns_kernel = xp.transpose(rhs.kernel, _batch_axes(batch) + given_axes + tuple(sites) + tuple(blades))
    other_shape = tuple(rhs.kernel.shape[axis] for axis in sites + blades)
    solution = solver(matrix, xp.reshape(columns_kernel, batch + (height, prod(other_shape))))

    unknown_shape = tuple(value.kernel.shape[axis] for axis in columns)
    solution = xp.reshape(solution, batch + unknown_shape + other_shape)
    # Back to the layout of the solution: its sites and the other sites, then its blades and the others'.
    b, u, s = len(batch), len(unknown_shape), len(sites)
    order = _batch_axes(batch) + tuple(range(b, b + u - 1)) + tuple(range(b + u, b + u + s)) \
        + (b + u - 1,) + tuple(range(b + u + s, b + u + s + len(blades)))
    solution = xp.transpose(solution, order)

    input_slots = [slot for slot in kept if slot]
    unknown_sites = dict(value.gatype.fields).get(unknowns)
    rhs_sites = dict(rhs.gatype.fields)
    fields = ((0, unknown_sites),) * (unknown_sites is not None) + tuple(
        (index + 1, rhs_sites[slot]) for index, slot in enumerate(input_slots) if slot in rhs_sites
    )
    subspaces = (value.axes[unknowns],) + tuple(rhs.axes[slot] for slot in input_slots)
    gatype = value.algebra.gatype(subspaces, fields=fields)
    # A form's right-hand side leaves its scalar output over, one blade long.
    return _result(value, gatype, xp.reshape(solution, batch + gatype.structural_shape))


def _batch_axes(batch: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(range(len(batch)))


def _slot_axes(value: Extensor, slot: int) -> tuple[int, ...]:
    """The kernel axes of one slot: its site axis, if it ranges over sites, then its blade axis."""
    field_slots = [field for field, _ in value.gatype.fields]
    sites = (value.ndim + field_slots.index(slot),) if slot in field_slots else ()
    return sites + (value.ndim + len(field_slots) + slot,)


@Extensor.adjoint.register(lambda value: value.arity == 1, fields=True)
def adjoint_field_map(value: Extensor) -> Extensor:
    """The map the metric carries over, summed over the sites: `A.adjoint()(b).scalar_product(a)`
    summed over the sites of a equals `b.scalar_product(A(a))` summed over the sites of b. Each block's
    adjoint, the sites of the output and the input exchanged."""
    return _exchanged(value, value.batch().adjoint())


@Extensor.adjugate.register(lambda value: value.arity == 1, fields=True)
def adjugate_field_map(value: Extensor) -> Extensor:
    """The map on complements that incidence carries over, summed over the sites:
    `A.adjugate()(c) & x` summed equals `c & A(x)` summed. Each block's adjugate, the sites of the
    output and the input exchanged."""
    return _exchanged(value, value.batch().adjugate())


def _exchanged(value: Extensor, blocks: Extensor) -> Extensor:
    """Blocks over the site pairs of a unary field map, read with its output's sites on the input
    and its input's sites on the output."""
    sites = dict(value.gatype.fields)
    kernel = blocks.kernel
    if len(sites) == 2:
        batch = blocks.ndim - 2
        kernel = value.context.xp.swapaxes(kernel, batch, batch + 1)
    blocks = Extensor._from_prepared_kernel(blocks.context, blocks.gatype, kernel)
    # The input's sites become the output's and the output's the input's.
    return blocks.field(*((0,) * (1 in sites) + (1,) * (0 in sites)))


# --- spectra and factors of the blocks read as one matrix -----------------------------------
def _is_field_square(value: GAType) -> bool:
    """A map, or a scalar form of two slots, between one space over the same sites, either side
    ranging over them."""
    sites = dict(value.fields)
    if value.arity == 1:
        rows, columns = 0, 1
    elif value.arity == 2 and value.output_subspace.same_support(value.algebra.subspace.scalar()) and 0 not in sites:
        rows, columns = 1, 2
    else:
        return False
    return value.subspaces[rows] == value.subspaces[columns] and sites.get(rows) == sites.get(columns)


def _is_field_endomorphism(value: GAType) -> bool:
    return value.arity == 1 and _is_field_square(value)


def _same_slots(value: GAType, other: GAType) -> bool:
    """The same spaces over the same sites, whatever facts either type carries."""
    return value.subspaces == other.subspaces and value.fields == other.fields


def _square_slots(value: Extensor) -> tuple[int, int]:
    return (0, 1) if value.arity == 1 else (1, 2)


@Extensor.det.register(_is_field_endomorphism, fields=True)
def det_field_map(value: Extensor) -> Extensor:
    """The determinant over every pair of a site and a blade."""
    return _scalars(value, value.context.xp.linalg.det(_flattened(value, 0, 1)[0]))


@Extensor.trace.register(_is_field_endomorphism, fields=True)
def trace_field_map(value: Extensor) -> Extensor:
    """The trace: each site's block traced, summed over the sites."""
    return _scalars(value, value.context.xp.trace(_flattened(value, 0, 1)[0], axis1=-2, axis2=-1))


@Extensor.inverse.register(_is_field_endomorphism, fields=True)
def inverse_field_map(value: Extensor) -> Extensor:
    """The inverse under composition, coupled over the sites."""
    matrix, rows, columns = _flattened(value, 0, 1)
    gatype = value.algebra.gatype((value.axes[1], value.axes[0]), fields=_exchanged_fields(value.gatype))
    return _from_matrix(value, value.context.xp.linalg.inv(matrix), columns, rows, gatype)


@Extensor.cholesky.register(_is_field_square, fields=True)
def cholesky_field(value: Extensor) -> Extensor:
    """The lower factor L of the blocks read as one Hermitian positive-definite matrix, `A = L L^H`, in
    the same layout."""
    first, second = _square_slots(value)
    matrix, rows, columns = _flattened(value, first, second)
    return _from_matrix(value, value.context.xp.linalg.cholesky(matrix), rows, columns, value.gatype.derive.structural)


@Extensor.eigh.register(_is_field_square, fields=True)
def eigh_field(value: Extensor) -> tuple[Extensor, Extensor]:
    """Ascending eigenvalues `[..., modes] Scalar` and coefficient-orthonormal modes, a batch of fields
    `[..., modes] X[n]`, of a Hermitian map or form over a field."""
    first, second = _square_slots(value)
    matrix, _, columns = _flattened(value, first, second)
    values, vectors = value.context.xp.linalg.eigh(matrix, UPLO="L")
    return _scalars(value, values), _modes(value, second, columns, vectors)


@Extensor.eigvalsh.register(_is_field_square, fields=True)
def eigvalsh_field(value: Extensor) -> Extensor:
    first, second = _square_slots(value)
    return _scalars(value, value.context.xp.linalg.eigvalsh(_flattened(value, first, second)[0], UPLO="L"))


@Extensor.eigh.register(lambda value, metric: _is_field_square(value) and _same_slots(value, metric), fields=True)
def eigh_field_metric(value: Extensor, metric: Extensor) -> tuple[Extensor, Extensor]:
    """The modes of `value(x) == eigenvalue * metric(x)`, for a Hermitian value and a positive-definite
    metric over the same field: ascending eigenvalues, and modes orthonormal in the metric."""
    xp = value.context.xp
    first, second = _square_slots(value)
    matrix, _, columns = _flattened(value, first, second)
    lower = xp.linalg.cholesky(_flattened(metric, first, second)[0])
    reduced = xp.linalg.solve(lower, _conjugate_transpose(xp, xp.linalg.solve(lower, matrix)))
    values, vectors = xp.linalg.eigh(_conjugate_transpose(xp, reduced), UPLO="L")
    return _scalars(value, values), _modes(value, second, columns, xp.linalg.solve(_conjugate_transpose(xp, lower), vectors))


@Extensor.eigvalsh.register(lambda value, metric: _is_field_square(value) and _same_slots(value, metric), fields=True)
def eigvalsh_field_metric(value: Extensor, metric: Extensor) -> Extensor:
    return eigh_field_metric(value, metric)[0]


@Extensor.eig.register(_is_field_endomorphism, fields=True)
def eig_field_map(value: Extensor) -> tuple[Extensor, Extensor]:
    """Eigenvalues `[..., modes] Scalar` and right modes, a batch of fields, of a field map."""
    matrix, _, columns = _flattened(value, 0, 1)
    values, vectors = value.context.xp.linalg.eig(matrix)
    return _scalars(value, values), _modes(value, 1, columns, vectors)


@Extensor.eigvals.register(_is_field_endomorphism, fields=True)
def eigvals_field_map(value: Extensor) -> Extensor:
    return _scalars(value, value.context.xp.linalg.eigvals(_flattened(value, 0, 1)[0]))


@Extensor.eig.register(lambda value, metric: _is_field_endomorphism(value) and _same_slots(value, metric), fields=True)
def eig_field_map_metric(value: Extensor, metric: Extensor) -> tuple[Extensor, Extensor]:
    """The modes of `value(x) == eigenvalue * metric(x)`: those of `metric.solve(value)`."""
    return eig_field_map(metric.solve(value))


@Extensor.eigvals.register(lambda value, metric: _is_field_endomorphism(value) and _same_slots(value, metric), fields=True)
def eigvals_field_map_metric(value: Extensor, metric: Extensor) -> Extensor:
    return eigvals_field_map(metric.solve(value))


@Extensor.svd.register(lambda value: value.arity == 1, fields=True)
def svd_field_map(value: Extensor) -> tuple[Extensor, Extensor, Extensor]:
    """Reduced SVD over every pair of a site and a blade: left modes, a batch of fields over the output,
    descending singular values, and right modes over the input; `A(v_i) == s_i u_i`."""
    xp = value.context.xp
    matrix, rows, columns = _flattened(value, 0, 1)
    left, singular, right = xp.linalg.svd(matrix, full_matrices=False)
    return _modes(value, 0, rows, left), _scalars(value, singular), _modes(value, 1, columns, _conjugate_transpose(xp, right))


@Extensor.svdvals.register(lambda value: value.arity == 1, fields=True)
def svdvals_field_map(value: Extensor) -> Extensor:
    return _scalars(value, value.context.xp.linalg.svd(_flattened(value, 0, 1)[0], compute_uv=False))


def _flattened(value: Extensor, rows: int, columns: int):
    """The blocks of value as one matrix, its rows the sites and blades of one slot and its columns those
    of another, with the shapes of both."""
    xp = value.context.xp
    batch = value.shape
    row_axes, column_axes = _slot_axes(value, rows), _slot_axes(value, columns)
    # A form's scalar output is one blade long and folds into the matrix.
    rest = tuple(axis for axis in range(len(batch), value.kernel.ndim) if axis not in row_axes + column_axes)
    matrix = xp.transpose(value.kernel, _batch_axes(batch) + row_axes + column_axes + rest)
    row_shape = tuple(value.kernel.shape[axis] for axis in row_axes)
    column_shape = tuple(value.kernel.shape[axis] for axis in column_axes)
    return xp.reshape(matrix, batch + (prod(row_shape), prod(column_shape))), row_shape, column_shape


def _from_matrix(value: Extensor, matrix, rows: tuple[int, ...], columns: tuple[int, ...], gatype: GAType) -> Extensor:
    """A matrix whose rows are the sites and blades of the first slot of gatype and whose columns those of
    the second, laid out as gatype's kernel: its sites, then its blades. A form keeps its scalar output."""
    xp = value.context.xp
    batch = matrix.shape[:-2]
    kernel = xp.reshape(matrix, batch + rows + columns)
    b, r, c = len(batch), len(rows), len(columns)
    order = (_batch_axes(batch) + tuple(range(b, b + r - 1)) + tuple(range(b + r, b + r + c - 1))
             + (b + r - 1, b + r + c - 1))
    return _result(value, gatype, xp.reshape(xp.transpose(kernel, order), batch + gatype.structural_shape))


def _modes(value: Extensor, slot: int, shape: tuple[int, ...], vectors) -> Extensor:
    """The columns of a matrix over the sites and blades of a slot, as a batch of fields over them."""
    xp = value.context.xp
    batch = vectors.shape[:-2]
    modes = xp.reshape(xp.swapaxes(vectors, -1, -2), batch + (vectors.shape[-1],) + shape)
    sites = dict(value.gatype.fields).get(slot)
    gatype = value.algebra.gatype(value.axes[slot], fields=((0, sites),) * (sites is not None))
    return _result(value, gatype, modes)


def _exchanged_fields(gatype: GAType) -> tuple[tuple[int, int], ...]:
    sites = dict(gatype.fields)
    return tuple((1 - slot, count) for slot, count in sorted(sites.items(), reverse=True))


def _conjugate_transpose(xp, matrix):
    return xp.conj(xp.swapaxes(matrix, -1, -2))
