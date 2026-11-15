"""Field extensions couple linear algebra across discrete sites and algebraic blades. A unary field
map (`Y[m] <- X[n]`) or a bilinear form over fields (`Scalar <- (X[m], Y[n])`) assembles every
site-and-blade pair into one block system: `solve` inverts square coupled systems while preserving
any input axes on `rhs` (`A.solve(A(Y)) == Y`), and `lstsq` and `pinv` compute minimum-coefficient-norm
solutions across all sites and blades so unconstrained gauge directions stay at zero. When the map or
form carries no site axes and only `rhs` is a field, `solve` and `lstsq` broadcast and solve the
local system independently at every site while preserving site assignments on the solution's axes.

Transpose operations act blockwise while exchanging input and output site axes. `adjoint` transports
each block through the metric `scalar_product` so `A.adjoint()(b).scalar_product(a)` summed over the
input sites equals `b.scalar_product(A(a))` summed over the output sites, whereas `adjugate`
transports each block onto the oppositely oriented complement spaces via `&` so `A.adjugate()(c) & x`
summed over the input sites equals `c & A(x)` summed over the output sites without requiring a metric.

Matrix factorizations and spectral decompositions flatten each axis's site and blade dimensions into
a single coupled matrix dimension. `inverse`, `det`, `trace` (the sum of diagonal site-block traces),
`cholesky`, `eig`, `eigh`, and reduced `svd` treat the coupled field operator or form as one operator
and return modal fields as batches `[..., modes] X[n]`. Supplying a positive-definite `metric` field
map or correlation solves the generalized problem `operator(x) == eigenvalue * metric(x)`, producing
`metric`-orthonormal field modes in `eigh` and reducing via `metric.solve(operator)` in `eig`.
Coupled field maps and site-axis semantics are detailed in `docs/fields.md`.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from math import prod
from typing import Any

import numpy as np

from numga.extensor import Extensor
from numga.extensor.fields import _aligned
from numga.gatype import GAType

from numga.extensions._linalg import _result, _scalars


@Extensor.solve.register(lambda t, rhs: t.over_sites.is_map_system(rhs), fields=True)
def solve_field_map(operator: Extensor, rhs: Extensor) -> Extensor:
    """Solving a coupled linear system finds `x` with `A(x) == rhs` across all sites and blades of
    `A`, carrying any input axes of `rhs` onto the solution (`A.solve(A(Y)) == Y`)."""
    return _solve(operator, 0, 1, _on_output_sites(operator, rhs), 0, _exactly(operator))


@Extensor.lstsq.register(lambda t, rhs: t.over_sites.is_map_system(rhs), fields=True)
def lstsq_field_map(operator: Extensor, rhs: Extensor, *, rcond: float = 1e-15) -> Extensor:
    """Solving a coupled system in least squares minimizes the sum of squared residual coefficients of
    `A(x) - rhs` over all sites and blades, zeroing singular modes at or below `rcond` times the
    largest singular value."""
    return _solve(operator, 0, 1, _on_output_sites(operator, rhs), 0, _least_squares(operator, rcond))


@Extensor.pinv.register(lambda t: t.arity == 1, fields=True)
def pinv_field_map(operator: Extensor, *, rcond: float = 1e-15) -> Extensor:
    """The Moore-Penrose pseudoinverse maps the output's sites and blades back to the input's in the
    coupled Euclidean coefficient norm."""
    identity = operator.context.lower(operator.algebra.operator.identity(operator.gatype.derive.output))  # X <- X
    sites = dict(operator.gatype.fields).get(0)
    if sites:
        identity = identity[..., None].broadcast_to((sites,)).field().on_diagonal()                       # X[n] <- X[n]
    return lstsq_field_map(operator, identity, rcond=rcond)


@Extensor.solve.register(lambda t, rhs: t.over_sites.is_form_system(rhs), fields=True)
def solve_field_form(form: Extensor, rhs: Extensor) -> Extensor:
    """Solving against a bilinear field form inverts partial evaluation on its first input across the
    coupled sites of both axes, satisfying `F(x, y) == rhs(..., y)` for all `y` (`F.solve(F(x)) == x`)."""
    return _solve(form, 2, 1, rhs, rhs.arity, _exactly(form))


@Extensor.lstsq.register(lambda t, rhs: t.over_sites.is_form_system(rhs), fields=True)
def lstsq_field_form(form: Extensor, rhs: Extensor, *, rcond: float = 1e-15) -> Extensor:
    """Solving a coupled bilinear form in least squares computes the minimum-coefficient-norm solution
    of `F(x, y) == rhs(..., y)` across both axes' sites, truncating singular values by `rcond`."""
    return _solve(form, 2, 1, rhs, rhs.arity, _least_squares(form, rcond))


@Extensor.solve.register(lambda t, rhs: t.over_sites.is_site_by_site_system(rhs), fields=True)
def solve_site_by_site(operator: Extensor, rhs: Extensor) -> Extensor:
    """When `operator` carries no sites and `rhs` is a field, solving applies the local map or form
    solve independently at every site so `A.solve(A(y)) == y` holds for fields `y`."""
    return _site_by_site(operator, rhs, Extensor.solve)


@Extensor.lstsq.register(lambda t, rhs: t.over_sites.is_site_by_site_system(rhs), fields=True)
def lstsq_site_by_site(operator: Extensor, rhs: Extensor, *, rcond: float = 1e-15) -> Extensor:
    """When `operator` carries no sites and `rhs` is a field, least squares solves the local system
    independently at every site with singular-value cutoff `rcond`."""
    return _site_by_site(operator, rhs, partial(Extensor.lstsq, rcond=rcond))


def _site_by_site(operator: Extensor, rhs: Extensor, solver: Callable) -> Extensor:
    # Treat rhs site axes as trailing batch axes during the local solve, then restore them as field axes.
    sites = rhs.gatype.site_shape
    solution = solver(_aligned(operator, (), sites), rhs.batch())
    # Map axes keep their axis positions, whereas a form's second input becomes the solution's output.
    places = [0 if operator.arity == 2 and axis == rhs.arity else axis for axis, _ in rhs.gatype.fields]
    order = sorted(range(len(places)), key=places.__getitem__)
    batch = solution.shape[:solution.ndim - len(sites)]
    kernel = solution.context.xp.transpose(
        solution.kernel, _batch_axes(batch) + tuple(len(batch) + site for site in order)
        + tuple(range(solution.ndim, solution.kernel.ndim)),
    )
    fields = tuple((places[site], sites[site]) for site in order)
    return Extensor._from_prepared_kernel(solution.context, solution.gatype.derive.with_fields(fields), kernel)


def _exactly(operator: Extensor) -> Callable:
    def solve(matrix: Any, columns: Any) -> Any:
        if matrix.shape[-1] != matrix.shape[-2]:
            raise TypeError(f"{operator.gatype.signature} is not square over its sites and blades")
        return operator.context.xp.linalg.solve(matrix, columns)
    return solve


def _least_squares(operator: Extensor, rcond: float) -> Callable:
    return lambda matrix, columns: operator.context.xp.linalg.pinv(matrix, rcond) @ columns


def _on_output_sites(operator: Extensor, rhs: Extensor) -> Extensor:
    rhs = rhs.cast(operator.output_subspace)
    sites = dict(operator.gatype.fields).get(0)
    # A site-free right-hand side broadcasts identically across the output sites of a field map.
    if not rhs.gatype.fields and sites:
        rhs = rhs[..., None].broadcast_to(rhs.shape + (sites,)).field()
    if dict(rhs.gatype.fields).get(0) != sites:
        raise TypeError(f"{rhs.gatype.signature} does not range over the sites of {operator.gatype.signature}")
    return rhs


def _solve(operator: Extensor, equations: int, unknowns: int, rhs: Extensor, given: int, solver: Callable) -> Extensor:
    """Flattening the `equations` and `unknowns` axes into one coupled matrix solves against the `given`
    axis of `rhs`, placing `unknowns` at the solution's output and any remaining axes of `rhs` as inputs."""
    xp = operator.context.xp
    batch = np.broadcast_shapes(operator.shape, rhs.shape)
    operator, rhs = operator.broadcast_to(batch), rhs.broadcast_to(batch)
    rows = _kernel_axes(operator, equations)
    columns = _kernel_axes(operator, unknowns)
    # A form's scalar output is one blade long and folds into the matrix.
    rest = tuple(dimension for dimension in range(len(batch), operator.kernel.ndim) if dimension not in rows + columns)
    matrix = xp.transpose(operator.kernel, _batch_axes(batch) + rows + columns + rest)
    height = prod(operator.kernel.shape[dimension] for dimension in rows)
    width = prod(operator.kernel.shape[dimension] for dimension in columns)
    matrix = xp.reshape(matrix, batch + (height, width))

    # Keep the remaining axes of rhs (its scalar output for a form and any extra inputs) as right-hand columns.
    kept = [axis for axis in range(rhs.arity + 1) if axis != given]
    sites = [dimension for axis in kept for dimension in _kernel_axes(rhs, axis)[:-1]]
    blades = [_kernel_axes(rhs, axis)[-1] for axis in kept]
    given_axes = _kernel_axes(rhs, given)
    columns_kernel = xp.transpose(rhs.kernel, _batch_axes(batch) + given_axes + tuple(sites) + tuple(blades))
    other_shape = tuple(rhs.kernel.shape[dimension] for dimension in sites + blades)
    solution = solver(matrix, xp.reshape(columns_kernel, batch + (height, prod(other_shape))))

    unknown_shape = tuple(operator.kernel.shape[dimension] for dimension in columns)
    solution = xp.reshape(solution, batch + unknown_shape + other_shape)
    # Restore kernel axis order: all site axes first, followed by all blade axes.
    b, u, s = len(batch), len(unknown_shape), len(sites)
    order = _batch_axes(batch) + tuple(range(b, b + u - 1)) + tuple(range(b + u, b + u + s)) \
        + (b + u - 1,) + tuple(range(b + u + s, b + u + s + len(blades)))
    solution = xp.transpose(solution, order)

    input_axes = [axis for axis in kept if axis]
    unknown_sites = dict(operator.gatype.fields).get(unknowns)
    rhs_sites = dict(rhs.gatype.fields)
    fields = ((0, unknown_sites),) * (unknown_sites is not None) + tuple(
        (index + 1, rhs_sites[axis]) for index, axis in enumerate(input_axes) if axis in rhs_sites
    )
    subspaces = (operator.axes[unknowns],) + tuple(rhs.axes[axis] for axis in input_axes)
    orientations = (operator.gatype.orientations[unknowns],) + tuple(rhs.gatype.orientations[axis] for axis in input_axes)
    gatype = operator.algebra.gatype.oriented(subspaces, orientations, fields=fields)
    return _result(operator, gatype, xp.reshape(solution, batch + gatype.structural_shape))


def _batch_axes(batch: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(range(len(batch)))


def _kernel_axes(operator: Extensor, axis: int) -> tuple[int, ...]:
    field_axes = [field for field, _ in operator.gatype.fields]
    sites = (operator.ndim + field_axes.index(axis),) if axis in field_axes else ()
    return sites + (operator.ndim + len(field_axes) + axis,)


@Extensor.adjoint.register(lambda t: t.arity == 1, fields=True)
def adjoint_field_map(operator: Extensor) -> Extensor:
    """Taking the metric adjoint of each site-pair block and exchanging input and output sites makes
    `A.adjoint()(b).scalar_product(a)` summed over input sites equal `b.scalar_product(A(a))` summed
    over output sites."""
    return _exchanged(operator, operator.batch().adjoint())


@Extensor.adjugate.register(lambda t: t.arity == 1, fields=True)
def adjugate_field_map(operator: Extensor) -> Extensor:
    """Taking the metric-free adjugate of each site-pair block and exchanging input and output sites
    makes `A.adjugate()(c) & x` summed over input sites equal `c & A(x)` summed over output sites
    between intrinsic spaces."""
    return _exchanged(operator, operator.batch().adjugate())


def _exchanged(operator: Extensor, blocks: Extensor) -> Extensor:
    sites = dict(operator.gatype.fields)
    kernel = blocks.kernel
    if len(sites) == 2:
        batch = blocks.ndim - 2
        kernel = operator.context.xp.swapaxes(kernel, batch, batch + 1)
    blocks = Extensor._from_prepared_kernel(blocks.context, blocks.gatype, kernel)
    # Assign the former input sites to the output (axis 0) and the former output sites to the input (axis 1).
    return blocks.field(*((0,) * (1 in sites) + (1,) * (0 in sites)))


# --- spectra and factors of the blocks read as one matrix -----------------------------------
def _square_axes(operator: Extensor) -> tuple[int, int]:
    return (0, 1) if operator.arity == 1 else (1, 2)


@Extensor.det.register(lambda t: t.over_sites.is_endomorphism, fields=True)
def det_field_map(operator: Extensor) -> Extensor:
    """Flattening sites and blades computes the determinant of the coupled field endomorphism."""
    return _scalars(operator, operator.context.xp.linalg.det(_flattened(operator, 0, 1)[0]))


@Extensor.trace.register(lambda t: t.over_sites.is_endomorphism, fields=True)
def trace_field_map(operator: Extensor) -> Extensor:
    """Tracing a coupled field endomorphism sums the traces of its diagonal site blocks."""
    return _scalars(operator, operator.context.xp.trace(_flattened(operator, 0, 1)[0], axis1=-2, axis2=-1))


@Extensor.inverse.register(lambda t: t.over_sites.is_endomorphism, fields=True)
def inverse_field_map(operator: Extensor) -> Extensor:
    """Inverting a square field map computes its inverse under coupled field composition."""
    matrix, rows, columns = _flattened(operator, 0, 1)
    gatype = operator.gatype.derive.transposed.derive.structural.derive.with_fields(_exchanged_fields(operator.gatype))
    return _from_matrix(operator, operator.context.xp.linalg.inv(matrix), columns, rows, gatype)


@Extensor.cholesky.register(lambda t: t.over_sites.is_square, fields=True)
def cholesky_field(operator: Extensor) -> Extensor:
    """The Cholesky factorization returns the lower triangular factor `L` with `A == L L^H` over the
    coupled site-and-blade matrix in the same field layout."""
    first, second = _square_axes(operator)
    matrix, rows, columns = _flattened(operator, first, second)
    return _from_matrix(operator, operator.context.xp.linalg.cholesky(matrix), rows, columns, operator.gatype.derive.structural)


@Extensor.eigh.register(lambda t: t.over_sites.is_square, fields=True)
def eigh_field(operator: Extensor) -> tuple[Extensor, Extensor]:
    """The Hermitian eigendecomposition returns ascending eigenvalues `[..., modes] Scalar` and
    coefficient-orthonormal modal fields `[..., modes] X[n]`."""
    first, second = _square_axes(operator)
    matrix, _, columns = _flattened(operator, first, second)
    values, vectors = operator.context.xp.linalg.eigh(matrix, UPLO="L")
    return _scalars(operator, values), _modes(operator, second, columns, vectors)


@Extensor.eigvalsh.register(lambda t: t.over_sites.is_square, fields=True)
def eigvalsh_field(operator: Extensor) -> Extensor:
    """Computing Hermitian eigenvalues returns ascending real eigenvalues `[..., modes] Scalar` of a
    coupled field map or form without modal fields."""
    first, second = _square_axes(operator)
    return _scalars(operator, operator.context.xp.linalg.eigvalsh(_flattened(operator, first, second)[0], UPLO="L"))


@Extensor.eigh.register(lambda t, metric: (t.over_sites.is_square or t.over_sites.is_correlation)
                        and t.axis_roles.same_as(metric), fields=True)
def eigh_field_metric(operator: Extensor, metric: Extensor) -> tuple[Extensor, Extensor]:
    """The generalized Hermitian eigendecomposition solves `operator(x) == eigenvalue * metric(x)` for
    matching endomorphisms, forms, or correlations, returning ascending eigenvalues `[..., modes] Scalar`
    and `metric`-orthonormal modal fields `[..., modes] X[n]`."""
    xp = operator.context.xp
    first, second = _square_axes(operator)
    matrix, _, columns = _flattened(operator, first, second)
    lower = xp.linalg.cholesky(_flattened(metric, first, second)[0])
    reduced = xp.linalg.solve(lower, _conjugate_transpose(xp, xp.linalg.solve(lower, matrix)))
    values, vectors = xp.linalg.eigh(_conjugate_transpose(xp, reduced), UPLO="L")
    return _scalars(operator, values), _modes(operator, second, columns, xp.linalg.solve(_conjugate_transpose(xp, lower), vectors))


@Extensor.eigvalsh.register(lambda t, metric: (t.over_sites.is_square or t.over_sites.is_correlation)
                            and t.axis_roles.same_as(metric), fields=True)
def eigvalsh_field_metric(operator: Extensor, metric: Extensor) -> Extensor:
    """Computing generalized Hermitian eigenvalues returns ascending eigenvalues `[..., modes] Scalar`
    of `operator(x) == eigenvalue * metric(x)`."""
    return eigh_field_metric(operator, metric)[0]


@Extensor.eig.register(lambda t: t.over_sites.is_endomorphism, fields=True)
def eig_field_map(operator: Extensor) -> tuple[Extensor, Extensor]:
    """The general eigendecomposition returns eigenvalues `[..., modes] Scalar` and right modal fields
    `[..., modes] X[n]` of a coupled field endomorphism."""
    matrix, _, columns = _flattened(operator, 0, 1)
    values, vectors = operator.context.xp.linalg.eig(matrix)
    return _scalars(operator, values), _modes(operator, 1, columns, vectors)


@Extensor.eigvals.register(lambda t: t.over_sites.is_endomorphism, fields=True)
def eigvals_field_map(operator: Extensor) -> Extensor:
    """Computing general eigenvalues returns the spectrum `[..., modes] Scalar` of a coupled field
    endomorphism without modal fields."""
    return _scalars(operator, operator.context.xp.linalg.eigvals(_flattened(operator, 0, 1)[0]))


@Extensor.eig.register(lambda t, metric: t.over_sites.is_endomorphism and t.axis_roles.same_as(metric), fields=True)
def eig_field_map_metric(operator: Extensor, metric: Extensor) -> tuple[Extensor, Extensor]:
    """The generalized eigendecomposition solves `operator(x) == eigenvalue * metric(x)` via
    `metric.solve(operator)`."""
    return eig_field_map(metric.solve(operator))


@Extensor.eigvals.register(lambda t, metric: t.over_sites.is_endomorphism and t.axis_roles.same_as(metric), fields=True)
def eigvals_field_map_metric(operator: Extensor, metric: Extensor) -> Extensor:
    """Computing generalized eigenvalues returns the spectrum `[..., modes] Scalar` of
    `metric.solve(operator)`."""
    return eigvals_field_map(metric.solve(operator))


@Extensor.svd.register(lambda t: t.arity == 1, fields=True)
def svd_field_map(operator: Extensor) -> tuple[Extensor, Extensor, Extensor]:
    """The reduced singular value decomposition over coupled sites and blades returns left output modal
    fields, descending singular values, and right input modal fields satisfying `A(v_i) == s_i * u_i`."""
    xp = operator.context.xp
    matrix, rows, columns = _flattened(operator, 0, 1)
    left, singular, right = xp.linalg.svd(matrix, full_matrices=False)
    return _modes(operator, 0, rows, left), _scalars(operator, singular), _modes(operator, 1, columns, _conjugate_transpose(xp, right))


@Extensor.svdvals.register(lambda t: t.arity == 1, fields=True)
def svdvals_field_map(operator: Extensor) -> Extensor:
    """Computing singular values returns descending singular values `[..., modes] Scalar` of the
    coupled site-and-blade matrix without modal fields."""
    return _scalars(operator, operator.context.xp.linalg.svd(_flattened(operator, 0, 1)[0], compute_uv=False))


def _flattened(operator: Extensor, rows: int, columns: int) -> tuple:
    xp = operator.context.xp
    batch = operator.shape
    row_axes, column_axes = _kernel_axes(operator, rows), _kernel_axes(operator, columns)
    # A form's scalar output is one blade long and folds into the matrix.
    rest = tuple(axis for axis in range(len(batch), operator.kernel.ndim) if axis not in row_axes + column_axes)
    matrix = xp.transpose(operator.kernel, _batch_axes(batch) + row_axes + column_axes + rest)
    row_shape = tuple(operator.kernel.shape[axis] for axis in row_axes)
    column_shape = tuple(operator.kernel.shape[axis] for axis in column_axes)
    return xp.reshape(matrix, batch + (prod(row_shape), prod(column_shape))), row_shape, column_shape


def _from_matrix(operator: Extensor, matrix: Any, rows: tuple[int, ...], columns: tuple[int, ...], gatype: GAType) -> Extensor:
    xp = operator.context.xp
    batch = matrix.shape[:-2]
    kernel = xp.reshape(matrix, batch + rows + columns)
    b, r, c = len(batch), len(rows), len(columns)
    order = (_batch_axes(batch) + tuple(range(b, b + r - 1)) + tuple(range(b + r, b + r + c - 1))
             + (b + r - 1, b + r + c - 1))
    return _result(operator, gatype, xp.reshape(xp.transpose(kernel, order), batch + gatype.structural_shape))


def _modes(operator: Extensor, axis: int, shape: tuple[int, ...], vectors: Any) -> Extensor:
    xp = operator.context.xp
    batch = vectors.shape[:-2]
    modes = xp.reshape(xp.swapaxes(vectors, -1, -2), batch + (vectors.shape[-1],) + shape)
    sites = dict(operator.gatype.fields).get(axis)
    gatype = operator.gatype.derive.axis(axis).derive.with_fields(((0, sites),) * (sites is not None))
    return _result(operator, gatype, modes)


def _exchanged_fields(gatype: GAType) -> tuple:
    sites = dict(gatype.fields)
    return tuple((1 - axis, count) for axis, count in sorted(sites.items(), reverse=True))


def _conjugate_transpose(xp: Any, matrix: Any) -> Any:
    return xp.conj(xp.swapaxes(matrix, -1, -2))
