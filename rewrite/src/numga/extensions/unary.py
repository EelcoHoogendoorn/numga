"""Linear algebra on unary extensors and scalar-output binary forms.

Eigenproblems and determinants require matching input/output blade support;
signed or reordered output layouts are aligned to the input first. Eigh, SVD,
pseudoinverse and least-squares use the Euclidean/Hermitian coefficient inner
product, independently of the Clifford metric. Eigh assumes Hermitian input.

Eigenvectors and singular vectors are nullary extensors: the last batch axis
enumerates modes. SVD returns (left_vectors, singular_values, right_vectors),
with right vectors rather than conjugate-transposed rows. Only the reduced
SVD is returned. Spectral values are scalar extensors on the same mode axis.
Complex eigenpairs promote the result context; they are never truncated to
real coefficients. Numerical methods require a concrete NumPy/JAX context.

Forms expose their two slot axes as matrix axes by indexing the scalar-output
axis explicitly. Eigenproblems require matching slot support. Transpose swaps
the slots and retains the scalar output and all batch axes. Generalized form
eigenproblems use SciPy on the NumPy backend (the optional linalg extra).
"""

from __future__ import annotations

from functools import lru_cache
from itertools import combinations
from math import prod
from typing import Any

import numpy as np

from numga.backend.context import binding_context
from numga.binding import AxisTransform, AxisTransformKind
from numga.extensor import Extensor
from numga.gatype import GAType, GATypePattern


def _is_form(gatype: GAType) -> bool:
    return gatype.arity == 2 and gatype.output_subspace.same_support(gatype.algebra.subspace.scalar())


def _is_square_form(gatype: GAType) -> bool:
    return _is_form(gatype) and gatype.subspaces[1].same_support(gatype.subspaces[2])


def _matching_forms(left: GAType, right: GAType) -> bool:
    return (_is_square_form(left) and _is_square_form(right)
            and left.subspaces[2].same_support(right.subspaces[2]))


def _is_endomorphism(gatype: GAType) -> bool:
    return gatype.arity == 1 and gatype.subspaces[0].same_support(gatype.subspaces[1])


def _result(value: Extensor, gatype: GAType, kernel: Any) -> Extensor:
    context = value.context
    dtype = np.result_type(context.dtype, kernel.dtype)
    if dtype != context.dtype:
        context = type(context)(value.algebra, dtype=dtype, execution=context.execution)
    return Extensor._from_prepared_kernel(
        context, gatype, context.xp.asarray(kernel, dtype=dtype),
    )


def _scalars(value: Extensor, kernel: Any) -> Extensor:
    return _result(value, value.algebra.gatype.scalar(), kernel[..., None])


def _eigenpairs(value: Extensor, values: Any, vectors: Any) -> tuple[Extensor, Extensor]:
    vectors = _result(
        value, value.algebra.gatype(value.axes[1]),
        value.context.xp.swapaxes(vectors, -1, -2),
    )
    return _scalars(vectors, values), vectors


def _form_map(value: Extensor) -> Extensor:
    value = value.cast(value.algebra.subspace.scalar())
    return Extensor._from_prepared_kernel(
        value.context, value.algebra.gatype(value.input_subspaces), value._kernel[..., 0, :, :],
    )


def _pencil(value: Extensor, metric: Extensor) -> tuple[Extensor, Extensor]:
    context = binding_context(value.context, (metric.context,))
    left = _form_map(context.lower(value)).cast(value.axes[2])
    right = _form_map(context.lower(metric))
    right = right(value.algebra.operator.identity(value.axes[2])).cast(value.axes[2])
    return left, right


def _linear_system(value: Extensor, rhs: Extensor) -> tuple[Extensor, Extensor]:
    context = binding_context(value.context, (rhs.context,))
    return context.lower(value), context.lower(rhs).cast(value.axes[0])


@Extensor.transpose.register(GATypePattern.map())
def transpose_map(value: Extensor) -> Extensor:
    permutation = tuple(range(value.ndim)) + (value.ndim + 1, value.ndim)
    return Extensor._from_prepared_kernel(
        value.context, value.gatype.transposed, value._kernel.transpose(permutation),
    )


@Extensor.transpose.register(_is_form)
def transpose_form(value: Extensor) -> Extensor:
    permutation = tuple(range(value.ndim + 1)) + (value.ndim + 2, value.ndim + 1)
    gatype = value.algebra.gatype((value.axes[0], value.axes[2], value.axes[1]))
    return Extensor._from_prepared_kernel(value.context, gatype, value._kernel.transpose(permutation))


@Extensor.inverse.register(lambda t: t.is_square_map)
def inverse_linear(value: Extensor) -> Extensor:
    """Composition inverse, swapping input and output coefficient layouts."""
    return Extensor._from_prepared_kernel(
        value.context, value.gatype.transposed.structural,
        value.context.matrix_inverse(value._kernel),
    )


@Extensor.det.register(_is_endomorphism)
def det(value: Extensor) -> Extensor:
    """Determinant of an endomorphism, as a scalar extensor per batch item."""
    value = value.cast(value.axes[1])
    return _scalars(value, value.context.xp.linalg.det(value._kernel))


@Extensor.solve.register(
    lambda t, r: t.is_square_map
    and r.output_subspace.support_is_subset_of(t.output_subspace)
)
def solve(value: Extensor, rhs: Extensor) -> Extensor:
    """Solve A(x)=rhs, preserving every RHS slot and broadcasting both batches."""
    value, rhs = _linear_system(value, rhs)
    columns = rhs._kernel.reshape(rhs.shape + (len(rhs.output_subspace), prod(rhs.structural_shape[1:])))
    solution = value.context.xp.linalg.solve(value._kernel, columns)
    gatype = value.algebra.gatype((value.axes[1],) + rhs.input_subspaces)
    return _result(value, gatype, solution.reshape(solution.shape[:-2] + gatype.structural_shape))


@Extensor.pinv.register(GATypePattern.map())
def pinv(value: Extensor, *, rcond: float = 1e-15) -> Extensor:
    """Moore-Penrose inverse; discard singular values <= rcond times the largest."""
    return _result(
        value, value.gatype.transposed.structural,
        value.context.xp.linalg.pinv(value._kernel, rcond),
    )


@Extensor.lstsq.register(
    lambda t, r: t.arity == 1
    and r.output_subspace.support_is_subset_of(t.output_subspace)
)
def lstsq(value: Extensor, rhs: Extensor, *, rcond: float = 1e-15) -> Extensor:
    """Minimum coefficient-norm least-squares solution, using pinv's cutoff.

    Returns the solution extensor alone; residuals are A(solution)-rhs.
    """
    value, rhs = _linear_system(value, rhs)
    return value.pinv(rcond=rcond)(rhs)


@lru_cache(maxsize=None)
def _grouped_lstsq_plan(value: GAType, rhs: GAType):
    if rhs.arity >= value.arity:
        raise TypeError("lstsq RHS must leave at least one input slot to solve")
    axes, matches = value.subspaces, []
    for inputs in combinations(range(1, value.arity + 1), rhs.arity):
        retained = (0,) + inputs
        alignment = tuple(AxisTransform.plan(source, axes[i]) for source, i in zip(rhs.subspaces, retained))
        if all(transform.is_implicit_bind_compatible for transform in alignment):
            matches.append((retained, alignment))
    if len(matches) != 1:
        raise TypeError("lstsq RHS must match exactly one ordered sequence of input slots without projection")
    retained, alignment = matches[0]
    selected = tuple(axis for axis in range(1, value.arity + 1) if axis not in retained)
    transforms = tuple((axis, transform) for axis, transform in enumerate(alignment)
                       if transform.kind is not AxisTransformKind.EXACT)
    result = value.algebra.gatype(tuple(axes[i] for i in selected))
    shape = prod(len(axes[i]) for i in retained), prod(len(axes[i]) for i in selected)
    return retained + selected, shape, result, transforms


@Extensor.lstsq.register(
    lambda t, r: t.arity > 1
    and r.output_subspace.support_is_subset_of(t.output_subspace)
)
def lstsq_tensor(value: Extensor, rhs: Extensor, *, rcond: float = 1e-15) -> Extensor:
    """Infer and solve a coefficient contraction from the two extensor signatures.

    RHS inputs must match exactly one ordered subsequence of the operator inputs.
    The unmatched inputs form the solution's output-first signature in their
    original order. The operator output and matched inputs form the equations.
    This is a linear solve over a tensor product, not a nonlinear factor solve;
    the unknown need not separate into one multivector per unmatched slot.
    """
    permutation, (rows, columns), gatype, transforms = _grouped_lstsq_plan(value.gatype, rhs.gatype)
    context = binding_context(value.context, (rhs.context,))
    value, rhs = context.lower(value), context.lower(rhs)
    coefficients = rhs._kernel
    for axis, transform in transforms:
        coefficients = value.context.transform_axis(coefficients, rhs.arity + 1, axis, transform)
    matrix = value._kernel.transpose(tuple(range(value.ndim)) + tuple(value.ndim + i for i in permutation))
    matrix = matrix.reshape(value.shape + (rows, columns))
    solution = value.context.xp.linalg.pinv(matrix, rcond) @ coefficients.reshape(rhs.shape + (rows, 1))
    return _result(value, gatype, solution.reshape(solution.shape[:-2] + gatype.structural_shape))


@Extensor.eig.register(_is_endomorphism)
def eig(value: Extensor) -> tuple[Extensor, Extensor]:
    """Eigenvalues and right eigenvectors, with modes on the last batch axis."""
    value = value.cast(value.axes[1])
    return _eigenpairs(value, *value.context.xp.linalg.eig(value._kernel))


@Extensor.cholesky.register(_is_endomorphism)
def cholesky(value: Extensor) -> Extensor:
    """Lower coefficient factor L with A = L L^H, in the input layout.

    Assumes Hermitian positive-definite input; validity is left to the backend.
    """
    value = value.cast(value.axes[1])
    return _result(value, value.gatype.structural, value.context.xp.linalg.cholesky(value._kernel))


@Extensor.eigvals.register(_is_endomorphism)
def eigvals(value: Extensor) -> Extensor:
    """Eigenvalues without computing eigenvectors."""
    value = value.cast(value.axes[1])
    return _scalars(value, value.context.xp.linalg.eigvals(value._kernel))


@Extensor.eigh.register(_is_endomorphism)
def eigh(value: Extensor) -> tuple[Extensor, Extensor]:
    """Ascending eigenvalues and coefficient-orthonormal Hermitian eigenvectors."""
    value = value.cast(value.axes[1])
    return _eigenpairs(value, *value.context.xp.linalg.eigh(value._kernel, UPLO="L"))


@Extensor.eigvalsh.register(_is_endomorphism)
def eigvalsh(value: Extensor) -> Extensor:
    """Ascending eigenvalues of a Hermitian endomorphism."""
    value = value.cast(value.axes[1])
    return _scalars(value, value.context.xp.linalg.eigvalsh(value._kernel, UPLO="L"))


@Extensor.svd.register(GATypePattern.map())
def svd(value: Extensor) -> tuple[Extensor, Extensor, Extensor]:
    """Reduced SVD as batches (left vectors, singular values, right vectors).

    A(v_i) = s_i u_i; A = sum_i s_i u_i v_i^H. Vector types are the
    original output and input spaces; singular values descend by magnitude.
    """
    xp = value.context.xp
    left, singular, right_h = xp.linalg.svd(value._kernel, full_matrices=False)
    return (
        _result(value, value.algebra.gatype(value.axes[0]), xp.swapaxes(left, -1, -2)),
        _scalars(value, singular),
        _result(value, value.algebra.gatype(value.axes[1]), xp.conj(right_h)),
    )


@Extensor.svdvals.register(GATypePattern.map())
def svdvals(value: Extensor) -> Extensor:
    """Descending singular values without computing singular vectors."""
    return _scalars(value, value.context.xp.linalg.svd(value._kernel, compute_uv=False))


@Extensor.eig.register(_is_square_form)
def eig_form(value: Extensor) -> tuple[Extensor, Extensor]:
    return _form_map(value).eig()


@Extensor.eigh.register(_is_square_form)
def eigh_form(value: Extensor) -> tuple[Extensor, Extensor]:
    return _form_map(value).eigh()


@Extensor.eigvals.register(_is_square_form)
def eigvals_form(value: Extensor) -> Extensor:
    return _form_map(value).eigvals()


@Extensor.eigvalsh.register(_is_square_form)
def eigvalsh_form(value: Extensor) -> Extensor:
    return _form_map(value).eigvalsh()


@Extensor.det.register(_is_square_form)
def det_form(value: Extensor) -> Extensor:
    return _form_map(value).det()


@Extensor.cholesky.register(_is_square_form)
def cholesky_form(value: Extensor) -> Extensor:
    """Lower unary coefficient factor, with both axes in the second slot layout."""
    return _form_map(value).cholesky()


@Extensor.trace.register(_is_square_form, position=0)
def trace_form(value: Extensor) -> Extensor:
    return _form_map(value).trace()


@Extensor.svd.register(_is_form)
def svd_form(value: Extensor) -> tuple[Extensor, Extensor, Extensor]:
    return _form_map(value).svd()


@Extensor.svdvals.register(_is_form)
def svdvals_form(value: Extensor) -> Extensor:
    return _form_map(value).svdvals()


@Extensor.eig.register(_matching_forms)
def eig_forms(value: Extensor, metric: Extensor) -> tuple[Extensor, Extensor]:
    """Generalized form eigenpairs, including infinite eigenvalues for singular metrics.

    The NumPy backend uses SciPy (the linalg extra); mode selection belongs to the caller.
    """
    left, right = _pencil(value, metric)
    return _eigenpairs(left, *left.context.generalized_eig(left._kernel, right._kernel))


@Extensor.eigh.register(_matching_forms)
def eigh_forms(value: Extensor, metric: Extensor) -> tuple[Extensor, Extensor]:
    """Generalized Hermitian form eigenpairs, normalized in a positive-definite metric."""
    left, right = _pencil(value, metric)
    return _eigenpairs(left, *left.context.generalized_eigh(left._kernel, right._kernel))


@Extensor.eigvals.register(_matching_forms)
def eigvals_forms(value: Extensor, metric: Extensor) -> Extensor:
    left, right = _pencil(value, metric)
    return _scalars(left, left.context.generalized_eigvals(left._kernel, right._kernel))


@Extensor.eigvalsh.register(_matching_forms)
def eigvalsh_forms(value: Extensor, metric: Extensor) -> Extensor:
    left, right = _pencil(value, metric)
    return _scalars(left, left.context.generalized_eigvalsh(left._kernel, right._kernel))
