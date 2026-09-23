"""Spectra of maps: eigenproblems, determinant, Cholesky factor and SVD.

Eigenproblems and determinants require matching input/output blade support;
signed or reordered output layouts are aligned to the input first. Eigh and SVD
use the Euclidean/Hermitian coefficient inner product, independently of the
Clifford metric. Eigh assumes Hermitian input.

SVD returns (left_vectors, singular_values, right_vectors), with right vectors
rather than conjugate-transposed rows. Only the reduced SVD is returned.
"""

from __future__ import annotations

from numga.extensor import Extensor
from numga.gatype import GATypePattern

from numga.extensions._linalg import _eigenpairs, _is_endomorphism, _result, _scalars


@Extensor.det.register(_is_endomorphism)
def det(value: Extensor) -> Extensor:
    """Determinant of an endomorphism, as a scalar extensor per batch item."""
    value = value.cast(value.axes[1])
    return _scalars(value, value.context.xp.linalg.det(value._kernel))


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
