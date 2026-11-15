"""Spectral decompositions, determinants, and factorizations operate on unary linear maps
(`arity == 1`). Endomorphism operations (`det`, `eig`, `eigvals`, `eigh`, `eigvalsh`, `cholesky`)
require matching input and output blade support and first align the output subspace to the input's
blade order and signs via `operator.cast(operator.axes[1])`, keeping determinants and spectra
invariant when input and output carry signed or permuted layouts of the same geometric subspace.

Whereas bilinear forms measure spectra relative to a second partner form, unary endomorphisms map a
space directly into itself for `det` and `eig`, while `eigh`, `cholesky`, and `svd` use the Euclidean
or Hermitian inner product on blade coefficients independently of the Clifford metric. `eig` and
`eigh` place the mode index on the trailing batch axis (`[..., modes]`), and `svd` returns the
reduced triple `(left_vectors, singular_values, right_vectors)` with right singular vectors returned
as input-space extensors satisfying `operator(v_i) == s_i * u_i` rather than conjugate-transposed
matrix rows. Spectral conventions for maps and forms are compared in `articles/frameworks.md` and
`docs/tutorial/extensors/11-eigenvalues-and-singular-values.md`.
"""

from __future__ import annotations

from numga.extensor import Extensor
from numga.gatype import GATypePattern

from numga.extensions._linalg import _eigenpairs, _result, _scalars


@Extensor.det.register(lambda t: t.is_endomorphism)
def det(operator: Extensor) -> Extensor:
    """Aligning the output layout to the input yields the basis-independent determinant of an
    endomorphism as a `Scalar` extensor."""
    operator = operator.cast(operator.axes[1])
    return _scalars(operator, operator.context.xp.linalg.det(operator._kernel))


@Extensor.eig.register(lambda t: t.is_endomorphism)
def eig(operator: Extensor) -> tuple[Extensor, Extensor]:
    """The general eigendecomposition returns eigenvalues `[..., modes] Scalar` and right eigenvectors
    `[..., modes]` of an endomorphism with the mode index on the last batch axis."""
    operator = operator.cast(operator.axes[1])
    return _eigenpairs(operator, *operator.context.xp.linalg.eig(operator._kernel))


@Extensor.cholesky.register(lambda t: t.is_endomorphism)
def cholesky(operator: Extensor) -> Extensor:
    """Factoring a coefficient-Hermitian positive-definite endomorphism produces the lower triangular
    map `L` satisfying `A == L(L^H)` in the input layout."""
    operator = operator.cast(operator.axes[1])
    return _result(operator, operator.gatype.derive.structural, operator.context.xp.linalg.cholesky(operator._kernel))


@Extensor.eigvals.register(lambda t: t.is_endomorphism)
def eigvals(operator: Extensor) -> Extensor:
    """Computing general eigenvalues returns the spectrum of an endomorphism as `[..., modes] Scalar`
    without eigenvectors."""
    operator = operator.cast(operator.axes[1])
    return _scalars(operator, operator.context.xp.linalg.eigvals(operator._kernel))


@Extensor.eigh.register(lambda t: t.is_endomorphism)
def eigh(operator: Extensor) -> tuple[Extensor, Extensor]:
    """The Hermitian eigendecomposition returns ascending real eigenvalues `[..., modes] Scalar` and
    coefficient-orthonormal eigenvectors `[..., modes]` on the trailing batch axis."""
    operator = operator.cast(operator.axes[1])
    return _eigenpairs(operator, *operator.context.xp.linalg.eigh(operator._kernel, UPLO="L"))


@Extensor.eigvalsh.register(lambda t: t.is_endomorphism)
def eigvalsh(operator: Extensor) -> Extensor:
    """Computing Hermitian eigenvalues returns ascending real eigenvalues `[..., modes] Scalar` of a
    coefficient-Hermitian endomorphism without eigenvectors."""
    operator = operator.cast(operator.axes[1])
    return _scalars(operator, operator.context.xp.linalg.eigvalsh(operator._kernel, UPLO="L"))


@Extensor.svd.register(GATypePattern.map())
def svd(operator: Extensor) -> tuple[Extensor, Extensor, Extensor]:
    """The reduced singular value decomposition returns `(left_vectors, singular_values, right_vectors)`
    on `[..., modes]` in the map's output and input spaces with descending singular values and
    `A(v_i) == s_i * u_i`."""
    xp = operator.context.xp
    left, singular, right_h = xp.linalg.svd(operator._kernel, full_matrices=False)
    return (
        _result(operator, operator.gatype.derive.output, xp.swapaxes(left, -1, -2)),
        _scalars(operator, singular),
        _result(operator, operator.gatype.derive.axis(1), xp.conj(right_h)),
    )


@Extensor.svdvals.register(GATypePattern.map())
def svdvals(operator: Extensor) -> Extensor:
    """Computing singular values returns descending coefficient singular values `[..., modes] Scalar`
    of a linear map without singular vectors."""
    return _scalars(operator, operator.context.xp.linalg.svd(operator._kernel, compute_uv=False))
