"""Linear maps carry two distinct notions of transpose in `numga`. `adjugate` is metric-free and
transports a map `A` onto the oppositely oriented complement spaces of its output and input by
fixing the extrinsic factor on the left of each regressive `&` pairing
(`A.adjugate()(c) & x == c & A(x)` between intrinsic spaces), so `A.adjugate().adjugate() == A`
holds in every dimension and degenerate signature without a metric or non-null pseudoscalar.
`adjoint` transports `A` through the metric `scalar_product`
(`A.adjoint()(b).scalar_product(a) == b.scalar_product(A(a))`) and requires a non-degenerate metric
on `A`'s input space.

Determined linear systems dispatch across both unary maps (`arity == 1`) and bilinear forms
(`arity >= 2`). On a map between spaces of equal dimension, `inverse` reverses the map's arrow and
`operator.solve(rhs)` solves `operator(x) == rhs` while carrying any input axes of `rhs` onto `x`
(`A.solve(A(Y)) == Y`). On a form `F`, `F.solve(rhs)` views `F(x, y) == rhs(..., y)` across all
test inputs `y`, inverting partial evaluation on the form's first input (`F.solve(F(x)) == x`).

Least-squares solvers separate coordinate norms from metric norms. Two-argument
`operator.lstsq(rhs)`, `form.lstsq(rhs)`, `basis.lstsq(rhs)` on batched nullary values,
higher-arity tensor `lstsq`, and `operator.pinv()` compute Moore-Penrose pseudoinverses directly in
blade coefficient coordinates independently of the Clifford metric. Three-argument
`form.lstsq(rhs, metric)` diagonalizes a symmetric `form` against a positive-definite `metric`
form, selecting the minimum-`metric(x, x)` solution and applying `rcond` to generalized eigenvalues
so mode truncation is invariant to basis choice and units. Pairings, adjoints, adjugates, and
solving on maps and forms are developed in `articles/frameworks.md`, `articles/common_practice.md`,
and `docs/tutorial/extensors/07-solving.md` through `10-the-adjugate.md`.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import combinations
from math import prod

import numpy as np

from numga.backend.context import binding_context
from numga.binding import AxisTransform, AxisTransformKind
from numga.extensor import Extensor
from numga.gatype import GAType, GATypeError, GATypePattern

from numga.extensions._linalg import _linear_system, _reading, _result, _scalars


@Extensor.inverse.register(lambda t: t.maps_equal_dimensions)
def inverse_linear(operator: Extensor) -> Extensor:
    """Inverting a square linear map reverses its arrow, swapping input and output subspaces and
    orientations."""
    return Extensor._from_prepared_kernel(
        operator.context, operator.gatype.derive.transposed.derive.structural,
        operator.context.matrix_inverse(operator._kernel),
    )


def _orientation_mismatch(operator: Extensor, rhs: Extensor, *_, **__) -> Extensor:
    raise GATypeError(
        f"cannot solve for an {rhs.gatype.orientation.value} right-hand side against a map whose "
        f"output takes an {operator.gatype.orientation.value} orientation"
    )


# Refuse right-hand sides whose orientation differs from the map's output orientation.
Extensor.solve.register(lambda t, r: t.arity == 1 and r.orientation is not t.orientation, position=0)(_orientation_mismatch)
Extensor.lstsq.register(lambda t, r: t.arity <= 1 and r.orientation is not t.orientation, position=0)(_orientation_mismatch)


@Extensor.solve.register(
    lambda t, r: t.maps_equal_dimensions
    and r.output_subspace.support_is_subset_of(t.output_subspace)
)
def solve(operator: Extensor, rhs: Extensor) -> Extensor:
    """Input axes on `rhs` carry through to the solution, so `A.solve(A(Y)) == Y` holds for maps `Y`
    as well as multivectors."""
    if operator.context.is_exact:
        return operator.inverse()(rhs)
    operator, rhs = _linear_system(operator, rhs)
    batch_shape = np.broadcast_shapes(operator.shape, rhs.shape)
    operator = operator.broadcast_to(batch_shape)
    rhs = rhs.broadcast_to(batch_shape)
    columns = rhs._kernel.reshape(batch_shape + (len(rhs.output_subspace), prod(rhs.structural_shape[1:])))
    solution = operator.context.xp.linalg.solve(operator._kernel, columns)
    gatype = operator.algebra.gatype.oriented(
        (operator.axes[1],) + rhs.input_subspaces,
        (operator.gatype.input_orientations[0],) + rhs.gatype.input_orientations,
    )
    return _result(operator, gatype, solution.reshape(batch_shape + gatype.structural_shape))


@Extensor.adjugate.register(GATypePattern.map())
def adjugate(operator: Extensor) -> Extensor:
    """The adjugate transports a linear map onto the oppositely oriented complement spaces of its
    output and input, fixing the extrinsic factor on the left of each regressive pairing:
    `A.adjugate()(c) & x == c & A(x)` between intrinsic spaces,
    `x & A.adjugate()(c) == A(x) & c` between extrinsic spaces,
    `A.adjugate()(c) & x == A(x) & c` from intrinsic to extrinsic, and
    `x & A.adjugate()(c) == c & A(x)` from extrinsic to intrinsic.

    In even dimensions swapping complementary grades flips sign on odd grades, so fixing
    extrinsic-first pairings ensures `A.adjugate().adjugate() == A`, keeps pulled-back positive
    forms `G.adjugate()(M(G))` positive, and needs neither a metric nor a non-null pseudoscalar.
    """
    on_input, on_output = _adjugate_pairings(operator.gatype)
    return on_input.solve(on_output.bind({1: operator}))


@lru_cache(maxsize=None)
def _adjugate_pairings(gatype: GAType) -> tuple[Extensor, Extensor]:
    return tuple(_reading(_complement(space), space) for space in (gatype.derive.axis(1), gatype.derive.output))


def _complement(space: GAType) -> GAType:
    return space.algebra.gatype.oriented((space.output_subspace.complement(),), (space.orientation.flipped,))


@Extensor.adjoint.register(GATypePattern.map())
def adjoint(operator: Extensor) -> Extensor:
    """The metric adjoint maps `A`'s output space back to its input space through the scalar
    product, `A.adjoint()(b).scalar_product(a) == b.scalar_product(A(a))`, and therefore exists only
    when the metric on `A`'s input space is non-degenerate."""
    on_input, on_output = _adjoint_pairing(operator.gatype)
    return on_input.solve(on_output(operator.gatype.derive.output, operator))


@lru_cache(maxsize=None)
def _adjoint_pairing(gatype: GAType) -> tuple[Extensor, Extensor]:
    output, input = gatype.derive.output, gatype.derive.axis(1)
    product = gatype.algebra.operator.scalar_product
    return product(input, input), product(output, output)


def _form_system(form: Extensor, rhs: Extensor) -> tuple[Extensor, Extensor]:
    """Viewing `form(x, y) == rhs(..., y)` over test inputs `y` turns the first argument `x` into
    the column space of a unary linear system."""
    scalar = form.algebra.subspace.scalar()
    form, rhs = form.cast(scalar), rhs.cast(scalar)
    oriented = form.algebra.gatype.oriented
    matrix = Extensor._from_prepared_kernel(
        form.context, oriented((form.axes[2], form.axes[1]), (form.gatype.orientations[2], form.gatype.orientations[1])),
        form.context.first_last_input(form._kernel, form.ndim),
    )
    covector = Extensor._from_prepared_kernel(
        rhs.context,
        oriented((rhs.axes[-1],) + tuple(rhs.axes[1:-1]), (rhs.gatype.orientations[-1],) + rhs.gatype.orientations[1:-1]),
        rhs.context.first_last_input(rhs._kernel, rhs.ndim),
    )
    return matrix, covector


@Extensor.solve.register(lambda form, rhs: form.form.is_system(rhs), position=0)
def solve_form(form: Extensor, rhs: Extensor) -> Extensor:
    """Solving `form(x, y) == rhs(..., y)` for `x` across all test inputs `y` inverts binding the
    form's first input (`F.solve(F(x)) == x`) while keeping leading inputs of `rhs` on the solution."""
    matrix, covector = _form_system(form, rhs)
    return matrix.solve(covector)


@Extensor.lstsq.register(lambda form, rhs: form.form.is_system(rhs), position=0)
def lstsq_form(form: Extensor, rhs: Extensor, *, rcond: float = 1e-15) -> Extensor:
    """Unweighted form least squares solves `form(x, y) == rhs(..., y)` in blade coordinates,
    whereas supplying a third `metric` form argument measures norms and mode cutoffs in that metric."""
    matrix, covector = _form_system(form, rhs)
    return matrix.lstsq(covector, rcond=rcond)


@Extensor.lstsq.register(lambda form, rhs, metric: form.form.is_system(rhs) and form.form.matches(metric))
def lstsq_form_metric(form: Extensor, rhs: Extensor, metric: Extensor, *, rcond: float = 1e-15) -> Extensor:
    """Among solutions of `form(x, y) == rhs(..., y)`, diagonalizing `form` against `metric` selects
    the one of minimum `metric(x, x)` and applies `rcond` to generalized eigenvalues so the cutoff
    is independent of basis and units."""
    values, modes = form.eigh(metric)
    xp = values.context.xp
    curvatures = values._kernel[..., 0]
    kept = xp.abs(curvatures) > rcond * xp.max(xp.abs(curvatures), axis=-1, keepdims=True)
    weights = _scalars(values, xp.where(kept, 1 / xp.where(kept, curvatures, 1), 0))
    readouts = rhs[..., None].bind({rhs.arity - 1: modes})
    return (modes * (readouts * weights)).sum(axis=-1)


@Extensor.pinv.register(GATypePattern.map())
def pinv(operator: Extensor, *, rcond: float = 1e-15) -> Extensor:
    """The Moore-Penrose pseudoinverse acts on blade coordinates independently of the Clifford
    metric, discarding singular values up to `rcond` times the largest."""
    return _result(
        operator, operator.gatype.derive.transposed.derive.structural,
        operator.context.xp.linalg.pinv(operator._kernel, rcond),
    )


@Extensor.lstsq.register(lambda t, r: t.arity == 0 and r.arity == 0)
def lstsq_nullary(basis: Extensor, rhs: Extensor, *, rcond: float = 1e-15) -> Extensor:
    """Scalar weights along the trailing batch axis fit `(c * basis).sum(axis=-1)` to `rhs` in least
    squares."""
    if basis.ndim < 1:
        raise ValueError("nullary lstsq requires at least one batch axis on value")
    basis, rhs = _linear_system(basis, rhs)
    matrix = basis.context.xp.swapaxes(basis._kernel, -2, -1)
    columns = rhs._kernel[..., None]
    solution = basis.context.xp.linalg.pinv(matrix, rcond) @ columns
    return _scalars(basis, solution[..., 0])


@Extensor.lstsq.register(
    lambda t, r: t.arity == 1
    and r.output_subspace.support_is_subset_of(t.output_subspace)
)
def lstsq(operator: Extensor, rhs: Extensor, *, rcond: float = 1e-15) -> Extensor:
    """Unweighted least squares minimizes Euclidean residual and solution norms in blade coordinates
    while preserving any input axes on `rhs`."""
    operator, rhs = _linear_system(operator, rhs)
    return operator.pinv(rcond=rcond)(rhs)


@lru_cache(maxsize=None)
def _grouped_lstsq_plan(operator: GAType, rhs: GAType) -> tuple:
    if rhs.arity >= operator.arity:
        raise TypeError("lstsq RHS must leave at least one input to solve")
    axes, matches = operator.subspaces, []
    for inputs in combinations(range(1, operator.arity + 1), rhs.arity):
        retained = (0,) + inputs
        alignment = tuple(AxisTransform.plan(source, axes[i]) for source, i in zip(rhs.subspaces, retained))
        if all(transform.is_implicit_bind_compatible for transform in alignment):
            matches.append((retained, alignment))
    if len(matches) != 1:
        raise TypeError("lstsq RHS must match exactly one ordered sequence of inputs without projection")
    retained, alignment = matches[0]
    selected = tuple(axis for axis in range(1, operator.arity + 1) if axis not in retained)
    transforms = tuple((axis, transform) for axis, transform in enumerate(alignment)
                       if transform.kind is not AxisTransformKind.EXACT)
    result = operator.algebra.gatype.oriented(
        tuple(axes[i] for i in selected), tuple(operator.orientations[i] for i in selected),
    )
    shape = prod(len(axes[i]) for i in retained), prod(len(axes[i]) for i in selected)
    return retained + selected, shape, result, transforms


@Extensor.lstsq.register(
    lambda t, r: t.arity > 1
    and r.output_subspace.support_is_subset_of(t.output_subspace)
)
def lstsq_tensor(operator: Extensor, rhs: Extensor, *, rcond: float = 1e-15) -> Extensor:
    """Matching `rhs`'s inputs to a unique ordered subsequence of `operator`'s inputs solves
    linearly over the tensor product of the remaining inputs (without requiring a rank-one
    factorization)."""
    permutation, (rows, columns), gatype, transforms = _grouped_lstsq_plan(operator.gatype, rhs.gatype)
    context = binding_context(operator.context, (rhs.context,))
    operator, rhs = context.lower(operator), context.lower(rhs)
    coefficients = rhs._kernel
    for axis, transform in transforms:
        coefficients = operator.context.transform_axis(coefficients, rhs.arity + 1, axis, transform)
    matrix = operator.context.xp.transpose(
        operator._kernel, tuple(range(operator.ndim)) + tuple(operator.ndim + i for i in permutation),
    )
    matrix = matrix.reshape(operator.shape + (rows, columns))
    solution = operator.context.xp.linalg.pinv(matrix, rcond) @ coefficients.reshape(rhs.shape + (rows, 1))
    return _result(operator, gatype, solution.reshape(solution.shape[:-2] + gatype.structural_shape))
