"""Linear systems: inverse, solve, least squares, pseudoinverse and adjugate on maps and forms.

Pseudoinverse and least-squares use the Euclidean/Hermitian coefficient inner
product, independently of the Clifford metric.
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
def inverse_linear(value: Extensor) -> Extensor:
    """Composition inverse, swapping input and output coefficient layouts."""
    return Extensor._from_prepared_kernel(
        value.context, value.gatype.derive.transposed.derive.structural,
        value.context.matrix_inverse(value._kernel),
    )


def _orientation_mismatch(value: Extensor, rhs: Extensor, *_: object, **__: object) -> Extensor:
    raise GATypeError(
        f"cannot solve for an {rhs.gatype.orientation.value} right-hand side against a map whose "
        f"output takes an {value.gatype.orientation.value} orientation"
    )


# A right-hand side of the other orientation than the map's output has no solution, by its type.
Extensor.solve.register(lambda t, r: t.arity == 1 and r.orientation is not t.orientation, position=0)(_orientation_mismatch)
Extensor.lstsq.register(lambda t, r: t.arity <= 1 and r.orientation is not t.orientation, position=0)(_orientation_mismatch)


@Extensor.solve.register(
    lambda t, r: t.maps_equal_dimensions
    and r.output_subspace.support_is_subset_of(t.output_subspace)
)
def solve(value: Extensor, rhs: Extensor) -> Extensor:
    """Solve A(x) == rhs for x: the inverse of composing into A's input, so A.solve(A(y)) == y.

    Every input of rhs is kept as an input of the solution, so A.solve(A(Y)) == Y for a
    map Y too; both batches broadcast. An exact A is solved by binding its exact inverse, cached
    on its kernel, into rhs: one contraction with a constant.
    """
    if value.context.is_exact:
        return value.inverse()(rhs)
    value, rhs = _linear_system(value, rhs)
    batch_shape = np.broadcast_shapes(value.shape, rhs.shape)
    value = value.broadcast_to(batch_shape)
    rhs = rhs.broadcast_to(batch_shape)
    columns = rhs._kernel.reshape(batch_shape + (len(rhs.output_subspace), prod(rhs.structural_shape[1:])))
    solution = value.context.xp.linalg.solve(value._kernel, columns)
    gatype = value.algebra.gatype.oriented(
        (value.axes[1],) + rhs.input_subspaces, (value.gatype.input_orientations[0],) + rhs.gatype.input_orientations,
    )
    return _result(value, gatype, solution.reshape(batch_shape + gatype.structural_shape))


@Extensor.adjugate.register(GATypePattern.map())
def adjugate(value: Extensor) -> Extensor:
    """The map on complements that incidence carries over, each pairing read with its extrinsic factor first.

    The input is the complement of A's output and the output the complement of A's input, each oppositely
    oriented, as a dual is, so that it pairs with its space to an ordinary number through the scalar part of the
    regressive product. Both pairings take the extrinsic factor first, as every reading does:
    `A.adjugate()(c) & x == c & A(x)` for a map between intrinsic spaces, `x & A.adjugate()(c) == A(x) & c`
    between extrinsic ones, `A.adjugate()(c) & x == A(x) & c` from intrinsic to extrinsic, and
    `x & A.adjugate()(c) == c & A(x)` from extrinsic to intrinsic. Swapping the factors of a pairing of
    complementary grades costs nothing in an odd dimension and the grade involute in an even one, so there the
    order is part of the definition, built into the exact pairings, which the types key. Read so, in every
    dimension the adjugate applied twice is A, a form symmetric under the reading is its own adjugate, a positive
    form pulled back, `G.adjugate()(M(G))`, is positive, and `A(B).adjugate() == B.adjugate()(A.adjugate())`.

    It needs no metric and does not depend on the scale of the pseudoscalar, so it exists in degenerate algebras
    and for singular A. For an outermorphism it is the adjugate in the classical sense: the determinant times the
    inverse of the outermorphism on the complementary grade. A map into the complement of its own input has an
    adjugate of its own type, and is symmetric when it equals its adjugate.
    """
    on_input, on_output = _adjugate_pairings(value.gatype)
    return on_input.solve(on_output.bind({1: value}))


@lru_cache(maxsize=None)
def _adjugate_pairings(gatype: GAType) -> tuple[Extensor, Extensor]:
    """The exact readings of a map's input and of its output by their complements, each a form taking the
    complement first."""
    return tuple(_reading(_complement(space), space) for space in (gatype.derive.axis(1), gatype.derive.output))


def _complement(space: GAType) -> GAType:
    """The complementary space, oppositely oriented, as a dual is, so that it pairs with the space to an ordinary
    number."""
    return space.algebra.gatype.oriented((space.output_subspace.complement(),), (space.orientation.flipped,))


@Extensor.adjoint.register(GATypePattern.map())
def adjoint(value: Extensor) -> Extensor:
    """The map the metric carries over, the adjoint of Hestenes and Sobczyk: from A's output to its
    input, with A.adjoint()(b).scalar_product(a) == b.scalar_product(A(a)).

    The input and the output are paired with themselves by the scalar product, the scalar part of
    the geometric product. The adjoint exists where that pairing on the input is invertible: in a
    degenerate algebra it fails, with LinAlgError, wherever the ideal blades enter. A versor's
    sandwich has its inverse sandwich as its adjoint, and a map into its own input is symmetric when
    it equals its adjoint.
    """
    on_input, on_output = _adjoint_pairing(value.gatype)
    return on_input.solve(on_output(value.gatype.derive.output, value))


@lru_cache(maxsize=None)
def _adjoint_pairing(gatype: GAType) -> tuple[Extensor, Extensor]:
    """The scalar-product pairings of a map's input with itself, and of its output with itself."""
    output, input = gatype.derive.output, gatype.derive.axis(1)
    product = gatype.algebra.operator.scalar_product
    return product(input, input), product(output, output)


def _form_system(value: Extensor, rhs: Extensor) -> tuple[Extensor, Extensor]:
    """Rewrite value(x, y) = rhs(..., y) as a coefficient system over y.

    The matrix maps the unknown x to the coefficients of value(x, .) over the last axis;
    the right-hand side is rhs read out over that same axis, with its leading axes kept
    as inputs of the solution. Each stays in its own context, so an exact form gives an
    exact matrix.
    """
    scalar = value.algebra.subspace.scalar()
    value, rhs = value.cast(scalar), rhs.cast(scalar)
    oriented = value.algebra.gatype.oriented
    matrix = Extensor._from_prepared_kernel(
        value.context, oriented((value.axes[2], value.axes[1]), (value.gatype.orientations[2], value.gatype.orientations[1])),
        value.context.first_last_input(value._kernel, value.ndim),
    )
    covector = Extensor._from_prepared_kernel(
        rhs.context,
        oriented((rhs.axes[-1],) + tuple(rhs.axes[1:-1]), (rhs.gatype.orientations[-1],) + rhs.gatype.orientations[1:-1]),
        rhs.context.first_last_input(rhs._kernel, rhs.ndim),
    )
    return matrix, covector


@Extensor.solve.register(lambda value, rhs: value.form.is_system(rhs), position=0)
def solve_form(value: Extensor, rhs: Extensor) -> Extensor:
    """Solve value(x, y) == rhs(..., y) for all y, with x in the form's first axis: the inverse
    of binding that axis, so F.solve(F(x)) == x.

    Leading inputs of rhs are kept as inputs of the solution, so a bilinear
    right-hand side yields a map. The last input of rhs must match the form's last input.
    Only the first axis is solved for.
    """
    matrix, covector = _form_system(value, rhs)
    return matrix.solve(covector)


@Extensor.lstsq.register(lambda value, rhs: value.form.is_system(rhs), position=0)
def lstsq_form(value: Extensor, rhs: Extensor, *, rcond: float = 1e-15) -> Extensor:
    """Least-squares version of solve_form, using pinv's cutoff on the form's coefficients."""
    matrix, covector = _form_system(value, rhs)
    return matrix.lstsq(covector, rcond=rcond)


@Extensor.lstsq.register(lambda value, rhs, metric: value.form.is_system(rhs) and value.form.matches(metric))
def lstsq_form_metric(value: Extensor, rhs: Extensor, metric: Extensor, *, rcond: float = 1e-15) -> Extensor:
    """Least squares of a symmetric form in a metric: of the x with value(x, .) == rhs(..., .),
    the one smallest in metric(x, x).

    The form is diagonalized against the metric, and each mode is solved on its own. Modes whose
    curvature per unit metric is at most rcond times the largest are dropped, so the cutoff is a
    ratio of curvatures, independent of basis and units. Leading axes of rhs are kept as input
    axes of the solution.
    """
    values, modes = value.eigh(metric)                       # [..., modes] Scalar, [..., modes] Axis
    xp = values.context.xp
    curvatures = values._kernel[..., 0]
    kept = xp.abs(curvatures) > rcond * xp.max(xp.abs(curvatures), axis=-1, keepdims=True)
    weights = _scalars(values, xp.where(kept, 1 / xp.where(kept, curvatures, 1), 0))
    readouts = rhs[..., None].bind({rhs.arity - 1: modes})   # [..., modes] Scalar <- (...)
    return (modes * (readouts * weights)).sum(axis=-1)


@Extensor.pinv.register(GATypePattern.map())
def pinv(value: Extensor, *, rcond: float = 1e-15) -> Extensor:
    """Moore-Penrose inverse; discard singular values <= rcond times the largest."""
    return _result(
        value, value.gatype.derive.transposed.derive.structural,
        value.context.xp.linalg.pinv(value._kernel, rcond),
    )


@Extensor.lstsq.register(lambda t, r: t.arity == 0 and r.arity == 0)
def lstsq_nullary(value: Extensor, rhs: Extensor, *, rcond: float = 1e-15) -> Extensor:
    """Solve for the scalar coefficients of a linear combination along the trailing batch axis.

    Finds c such that sum_i c_i * value[..., i] ≈ rhs.
    """
    if value.ndim < 1:
        raise ValueError("nullary lstsq requires at least one batch axis on value")
    value, rhs = _linear_system(value, rhs)
    matrix = value.context.xp.swapaxes(value._kernel, -2, -1)
    columns = rhs._kernel[..., None]
    solution = value.context.xp.linalg.pinv(matrix, rcond) @ columns
    return _scalars(value, solution[..., 0])


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
        raise TypeError("lstsq RHS must leave at least one input to solve")
    axes, matches = value.subspaces, []
    for inputs in combinations(range(1, value.arity + 1), rhs.arity):
        retained = (0,) + inputs
        alignment = tuple(AxisTransform.plan(source, axes[i]) for source, i in zip(rhs.subspaces, retained))
        if all(transform.is_implicit_bind_compatible for transform in alignment):
            matches.append((retained, alignment))
    if len(matches) != 1:
        raise TypeError("lstsq RHS must match exactly one ordered sequence of inputs without projection")
    retained, alignment = matches[0]
    selected = tuple(axis for axis in range(1, value.arity + 1) if axis not in retained)
    transforms = tuple((axis, transform) for axis, transform in enumerate(alignment)
                       if transform.kind is not AxisTransformKind.EXACT)
    result = value.algebra.gatype.oriented(
        tuple(axes[i] for i in selected), tuple(value.orientations[i] for i in selected),
    )
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
    the unknown need not separate into one multivector per unmatched axis.
    """
    permutation, (rows, columns), gatype, transforms = _grouped_lstsq_plan(value.gatype, rhs.gatype)
    context = binding_context(value.context, (rhs.context,))
    value, rhs = context.lower(value), context.lower(rhs)
    coefficients = rhs._kernel
    for axis, transform in transforms:
        coefficients = value.context.transform_axis(coefficients, rhs.arity + 1, axis, transform)
    matrix = value.context.xp.transpose(
        value._kernel, tuple(range(value.ndim)) + tuple(value.ndim + i for i in permutation),
    )
    matrix = matrix.reshape(value.shape + (rows, columns))
    solution = value.context.xp.linalg.pinv(matrix, rcond) @ coefficients.reshape(rhs.shape + (rows, 1))
    return _result(value, gatype, solution.reshape(solution.shape[:-2] + gatype.structural_shape))
