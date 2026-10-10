"""Linear systems over fields: a field map, or a bilinear form over fields, solved as one coupled
system over every pair of a site and a blade."""

from __future__ import annotations

from math import prod

import numpy as np

from numga.extensor import Extensor
from numga.gatype import GAType

from numga.extensions._linalg import _result


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
    rhs = rhs.cast(value.output_subspace)
    sites = dict(value.gatype.fields).get(0)
    if not rhs.gatype.fields and sites:
        # A value without sites is the same at every site.
        rhs = rhs[..., None].broadcast_to(rhs.shape + (sites,)).field()
    if dict(rhs.gatype.fields).get(0) != sites:
        raise TypeError(f"{rhs.gatype.signature} does not range over the sites of {value.gatype.signature}")
    return _solve(value, 0, 1, rhs, 0)


@Extensor.solve.register(_is_field_form_system, fields=True)
def solve_field_form(value: Extensor, rhs: Extensor) -> Extensor:
    """Solve F(x, y) == rhs(..., y) for all y, x in the form's first slot, coupled over the sites of
    both slots: F.solve(F(x)) == x. For a Hessian and a gradient, the Newton step up to sign."""
    return _solve(value, 2, 1, rhs, rhs.arity)


def _solve(value: Extensor, equations: int, unknowns: int, rhs: Extensor, given: int) -> Extensor:
    """Solve for the unknown slot of value, pairing its equation slot with the given slot of rhs.
    The solution has the unknown slot as its output and the other slots of rhs as inputs."""
    xp = value.context.xp
    batch = np.broadcast_shapes(value.shape, rhs.shape)
    value, rhs = value.broadcast_to(batch), rhs.broadcast_to(batch)
    rows = _slot_axes(value, equations)
    columns = _slot_axes(value, unknowns)
    # A form's scalar output is one blade long and folds into the matrix.
    rest = tuple(axis for axis in range(len(batch), value.kernel.ndim) if axis not in rows + columns)
    matrix = xp.transpose(value.kernel, _batch_axes(batch) + rows + columns + rest)
    size = prod(value.kernel.shape[axis] for axis in rows)
    if size != prod(value.kernel.shape[axis] for axis in columns):
        raise TypeError(f"{value.gatype.signature} is not square over its sites and blades")
    matrix = xp.reshape(matrix, batch + (size, size))

    # The other slots of rhs: its scalar output for a form, its open inputs either way.
    kept = [slot for slot in range(rhs.arity + 1) if slot != given]
    sites = [axis for slot in kept for axis in _slot_axes(rhs, slot)[:-1]]
    blades = [_slot_axes(rhs, slot)[-1] for slot in kept]
    given_axes = _slot_axes(rhs, given)
    columns_kernel = xp.transpose(rhs.kernel, _batch_axes(batch) + given_axes + tuple(sites) + tuple(blades))
    other_shape = tuple(rhs.kernel.shape[axis] for axis in sites + blades)
    solution = xp.linalg.solve(matrix, xp.reshape(columns_kernel, batch + (size, prod(other_shape))))

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


def _planned(method: str):
    """An operation on field maps that is intended and not implemented yet."""
    def implementation(value: Extensor, *arguments, **kwargs) -> Extensor:
        raise NotImplementedError(f"{method} of a field map ({value.gatype.signature}) is planned and not implemented yet")
    return implementation


# Operations a field map is meant to have, coupled over its sites as solve is. Fields of values run
# the operations over blades site by site instead.
for _method in ("inverse", "pinv", "cholesky", "det", "trace",
                "eig", "eigvals", "eigh", "eigvalsh", "svd", "svdvals"):
    getattr(Extensor, _method).register(lambda value: value.arity > 0, fields=True)(_planned(_method))
for _method in ("lstsq", "eig", "eigvals", "eigh", "eigvalsh"):
    getattr(Extensor, _method).register(lambda value, other: value.arity > 0, fields=True)(_planned(_method))
