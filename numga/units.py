"""Static per-blade and per-entry physical dimension tracking on `GAType`.

Every `Unit` is an immutable, interned product of rational powers of base dimensions (`dim.length`,
`dim.mass`, `dim.time`, `dim.extent`, `dim.none`, ...). Every `GAType` carries a sparse interned
`UnitGrid` matching its blade axes, with unchecked types represented as a state of `UnitGrid`
itself (`grid.is_unchecked`) and polymorphic structural zeros as `grid.is_zero`. Algebras carry no
physical units.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Mapping
from fractions import Fraction
from functools import lru_cache
from itertools import combinations
from math import isinf, prod
from types import NotImplementedType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from numga.extensor import Extensor
    from numga.gatype import GAType
    from numga.subspace import SubSpace


# --- Unit value type and dim namespace --------------------------------------------------

def _to_exponent(power: int | Fraction, name: str = "") -> Fraction:
    if isinstance(power, bool) or not isinstance(power, (int, Fraction)):
        raise TypeError(f"dimension exponent{f' for {name!r}' if name else ''} must be an int or Fraction; got {power!r}")
    return Fraction(power)


def _format_factor(name: str, power: Fraction) -> str:
    if power == 1:
        return name
    if power.denominator == 1:
        return f"{name}^{power.numerator}"
    return f"{name}^({power.numerator}/{power.denominator})"


def _scale_value(value: object, factor: Unit) -> Any:
    """Scaling an `Extensor` or `SparseExtensor` by a `Unit` updates only static units."""
    from numga.extensor import Extensor

    if isinstance(value, Extensor):
        if value.gatype.units.is_unchecked:
            return value
        return Extensor._from_prepared_kernel(
            value.context, value.gatype.derive.scaled_unit(factor), value._kernel,
        )

    from numga.sparse.extensor import SparseExtensor

    if isinstance(value, SparseExtensor):
        return SparseExtensor(
            _scale_value(value.cells, factor), value.rows, value.columns, value.shape,
        )
    return NotImplemented


class Unit:
    """An immutable, canonically ordered product of rational powers of physical dimensions."""

    __slots__ = ("_exponents", "_hash")
    _pool: dict[tuple[tuple[str, Fraction], ...], Unit] = {}

    def __new__(cls, exponents: Iterable[tuple[str, int | Fraction]] = ()) -> Unit:
        key = tuple(sorted((str(name), _to_exponent(power, str(name))) for name, power in exponents if power))
        cached = cls._pool.get(key)
        if cached is not None:
            return cached
        instance = super().__new__(cls)
        object.__setattr__(instance, "_exponents", key)
        object.__setattr__(instance, "_hash", hash(key))
        cls._pool[key] = instance
        return instance

    @classmethod
    def require(cls, value: object) -> Unit:
        if not isinstance(value, cls):
            raise TypeError(f"expected a Unit from `dim`; got {value!r}")
        return value

    @property
    def exponents(self) -> dict[str, Fraction]:
        return dict(self._exponents)

    @property
    def is_dimensionless(self) -> bool:
        return not self._exponents

    def inverse(self) -> Unit:
        return Unit((name, -power) for name, power in self._exponents)

    def root(self, degree: int) -> Unit:
        if isinstance(degree, bool) or not isinstance(degree, int) or degree <= 0:
            raise ValueError(f"root degree must be a positive integer; got {degree!r}")
        return Unit((name, power / degree) for name, power in self._exponents)

    def __mul__(self, other: object) -> Any:
        if isinstance(other, Unit):
            combined = dict(self._exponents)
            for name, power in other._exponents:
                total = combined.get(name, Fraction(0)) + power
                if total:
                    combined[name] = total
                else:
                    combined.pop(name, None)
            return Unit(combined.items())
        return _scale_value(other, self)

    def __rmul__(self, other: object) -> Any:
        if other == 1 and not isinstance(other, bool):
            return self
        return _scale_value(other, self)

    def __truediv__(self, other: object) -> Any:
        if isinstance(other, Unit):
            return self * other.inverse()
        return NotImplemented

    def __rtruediv__(self, other: object) -> Any:
        if other == 1 and not isinstance(other, bool):
            return self.inverse()
        return _scale_value(other, self.inverse())

    def __pow__(self, power: int | Fraction) -> Unit:
        exponent = _to_exponent(power)
        return Unit((name, p * exponent) for name, p in self._exponents)

    def __eq__(self, other: object) -> bool | NotImplementedType:
        if isinstance(other, Unit):
            return self._exponents == other._exponents
        return NotImplemented

    def __hash__(self) -> int:
        return self._hash

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("Unit is immutable")

    def __reduce__(self) -> tuple:
        return type(self), (self._exponents,)

    def __str__(self) -> str:
        if not self._exponents:
            return "1"
        numerator = [_format_factor(name, p) for name, p in self._exponents if p > 0]
        denominator = [_format_factor(name, -p) for name, p in self._exponents if p < 0]
        top = "*".join(numerator) if numerator else "1"
        if not denominator:
            return top
        bottom = "*".join(denominator)
        return f"{top}/({bottom})" if len(denominator) > 1 else f"{top}/{bottom}"

    def __repr__(self) -> str:
        return f"Unit({str(self)!r})"


_ONE = Unit()


class _Dimensions:
    """Attribute namespace producing base physical `Unit` dimensions (`dim.length`, `dim.none`, ...)."""

    __slots__ = ()

    @property
    def none(self) -> Unit:
        """Dimensionless unit (`1`)."""
        return _ONE

    def __getattr__(self, name: str) -> Unit:
        if name.startswith("_"):
            raise AttributeError(name)
        return Unit(((name, Fraction(1)),))


dim = _Dimensions()


# --- sparse interned per-entry UnitGrid -------------------------------------------------

def _strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    strides = [1] * len(shape)
    for axis in range(len(shape) - 2, -1, -1):
        strides[axis] = strides[axis + 1] * shape[axis + 1]
    return tuple(strides)


def _unravel(flat: int, strides: tuple[int, ...]) -> tuple[int, ...]:
    coords = []
    for stride in strides:
        q, flat = divmod(flat, stride)
        coords.append(q)
    return tuple(coords)


def _ravel(coords: tuple[int, ...], strides: tuple[int, ...]) -> int:
    return sum(c * s for c, s in zip(coords, strides))


class UnitGrid:
    """Sparse interned per-entry physical unit table matching a `GAType`'s blade axes.

    Unchecked types carry `is_unchecked=True`; polymorphic structural zeros carry `is_zero=True`;
    checked nonzero types store sorted `((flat_index, Unit), ...)` entries.
    """

    __slots__ = ("_shape", "_items", "_unchecked", "_uniform", "_hash", "_by_index")
    _pool: dict[tuple[tuple[int, ...], bool, tuple[tuple[int, Unit], ...]], UnitGrid] = {}

    def __new__(
        cls,
        shape: tuple[int, ...],
        items: tuple[tuple[int, Unit], ...] = (),
        *,
        unchecked: bool = False,
    ) -> UnitGrid:
        shape = tuple(int(n) for n in shape)
        items = () if unchecked else tuple(sorted(items))
        key = (shape, bool(unchecked), items)
        cached = cls._pool.get(key)
        if cached is not None:
            return cached
        instance = super().__new__(cls)
        object.__setattr__(instance, "_shape", shape)
        object.__setattr__(instance, "_items", items)
        object.__setattr__(instance, "_unchecked", bool(unchecked))
        if unchecked or not items:
            uniform: Unit | None = None
        else:
            first = items[0][1]
            uniform = first if all(u == first for _, u in items) else None
        object.__setattr__(instance, "_uniform", uniform)
        object.__setattr__(instance, "_hash", hash(key))
        object.__setattr__(instance, "_by_index", None)
        cls._pool[key] = instance
        return instance

    @classmethod
    def unchecked(cls, shape: tuple[int, ...]) -> UnitGrid:
        return cls(shape, (), unchecked=True)

    @classmethod
    def zero(cls, shape: tuple[int, ...]) -> UnitGrid:
        return cls(shape, ())

    @classmethod
    def uniform_grid(cls, shape: tuple[int, ...], unit: Unit) -> UnitGrid:
        unit = Unit.require(unit)
        total = prod(shape)
        if total == 0:
            return cls.zero(shape)
        return cls(shape, tuple((i, unit) for i in range(total)))

    @classmethod
    def from_entries(cls, shape: tuple[int, ...], entries: Mapping[int, Unit]) -> UnitGrid:
        if not entries:
            return cls.zero(shape)
        return cls(shape, tuple(sorted(entries.items())))

    @classmethod
    def from_nonzero_mask(cls, array: Any) -> UnitGrid:
        """Build a dimensionless grid over the nonzero entries of an exact operator kernel."""
        import numpy as np

        arr = np.asarray(array)
        flat = arr.reshape(-1)
        items = tuple((int(i), _ONE) for i in np.flatnonzero(flat))
        return cls(arr.shape, items)

    @classmethod
    def from_sector_map(cls, subspace: SubSpace, sector_units: Mapping[Any, Unit]) -> UnitGrid:
        """Build a 1D `UnitGrid` on `subspace` by assigning units to disjoint sector keys
        (`GAType`, `SubSpace`, or blade name `str`)."""
        from numga.gatype import GAType
        from numga.gatype.errors import GATypeError
        from numga.subspace import SubSpace as _SubSpace

        index_by_mask = {mask: i for i, mask in enumerate(subspace.masks)}
        entries: dict[int, Unit] = {}
        for key, raw_unit in sector_units.items():
            unit = Unit.require(raw_unit)
            if isinstance(key, GAType):
                if key.arity:
                    raise GATypeError(f"sector unit key must be nullary; got {key}")
                masks = key.output_subspace.masks
            elif isinstance(key, _SubSpace):
                masks = key.masks
            elif isinstance(key, str):
                masks = subspace.algebra.subspace(key).masks
            else:
                raise TypeError(f"sector unit key must be a GAType, SubSpace, or blade name; got {key!r}")
            for mask in masks:
                idx = index_by_mask.get(mask)
                if idx is None:
                    raise GATypeError(f"sector blade {subspace.algebra.blade_name(mask)!r} is not in {subspace}")
                prev = entries.get(idx)
                if prev is not None and prev != unit:
                    raise GATypeError(
                        f"sector map assigns conflicting units {prev} and {unit} to blade "
                        f"{subspace.algebra.blade_name(mask)!r}"
                    )
                entries[idx] = unit
        return cls.from_entries((len(subspace),), entries)

    @classmethod
    def outer_quotient(cls, output: UnitGrid, inputs: tuple[UnitGrid, ...]) -> UnitGrid:
        """Build a map grid `entry[o..., i1..., ...] = output[o...] / (inputs[0][i1...] * ...)`."""
        shape = output._shape + tuple(n for g in inputs for n in g._shape)
        if output._unchecked or any(g._unchecked for g in inputs):
            return cls.unchecked(shape)
        if output.is_zero or any(g.is_zero for g in inputs):
            return cls.zero(shape)
        out_strides = _strides(output._shape)
        in_strides = tuple(_strides(g._shape) for g in inputs)
        strides = _strides(shape)
        entries: dict[int, Unit] = {}

        def visit(axis: int, coords: tuple[int, ...], acc: Unit) -> None:
            if axis == len(inputs):
                entries[_ravel(coords, strides)] = acc
                return
            g_strides = in_strides[axis]
            for flat, u in inputs[axis]._items:
                visit(axis + 1, coords + _unravel(flat, g_strides), acc / u)

        for o_flat, o_u in output._items:
            visit(0, _unravel(o_flat, out_strides), o_u)
        return cls.from_entries(shape, entries)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def items(self) -> tuple[tuple[int, Unit], ...]:
        return self._items

    @property
    def is_unchecked(self) -> bool:
        return self._unchecked

    @property
    def is_zero(self) -> bool:
        return not self._unchecked and not self._items

    @property
    def uniform(self) -> Unit | None:
        return self._uniform

    def as_dict(self) -> dict[int, Unit]:
        mapping = self._by_index
        if mapping is None:
            mapping = dict(self._items)
            object.__setattr__(self, "_by_index", mapping)
        return mapping

    def get(self, flat_index: int) -> Unit | None:
        return self.as_dict().get(flat_index)

    def scale(self, unit: Unit) -> UnitGrid:
        unit = Unit.require(unit)
        if self._unchecked or self.is_zero or unit is _ONE:
            return self
        return UnitGrid(self._shape, tuple((idx, u * unit) for idx, u in self._items))

    def reverse_axes(self) -> UnitGrid:
        new_shape = self._shape[::-1]
        if self._unchecked:
            return UnitGrid.unchecked(new_shape)
        if len(self._shape) <= 1 or not self._items:
            return UnitGrid(new_shape, self._items)
        old_s, new_s = _strides(self._shape), _strides(new_shape)
        return UnitGrid.from_entries(new_shape, {
            _ravel(_unravel(flat, old_s)[::-1], new_s): u for flat, u in self._items
        })

    def remap_axis(self, axis: int, new_size: int, old_to_new: Mapping[int, int]) -> UnitGrid:
        new_shape = self._shape[:axis] + (new_size,) + self._shape[axis + 1:]
        if self._unchecked:
            return UnitGrid.unchecked(new_shape)
        if not self._items:
            return UnitGrid.zero(new_shape)
        old_s, new_s = _strides(self._shape), _strides(new_shape)
        entries: dict[int, Unit] = {}
        for flat, u in self._items:
            coords = _unravel(flat, old_s)
            mapped = old_to_new.get(coords[axis])
            if mapped is not None:
                entries[_ravel(coords[:axis] + (mapped,) + coords[axis + 1:], new_s)] = u
        return UnitGrid.from_entries(new_shape, entries)

    def take_axis(self, axis: int, indices: tuple[int, ...]) -> UnitGrid:
        new_shape = self._shape[:axis] + (len(indices),) + self._shape[axis + 1:]
        if self._unchecked:
            return UnitGrid.unchecked(new_shape)
        if not self._items:
            return UnitGrid.zero(new_shape)
        old_to_new: dict[int, list[int]] = {}
        for new_i, old_i in enumerate(indices):
            old_to_new.setdefault(int(old_i), []).append(new_i)
        old_s, new_s = _strides(self._shape), _strides(new_shape)
        entries: dict[int, Unit] = {}
        for flat, u in self._items:
            coords = _unravel(flat, old_s)
            for new_i in old_to_new.get(coords[axis], ()):
                entries[_ravel(coords[:axis] + (new_i,) + coords[axis + 1:], new_s)] = u
        return UnitGrid.from_entries(new_shape, entries)

    def format_for(self, subspaces: tuple[SubSpace, ...]) -> str:
        """Compact human-readable unit suffix for `GAType.signature`."""
        if self._unchecked:
            return ""
        if self.is_zero:
            return "0"
        if self._uniform is _ONE:
            return ""
        if self._uniform is not None:
            return str(self._uniform)
        distinct = tuple(dict.fromkeys(u for _, u in self._items))
        return " | ".join(str(u) for u in distinct)

    def __eq__(self, other: object) -> bool | NotImplementedType:
        if isinstance(other, UnitGrid):
            return self._shape == other._shape and self._unchecked == other._unchecked and self._items == other._items
        return NotImplemented

    def __hash__(self) -> int:
        return self._hash

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("UnitGrid is immutable")

    def __reduce__(self) -> tuple:
        if self._unchecked:
            return UnitGrid.unchecked, (self._shape,)
        return type(self), (self._shape, self._items)

    def __repr__(self) -> str:
        if self._unchecked:
            return f"UnitGrid.unchecked({self._shape})"
        if self.is_zero:
            return f"UnitGrid.zero({self._shape})"
        if self._uniform is not None and len(self._items) == prod(self._shape):
            return f"UnitGrid({self._shape}, {self._uniform})"
        return f"UnitGrid({self._shape}, {dict(self._items)!r})"


def coerce_unit_grid(spec: object, subspaces: tuple[SubSpace, ...]) -> UnitGrid:
    """Convert a public `unit=` argument (`None`, `Unit`, `UnitGrid`, or sector `Mapping`) into a
    shape-matched `UnitGrid`."""
    from numga.gatype.errors import GATypeError

    shape = tuple(len(space) for space in subspaces)
    if spec is None:
        return UnitGrid.unchecked(shape)
    if isinstance(spec, Unit):
        return UnitGrid.uniform_grid(shape, spec)
    if isinstance(spec, UnitGrid):
        if spec.shape != shape:
            raise GATypeError(f"UnitGrid shape {spec.shape} does not match type shape {shape}")
        return spec
    if isinstance(spec, Mapping):
        if len(subspaces) != 1:
            raise GATypeError("sector unit mappings apply to nullary types; build map types from typed axes")
        return UnitGrid.from_sector_map(subspaces[0], spec)
    raise TypeError(f"expected a Unit, UnitGrid, sector mapping, or None; got {spec!r}")


def _blade_label(subspaces: tuple[SubSpace, ...], coords: tuple[int, ...]) -> str:
    parts = [space.algebra.blade_name(space.masks[idx]) for space, idx in zip(subspaces, coords)]
    return parts[0] if len(parts) == 1 else f"({', '.join(parts)})"


# --- core unit combination laws (merge, embed, contract) --------------------------------

def merge_grids(grids: tuple[UnitGrid, ...], subspaces: tuple[SubSpace, ...], operation: str = "combine") -> UnitGrid:
    """Merge same-shape `UnitGrid`s, requiring identical `Unit` on every shared nonzero entry."""
    from numga.gatype.errors import GATypeError

    shape = tuple(len(space) for space in subspaces)
    if any(grid.is_unchecked for grid in grids):
        return UnitGrid.unchecked(shape)
    merged: dict[int, Unit] = {}
    strides = _strides(shape)
    for grid in grids:
        for idx, unit in grid.items:
            prev = merged.get(idx)
            if prev is not None and prev != unit:
                raise GATypeError(
                    f"cannot {operation} on blade {_blade_label(subspaces, _unravel(idx, strides))}: "
                    f"conflicting units {prev} and {unit}"
                )
            merged[idx] = unit
    return UnitGrid.from_entries(shape, merged)


def remap_grid_subspaces(
    grid: UnitGrid, old_subspaces: tuple[SubSpace, ...], new_subspaces: tuple[SubSpace, ...],
) -> UnitGrid:
    """Embed or project `grid` from `old_subspaces` onto `new_subspaces` along every differing axis."""
    result = grid
    for axis, (old_space, new_space) in enumerate(zip(old_subspaces, new_subspaces)):
        if old_space is new_space:
            continue
        new_index = {mask: i for i, mask in enumerate(new_space.masks)}
        mapping = {old_i: new_index[mask] for old_i, mask in enumerate(old_space.masks) if mask in new_index}
        result = result.remap_axis(axis, len(new_space), mapping)
    return result


def embed_and_merge(operands: tuple[GAType, ...], axes: tuple[SubSpace, ...], operation: str = "combine") -> UnitGrid:
    """Embed each operand's `UnitGrid` onto `axes` and merge; unchecked grids absorb."""
    shape = tuple(len(space) for space in axes)
    if any(operand.units.is_unchecked for operand in operands):
        return UnitGrid.unchecked(shape)
    return merge_grids(
        tuple(remap_grid_subspaces(op.units, op.subspaces, axes) for op in operands),
        axes,
        operation,
    )


def contract_units(
    target_gatype: GAType,
    bindings: tuple,
    result_subspaces: tuple[SubSpace, ...],
    output_indices: tuple[int, ...],
) -> UnitGrid:
    """Contract per-entry `UnitGrid`s across a `BindingPlan`, verifying homogeneity on every output entry."""
    from numga.binding import AxisTransformKind
    from numga.gatype.errors import GATypeError

    out_shape = tuple(len(space) for space in result_subspaces)
    if target_gatype.units.is_unchecked or any(b.operand_gatype.units.is_unchecked for b in bindings):
        return UnitGrid.unchecked(out_shape)
    if target_gatype.units.is_zero or any(b.operand_gatype.units.is_zero for b in bindings):
        return UnitGrid.zero(out_shape)

    bound_by_axis: dict[int, dict[int, list[tuple[tuple[int, ...], Unit]]]] = {}
    for binding in bindings:
        op_grid = binding.operand_gatype.units
        transform = binding.transform
        if transform.kind is not AxisTransformKind.EXACT:
            mapping = {
                src: tgt
                for tgt, src in enumerate(transform.target_from_source)
                if src is not None
            }
            op_grid = op_grid.remap_axis(0, len(transform.target), mapping)
        op_strides = _strides(op_grid.shape)
        grouped: dict[int, list[tuple[tuple[int, ...], Unit]]] = {}
        for flat, u in op_grid.items:
            coords = _unravel(flat, op_strides)
            grouped.setdefault(coords[0], []).append((coords[1:], u))
        bound_by_axis[binding.axis] = grouped

    out_filter: dict[int, int] | None = None
    if result_subspaces[0] is not target_gatype.output_subspace:
        out_filter = {int(old_i): new_i for new_i, old_i in enumerate(output_indices)}

    target_strides = _strides(target_gatype.units.shape)
    out_strides = _strides(out_shape)
    result_entries: dict[int, Unit] = {}
    arity = target_gatype.arity

    for flat, target_u in target_gatype.units.items:
        t_coords = _unravel(flat, target_strides)
        out0 = t_coords[0]
        if out_filter is not None:
            mapped_out0 = out_filter.get(out0)
            if mapped_out0 is None:
                continue
            out0 = mapped_out0

        partials: list[tuple[tuple[int, ...], Unit]] = [((out0,), target_u)]
        for axis in range(arity):
            k = t_coords[axis + 1]
            grouped = bound_by_axis.get(axis)
            if grouped is None:
                partials = [(c + (k,), u) for c, u in partials]
            else:
                matches = grouped.get(k)
                if not matches:
                    partials = []
                    break
                partials = [
                    (c + op_tail, u * op_u)
                    for c, u in partials
                    for op_tail, op_u in matches
                ]
        for out_coords, out_u in partials:
            out_flat = _ravel(out_coords, out_strides)
            prev = result_entries.get(out_flat)
            if prev is not None and prev != out_u:
                raise GATypeError(
                    f"incompatible physical units {prev} and {out_u} on "
                    f"{_blade_label(result_subspaces, out_coords)}"
                )
            result_entries[out_flat] = out_u

    return UnitGrid.from_entries(out_shape, result_entries)


def divide_by_input(output: UnitGrid, argument: UnitGrid) -> UnitGrid:
    """Per-entry unit grid for differentiation `d(output) / d(argument)`."""
    if not output.is_unchecked and not argument.is_unchecked and len(argument.items) < prod(argument.shape):
        from numga.gatype.errors import GATypeError

        raise GATypeError(
            "differentiation requires concrete units on every blade of the input value; "
            f"got {argument}"
        )
    return UnitGrid.outer_quotient(output, (argument,))


# --- 2D rank-1 factorization for linear algebra -----------------------------------------

def _require_grid(gatype: GAType, method: str) -> UnitGrid:
    if gatype.units.is_unchecked:
        raise ValueError(f"{method!r} unit law called on unchecked type {gatype}")
    return gatype.units


def _require_uniform(gatype: GAType, method: str) -> Unit:
    from numga.gatype.errors import GATypeError

    grid = _require_grid(gatype, method)
    if grid.is_zero:
        return _ONE
    if grid.uniform is not None:
        return grid.uniform
    raise GATypeError(f"{method!r} requires uniform blade units; got {gatype.signature}")


def _dual_permutation(source: SubSpace, target: SubSpace) -> dict[int, int]:
    algebra = source.algebra
    target_idx = {mask: i for i, mask in enumerate(target.masks)}
    return {
        i: target_idx[algebra.complement(mask)]
        for i, mask in enumerate(source.masks)
        if algebra.complement(mask) in target_idx
    }


def _grid_2d(gatype: GAType, method: str) -> tuple[tuple[int, int], dict[tuple[int, int], Unit]]:
    from numga.gatype.errors import GATypeError

    grid = _require_grid(gatype, method)
    if gatype.arity == 1:
        strides = _strides(grid.shape)
        return grid.shape, {_unravel(flat, strides): u for flat, u in grid.items}
    if gatype.is_form:
        strides = _strides(grid.shape)
        entries: dict[tuple[int, int], Unit] = {}
        for flat, u in grid.items:
            _, i, j = _unravel(flat, strides)
            entries[(i, j)] = u
        return (grid.shape[1], grid.shape[2]), entries
    raise GATypeError(f"{method!r} expects a unary map or bilinear form; got {gatype.signature}")


def _align_rows(row_space: SubSpace, col_space: SubSpace, entries: Mapping[tuple[int, int], Unit]) -> dict[tuple[int, int], Unit]:
    if row_space is col_space:
        return dict(entries)
    col_idx = {mask: i for i, mask in enumerate(col_space.masks)}
    row_map = {i: col_idx[mask] for i, mask in enumerate(row_space.masks) if mask in col_idx}
    return {(row_map[i], j): u for (i, j), u in entries.items() if i in row_map}


def rank1_factor(
    shape: tuple[int, int], entries: Mapping[tuple[int, int], Unit], operation: str = "linear algebra",
) -> list[tuple[dict[int, Unit], dict[int, Unit]]]:
    """Factor nonzero 2D entries per connected component as `entries[i, j] == row[i] / col[j]`."""
    from numga.gatype.errors import GATypeError

    rows_adj: dict[int, list[tuple[int, Unit]]] = {}
    cols_adj: dict[int, list[tuple[int, Unit]]] = {}
    for (i, j), u in entries.items():
        rows_adj.setdefault(i, []).append((j, u))
        cols_adj.setdefault(j, []).append((i, u))

    visited_r: set[int] = set()
    visited_c: set[int] = set()
    components: list[tuple[dict[int, Unit], dict[int, Unit]]] = []

    for r0 in sorted(rows_adj):
        if r0 in visited_r:
            continue
        row_u: dict[int, Unit] = {r0: _ONE}
        col_u: dict[int, Unit] = {}
        queue: deque[tuple[str, int]] = deque([("r", r0)])
        visited_r.add(r0)
        while queue:
            kind, idx = queue.popleft()
            if kind == "r":
                ru = row_u[idx]
                for j, u in rows_adj[idx]:
                    expected = ru / u
                    prev = col_u.get(j)
                    if prev is None:
                        col_u[j] = expected
                        visited_c.add(j)
                        queue.append(("c", j))
                    elif prev != expected:
                        raise GATypeError(
                            f"{operation} matrix is dimensionally inhomogeneous: "
                            f"column {j} requires both {prev} and {expected}"
                        )
            else:
                cu = col_u[idx]
                for i, u in cols_adj[idx]:
                    expected = u * cu
                    prev = row_u.get(i)
                    if prev is None:
                        row_u[i] = expected
                        visited_r.add(i)
                        queue.append(("r", i))
                    elif prev != expected:
                        raise GATypeError(
                            f"{operation} matrix is dimensionally inhomogeneous: "
                            f"row {i} requires both {prev} and {expected}"
                        )
        components.append((row_u, col_u))
    return components


def invert_map_units(gatype: GAType) -> UnitGrid:
    """Invert a square unary map's `UnitGrid`: `inv[j, i] = col[j] / row[i]` per component."""
    if gatype.units.is_unchecked:
        return UnitGrid.unchecked(gatype.units.shape[::-1])
    shape, entries = _grid_2d(gatype, "inverse")
    inv_shape = (shape[1], shape[0])
    if not entries:
        return UnitGrid.zero(inv_shape)
    strides = _strides(inv_shape)
    inv_entries: dict[int, Unit] = {}
    for row_u, col_u in rank1_factor(shape, entries, "inverse"):
        for j, cu in col_u.items():
            for i, ru in row_u.items():
                inv_entries[_ravel((j, i), strides)] = cu / ru
    return UnitGrid.from_entries(inv_shape, inv_entries)


# --- algebraic unit closure for exp, log, square_root, inverse --------------------------

def _binary_nullary(
    op: Extensor, left: UnitGrid, right: UnitGrid, op_name: str,
) -> tuple[SubSpace, UnitGrid]:
    """Contract two nullary `UnitGrid`s through an exact binary operator `op`."""
    from numga.gatype.errors import GATypeError

    out_space = op.output_subspace
    if left.is_zero or right.is_zero:
        return out_space, UnitGrid.zero((len(out_space),))
    strides = _strides(op.gatype.units.shape)
    ld, rd = left.as_dict(), right.as_dict()
    out_entries: dict[int, Unit] = {}
    for flat, _ in op.gatype.units.items:
        o, i, j = _unravel(flat, strides)
        lu = ld.get(i)
        if lu is None:
            continue
        ru = rd.get(j)
        if ru is None:
            continue
        pu = lu * ru
        prev = out_entries.get(o)
        if prev is not None and prev != pu:
            name = out_space.algebra.blade_name(out_space.masks[o])
            raise GATypeError(f"cannot {op_name} on blade {name}: conflicting units {prev} and {pu}")
        out_entries[o] = pu
    return out_space, UnitGrid.from_entries((len(out_space),), out_entries)


def _multiply_nullary(
    left_space: SubSpace, left: UnitGrid, right_space: SubSpace, right: UnitGrid, op_name: str,
) -> tuple[SubSpace, UnitGrid]:
    from numga.gatype import Orientation

    algebra = left_space.algebra
    op = algebra.operator.geometric_product(
        algebra.gatype(left_space, orientation=Orientation.extrinsic),
        algebra.gatype(right_space, orientation=Orientation.extrinsic),
    )
    return _binary_nullary(op, left, right, op_name)


def _scalar_index(space: SubSpace) -> int | None:
    for i, mask in enumerate(space.masks):
        if mask == 0:
            return i
    return None


@lru_cache(maxsize=None)
def _exp_closure(gatype: GAType) -> tuple[SubSpace, UnitGrid]:
    """Fixed-point blade-unit closure of `1 + x + x^2/2 + ...`."""
    from numga.gatype.errors import GATypeError

    algebra = gatype.algebra
    x_space = gatype.output_subspace
    x_grid = _require_grid(gatype, "exp")
    scalar_space = algebra.subspace.scalar()
    acc_space = scalar_space.union(x_space)
    acc_grid = merge_grids(
        (
            remap_grid_subspaces(UnitGrid.uniform_grid((1,), _ONE), (scalar_space,), (acc_space,)),
            remap_grid_subspaces(x_grid, (x_space,), (acc_space,)),
        ),
        (acc_space,),
        "exp",
    )
    for _ in range(1 << algebra.dimension):
        prod_space, prod_grid = _multiply_nullary(acc_space, acc_grid, x_space, x_grid, "exp")
        next_space = acc_space.union(prod_space)
        next_grid = merge_grids(
            (
                remap_grid_subspaces(acc_grid, (acc_space,), (next_space,)),
                remap_grid_subspaces(prod_grid, (prod_space,), (next_space,)),
            ),
            (next_space,),
            "exp",
        )
        if next_space is acc_space and next_grid is acc_grid:
            return acc_space, acc_grid
        acc_space, acc_grid = next_space, next_grid
    raise GATypeError(f"exp power series units did not stabilize for {gatype.signature}")


def validate_reverse_product_one(gatype: GAType) -> None:
    """Certifying `ReverseProductOne` (`R * ~R == 1`) requires a dimensionless scalar part of `R * ~R`."""
    from numga.gatype.errors import GATypeError

    if gatype.arity or gatype.units.is_unchecked:
        return
    space = gatype.output_subspace
    deg = gatype.algebra.degenerate_mask
    for idx, u in gatype.units.items:
        if not (space.masks[idx] & deg) and u is not _ONE:
            name = space.algebra.blade_name(space.masks[idx])
            raise GATypeError(
                f"unit-norm trait requires dimensionless non-null blade {name!r}; got {u} in {gatype.signature}"
            )


# --- deferred subspace-matched stamp for dispatch ---------------------------------------

class _OnSubspaces:
    """UnitGrid paired with its reference subspaces, remapped onto the implementation's output axes."""

    __slots__ = ("subspaces", "grid")

    def __init__(self, subspaces: tuple[SubSpace, ...], grid: UnitGrid) -> None:
        self.subspaces = subspaces
        self.grid = grid

    def for_gatype(self, gatype: GAType) -> UnitGrid:
        return remap_grid_subspaces(self.grid, self.subspaces, gatype.subspaces)


def resolve_stamp(spec: object, gatype: GAType) -> UnitGrid:
    """Resolve a unit-law output (`UnitGrid`, `Unit`, or `_OnSubspaces`) onto `gatype`."""
    if isinstance(spec, _OnSubspaces):
        return spec.for_gatype(gatype)
    blade_shape = tuple(len(space) for space in gatype.subspaces)
    if isinstance(spec, Unit):
        return UnitGrid.uniform_grid(blade_shape, spec)
    if isinstance(spec, UnitGrid):
        if spec.is_unchecked:
            return UnitGrid.unchecked(blade_shape)
        if spec.shape == blade_shape:
            return spec
        if spec.is_zero:
            return UnitGrid.zero(blade_shape)
        if spec.uniform is not None:
            return UnitGrid.uniform_grid(blade_shape, spec.uniform)
        from numga.gatype.errors import GATypeError
        raise GATypeError(f"unit grid shape {spec.shape} does not match result {gatype.signature}")
    raise TypeError(f"unsupported unit law result: {spec!r}")


# --- extension method unit laws (attached at registration) ------------------------------

def with_arguments(law: Callable) -> Callable:
    """Mark a unit law to receive raw extra non-operand arguments and kwargs from dispatch."""
    law.uses_extra_args = True
    return law


def keep(operand: GAType) -> UnitGrid:
    return _require_grid(operand, "keep")


def dimensionless(*operands: GAType) -> UnitGrid:
    from numga.gatype.errors import GATypeError

    for operand in operands:
        grid = _require_grid(operand, "transcendental")
        if not grid.is_zero and grid.uniform is not _ONE:
            raise GATypeError(f"requires dimensionless input; got {operand.signature}")
    return UnitGrid.uniform_grid(tuple(len(space) for space in operands[0].subspaces), _ONE)


def exp_unit(operand: GAType) -> _OnSubspaces:
    space, grid = _exp_closure(operand)
    return _OnSubspaces((space,), grid)


def log_unit(operand: GAType) -> _OnSubspaces:
    """`log(R)` requires a dimensionless scalar blade and closure under squaring `R * R`."""
    from numga.gatype.errors import GATypeError

    grid = _require_grid(operand, "log")
    space = operand.output_subspace
    if operand.is_scalar:
        if not grid.is_zero and grid.uniform is not _ONE:
            raise GATypeError(f"'log' requires dimensionless scalar input; got {operand.signature}")
        return _OnSubspaces((space,), UnitGrid.uniform_grid((1,), _ONE))
    s_idx = _scalar_index(space)
    s_unit = grid.get(s_idx) if s_idx is not None else None
    if s_unit is not None and s_unit is not _ONE:
        raise GATypeError(f"'log' requires dimensionless scalar part; got {s_unit} in {operand.signature}")
    prod_space, prod_grid = _multiply_nullary(space, grid, space, grid, "log")
    union_space = space.union(prod_space)
    merged = merge_grids(
        (
            remap_grid_subspaces(grid, (space,), (union_space,)),
            remap_grid_subspaces(prod_grid, (prod_space,), (union_space,)),
        ),
        (union_space,),
        "log",
    )
    return _OnSubspaces((union_space,), merged)


def _scalar_norm_unit(operand: GAType, method: str) -> Unit:
    from numga.gatype.errors import GATypeError

    grid = _require_grid(operand, method)
    space = operand.output_subspace
    deg = operand.algebra.degenerate_mask
    found: Unit | None = None
    for idx, u in grid.items:
        if space.masks[idx] & deg:
            continue
        if found is None:
            found = u
        elif found != u:
            raise GATypeError(
                f"{method!r} mixes non-null blades with conflicting units {found} and {u} in {operand.signature}"
            )
    if found is not None:
        return found
    if grid.uniform is not None:
        return grid.uniform
    if grid.is_zero:
        return _ONE
    raise GATypeError(f"{method!r} has no single norm unit on {operand.signature}")


def normalized_unit(operand: GAType) -> UnitGrid:
    grid = _require_grid(operand, "normalized")
    return grid.scale(_scalar_norm_unit(operand, "normalized").inverse())


def _root_unit(operand: GAType, *, inverse: bool) -> UnitGrid | _OnSubspaces:
    from numga.gatype.errors import GATypeError

    method = "inverse_square_root" if inverse else "square_root"
    grid = _require_grid(operand, method)
    if operand.is_scalar:
        if grid.is_zero:
            return grid
        u = grid.uniform.root(2)
        return UnitGrid.uniform_grid((1,), u.inverse() if inverse else u)
    space = operand.output_subspace
    s_idx = _scalar_index(space)
    s_unit = grid.get(s_idx) if s_idx is not None else None
    if s_unit is None:
        raise GATypeError(f"{method!r} requires a nonzero scalar blade unit in {operand.signature}")
    norm_grid = grid.scale(s_unit.inverse())
    prod_space, prod_grid = _multiply_nullary(space, norm_grid, space, norm_grid, method)
    union_space = space.union(prod_space)
    merged = merge_grids(
        (
            remap_grid_subspaces(norm_grid, (space,), (union_space,)),
            remap_grid_subspaces(prod_grid, (prod_space,), (union_space,)),
        ),
        (union_space,),
        method,
    )
    scale = s_unit.root(2).inverse() if inverse else s_unit.root(2)
    return _OnSubspaces((union_space,), merged.scale(scale))


def root_unit(operand: GAType) -> UnitGrid | _OnSubspaces:
    return _root_unit(operand, inverse=False)


def inverse_root_unit(operand: GAType) -> UnitGrid | _OnSubspaces:
    return _root_unit(operand, inverse=True)


def invert_unit_law(operand: GAType) -> UnitGrid | _OnSubspaces:
    """Invert a unary linear map via rank-1 factorization or a multivector via left-multiply."""
    if operand.arity == 1:
        return invert_map_units(operand)
    grid = _require_grid(operand, "inverse")
    if grid.is_zero:
        return grid
    if grid.uniform is not None:
        return grid.scale(grid.uniform.inverse() ** 2)
    from numga.binding import BindingPlan
    from numga.gatype import Orientation

    algebra = operand.algebra
    sub_type = operand.derive.minimal_subalgebra
    sub_space = sub_type.output_subspace
    prod_op = algebra.operator.geometric_product(
        algebra.gatype(operand.output_subspace, orientation=Orientation.extrinsic), sub_type,
    )
    plan = BindingPlan.from_types(prod_op.gatype, ((0, operand),))
    aligned = remap_grid_subspaces(plan.result_unit, plan.result_subspaces, (sub_space, sub_space))
    strides = _strides(aligned.shape)
    mat_entries = {_unravel(flat, strides): u for flat, u in aligned.items}
    s_idx = _scalar_index(sub_space)
    sol_entries: dict[int, Unit] = {}
    for row_u, col_u in rank1_factor(aligned.shape, mat_entries, "inverse"):
        if s_idx in row_u:
            scale = _ONE / row_u[s_idx]
            for j, cu in col_u.items():
                sol_entries[j] = scale * cu
    return _OnSubspaces((sub_space,), UnitGrid.from_entries((len(sub_space),), sol_entries))


def solve_unit(operator: GAType, rhs: GAType, *_extra: GAType) -> UnitGrid:
    """Solve `operator(x) == rhs` for `x`'s per-entry `UnitGrid` using rank-1 factorization."""
    from numga.gatype.errors import GATypeError

    rhs_grid = _require_grid(rhs, "solve")
    if operator.arity == 1:
        shape, entries = _grid_2d(operator, "solve")
        aligned = remap_grid_subspaces(rhs_grid, rhs.subspaces, (operator.subspaces[0],) + rhs.input_subspaces)
        strides = _strides(aligned.shape)
        tail_shape = aligned.shape[1:]
        rhs_slices: dict[tuple[int, ...], dict[int, Unit]] = {}
        for flat, u in aligned.items:
            coords = _unravel(flat, strides)
            rhs_slices.setdefault(coords[1:], {})[coords[0]] = u
    elif operator.is_form:
        f_shape, f_entries = _grid_2d(operator, "solve")
        shape = (f_shape[1], f_shape[0])
        entries = {(y, x): u for (x, y), u in f_entries.items()}
        aligned = remap_grid_subspaces(rhs_grid, rhs.subspaces, rhs.subspaces[:-1] + (operator.subspaces[2],))
        strides = _strides(aligned.shape)
        tail_shape = aligned.shape[1:-1]
        rhs_slices = {}
        for flat, u in aligned.items:
            coords = _unravel(flat, strides)
            rhs_slices.setdefault(coords[1:-1], {})[coords[-1]] = u
    else:
        raise GATypeError(f"'solve' expects a unary map or bilinear form; got {operator.signature}")

    components = rank1_factor(shape, entries, "solve")
    sol_shape = (shape[1],) + tail_shape
    sol_strides = _strides(sol_shape)
    sol_entries: dict[int, Unit] = {}
    for tail, rhs_dict in rhs_slices.items():
        for row_u, col_u in components:
            scales = {rhs_dict[i] / ru for i, ru in row_u.items() if i in rhs_dict}
            if not scales:
                continue
            if len(scales) > 1:
                raise GATypeError(
                    f"right-hand side {rhs.signature} is dimensionally inconsistent with {operator.signature}"
                )
            scale = next(iter(scales))
            for j, cu in col_u.items():
                sol_entries[_ravel((j,) + tail, sol_strides)] = scale * cu
    return UnitGrid.from_entries(sol_shape, sol_entries)


def lstsq_unit(operator: GAType, rhs: GAType) -> Unit:
    return _require_uniform(rhs, "lstsq") / _require_uniform(operator, "lstsq")


def lstsq_weighted_unit(operator: GAType, rhs: GAType, metric: GAType) -> UnitGrid:
    _pencil_ratio_and_modes(operator, metric, "lstsq")
    return solve_unit(operator, rhs)


def pinv_unit(operator: GAType) -> UnitGrid:
    op_u = _require_uniform(operator, "pinv")
    return UnitGrid.uniform_grid((operator.units.shape[1], operator.units.shape[0]), op_u.inverse())


def adjugate_unit(operator: GAType) -> UnitGrid:
    grid = _require_grid(operator, "adjugate")
    out_space, in_space = operator.subspaces[0], operator.subspaces[1]
    adj_out = in_space.algebra.subspace.from_masks(in_space.algebra.complement(m) for m in in_space.masks)
    adj_in = out_space.algebra.subspace.from_masks(out_space.algebra.complement(m) for m in out_space.masks)
    j_map, i_map = _dual_permutation(in_space, adj_out), _dual_permutation(out_space, adj_in)
    old_s, new_s = _strides(grid.shape), _strides((len(adj_out), len(adj_in)))
    return UnitGrid.from_entries((len(adj_out), len(adj_in)), {
        _ravel((j_map[j], i_map[i]), new_s): u
        for flat, u in grid.items
        for i, j in (_unravel(flat, old_s),)
    })


def _component_det(operator: GAType, method: str) -> Unit | None:
    shape, entries = _grid_2d(operator, method)
    if operator.arity == 1:
        entries = _align_rows(operator.subspaces[0], operator.subspaces[1], entries)
    elif operator.is_form:
        entries = _align_rows(operator.subspaces[1], operator.subspaces[2], entries)
    det_u = _ONE
    covered = 0
    for row_u, col_u in rank1_factor(shape, entries, method):
        if len(row_u) != len(col_u):
            return None
        covered += len(row_u)
        for ru in row_u.values():
            det_u = det_u * ru
        for cu in col_u.values():
            det_u = det_u / cu
    return det_u if covered == shape[0] else None


def det_unit(operator: GAType, partner: GAType | None = None) -> UnitGrid:
    from numga.gatype.errors import GATypeError

    op_det = _component_det(operator, "det")
    if op_det is None:
        return UnitGrid.zero((1,))
    if partner is not None:
        p_det = _component_det(partner, "det")
        if p_det is None:
            raise GATypeError(f"'det' partner {partner.signature} is singular")
        op_det = op_det / p_det
    axis = 1 if operator.arity == 1 else 2
    sites = prod(size for field_axis, size in operator.fields if field_axis == axis)
    return UnitGrid.uniform_grid((1,), op_det ** sites)


def _symmetric_scales(
    components: list[tuple[dict[int, Unit], dict[int, Unit]]], signature: str, method: str,
) -> dict[int, Unit]:
    from numga.gatype.errors import GATypeError

    w_units: dict[int, Unit] = {}
    for row_u, col_u in components:
        products = {row_u[i] * col_u[i] for i in row_u if i in col_u}
        if len(products) != 1:
            raise GATypeError(f"{method!r} requires symmetric unit scaling on {signature}")
        c_root = next(iter(products)).root(2)
        for i, ru in row_u.items():
            w_units[i] = ru / c_root
    return w_units


def _endomorphism_eig_units(
    n: int, aligned: dict[tuple[int, int], Unit], signature: str, method: str,
) -> tuple[Unit, UnitGrid]:
    """Solve `A_{ij} v_j == lambda v_i` across the directed sparsity graph of `aligned`."""
    from numga.gatype.errors import GATypeError

    adj: dict[int, list[tuple[int, Unit, int]]] = {}
    for (i, j), u in aligned.items():
        adj.setdefault(i, []).append((j, u.inverse(), 1))
        adj.setdefault(j, []).append((i, u, -1))

    # Each visited index stores `(base, power)` representing `v_k == base * lambda**power`.
    assigned: dict[int, tuple[Unit, int]] = {}
    eig_u: Unit | None = None
    for root in sorted(adj):
        if root in assigned:
            continue
        assigned[root] = (_ONE, 0)
        queue = deque([root])
        while queue:
            node = queue.popleft()
            base, power = assigned[node]
            for nxt, factor, step in adj[node]:
                cand_base, cand_power = base * factor, power + step
                prev = assigned.get(nxt)
                if prev is None:
                    assigned[nxt] = (cand_base, cand_power)
                    queue.append(nxt)
                    continue
                prev_base, prev_power = prev
                dp = prev_power - cand_power
                if dp == 0:
                    if prev_base != cand_base:
                        raise GATypeError(f"{method!r} has incompatible entry units on {signature}")
                else:
                    ratio = cand_base / prev_base if dp > 0 else prev_base / cand_base
                    cand_eig = ratio.root(abs(dp))
                    if eig_u is None:
                        eig_u = cand_eig
                    elif eig_u != cand_eig:
                        raise GATypeError(f"{method!r} requires a single eigenvalue unit across {signature}")

    if eig_u is None:
        raise GATypeError(f"{method!r} requires a single eigenvalue unit across {signature}")
    mode_entries = {i: base * (eig_u ** power) for i, (base, power) in assigned.items()}
    return eig_u, UnitGrid.from_entries((n,), mode_entries)


def _pencil_ratio_and_modes(operator: GAType, partner: GAType | None, method: str) -> tuple[Unit, UnitGrid]:
    from numga.gatype.errors import GATypeError

    shape, entries = _grid_2d(operator, method)
    n = shape[0]

    if partner is None and operator.arity == 1:
        if method in ("eigh", "eigvalsh"):
            return _require_uniform(operator, method), UnitGrid.uniform_grid((n,), _ONE)
        aligned = _align_rows(operator.subspaces[0], operator.subspaces[1], entries)
        return _endomorphism_eig_units(n, aligned, operator.signature, method)

    if partner is None and operator.is_form:
        if operator.has_fields:
            return _require_uniform(operator, method), UnitGrid.uniform_grid((n,), _ONE)
        aligned = _align_rows(operator.subspaces[1], operator.subspaces[2], entries)
        axis_space = operator.subspaces[2]
        deg = operator.algebra.degenerate_mask
        w_units = _symmetric_scales(rank1_factor(shape, aligned, method), operator.signature, method)
        measured_w = {w_units[i] for i, m in enumerate(axis_space.masks) if not (m & deg) and i in w_units}
        if len(measured_w) != 1:
            raise GATypeError(f"{method!r} mixes measured blades with distinct units in {operator.signature}")
        w0 = next(iter(measured_w))
        return w0 * w0, UnitGrid.from_entries((n,), {i: w0 / wi for i, wi in w_units.items()})

    assert partner is not None
    p_shape, p_entries = _grid_2d(partner, method)
    if operator.arity == 1 and operator.is_endomorphism:
        entries = _align_rows(operator.subspaces[0], operator.subspaces[1], entries)
        p_entries = _align_rows(partner.subspaces[0], partner.subspaces[1], p_entries)
    elif operator.arity == 1 and operator.is_correlation:
        r_map = _dual_permutation(operator.subspaces[0], operator.subspaces[1])
        entries = {(r_map[i], j): u for (i, j), u in entries.items() if i in r_map}
        pr_map = _dual_permutation(partner.subspaces[0], partner.subspaces[1])
        p_entries = {(pr_map[i], j): u for (i, j), u in p_entries.items() if i in pr_map}
    elif operator.is_form:
        entries = _align_rows(operator.subspaces[1], operator.subspaces[2], entries)
        p_entries = _align_rows(partner.subspaces[1], partner.subspaces[2], p_entries)
        if partner.subspaces[2] is not operator.subspaces[2]:
            op_idx = {m: i for i, m in enumerate(operator.subspaces[2].masks)}
            p_map = {i: op_idx[m] for i, m in enumerate(partner.subspaces[2].masks) if m in op_idx}
            p_entries = {(p_map[i], p_map[j]): u for (i, j), u in p_entries.items() if i in p_map and j in p_map}

    p_comps = rank1_factor(p_shape, p_entries, method)
    op_comps = rank1_factor(shape, entries, method)
    ratios = {u / p_entries[k] for k, u in entries.items() if k in p_entries}
    if len(ratios) != 1:
        raise GATypeError(
            f"{method!r} requires a single eigenvalue unit between {operator.signature} and {partner.signature}"
        )
    eig_u = next(iter(ratios))
    if method in ("eigh", "eigvalsh", "lstsq"):
        w_units = _symmetric_scales(p_comps, partner.signature, method)
        eig_root = eig_u.root(2)
        for i, ow in _symmetric_scales(op_comps, operator.signature, method).items():
            w_units.setdefault(i, ow / eig_root)
        for (i, j), u in entries.items():
            if i in w_units and j in w_units and u != eig_u * w_units[i] * w_units[j]:
                raise GATypeError(f"{method!r} units of {operator.signature} do not match {eig_u} * {partner.signature}")
        return eig_u, UnitGrid.from_entries((n,), {i: wi.inverse() for i, wi in w_units.items()})
    mode_entries = {}
    for _, col_u in p_comps:
        c0 = next(iter(col_u.values()))
        mode_entries.update({j: cu / c0 for j, cu in col_u.items()})
    for _, col_u in op_comps:
        anchor = next((j for j in col_u if j in mode_entries), None)
        scale = mode_entries[anchor] / col_u[anchor] if anchor is not None else _ONE / next(iter(col_u.values()))
        for j, cu in col_u.items():
            mode_entries.setdefault(j, cu * scale)
    return eig_u, UnitGrid.from_entries((n,), mode_entries)


def eigvals_unit(operator: GAType, partner: GAType | None = None) -> UnitGrid:
    return UnitGrid.uniform_grid((1,), _pencil_ratio_and_modes(operator, partner, "eigvals")[0])


def eig_unit(operator: GAType, partner: GAType | None = None) -> tuple[UnitGrid, UnitGrid]:
    eig_u, modes = _pencil_ratio_and_modes(operator, partner, "eig")
    return UnitGrid.uniform_grid((1,), eig_u), modes


def eigvalsh_unit(operator: GAType, partner: GAType | None = None) -> UnitGrid:
    return UnitGrid.uniform_grid((1,), _pencil_ratio_and_modes(operator, partner, "eigvalsh")[0])


def eigh_unit(operator: GAType, partner: GAType | None = None) -> tuple[UnitGrid, UnitGrid]:
    eig_u, modes = _pencil_ratio_and_modes(operator, partner, "eigh")
    return UnitGrid.uniform_grid((1,), eig_u), modes


def cholesky_unit(operator: GAType) -> UnitGrid:
    u = _require_uniform(operator, "cholesky")
    n = len(operator.subspaces[1 if operator.arity == 1 else 2])
    return UnitGrid.uniform_grid((n, n), u.root(2))


def svdvals_unit(operator: GAType) -> UnitGrid:
    return UnitGrid.uniform_grid((1,), _require_uniform(operator, "svdvals"))


def svd_unit(operator: GAType) -> tuple[UnitGrid, UnitGrid, UnitGrid]:
    u = _require_uniform(operator, "svd")
    out_n, in_n = operator.units.shape
    return (
        UnitGrid.uniform_grid((out_n,), _ONE),
        UnitGrid.uniform_grid((1,), u),
        UnitGrid.uniform_grid((in_n,), _ONE),
    )


def polar_unit(operand: GAType) -> tuple[UnitGrid, _OnSubspaces]:
    grid = _require_grid(operand, "decompose_polar")
    norm_u = _scalar_norm_unit(operand, "decompose_polar")
    rev_space = operand.derive.symmetric_reverse.output_subspace
    prod_space, prod_grid = _multiply_nullary(
        operand.output_subspace, grid, operand.output_subspace, grid, "decompose_polar",
    )
    scale_grid = remap_grid_subspaces(prod_grid, (prod_space,), (rev_space,)).scale(norm_u.inverse())
    return grid.scale(norm_u.inverse()), _OnSubspaces((rev_space,), scale_grid)


def compare_units(left: GAType, right: GAType) -> UnitGrid:
    return embed_and_merge((left, right), (left.output_subspace,), "compare")


@with_arguments
def clip_unit(value: GAType, minimum: object = None, maximum: object = None) -> UnitGrid:
    from numga.extensor import Extensor
    from numga.gatype.errors import GATypeError

    grid = _require_grid(value, "clip")
    unit = grid.uniform
    for bound in (minimum, maximum):
        if bound is None:
            continue
        if isinstance(bound, Extensor):
            if bound.gatype.units.is_unchecked:
                return UnitGrid.unchecked(grid.shape)
            if bound.gatype.units.is_zero:
                continue
            if not grid.is_zero and bound.gatype.units.uniform != unit:
                raise GATypeError(f"'clip' bound {bound.gatype.signature} does not match {value.signature}")
        elif not (isinstance(bound, (int, float)) and not isinstance(bound, bool) and (bound == 0 or isinf(bound))):
            if not grid.is_zero and unit is not _ONE:
                raise GATypeError(f"'clip' scalar bound {bound!r} is dimensionless; value has {value.signature}")
    return grid


def _trace_axes(operator: GAType, first: int, second: int, method: str) -> UnitGrid:
    from numga.gatype.errors import GATypeError

    grid = _require_grid(operator, method)
    s_a, s_b = operator.subspaces[first], operator.subspaces[second]
    if operator.orientations[first] is operator.orientations[second]:
        b_idx = {m: i for i, m in enumerate(s_b.masks)}
        pair = {i: b_idx[m] for i, m in enumerate(s_a.masks) if m in b_idx}
    else:
        pair = _dual_permutation(s_a, s_b)
    kept = [ax for ax in range(len(operator.subspaces)) if ax not in (first, second)]
    prepend_scalar = first == 0 or second == 0
    out_shape = ((1,) if prepend_scalar else ()) + tuple(grid.shape[ax] for ax in kept)
    out_spaces = ((operator.algebra.subspace.scalar(),) if prepend_scalar else ()) + tuple(
        operator.subspaces[ax] for ax in kept
    )
    old_s, out_s = _strides(grid.shape), _strides(out_shape)
    entries: dict[int, Unit] = {}
    for flat, u in grid.items:
        coords = _unravel(flat, old_s)
        if pair.get(coords[first]) != coords[second]:
            continue
        out_c = ((0,) if prepend_scalar else ()) + tuple(coords[ax] for ax in kept)
        out_flat = _ravel(out_c, out_s)
        prev = entries.get(out_flat)
        if prev is not None and prev != u:
            raise GATypeError(f"{method!r} sums incompatible units {prev} and {u} on {_blade_label(out_spaces, out_c)}")
        entries[out_flat] = u
    return UnitGrid.from_entries(out_shape, entries)


def trace_unnamed_unit(operator: GAType) -> UnitGrid:
    a, b = operator.axis_roles.pair
    return _trace_axes(operator, a, b, "trace")


@with_arguments
def trace_unit(operator: GAType, first: int, second: int) -> UnitGrid:
    a, b = operator.axis_roles.paired(first, second)
    return _trace_axes(operator, a, b, "trace")


def contract_unnamed_unit(operator: GAType) -> UnitGrid:
    a, b = operator.axis_roles.contracted(*operator.axis_roles.pair)
    return _trace_axes(operator, a, b, "contract")


@with_arguments
def contract_unit(operator: GAType, first: int, second: int) -> UnitGrid:
    a, b = operator.axis_roles.contracted(first, second)
    return _trace_axes(operator, a, b, "contract")


@with_arguments
def outermorphism_unit(operator: GAType, target: GAType) -> _OnSubspaces:
    """Compute the per-entry `UnitGrid` of `operator.outermorphism(target)`."""
    from operator import and_, xor
    from numga.extensions.outermorphism import _exterior, _image_product

    grid = _require_grid(operator, "outermorphism")
    space, target_subspace, out_space = operator.subspaces[1], target.output_subspace, operator.subspaces[0]
    algebra = operator.algebra
    product, _, count = _exterior(space, target_subspace)
    correlation = operator.orientation is not operator.input_orientations[0]
    image_product, _ = _image_product(space, out_space, product, correlation)
    op_strides = _strides(grid.shape)

    col_dicts: list[dict[int, Unit]] = [{} for _ in range(len(space))]
    for flat, u in grid.items:
        i, j = _unravel(flat, op_strides)
        col_dicts[j][i] = u
    col_grids = [UnitGrid.from_entries((len(out_space),), d) for d in col_dicts]

    def combine(s1: SubSpace, g1: UnitGrid, s2: SubSpace, g2: UnitGrid) -> tuple[SubSpace, UnitGrid]:
        t1, t2 = algebra.gatype(s1, orientation=operator.orientation), algebra.gatype(s2, orientation=operator.orientation)
        op = algebra.operator.wedge(t1, t2) if image_product is xor else algebra.operator.regressive(t1, t2)
        return _binary_nullary(op, g1, g2, "outermorphism")

    target_index = {mask: i for i, mask in enumerate(target_subspace.masks)}
    in_reg = product is and_
    col_results: dict[int, tuple[SubSpace, UnitGrid]] = {}
    for indices in combinations(range(len(space)), count):
        if count == 0:
            mask = ((1 << algebra.dimension) - 1) if in_reg else 0
        elif in_reg:
            acc = 0
            for idx in indices:
                acc ^= algebra.complement(space.masks[idx])
            mask = algebra.complement(acc)
        else:
            mask = 0
            for idx in indices:
                mask ^= space.masks[idx]
        col_j = target_index.get(mask)
        if col_j is None:
            continue
        if count == 0:
            s_cur = algebra.subspace.scalar() if image_product is xor else algebra.subspace.antiscalar()
            g_cur = UnitGrid.uniform_grid((1,), _ONE)
        else:
            s_cur, g_cur = out_space, col_grids[indices[0]]
            for idx in indices[1:]:
                s_cur, g_cur = combine(s_cur, g_cur, out_space, col_grids[idx])
        col_results[col_j] = (s_cur, g_cur)

    image_space = algebra.subspace.empty()
    for s_cur, _ in col_results.values():
        image_space = image_space.union(s_cur)
    shape = (len(image_space), len(target_subspace))
    strides = _strides(shape)
    entries: dict[int, Unit] = {}
    for col_j, (s_cur, g_cur) in col_results.items():
        for out_i, u in remap_grid_subspaces(g_cur, (s_cur,), (image_space,)).items:
            entries[_ravel((out_i, col_j), strides)] = u
    return _OnSubspaces((image_space, target_subspace), UnitGrid.from_entries(shape, entries))
