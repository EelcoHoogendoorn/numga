"""`Extensor` is the single immutable runtime type for multivectors (`arity == 0`), linear maps
(`arity == 1`), and multilinear maps or forms (`arity >= 2`) across symbolic (`ExactContext`) and
array (`NumPyContext`, `JaxContext`) backends. Every `Extensor` pairs a backend `Context`, a
`GAType` carrying static output and input blade subspaces, intrinsic or extrinsic orientations,
coupled field sites, and proven value traits, and a `kernel` of blade coefficients. Evaluating an
algebraic expression on a nullary `GAType` or passing a `GAType` as a map argument supplies the
identity map on that type, so linear and multilinear maps are constructed directly from algebraic
expressions.

On array backends, leading `shape` axes hold independent batch copies, while trailing
`structural_shape` axes belong to the `GAType` (coupled field site axes first, followed by output
and input blade axes). Indexing, iteration, `reshape`, `broadcast_to`, `sum`, `mean`, `stack`, and
`concatenate` act exclusively on batch axes; on `ExactContext`, `kernel` holds unbatched symbolic
expressions. `.field(*structural_axes)` promotes trailing batch axes into coupled site axes without
moving coefficients, `.sites` reduces over site axes while preserving batch axes, and `.batch()`
demotes site axes back to independent batch axes.

Static blade support propagates through every operation. Selecting a subspace or grade (`select`,
`select_subspace`, `select_grade`) projects onto the target blades and zero-fills missing ones so
the output spans the full target subspace, whereas restricting (`restrict`, `restrict_subspace`,
`restrict_grade`) intersects with existing support without adding zero-filled blades. `cast`
projects or embeds output blade layout without transferring value traits or changing orientation,
`with_traits` asserts traits trusted from the caller, and `retyped` widens blade support or forgets
traits so a narrower initial value can enter a loop carry of wider type.

Algebraic products and extensions follow orientation and metric rules encoded on `GAType`. The
exterior product (`^`), regressive product (`&`), and interior products depend on subspace
incidence and orientation alone without consulting the metric; `dual` and `dual_inverse` apply the
right-Hodge complement while flipping orientation; `dot` pairs same-orientation values (directly on
extrinsic elements and through duals on intrinsic ones) while cross-orientation pairings belong to
`&`; and `>>` / `<<` compose reflections with the parity sign required by intrinsic versus
extrinsic passengers. Domain-specific operations (`inverse`, `exp`, `log`, `solve`, `lstsq`, `eigh`,
`svd`, and scalar functions) attach via `ExtensionMethod` descriptors and dispatch on the full
`GAType` signature. The extensor model, operator syntax, sign and orientation conventions, and
internal table execution are documented in `docs/extensors.md`, `docs/extensor_syntax.md`,
`docs/conventions.md`, and `docs/internals.md`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from functools import reduce
from operator import index as integer_index
from types import NotImplementedType
from typing import TYPE_CHECKING

import numga
from numga.backend.context import binding_context
from numga.binding import AxisTransform, BindingPlan, TypeRules, normalize_bind_arguments, nullary_identity_groups
from numga.extension import ExtensionMethod
from numga.gatype import GAType, GATypeError, Orientation, Trait, UnorientedSubspace, rules
from numga.subspace import SubSpace
from numga.units import Unit, UnitGrid, merge_grids, remap_grid_subspaces

from . import operands

from .namespaces import RestrictNamespace, SelectNamespace, SitesNamespace
from .application import application, unary_application
from .fields import bind_fields

if TYPE_CHECKING:
    import numpy as np

    from numga.algebra import Algebra
    from numga.backend.context import Context


class Extensor:
    """Multivectors, linear maps, and multilinear forms share one immutable representation pairing a
    backend `Context`, a `GAType`, and a coefficient `kernel`."""

    __slots__ = ("_context", "_gatype", "_kernel")
    __hash__ = None
    # Disable NumPy ufunc coercion so array * extensor uses Extensor.__rmul__ instead of building an object array.
    __array_ufunc__ = None
    inverse = ExtensionMethod("inverse", takes="a value, or a map between spaces of equal dimension")
    norm_squared = ExtensionMethod("norm_squared", takes="a value", output_orientation=Orientation.extrinsic)
    norm = ExtensionMethod("norm", takes="a value", output_orientation=Orientation.extrinsic)
    antinorm = ExtensionMethod("antinorm", takes="a value", output_orientation=Orientation.extrinsic)
    normalized = ExtensionMethod("normalized", takes="a value")
    square_root = ExtensionMethod("square_root", takes="a value", input_orientation=Orientation.extrinsic)
    inverse_square_root = ExtensionMethod("inverse_square_root", takes="a value", input_orientation=Orientation.extrinsic)
    study_norm_squared = ExtensionMethod("study_norm_squared", takes="a Study number: a scalar plus a part whose square is scalar", output_orientation=Orientation.extrinsic)
    study_norm = ExtensionMethod("study_norm", takes="a Study number: a scalar plus a part whose square is scalar", output_orientation=Orientation.extrinsic)
    exp = ExtensionMethod("exp", takes="a value", input_orientation=Orientation.extrinsic)
    log = ExtensionMethod("log", takes="a value", input_orientation=Orientation.extrinsic)
    exp_linear = ExtensionMethod("exp_linear", takes="a bivector", input_orientation=Orientation.extrinsic)
    exp_linear_normalized = ExtensionMethod("exp_linear_normalized", takes="a bivector", input_orientation=Orientation.extrinsic)
    exp_quadratic = ExtensionMethod("exp_quadratic", takes="a bivector", input_orientation=Orientation.extrinsic)
    exp_cayley = ExtensionMethod("exp_cayley", takes="a bivector", input_orientation=Orientation.extrinsic)
    exp_bisect = ExtensionMethod("exp_bisect", takes="a bivector", input_orientation=Orientation.extrinsic)
    exp_derivative = ExtensionMethod("exp_derivative", takes="a bivector")
    log_linear = ExtensionMethod("log_linear", takes="an even value", input_orientation=Orientation.extrinsic)
    log_linear_normalized = ExtensionMethod("log_linear_normalized", takes="a rotor", input_orientation=Orientation.extrinsic)
    log_quadratic = ExtensionMethod("log_quadratic", takes="a rotor", input_orientation=Orientation.extrinsic)
    log_pade = ExtensionMethod("log_pade", takes="a rotor", input_orientation=Orientation.extrinsic)
    square_root_denman_beavers = ExtensionMethod("square_root_denman_beavers", takes="a rotor", input_orientation=Orientation.extrinsic)
    geometric_mean = ExtensionMethod("geometric_mean", takes="two rotors")
    decompose_polar = ExtensionMethod("decompose_polar", takes="a bivector whose product with its reverse is a Study number")
    decompose_invariant = ExtensionMethod("decompose_invariant", takes="a bivector whose square is a scalar or a Study number")
    motor_translator = ExtensionMethod("motor_translator", takes="a rotor")
    motor_rotor = ExtensionMethod("motor_rotor", takes="a rotor")
    motor_split = ExtensionMethod("motor_split", takes="a rotor, and the antivector to split it about")
    inverse_shirokov = ExtensionMethod("inverse_shirokov", takes="a value")
    inverse_factor = ExtensionMethod("inverse_factor", takes="a value")
    inverse_hitzer = ExtensionMethod("inverse_hitzer", takes="a value whose product with its reverse, times its Clifford conjugate, is a scalar")
    trace = ExtensionMethod("trace", takes="an extensor with two axes that carry an index, or the two axes to pair", overloads=(1, 3))
    contract = ExtensionMethod("contract", takes="an extensor with two inputs of one space, or the two axes to pair", overloads=(1, 3))
    outermorphism = ExtensionMethod("outermorphism", takes="a map from vectors to vectors, or from antivectors to antivectors")
    det = ExtensionMethod("det", takes="a map from a space to itself, or a square form, with an optional metric form on the same space", overloads=(1, 2))
    real = ExtensionMethod("real", takes="an extensor")
    solve = ExtensionMethod("solve", takes="a map between spaces of equal dimension, or a form, with a right-hand side its output takes")
    lstsq = ExtensionMethod("lstsq", takes="a map or a form, with a right-hand side its output takes and an optional metric form", overloads=(2, 3))
    pinv = ExtensionMethod("pinv", takes="a map")
    adjugate = ExtensionMethod("adjugate", takes="a map")
    adjoint = ExtensionMethod("adjoint", takes="a map")
    cholesky = ExtensionMethod("cholesky", takes="a map from a space to itself, or a square form")
    eig = ExtensionMethod("eig", takes="a map from a space to itself, or a square form, with an optional metric form on the same space", overloads=(1, 2))
    eigvals = ExtensionMethod("eigvals", takes="a map from a space to itself, or a square form, with an optional metric form on the same space", overloads=(1, 2))
    eigh = ExtensionMethod("eigh", takes="a map from a space to itself, or a square form, with an optional metric form on the same space", overloads=(1, 2))
    eigvalsh = ExtensionMethod("eigvalsh", takes="a map from a space to itself, or a square form, with an optional metric form on the same space", overloads=(1, 2))
    svd = ExtensionMethod("svd", takes="a map")
    svdvals = ExtensionMethod("svdvals", takes="a map")
    sin = ExtensionMethod("sin", takes="a scalar")
    cos = ExtensionMethod("cos", takes="a scalar", output_orientation=Orientation.extrinsic)
    tan = ExtensionMethod("tan", takes="a scalar")
    arcsin = ExtensionMethod("arcsin", takes="a scalar")
    arccos = ExtensionMethod("arccos", takes="a scalar", input_orientation=Orientation.extrinsic)
    arctan = ExtensionMethod("arctan", takes="a scalar")
    sinh = ExtensionMethod("sinh", takes="a scalar")
    cosh = ExtensionMethod("cosh", takes="a scalar", output_orientation=Orientation.extrinsic)
    tanh = ExtensionMethod("tanh", takes="a scalar")
    arcsinh = ExtensionMethod("arcsinh", takes="a scalar")
    arccosh = ExtensionMethod("arccosh", takes="a scalar", input_orientation=Orientation.extrinsic)
    arctanh = ExtensionMethod("arctanh", takes="a scalar")
    clip = ExtensionMethod("clip", takes="a scalar")
    isnan = ExtensionMethod("isnan", takes="a scalar")
    isfinite = ExtensionMethod("isfinite", takes="a scalar")
    isinf = ExtensionMethod("isinf", takes="a scalar")
    argsort = ExtensionMethod("argsort", takes="a scalar")
    argmax = ExtensionMethod("argmax", takes="a scalar")
    argmin = ExtensionMethod("argmin", takes="a scalar")
    min = ExtensionMethod("min", takes="a scalar")
    max = ExtensionMethod("max", takes="a scalar")
    abs = ExtensionMethod("abs", takes="a scalar")
    less = ExtensionMethod("less", takes="two scalars")
    less_equal = ExtensionMethod("less_equal", takes="two scalars")
    greater = ExtensionMethod("greater", takes="two scalars")
    greater_equal = ExtensionMethod("greater_equal", takes="two scalars")
    to_array = ExtensionMethod("to_array", takes="a scalar")
    add = ExtensionMethod("add", takes="two extensors of equal arity, or a value and a number or array")
    subtract = ExtensionMethod("subtract", takes="two extensors of equal arity, or a value and a number or array")
    divide = ExtensionMethod("divide", takes="two values, or any extensor and a scalar divisor")
    power = ExtensionMethod("power", takes="a value")
    reverse = ExtensionMethod("reverse", takes="an extensor")
    clifford_conjugate = ExtensionMethod("clifford_conjugate", takes="an extensor")
    involute = ExtensionMethod("involute", takes="an extensor")
    scalar_negation = ExtensionMethod("scalar_negation", takes="an extensor")
    pseudoscalar_negation = ExtensionMethod("pseudoscalar_negation", takes="an extensor")

    def __init__(self, context: Context, gatype: GAType, kernel) -> None:
        kernel = context.prepare_kernel(kernel)
        if gatype.algebra is not context.algebra:
            raise ValueError("GAType and Context belong to different algebras")
        if kernel.shape[-len(gatype.structural_shape):] != gatype.structural_shape:
            raise ValueError(
                f"kernel trailing shape does not match GAType structural shape {gatype.structural_shape}"
            )
        if context.is_exact and kernel.shape != gatype.structural_shape:
            raise ValueError("exact Extensors cannot carry batch dimensions")
        self._context = context
        self._gatype = gatype
        self._kernel = kernel

    @classmethod
    def _from_prepared_kernel(
        cls,
        context: Context,
        gatype: GAType,
        kernel,
    ) -> Extensor:
        """Trusted internal results bypass kernel coercion and shape validation."""
        self = cls.__new__(cls)
        self._context = context
        self._gatype = gatype
        self._kernel = kernel
        return self

    @property
    def context(self) -> Context:
        return self._context

    @property
    def gatype(self) -> GAType:
        return self._gatype

    @property
    def has_units(self) -> bool:
        """Whether this extensor carries checked physical units (false when unchecked)."""
        return self._gatype.has_units

    @property
    def unit(self) -> Unit:
        """Single physical unit shared by every nonzero entry; raises `GATypeError` if unchecked,
        structurally zero, or non-uniform."""
        return self._gatype.unit

    @property
    def units(self) -> UnitGrid:
        """Per-entry physical unit grid carried on this extensor's `GAType`."""
        return self._gatype.units

    @property
    def kernel(self):
        return self._kernel

    @property
    def algebra(self) -> Algebra:
        return self.gatype.algebra

    @property
    def axes(self) -> tuple[SubSpace, ...]:
        return self.gatype.subspaces

    @property
    def output_subspace(self) -> SubSpace:
        return self.gatype.output_subspace

    @property
    def subspace(self) -> SubSpace:
        return self.output_subspace

    @property
    def output_gatype(self) -> GAType:
        """Stripping a map or form's input axes leaves the nullary `GAType` (blade subspace and
        orientation) of the values it produces."""
        return self.gatype.derive.output

    @property
    def input_subspaces(self) -> tuple[SubSpace, ...]:
        return self.gatype.input_subspaces

    @property
    def arity(self) -> int:
        return self.gatype.arity

    @property
    def structural_shape(self) -> tuple[int, ...]:
        return self.gatype.structural_shape

    @property
    def shape(self) -> tuple[int, ...]:
        """Batch dimensions precede any field site axes and blade coefficient axes, which belong to
        `structural_shape`."""
        return self._kernel.shape[: -len(self.gatype.structural_shape)]

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def dtype(self) -> np.dtype:
        return self.context.dtype

    def __getitem__(self, index) -> Extensor:
        kernel_index = _batch_kernel_index(index, self.ndim, len(self.structural_shape))
        return type(self)._from_prepared_kernel(self.context, self.gatype, self._kernel[kernel_index])

    def __iter__(self):
        return (self[index] for index in range(self.shape[0]))

    def __len__(self) -> int:
        if not self.shape:
            raise TypeError("len() of an unbatched Extensor")
        return self.shape[0]

    def __invert__(self) -> Extensor:
        return self.reverse()

    def formula(self) -> str:
        """Symbolic extensors expand into human-readable coordinate expressions for each output
        blade."""
        from numga.operator.format import formula
        return formula(self)

    def coefficient_table(self) -> str:
        """Unbatched multivectors and linear maps format as a blade-labelled table of coefficients."""
        from numga.operator.format import coefficient_table
        return coefficient_table(self)

    def to_python(self, function_name: str = "apply") -> str:
        """Symbolic extensors compile into standalone Python code operating on plain coefficient
        arrays."""
        from numga.operator.format import python_code
        return python_code(self, function_name)

    @property
    def at(self) -> _AtIndexer:
        return _AtIndexer(self)

    def map_kernel(
        self, function: Callable, *args,
        preserve_traits: bool = False, **kwargs,
    ) -> Extensor:
        """Explicit traits are dropped unless `preserve_traits=True`, because arbitrary array
        operations over batch axes need not preserve whole-value invariants such as unit norm."""
        kernel = function(self._kernel, *args, **kwargs)
        gatype = self.gatype if preserve_traits else self.gatype.derive.structural
        return type(self)._from_prepared_kernel(self.context, gatype, kernel)

    def reshape(self, *shape: int | tuple[int, ...]) -> Extensor:
        batch_shape = _shape_arguments(shape)
        kernel = self.context.xp.reshape(
            self._kernel,
            batch_shape + self.structural_shape,
        )
        return type(self)._from_prepared_kernel(self.context, self.gatype, kernel)

    def broadcast_to(self, shape: int | tuple[int, ...]) -> Extensor:
        batch_shape = _shape_arguments((shape,))
        kernel = self.context.xp.broadcast_to(
            self._kernel,
            batch_shape + self.structural_shape,
        )
        return type(self)._from_prepared_kernel(self.context, self.gatype, kernel)

    def sum(
        self,
        axis: int | tuple[int, ...] | None = None,
        *,
        keepdims: bool = False,
    ) -> Extensor:
        axes = _batch_axes(axis, self.ndim)
        kernel = self.context.xp.sum(self._kernel, axis=axes, keepdims=keepdims)
        return type(self)._from_prepared_kernel(self.context, self.gatype.derive.structural, kernel)

    def cumsum(self, axis: int) -> Extensor:
        """Running sums accumulate independently along a batch axis while dropping nonlinear traits
        such as unit norm."""
        kernel = self.context.xp.cumsum(self._kernel, axis=_existing_axis(axis, self.ndim))
        return type(self)._from_prepared_kernel(self.context, self.gatype.derive.structural, kernel)

    def cumprod(self, axis: int) -> Extensor:
        """Later elements multiply on the left (`x0`, `x1 * x0`, `x2 * x1 * x0`) so a sequence of
        rotors composes in application order."""
        square = self.algebra.operator.geometric_product(self.gatype, self.gatype)
        if self.arity or not square.output_subspace.support_is_subset_of(self.output_subspace):
            raise GATypeError(f"cumprod needs a type closed under the geometric product; {self.gatype} is not")
        leading = (slice(None),) * _existing_axis(axis, self.ndim)
        running = [self[leading + (0,)]]
        for position in range(1, self.shape[len(leading)]):
            running.append(self[leading + (position,)] * running[-1])
        return type(self).stack(running, axis=len(leading))

    def mean(
        self,
        axis: int | tuple[int, ...] | None = None,
        *,
        keepdims: bool = False,
    ) -> Extensor:
        axes = _batch_axes(axis, self.ndim)
        kernel = self.context.xp.mean(self._kernel, axis=axes, keepdims=keepdims)
        return type(self)._from_prepared_kernel(self.context, self.gatype.derive.structural, kernel)

    @classmethod
    def stack(
        cls,
        extensors: Iterable[Extensor],
        axis: int = 0,
    ) -> Extensor:
        values = _collection_values(extensors)
        first = values[0]
        gatype = _stacked_type(values)
        logical_axis = _insertion_axis(axis, first.ndim)
        kernel = first.context.xp.stack(
            tuple(_embed_kernel(first.context, value, gatype.subspaces) for value in values),
            axis=logical_axis,
        )
        return cls._from_prepared_kernel(first.context, gatype, kernel)

    @classmethod
    def concatenate(
        cls,
        extensors: Iterable[Extensor],
        axis: int = 0,
    ) -> Extensor:
        values = _collection_values(extensors)
        first = values[0]
        gatype = _stacked_type(values)
        logical_axis = _existing_axis(axis, first.ndim)
        kernel = first.context.xp.concatenate(
            tuple(_embed_kernel(first.context, value, gatype.subspaces) for value in values),
            axis=logical_axis,
        )
        return cls._from_prepared_kernel(first.context, gatype, kernel)

    def field(self, *structural_axes: int) -> Extensor:
        """Trailing batch axes become coupled site axes on `structural_axes` (`0` for output,
        `1..arity` for inputs), distinguishing coupled field sites from independent batch copies
        without moving coefficients."""
        structural_axes = structural_axes or (0,)
        if self.context.is_exact or self.ndim < len(structural_axes):
            raise TypeError("a field takes its sites from the last batch axes of a backend value")
        if list(structural_axes) != sorted(set(structural_axes)) or any(
            field <= structural_axes[-1] for field, _ in self.gatype.fields
        ):
            raise GATypeError(
                f"axes {structural_axes} of {self.gatype.signature} are not in order before its field axes"
            )
        # Prepend the promoted batch axes ahead of any existing site axes so kernel layout stays unchanged.
        fields = tuple(zip(structural_axes, self.shape[self.ndim - len(structural_axes):])) + self.gatype.fields
        return type(self)._from_prepared_kernel(self.context, self.gatype.derive.with_fields(fields), self._kernel)

    def on_diagonal(self) -> Extensor:
        """A field of per-site maps `X[n] <- Y` becomes the site-diagonal field map `X[n] <- Y[n]`
        acting within each site alone."""
        if self.arity < 1 or [axis for axis, _ in self.gatype.fields] != [0]:
            raise GATypeError(f"on_diagonal places a field of maps, X[n] <- Y, on its sites; {self.gatype.signature} is not one")
        xp, sites = self.context.xp, self.gatype.site_shape[0]
        identity = xp.reshape(xp.eye(sites, dtype=self._kernel.dtype), (sites, sites) + (1,) * len(self.axes))
        kernel = xp.expand_dims(self._kernel, self.ndim + 1) * identity
        fields = ((0, sites), (1, sites))
        return type(self)._from_prepared_kernel(self.context, self.gatype.derive.structural.derive.with_fields(fields), kernel)

    def batch(self) -> Extensor:
        """Demoting coupled field site axes back to trailing batch axes treats every site as an
        independent copy again without moving coefficients."""
        return type(self)._from_prepared_kernel(self.context, self.gatype.derive.plain, self._kernel)

    def bind(self, *args: Extensor | dict[int, Extensor]) -> Extensor:
        raw_operands = normalize_bind_arguments(self.arity, args)
        if not raw_operands:
            return self
        if self._gatype.has_fields or any(operand.gatype.has_fields for operand in raw_operands.values()):
            return bind_fields(self, raw_operands)

        plan = BindingPlan.build(self.gatype, raw_operands)
        operands = {}
        prepared_operands = {}
        context = binding_context(
            self.context, tuple(raw_operands[axis].context for axis in plan.axes),
        )
        target = self
        output = plan.result_subspaces[0]
        if output is not self.output_subspace:
            # Slice away statically zero output blades before lowering or contracting.
            indices = plan.output_indices
            kernel = (
                self._kernel.take(indices, axis=0)
                if self.context.is_exact else self.context.xp.take(
                    self._kernel, self.context.xp.asarray(indices, dtype=int), axis=self.ndim,
                )
            )
            target = type(self)._from_prepared_kernel(self.context, plan.execution_gatype, kernel)
        for axis, operand in raw_operands.items():
            identity = id(operand)
            concrete = prepared_operands.get(identity)
            if concrete is None:
                concrete = context.lower(operand)
                # Lower each distinct argument once per bind so repeated arguments keep object identity.
                prepared_operands[identity] = concrete
            operands[axis] = concrete

        result_gatype = TypeRules.bind(plan)
        result = context.execute_bind(target, operands, plan)
        return type(self)._from_prepared_kernel(context, result_gatype, result)

    def __call__(self, *operands: Extensor | GAType) -> Extensor:
        """Fewer than `arity` arguments bind leading inputs into a lower-arity map, and passing a
        `GAType` supplies its identity map (or restricts that input to a subtype)."""
        if any(isinstance(operand, SubSpace) for operand in operands):
            raise UnorientedSubspace()
        if any(isinstance(operand, GAType) for operand in operands):
            bound = {}
            for axis, operand in enumerate(operands):
                if not isinstance(operand, GAType):
                    bound[axis] = operand
                elif (narrowed := self._gatype.axis_roles.narrowing(axis, operand)) is not None:
                    bound[axis] = narrowed
            return self.bind(bound)
        if len(operands) < self.arity:
            return self.bind(*operands)
        if len(operands) == 1:
            operand = operands[0]
            if not isinstance(operand, Extensor):
                # Delegate unary application to external operand types such as Sparse.
                return operand._applied_by(self)
            execute = unary_application(
                type(self), self._context, self._gatype,
                operand._context, operand._gatype,
            )
            return execute(self, operand)
        execute = application(
            type(self), self._context, self._gatype,
            tuple((operand._context, operand._gatype) for operand in operands),
            nullary_identity_groups(tuple(enumerate(operands))),
        )
        return execute(self, *operands)

    def squeeze_output(self) -> Extensor:
        """Dropping output blades whose symbolic expressions simplify to identically zero narrows an
        exact extensor's static support."""
        if not self.context.is_exact:
            raise TypeError("output squeezing requires an exact Extensor")
        indices = self.kernel.output_nonzero_indices()
        output = self.output_subspace.restrict(
            tuple(self.output_subspace.masks[index] for index in indices),
        )
        gatype = rules.SQUEEZE.result((self.gatype,), (output,) + self.input_subspaces)
        return type(self)._from_prepared_kernel(self.context, gatype, self.kernel.take(indices, axis=0))

    def materialize(self, context: Context) -> Extensor:
        return context.lower(self)

    def cast(self, target: SubSpace | GAType) -> Extensor:
        """Output blades are projected or zero-embedded into `target` without transferring algebraic
        traits, since matching a rotor's layout does not establish the rotor constraint."""
        target_subspace = target.output_subspace if isinstance(target, GAType) else target
        merged_units = None
        if isinstance(target, GAType):
            if target.orientation is not self.gatype.orientation:
                raise GATypeError(
                    f"a cast changes layout, not orientation: {self.gatype.signature} cannot become {target.signature}"
                )
            if self.has_units and target.has_units and self.arity == target.arity:
                projected = remap_grid_subspaces(self.units, self.axes, (target_subspace,) + target.input_subspaces)
                merged_units = merge_grids((projected, target.units), (target_subspace,) + target.input_subspaces, "cast")
        if self.output_subspace is target_subspace:
            result = self
        else:
            result = self.algebra.operator.cast(
                self.gatype.derive.output, target_subspace,
            ).bind({0: self})
        if merged_units is not None and result.units != merged_units:
            return type(self)._from_prepared_kernel(
                result.context, result.gatype.derive.unit(merged_units), result._kernel,
            )
        return result

    @property
    def select(self) -> SelectNamespace:
        """Selecting a grade or named subspace (`x.select.bivector`, `x.select[2]`) projects onto
        those output blades and zero-fills any missing ones so the output spans the full target
        subspace (unlike `restrict`, which keeps only blades already in `self`)."""
        return SelectNamespace(self)

    @property
    def sites(self) -> SitesNamespace:
        """Reductions on `.sites` contract coupled field site axes (`energy.sites.sum()`) while
        leaving batch axes intact."""
        return SitesNamespace(self)

    @property
    def restrict(self) -> RestrictNamespace:
        """Restricting to a grade or named subspace (`x.restrict.bivector`, `x.restrict[2]`)
        intersects output support with the target, keeping only blades already present in `self`
        without adding zero-filled blades (unlike `select`, which pads to the full target subspace)."""
        return RestrictNamespace(self)

    def select_subspace(self, subspace: SubSpace) -> Extensor:
        """Projecting onto `subspace` keeps matching output blades and zero-fills any missing ones
        so the output subspace becomes all of `subspace` (unlike `restrict_subspace`, which keeps
        only blades already in `self`)."""
        return self.cast(subspace)

    def restrict_subspace(self, subspace: SubSpace) -> Extensor:
        """Intersecting output support with `subspace` keeps only blades already present in `self`
        without zero-filling missing blades of `subspace` (unlike `select_subspace`, which pads to
        all of `subspace`)."""
        return self.algebra.operator.restrict(
            self.gatype.derive.output, subspace,
        ).bind({0: self})

    def select_grade(self, grade: int) -> Extensor:
        """Grade selection projects onto grade `grade` and zero-fills missing blades so the output
        spans the full grade subspace (unlike `restrict_grade`, which keeps only grade-`grade`
        blades already in `self`)."""
        return self.select_subspace(self.algebra.subspace.k_vector(grade))

    def restrict_grade(self, grade: int) -> Extensor:
        """Grade restriction keeps only grade-`grade` blades already present in `self` without
        adding zero-filled blades (unlike `select_grade`, which pads to the full grade subspace)."""
        return self.restrict_subspace(self.algebra.subspace.k_vector(grade))

    def with_traits(self, *traits: Trait) -> Extensor:
        """Caller-asserted traits are trusted without inspecting coefficients."""
        gatype = self.gatype.derive.with_traits(*traits)
        if gatype is self.gatype:
            return self
        return type(self)._from_prepared_kernel(self.context, gatype, self._kernel)

    def retyped(self, gatype: GAType) -> Extensor:
        """Forgetting traits that `gatype` omits lets a more specific initial value (such as a
        product of two reflections) enter a loop carry typed as a general motor."""
        if gatype.arity or self.gatype.arity:
            raise TypeError(f"retyped takes values, not maps; got {self.gatype} and {gatype}")
        embedded = self.cast(gatype)
        unproven = [trait for trait in gatype.traits if trait not in embedded.gatype.traits]
        if unproven:
            raise GATypeError(f"{self.gatype} does not carry {unproven}, which {gatype} states; retyped only forgets facts")
        if not gatype.has_units:
            result_units = embedded.gatype.units
        elif not embedded.gatype.has_units:
            result_units = UnitGrid.unchecked(tuple(len(space) for space in gatype.subspaces))
        else:
            result_units = merge_grids((embedded.gatype.units, gatype.units), gatype.subspaces, "retyped")
        return type(self)._from_prepared_kernel(self.context, gatype.derive.unit(result_units), embedded._kernel)

    def _symmetric_product(
        self, transform: str, *, scalar_only: bool = False,
    ) -> Extensor:
        operator = self.algebra.operator._symmetric_product(
            self.gatype.derive.plain, transform, scalar_only=scalar_only,
        )
        return operator(self, self)

    def squared(self) -> Extensor:
        """Squaring an extensor combines cross-blade cancellations from sharing one operand, giving
        tighter static support than a general `x * y` product."""
        return self._symmetric_product("identity")

    def symmetric_reverse_product(self) -> Extensor:
        """Coefficients are multiplied out even when `self` carries a unit-norm trait, so numerical
        drift from unity remains measurable."""
        return self._symmetric_product("reverse")

    def scalar_norm_squared(self) -> Extensor:
        """Evaluating only the scalar part of `self * self.reverse()` measures Euclidean coefficient
        magnitude squared without computing higher-grade blades."""
        return self._symmetric_product("reverse", scalar_only=True)

    def symmetric_conjugate_product(self) -> Extensor:
        return self._symmetric_product("clifford_conjugate")

    def symmetric_scalar_negation_product(self) -> Extensor:
        return self._symmetric_product("scalar_negation")

    def symmetric_pseudoscalar_negation_product(self) -> Extensor:
        return self._symmetric_product("pseudoscalar_negation")

    def symmetric_involute_product(self) -> Extensor:
        return self._symmetric_product("involute")

    def dual(self) -> Extensor:
        """The right-Hodge complement maps each basis blade to its complement signed by its
        negative-metric factors, flipping intrinsic and extrinsic orientation without degenerating
        along null directions."""
        return self.algebra.operator.dual(self.gatype.derive.plain.derive.output)(self)

    def dual_inverse(self) -> Extensor:
        """The inverse right-Hodge complement satisfies `x.dual().dual_inverse() == x`, undoing the
        grade-dependent sign that applying `dual()` twice introduces."""
        return self.algebra.operator.dual_inverse(self.gatype.derive.plain.derive.output)(self)

    def __pos__(self) -> Extensor:
        return self

    def __neg__(self) -> Extensor:
        gatype = rules.NEGATE.result((self.gatype,), self.gatype.subspaces).derive.with_fields(self.gatype.fields)
        return type(self)._from_prepared_kernel(self.context, gatype, -self._kernel)

    def __lt__(self, other: object):
        return self.less(operands.compared(self._context, other))

    def __le__(self, other: object):
        return self.less_equal(operands.compared(self._context, other))

    def __gt__(self, other: object):
        return self.greater(operands.compared(self._context, other))

    def __ge__(self, other: object):
        return self.greater_equal(operands.compared(self._context, other))

    def __add__(self, other: object) -> Extensor | NotImplementedType:
        other = operands.linear(self._context, other)
        return NotImplemented if other is None else self.add(other)

    def __radd__(self, other: object) -> Extensor | NotImplementedType:
        other = operands.linear(self._context, other)
        return NotImplemented if other is None else self.add(other)

    def __sub__(self, other: object) -> Extensor | NotImplementedType:
        other = operands.linear(self._context, other)
        return NotImplemented if other is None else self.subtract(other)

    def __rsub__(self, other: object) -> Extensor | NotImplementedType:
        other = operands.linear(self._context, other)
        return NotImplemented if other is None else Extensor.subtract(other, self)

    def __mul__(self, other: object) -> Extensor | NotImplementedType:
        if numga.expression.is_expression_operand(other):
            return numga.expression.geometric_product(self, other)
        found = operands.kind(other)
        if found is operands.Kind.ARRAY:
            return numga.expression.geometric_product(self, operands.batch(self._context, other))
        if found is not operands.Kind.NUMBER:
            return NotImplemented
        scalar = self.context.prepare_scalar(other)
        gatype = rules.SCALE.result((self.gatype,), self.gatype.subspaces).derive.with_fields(self.gatype.fields)
        return type(self)._from_prepared_kernel(self.context, gatype, self._kernel * scalar)

    def __rmul__(self, other: object) -> Extensor | NotImplementedType:
        if numga.expression.is_expression_operand(other):
            return numga.expression.geometric_product(other, self)
        return self * other

    def __truediv__(self, other: object) -> Extensor | NotImplementedType:
        if operands.kind(other) is operands.Kind.NUMBER:
            # The context validates the scalar before it is inverted.
            return self * (1 / self._context.prepare_scalar(other))
        other = operands.divisor(self._context, other)
        return NotImplemented if other is None else self.divide(other)

    def __rtruediv__(self, other: object) -> Extensor | NotImplementedType:
        other = operands.divisor(self._context, other)
        return NotImplemented if other is None else Extensor.divide(other, self)

    def __pow__(self, exponent: int) -> Extensor | NotImplementedType:
        if not isinstance(exponent, int):
            return NotImplemented
        return self.power(exponent)

    def wedge(self, other: Extensor | GAType) -> Extensor:
        """The exterior product (`self ^ other`) keeps grade `r + s` of the geometric product,
        vanishing along shared directions."""
        return numga.expression.wedge(self, other)

    def scalar_product(self, other: Extensor | GAType) -> Extensor:
        """The scalar product takes the grade-zero part of `self * other` without reversing either
        factor (unlike `dot`)."""
        return numga.expression.scalar_product(self, other)

    def dot(self, other: Extensor | GAType) -> Extensor:
        """Extrinsic values pair by the scalar part of `self * other.reverse()`, while intrinsic
        values (points, directions, and their joins) pair through their duals so a direction dotted
        with itself gives its squared length in space, spacetime, and projective algebras alike.
        Cross-orientation pairings belong to the regressive product `&` rather than `dot`."""
        return numga.expression.dot(self, other)

    def inner(self, other: Extensor | GAType) -> Extensor:
        """The symmetric metric inner product (`self | other`) keeps grade `|r - s|` of the
        geometric product for grades `r` and `s`."""
        return numga.expression.inner(self, other)

    def bivector_product(self, other: Extensor | GAType) -> Extensor:
        return numga.expression.bivector_product(self, other)

    def trivector_product(self, other: Extensor | GAType) -> Extensor:
        return numga.expression.trivector_product(self, other)

    def commutator(self, other: Extensor | GAType) -> Extensor:
        """The half-difference `(self * other - other * self) / 2` gives the bivector Lie bracket
        without an extra factor of two."""
        return numga.expression.commutator(self, other)

    def anticommutator(self, other: Extensor | GAType) -> Extensor:
        """The half-sum `(self * other + other * self) / 2` keeps the commuting part of
        `self * other`."""
        return numga.expression.anticommutator(self, other)

    def left_contraction(self, other: Extensor | GAType) -> Extensor:
        """The left metric contraction keeps grade `s - r` of `self * other` for grades `r` and `s`,
        vanishing when `r > s`."""
        return numga.expression.left_contraction(self, other)

    def right_contraction(self, other: Extensor | GAType) -> Extensor:
        """The right metric contraction keeps grade `r - s` of `self * other` for grades `r` and
        `s`, vanishing when `s > r`."""
        return numga.expression.right_contraction(self, other)

    def left_interior(self, other: Extensor | GAType) -> Extensor:
        """The left interior product anti-wedges the left complement of `self` with `other`, staying
        non-degenerate along null directions."""
        return numga.expression.left_interior(self, other)

    def right_interior(self, other: Extensor | GAType) -> Extensor:
        """The right interior product anti-wedges `self` with the right complement of `other`,
        staying non-degenerate along null directions."""
        return numga.expression.right_interior(self, other)

    def regressive(self, other: Extensor | GAType) -> Extensor:
        """The metric-free regressive product (`self & other`) measures complementary blades against
        the intrinsic unit volume (`left ^ right == (left & right) * antiscalar`), joining intrinsic
        elements and pairing opposite orientations."""
        return numga.expression.regressive(self, other)

    def transform(self, passenger: Extensor | GAType) -> Extensor:
        """Versor action composes reflections in vector factors; reflecting in unit `m` sends
        extrinsic `x` to `m * x.involute() * m.inverse()` and negates that result on intrinsic `x`."""
        return numga.expression.transform(self, passenger)

    def reverse_transform(self, passenger: Extensor | GAType) -> Extensor:
        """Applying the reversed versor (`self << passenger`) inverts `self >> passenger` whenever
        `self` has unit norm."""
        return numga.expression.reverse_transform(self, passenger)

    def __rshift__(self, other: Extensor | GAType) -> Extensor:
        return self.transform(other)

    def __lshift__(self, other: Extensor | GAType) -> Extensor:
        return self.reverse_transform(other)

    # Return NotImplemented for unknown operand types such as Sparse so their reflected operators run.
    def __xor__(self, other: Extensor | GAType) -> Extensor | NotImplementedType:
        return self.wedge(other) if numga.expression.is_expression_operand(other) else NotImplemented

    def __rxor__(self, other: Extensor | GAType) -> Extensor | NotImplementedType:
        return numga.expression.wedge(other, self) if numga.expression.is_expression_operand(other) else NotImplemented

    def __and__(self, other: Extensor | GAType) -> Extensor | NotImplementedType:
        return self.regressive(other) if numga.expression.is_expression_operand(other) else NotImplemented

    def __rand__(self, other: Extensor | GAType) -> Extensor | NotImplementedType:
        return numga.expression.regressive(other, self) if numga.expression.is_expression_operand(other) else NotImplemented

    def __or__(self, other: Extensor | GAType) -> Extensor | NotImplementedType:
        return self.inner(other) if numga.expression.is_expression_operand(other) else NotImplemented

    def __ror__(self, other: Extensor | GAType) -> Extensor | NotImplementedType:
        return numga.expression.inner(other, self) if numga.expression.is_expression_operand(other) else NotImplemented

    def _require_compatible(self, other: Extensor) -> None:
        if not self.context.is_compatible_with(other.context):
            raise ValueError("Extensor contexts are incompatible")
        if self.gatype != other.gatype:
            raise GATypeError("Extensor GATypes differ")

    def __repr__(self) -> str:
        return f"Extensor({self.gatype.signature}, shape={self.shape})"


class _AtIndexer:
    __slots__ = ("_extensor",)

    def __init__(self, extensor: Extensor) -> None:
        self._extensor = extensor

    def __getitem__(self, index: object) -> _AtUpdate:
        return _AtUpdate(self._extensor, index)


def _embed_kernel(
    context: Context,
    value: Extensor,
    target_axes: tuple[SubSpace, ...],
):
    kernel = value._kernel
    structural_ndim = len(value.axes)
    for axis, (source, target) in enumerate(zip(value.axes, target_axes)):
        transform = AxisTransform.plan(source, target)
        kernel = context.transform_axis(
            kernel,
            structural_ndim,
            axis,
            transform,
        )
    return kernel


class _AtUpdate:
    __slots__ = ("_extensor", "_index")

    def __init__(self, extensor: Extensor, index: object) -> None:
        self._extensor = extensor
        self._index = index

    def set(self, value: object) -> Extensor:
        return self._update(self._extensor.context.functional_set, value)

    def add(self, value: object) -> Extensor:
        return self._update(self._extensor.context.functional_add, value)

    def _update(self, update: Callable, value: object) -> Extensor:
        target = self._extensor
        kernel_index = _batch_kernel_index(
            self._index,
            target.ndim,
            len(target.structural_shape),
        )
        if isinstance(value, Extensor):
            if not target.context.is_compatible_with(value.context):
                raise ValueError("Extensor contexts are incompatible")
            if (
                target.gatype.subspaces != value.gatype.subspaces
                or target.gatype.orientations != value.gatype.orientations
                or target.gatype.fields != value.gatype.fields
            ):
                raise GATypeError("Extensor GATypes differ")
            merged_units = merge_grids((target.units, value.units), target.axes, "at")
            replacement = value._kernel
        else:
            merged_units = target.units
            replacement = value
        # Drop explicit traits because elementwise updates need not preserve whole-value invariants.
        kernel = update(target._kernel, kernel_index, replacement)
        return type(target)._from_prepared_kernel(
            target.context, target.gatype.derive.structural.derive.unit(merged_units), kernel,
        )


def _stacked_type(values: tuple[Extensor, ...]) -> GAType:
    gatypes = tuple(value.gatype for value in values)
    first = gatypes[0]
    if all(gatype == first for gatype in gatypes[1:]):
        return first
    if any(gatype.input_subspaces != first.input_subspaces for gatype in gatypes):
        raise GATypeError("a collection requires equal input axes")
    if any(gatype.fields != first.fields for gatype in gatypes):
        raise GATypeError("a collection requires equal fields")
    output = reduce(SubSpace.union, (gatype.output_subspace for gatype in gatypes))
    return rules.STACK.result(gatypes, (output,) + first.input_subspaces).derive.with_fields(first.fields)


def _collection_values(extensors: Iterable[Extensor]) -> tuple[Extensor, ...]:
    values = tuple(extensors)
    first = values[0]
    for value in values[1:]:
        if not first.context.is_compatible_with(value.context):
            raise ValueError("Extensor contexts are incompatible")
    return values


def _shape_arguments(arguments: tuple) -> tuple[int, ...]:
    if len(arguments) == 1 and isinstance(arguments[0], (tuple, list)):
        arguments = tuple(arguments[0])
    return tuple(integer_index(size) for size in arguments)


def _batch_axes(axis: int | tuple[int, ...] | None, ndim: int) -> tuple[int, ...]:
    if axis is None:
        return tuple(range(ndim))
    raw_axes = axis if isinstance(axis, tuple) else (axis,)
    return tuple(_existing_axis(item, ndim) for item in raw_axes)


def _existing_axis(axis: int, ndim: int) -> int:
    normalized = integer_index(axis)
    if normalized < 0:
        normalized += ndim
    if normalized < 0 or normalized >= ndim:
        raise IndexError(f"batch axis {axis} is out of range for ndim {ndim}")
    return normalized


def _insertion_axis(axis: int, ndim: int) -> int:
    normalized = integer_index(axis)
    if normalized < 0:
        normalized += ndim + 1
    if normalized < 0 or normalized > ndim:
        raise IndexError(f"stack axis {axis} is out of range for ndim {ndim}")
    return normalized


def _batch_kernel_index(
    index: object,
    batch_ndim: int,
    structural_ndim: int,
) -> tuple:
    items = list(index if isinstance(index, tuple) else (index,))
    ellipses = [position for position, item in enumerate(items) if item is Ellipsis]
    if len(ellipses) > 1:
        raise IndexError("a batch index may contain only one ellipsis")

    consumed = sum(_index_dimensions(item) for item in items if item is not Ellipsis)
    if consumed > batch_ndim:
        raise IndexError(
            f"batch index consumes {consumed} axes, but Extensor ndim is {batch_ndim}"
        )
    fill = [slice(None)] * (batch_ndim - consumed)
    if ellipses:
        position = ellipses[0]
        items[position : position + 1] = fill
    else:
        items.extend(fill)
    items.extend(slice(None) for _ in range(structural_ndim))
    return tuple(items)


def _index_dimensions(item: object) -> int:
    if item is None:
        return 0
    if isinstance(item, bool):
        raise TypeError("boolean scalar batch indices are not supported")
    dtype = getattr(item, "dtype", None)
    if getattr(dtype, "kind", None) == "b":
        return max(1, int(getattr(item, "ndim", 1)))
    return 1


def stack(extensors: Iterable[Extensor], axis: int = 0) -> Extensor:
    return Extensor.stack(extensors, axis=axis)


def concatenate(extensors: Iterable[Extensor], axis: int = 0) -> Extensor:
    return Extensor.concatenate(extensors, axis=axis)


def _branch(value: Extensor | int, other: Extensor) -> Extensor:
    if isinstance(value, Extensor):
        return value
    if value != 0:
        raise TypeError(f"where takes extensors or a plain zero, not {value!r}")
    return other.context.extensor(other.gatype.derive.structural, other.context.xp.zeros_like(other.kernel))


def where(condition, if_true: Extensor | int, if_false: Extensor | int) -> Extensor:
    """Elementwise selection avoids mask multiplication so undefined values (`0 / 0`) in the
    unchosen branch never propagate. On fields, `condition` ranges over batch axes followed by
    output sites (`depths > 0` selects site by site; `mask[..., None]` selects whole fields), and
    literal `0` stands for the zero of the other branch's type."""
    if_true, if_false = _branch(if_true, if_false), _branch(if_false, if_true)
    context = binding_context(if_true.context, (if_false.context,))
    values = (context.lower(if_true), context.lower(if_false))
    gatype = _stacked_type(values)
    kernels = tuple(_embed_kernel(context, value, gatype.subspaces) for value in values)
    mask = context.xp.asarray(condition)
    # Output site axes lead structural_shape; trailing mask axes must broadcast against them.
    sites = tuple(size for axis, size in gatype.fields if axis == 0)
    if mask.ndim < len(sites) or not all(size in (1, site)
                                         for size, site in zip(mask.shape[mask.ndim - len(sites):], sites)):
        raise ValueError(f"where on a field of sites {sites} takes a condition over the batch axes and the sites, "
                         f"not one of shape {mask.shape}")
    mask = context.xp.reshape(mask, mask.shape + (1,) * (len(gatype.structural_shape) - len(sites)))
    return Extensor._from_prepared_kernel(context, gatype, context.xp.where(mask, *kernels))
