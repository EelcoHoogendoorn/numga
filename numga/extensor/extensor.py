"""The single immutable Extensor value at every stage and arity."""

from __future__ import annotations

from functools import reduce
from operator import index as integer_index
from types import NotImplementedType
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping, Sequence, SupportsIndex

import numga
from numga.backend.context import binding_context
from numga.binding import AxisTransform, BindingPlan, TypeRules, normalize_bind_arguments, nullary_identity_groups
from numga.extension import ExtensionMethod
from numga.gatype import GAType, Orientation, Trait, rules
from numga.subspace import SubSpace

from . import operands

from .namespaces import RestrictNamespace, SelectNamespace, SitesNamespace
from .application import application, unary_application
from .fields import bind_fields

if TYPE_CHECKING:
    import numpy as np

    from numga.algebra import Algebra
    from numga.backend.context import Context


class Extensor:
    """Immutable exact or backend value representing every extensor arity."""

    __slots__ = ("_context", "_gatype", "_kernel")
    __hash__ = None
    # Arrays are coefficient storage, not operands implicitly converted into
    # object arrays of Extensors by NumPy's reflected arithmetic.
    __array_ufunc__ = None
    inverse = ExtensionMethod("inverse")
    norm_squared = ExtensionMethod("norm_squared", output_orientation=Orientation.extrinsic)
    norm = ExtensionMethod("norm", output_orientation=Orientation.extrinsic)
    normalized = ExtensionMethod("normalized")
    square_root = ExtensionMethod("square_root", input_orientation=Orientation.extrinsic)
    inverse_square_root = ExtensionMethod("inverse_square_root", input_orientation=Orientation.extrinsic)
    study_norm_squared = ExtensionMethod("study_norm_squared", output_orientation=Orientation.extrinsic)
    study_norm = ExtensionMethod("study_norm", output_orientation=Orientation.extrinsic)
    exp = ExtensionMethod("exp", input_orientation=Orientation.extrinsic)
    log = ExtensionMethod("log", input_orientation=Orientation.extrinsic)
    exp_linear = ExtensionMethod("exp_linear", input_orientation=Orientation.extrinsic)
    exp_linear_normalized = ExtensionMethod("exp_linear_normalized", input_orientation=Orientation.extrinsic)
    exp_quadratic = ExtensionMethod("exp_quadratic", input_orientation=Orientation.extrinsic)
    exp_cayley = ExtensionMethod("exp_cayley", input_orientation=Orientation.extrinsic)
    exp_bisect = ExtensionMethod("exp_bisect", input_orientation=Orientation.extrinsic)
    exp_derivative = ExtensionMethod("exp_derivative")
    log_linear = ExtensionMethod("log_linear", input_orientation=Orientation.extrinsic)
    log_linear_normalized = ExtensionMethod("log_linear_normalized", input_orientation=Orientation.extrinsic)
    log_quadratic = ExtensionMethod("log_quadratic", input_orientation=Orientation.extrinsic)
    log_pade = ExtensionMethod("log_pade", input_orientation=Orientation.extrinsic)
    square_root_denman_beavers = ExtensionMethod("square_root_denman_beavers", input_orientation=Orientation.extrinsic)
    geometric_mean = ExtensionMethod("geometric_mean")
    decompose_polar = ExtensionMethod("decompose_polar")
    decompose_invariant = ExtensionMethod("decompose_invariant")
    motor_translator = ExtensionMethod("motor_translator")
    motor_rotor = ExtensionMethod("motor_rotor")
    motor_split = ExtensionMethod("motor_split")
    inverse_shirokov = ExtensionMethod("inverse_shirokov")
    inverse_factor = ExtensionMethod("inverse_factor")
    inverse_hitzer = ExtensionMethod("inverse_hitzer")
    trace = ExtensionMethod("trace", operand_counts=(1, 3))
    contract = ExtensionMethod("contract", operand_counts=(1, 3))
    outermorphism = ExtensionMethod("outermorphism")
    det = ExtensionMethod("det", operand_counts=(1, 2))
    real = ExtensionMethod("real")
    solve = ExtensionMethod("solve")
    lstsq = ExtensionMethod("lstsq", operand_counts=(2, 3))
    pinv = ExtensionMethod("pinv")
    adjugate = ExtensionMethod("adjugate")
    adjoint = ExtensionMethod("adjoint")
    cholesky = ExtensionMethod("cholesky")
    eig = ExtensionMethod("eig", operand_counts=(1, 2))
    eigvals = ExtensionMethod("eigvals", operand_counts=(1, 2))
    eigh = ExtensionMethod("eigh", operand_counts=(1, 2))
    eigvalsh = ExtensionMethod("eigvalsh", operand_counts=(1, 2))
    svd = ExtensionMethod("svd")
    svdvals = ExtensionMethod("svdvals")
    sin = ExtensionMethod("sin")
    cos = ExtensionMethod("cos", output_orientation=Orientation.extrinsic)
    tan = ExtensionMethod("tan")
    arcsin = ExtensionMethod("arcsin")
    arccos = ExtensionMethod("arccos", input_orientation=Orientation.extrinsic)
    arctan = ExtensionMethod("arctan")
    sinh = ExtensionMethod("sinh")
    cosh = ExtensionMethod("cosh", output_orientation=Orientation.extrinsic)
    tanh = ExtensionMethod("tanh")
    arcsinh = ExtensionMethod("arcsinh")
    arccosh = ExtensionMethod("arccosh", input_orientation=Orientation.extrinsic)
    arctanh = ExtensionMethod("arctanh")
    clip = ExtensionMethod("clip")
    isnan = ExtensionMethod("isnan")
    isfinite = ExtensionMethod("isfinite")
    isinf = ExtensionMethod("isinf")
    argsort = ExtensionMethod("argsort")
    argmax = ExtensionMethod("argmax")
    argmin = ExtensionMethod("argmin")
    abs = ExtensionMethod("abs")
    less = ExtensionMethod("less")
    less_equal = ExtensionMethod("less_equal")
    greater = ExtensionMethod("greater")
    greater_equal = ExtensionMethod("greater_equal")
    to_array = ExtensionMethod("to_array")
    add = ExtensionMethod("add")
    subtract = ExtensionMethod("subtract")
    divide = ExtensionMethod("divide")
    power = ExtensionMethod("power")
    reverse = ExtensionMethod("reverse")
    clifford_conjugate = ExtensionMethod("clifford_conjugate")
    involute = ExtensionMethod("involute")
    scalar_negation = ExtensionMethod("scalar_negation")
    pseudoscalar_negation = ExtensionMethod("pseudoscalar_negation")

    def __init__(self, context: Context, gatype: GAType, kernel: Any) -> None:
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
        context: "Context",
        gatype: GAType,
        kernel: Any,
    ) -> "Extensor":
        """Wrap an internal result without coercion, copies, or validation."""

        self = cls.__new__(cls)
        self._context = context
        self._gatype = gatype
        self._kernel = kernel
        return self

    @property
    def context(self) -> "Context":
        return self._context

    @property
    def gatype(self) -> GAType:
        return self._gatype

    @property
    def kernel(self) -> Any:
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
        """The nullary type of what this extensor produces, with its orientation: for an expression in
        open types, the type its construction gives the result."""
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
        """The batch axes, in front of the site axes of any field slots and the coefficient axes."""
        return self._kernel.shape[: -len(self.gatype.structural_shape)]

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def dtype(self) -> np.dtype:
        return self.context.dtype

    def __getitem__(self, index: object) -> "Extensor":
        kernel_index = _batch_kernel_index(index, self.ndim, len(self.structural_shape))
        return type(self)._from_prepared_kernel(self.context, self.gatype, self._kernel[kernel_index])

    def __iter__(self):
        return (self[index] for index in range(self.shape[0]))

    def __len__(self) -> int:
        """The length of the first batch axis; an unbatched extensor has none."""
        if not self.shape:
            raise TypeError("len() of an unbatched Extensor")
        return self.shape[0]

    def __invert__(self) -> Extensor:
        return self.reverse()

    def formula(self) -> str:
        """Expand an exact extensor into equations for its blade coefficients."""
        from numga.operator.format import formula
        return formula(self)

    def coefficient_table(self) -> str:
        """Print-ready blade-labelled coefficients of an unbatched multivector or linear map."""
        from numga.operator.format import coefficient_table
        return coefficient_table(self)

    def to_python(self, name: str = "apply") -> str:
        """Generate standalone Python for an exact extensor's coefficient map."""
        from numga.operator.format import python_code
        return python_code(self, name)

    @property
    def at(self) -> "_AtIndexer":
        return _AtIndexer(self)

    def map_kernel(
        self, function: Callable[..., Any], *args: Any,
        preserve_traits: bool = False, **kwargs: Any,
    ) -> Extensor:
        """Apply a storage operation and wrap its result without validation or copying.

        The callable must not mutate the input and must return storage suitable
        for this context, retaining the trailing structural axes and their
        coordinate meaning. Batch axes may change. Explicit traits are dropped
        unless the caller asserts that this operation preserves them.
        """

        kernel = function(self._kernel, *args, **kwargs)
        gatype = self.gatype if preserve_traits else self.gatype.derive.structural
        return type(self)._from_prepared_kernel(self.context, gatype, kernel)

    def reshape(self, *shape: SupportsIndex | Sequence[SupportsIndex]) -> Extensor:
        batch_shape = _shape_arguments(shape)
        kernel = self.context.xp.reshape(
            self._kernel,
            batch_shape + self.structural_shape,
        )
        return type(self)._from_prepared_kernel(self.context, self.gatype, kernel)

    def broadcast_to(self, shape: SupportsIndex | Sequence[SupportsIndex]) -> Extensor:
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
    ) -> "Extensor":
        axes = _batch_axes(axis, self.ndim)
        kernel = self.context.xp.sum(self._kernel, axis=axes, keepdims=keepdims)
        return type(self)._from_prepared_kernel(self.context, self.gatype.derive.structural, kernel)

    def cumsum(self, axis: int) -> "Extensor":
        """Running sums along one batch axis."""
        kernel = self.context.xp.cumsum(self._kernel, axis=_existing_axis(axis, self.ndim))
        return type(self)._from_prepared_kernel(self.context, self.gatype.derive.structural, kernel)

    def cumprod(self, axis: int) -> "Extensor":
        """Running geometric products along one batch axis, each later element multiplying from the
        left: `x0`, `x1 * x0`, `x2 * x1 * x0`, so a running product of rotors applies them in turn.
        The type must be closed under the product, so that every running product has it."""
        square = self.algebra.operator.geometric_product(self.gatype, self.gatype)
        if self.arity or not square.output_subspace.support_is_subset_of(self.output_subspace):
            raise TypeError(f"cumprod needs a type closed under the geometric product; {self.gatype} is not")
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
    ) -> "Extensor":
        axes = _batch_axes(axis, self.ndim)
        kernel = self.context.xp.mean(self._kernel, axis=axes, keepdims=keepdims)
        return type(self)._from_prepared_kernel(self.context, self.gatype.derive.structural, kernel)

    @classmethod
    def stack(
        cls,
        extensors: Iterable[Extensor],
        axis: int = 0,
    ) -> "Extensor":
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
    ) -> "Extensor":
        values = _collection_values(extensors)
        first = values[0]
        gatype = _stacked_type(values)
        logical_axis = _existing_axis(axis, first.ndim)
        kernel = first.context.xp.concatenate(
            tuple(_embed_kernel(first.context, value, gatype.subspaces) for value in values),
            axis=logical_axis,
        )
        return cls._from_prepared_kernel(first.context, gatype, kernel)

    def field(self, *slots: int) -> Extensor:
        """The last batch axes read as the sites of slots, the output 0 and the inputs from 1, one axis
        per slot in slot order: `values.field()` is a field over the last batch axis, and
        `pair_maps.field(0, 1)` a field map from a batch of maps over pairs of an output and an input
        site. A field is a collection whose elements are coupled, where a batch holds independent
        copies. The coefficients are untouched, so slots that already range over sites must come after
        these."""
        slots = slots or (0,)
        if self.context.is_exact or self.ndim < len(slots):
            raise TypeError("a field takes its sites from the last batch axes of a backend value")
        if list(slots) != sorted(set(slots)) or any(field <= slots[-1] for field, _ in self.gatype.fields):
            raise TypeError(f"slots {slots} of {self.gatype.signature} are not in order before its field slots")
        # Their site axes come first among the site axes, where the last batch axes already sit.
        fields = tuple(zip(slots, self.shape[self.ndim - len(slots):])) + self.gatype.fields
        return type(self)._from_prepared_kernel(self.context, self.gatype.derive.with_fields(fields), self._kernel)

    def on_diagonal(self) -> Extensor:
        """A field of maps placed on the site diagonal: from `X[n] <- Y` the field map `X[n] <- Y[n]`
        that acts on the element at each site by the map at that site, and on no other."""
        if self.arity < 1 or [slot for slot, _ in self.gatype.fields] != [0]:
            raise TypeError(f"on_diagonal places a field of maps, X[n] <- Y, on its sites; {self.gatype.signature} is not one")
        xp, sites = self.context.xp, self.gatype.site_shape[0]
        identity = xp.reshape(xp.eye(sites, dtype=self._kernel.dtype), (sites, sites) + (1,) * len(self.axes))
        kernel = xp.expand_dims(self._kernel, self.ndim + 1) * identity
        fields = ((0, sites), (1, sites))
        return type(self)._from_prepared_kernel(self.context, self.gatype.derive.structural.derive.with_fields(fields), kernel)

    def batch(self) -> Extensor:
        """Every site axis read as a batch axis again, the coefficients untouched."""
        return type(self)._from_prepared_kernel(self.context, self.gatype.derive.plain, self._kernel)

    def bind(self, *args: Extensor | Mapping[int, Extensor]) -> Extensor:
        raw_operands = normalize_bind_arguments(self.arity, args)
        if not raw_operands:
            return self
        if self._gatype.has_fields or any(operand.gatype.has_fields for operand in raw_operands.values()):
            return bind_fields(self, raw_operands)

        plan = BindingPlan.build(self.gatype, raw_operands)
        operands: dict[int, Extensor] = {}
        prepared_operands: dict[int, Extensor] = {}
        context = binding_context(
            self.context, tuple(raw_operands[slot].context for slot in plan.slots),
        )
        target = self
        output = plan.result_subspaces[0]
        if output is not self.output_subspace:
            # Remove statically impossible rows before lowering or contraction.
            indices = plan.output_indices
            kernel = (
                self._kernel.take(indices, axis=0)
                if self.context.is_exact else self.context.xp.take(
                    self._kernel, self.context.xp.asarray(indices, dtype=int), axis=self.ndim,
                )
            )
            target = type(self)._from_prepared_kernel(self.context, plan.execution_gatype, kernel)
        for slot, operand in raw_operands.items():
            identity = id(operand)
            concrete = prepared_operands.get(identity)
            if concrete is None:
                concrete = context.lower(operand)
                # Lower each distinct argument once per atomic bind. Besides
                # avoiding duplicate work, this retains same-object evidence.
                prepared_operands[identity] = concrete
            operands[slot] = concrete

        result_gatype = TypeRules.bind(plan)
        result = context.execute_bind(target, operands, plan)
        return type(self)._from_prepared_kernel(context, result_gatype, result)

    def __call__(self, *operands: Extensor | GAType) -> Extensor:
        """Apply to operands in slot order. Fewer operands than slots bind the leading slots. A bare
        type is its identity map: the slot's own type leaves that slot open, form(Vector, v) binds
        only the second, and a type inside the slot narrows it to that type."""
        if any(isinstance(operand, (GAType, SubSpace)) for operand in operands):
            bound = {}
            for slot, operand in enumerate(operands):
                if not isinstance(operand, (GAType, SubSpace)):
                    bound[slot] = operand
                elif (narrowed := self._gatype.slots.narrowing(slot, operand)) is not None:
                    bound[slot] = narrowed
            return self.bind(bound)
        if len(operands) < self.arity:
            return self.bind(*operands)
        if len(operands) == 1:
            operand = operands[0]
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

    def squeeze_output(self) -> "Extensor":
        """Drop output coordinates proved zero by an exact kernel."""

        if not self.context.is_exact:
            raise TypeError("output squeezing requires an exact Extensor")
        indices = self.kernel.output_nonzero_indices()
        output = self.output_subspace.restrict(
            tuple(self.output_subspace.masks[index] for index in indices),
        )
        gatype = rules.SQUEEZE.result((self.gatype,), (output,) + self.input_subspaces)
        return type(self)._from_prepared_kernel(self.context, gatype, self.kernel.take(indices, axis=0))

    def materialize(self, context: "Context") -> "Extensor":
        return context.lower(self)

    def cast(self, target: SubSpace | GAType) -> "Extensor":
        """Explicitly project or embed every output into a subspace, or into a type's output subspace.

        A type's traits are not asserted: projecting onto rotor coefficients does not make a rotor.
        """

        target_subspace = target.output_subspace if isinstance(target, GAType) else target
        if isinstance(target, GAType) and target.orientation is not self.gatype.orientation:
            raise ValueError(
                f"a cast changes layout, not orientation: {self.gatype.signature} cannot become {target.signature}"
            )
        if self.output_subspace is target_subspace:
            return self
        return self.algebra.operator.cast(
            self.gatype.derive.output, target_subspace,
        ).bind({0: self})

    @property
    def select(self) -> SelectNamespace:
        """Select exact output support by grade or SubSpace constructor name."""

        return SelectNamespace(self)

    @property
    def sites(self) -> SitesNamespace:
        """Reductions over the sites of a field's output: `energy.sites.sum()`, `positions.sites.mean()`."""

        return SitesNamespace(self)

    @property
    def restrict(self) -> RestrictNamespace:
        """Intersect static output support by grade or constructor name."""

        return RestrictNamespace(self)

    def select_subspace(self, subspace: SubSpace) -> "Extensor":
        """Select every requested output blade, filling missing blades with zero."""

        return self.cast(subspace)

    def restrict_subspace(self, subspace: SubSpace) -> "Extensor":
        """Keep only output blades present in both static SubSpaces."""

        return self.algebra.operator.restrict(
            self.gatype.derive.output, subspace,
        ).bind({0: self})

    def select_grade(self, grade: int) -> "Extensor":
        return self.select_subspace(self.algebra.subspace.k_vector(grade))

    def restrict_grade(self, grade: int) -> "Extensor":
        return self.restrict_subspace(self.algebra.subspace.k_vector(grade))

    def with_traits(self, *traits: Trait) -> Extensor:
        """Trust additional immutable facts without changing coefficients."""

        gatype = self.gatype.derive.with_traits(*traits)
        if gatype is self.gatype:
            return self
        return type(self)._from_prepared_kernel(self.context, gatype, self._kernel)

    def _symmetric_product(
        self, transform: str, *, scalar_only: bool = False,
    ) -> "Extensor":
        """Measure coefficients on the carrier proved by this whole-value type."""

        operator = self.algebra.operator._symmetric_product(
            self.gatype.derive.plain, transform, scalar_only=scalar_only,
        )
        return operator(self, self)

    def squared(self) -> "Extensor":
        """Evaluate this value's square with structural cancellations combined."""

        return self._symmetric_product("identity")

    def symmetric_reverse_product(self) -> "Extensor":
        """Measure ``x * reverse(x)``, never substituting one for coefficients."""

        return self._symmetric_product("reverse")

    def scalar_norm_squared(self) -> "Extensor":
        """The scalar part of ``x * reverse(x)``, with only that part built."""

        return self._symmetric_product("reverse", scalar_only=True)

    def symmetric_conjugate_product(self) -> "Extensor":
        return self._symmetric_product("clifford_conjugate")

    def symmetric_scalar_negation_product(self) -> "Extensor":
        return self._symmetric_product("scalar_negation")

    def symmetric_pseudoscalar_negation_product(self) -> "Extensor":
        return self._symmetric_product("pseudoscalar_negation")

    def symmetric_involute_product(self) -> "Extensor":
        return self._symmetric_product("involute")

    def dual(self) -> Extensor:
        return self.algebra.operator.dual(self.gatype.derive.plain.derive.output)(self)

    def dual_inverse(self) -> Extensor:
        return self.algebra.operator.dual_inverse(self.gatype.derive.plain.derive.output)(self)

    def __pos__(self) -> "Extensor":
        return self

    def __neg__(self) -> "Extensor":
        gatype = rules.NEGATE.result((self.gatype,), self.gatype.subspaces).derive.with_fields(self.gatype.fields)
        return type(self)._from_prepared_kernel(self.context, gatype, -self._kernel)

    def __lt__(self, other: object) -> Any:
        return self.less(operands.compared(self._context, other))

    def __le__(self, other: object) -> Any:
        return self.less_equal(operands.compared(self._context, other))

    def __gt__(self, other: object) -> Any:
        return self.greater(operands.compared(self._context, other))

    def __ge__(self, other: object) -> Any:
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

    def __mul__(self, scalar: object) -> Extensor:
        if numga.expression.is_expression_operand(scalar):
            return numga.expression.geometric_product(self, scalar)
        found = operands.kind(scalar)
        if found is operands.Kind.ARRAY:
            return numga.expression.geometric_product(self, operands.batch(self._context, scalar))
        if found is not operands.Kind.NUMBER:
            return NotImplemented
        scalar = self.context.prepare_scalar(scalar)
        gatype = rules.SCALE.result((self.gatype,), self.gatype.subspaces).derive.with_fields(self.gatype.fields)
        return type(self)._from_prepared_kernel(self.context, gatype, self._kernel * scalar)

    def __rmul__(self, scalar: object) -> Extensor:
        if numga.expression.is_expression_operand(scalar):
            return numga.expression.geometric_product(scalar, self)
        return self * scalar

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

    def wedge(self, other: Extensor | GAType | SubSpace) -> Extensor:
        return numga.expression.wedge(self, other)

    def scalar_product(self, other: Extensor | GAType | SubSpace) -> Extensor:
        return numga.expression.scalar_product(self, other)

    def inner(self, other: Extensor | GAType | SubSpace) -> Extensor:
        return numga.expression.inner(self, other)

    def bivector_product(self, other: Extensor | GAType | SubSpace) -> Extensor:
        return numga.expression.bivector_product(self, other)

    def trivector_product(self, other: Extensor | GAType | SubSpace) -> Extensor:
        return numga.expression.trivector_product(self, other)

    def commutator(self, other: Extensor | GAType | SubSpace) -> Extensor:
        return numga.expression.commutator(self, other)

    def anticommutator(self, other: Extensor | GAType | SubSpace) -> Extensor:
        return numga.expression.anticommutator(self, other)

    def left_contraction(self, other: Extensor | GAType | SubSpace) -> Extensor:
        return numga.expression.left_contraction(self, other)

    def right_contraction(self, other: Extensor | GAType | SubSpace) -> Extensor:
        return numga.expression.right_contraction(self, other)

    def left_interior(self, other: Extensor | GAType | SubSpace) -> Extensor:
        return numga.expression.left_interior(self, other)

    def right_interior(self, other: Extensor | GAType | SubSpace) -> Extensor:
        return numga.expression.right_interior(self, other)

    def regressive(self, other: Extensor | GAType | SubSpace) -> Extensor:
        return numga.expression.regressive(self, other)

    def transform(self, passenger: Extensor | GAType | SubSpace) -> Extensor:
        """The versor's action on the passenger, `self >> passenger`: the composition of the reflections
        in its vector factors. A reflection in a unit vector `m` takes an extrinsic `x` to
        `m * x.involute() * m.inverse()`, and an intrinsic `x` to its negative."""
        return numga.expression.transform(self, passenger)

    def reverse_transform(self, passenger: Extensor | GAType | SubSpace) -> Extensor:
        """The action of the versor's reverse, `self << passenger`: defined for every value, and for a
        unit versor the inverse of `self >> passenger`."""
        return numga.expression.reverse_transform(self, passenger)

    def __rshift__(self, other: Extensor | GAType | SubSpace) -> Extensor:
        return self.transform(other)

    def __lshift__(self, other: Extensor | GAType | SubSpace) -> Extensor:
        return self.reverse_transform(other)

    # Operands numga does not know, a sparse extensor among them, get their reflected operator.
    def __xor__(self, other: Extensor | GAType | SubSpace) -> Extensor:
        return self.wedge(other) if numga.expression.is_expression_operand(other) else NotImplemented

    def __rxor__(self, other: Extensor | GAType | SubSpace) -> Extensor:
        return numga.expression.wedge(other, self) if numga.expression.is_expression_operand(other) else NotImplemented

    def __and__(self, other: Extensor | GAType | SubSpace) -> Extensor:
        return self.regressive(other) if numga.expression.is_expression_operand(other) else NotImplemented

    def __rand__(self, other: Extensor | GAType | SubSpace) -> Extensor:
        return numga.expression.regressive(other, self) if numga.expression.is_expression_operand(other) else NotImplemented

    def __or__(self, other: Extensor | GAType | SubSpace) -> Extensor:
        return self.inner(other) if numga.expression.is_expression_operand(other) else NotImplemented

    def __ror__(self, other: Extensor | GAType | SubSpace) -> Extensor:
        return numga.expression.inner(other, self) if numga.expression.is_expression_operand(other) else NotImplemented

    def _require_compatible(self, other: "Extensor") -> None:
        if not self.context.is_compatible_with(other.context):
            raise ValueError("Extensor contexts are incompatible")
        if self.gatype != other.gatype:
            raise ValueError("Extensor GATypes differ")

    def __repr__(self) -> str:
        return f"Extensor({self.gatype.signature}, shape={self.shape})"


class _AtIndexer:
    __slots__ = ("_extensor",)

    def __init__(self, extensor: Extensor) -> None:
        self._extensor = extensor

    def __getitem__(self, index: object) -> "_AtUpdate":
        return _AtUpdate(self._extensor, index)


def _embed_kernel(
    context: "Context",
    value: Extensor,
    target_axes: tuple[SubSpace, ...],
) -> Any:
    """Zero-embed every structural axis into the addition result axes."""

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
        """Add the value at the index; where the index repeats, the additions accumulate."""
        return self._update(self._extensor.context.functional_add, value)

    def _update(self, update: Callable[[Any, object, object], Any], value: object) -> Extensor:
        target = self._extensor
        kernel_index = _batch_kernel_index(
            self._index,
            target.ndim,
            len(target.structural_shape),
        )
        if isinstance(value, Extensor):
            target._require_compatible(value)
            replacement = value._kernel
        else:
            replacement = value
        # The new elements need not share the target's traits.
        kernel = update(target._kernel, kernel_index, replacement)
        return type(target)._from_prepared_kernel(target.context, target.gatype.derive.structural, kernel)


def _stacked_type(values: tuple[Extensor, ...]) -> GAType:
    """The type of values set side by side along a batch axis: each structural output axis the union
    of theirs, zero-embedded."""
    gatypes = tuple(value.gatype for value in values)
    first = gatypes[0]
    if all(gatype == first for gatype in gatypes[1:]):
        return first
    if any(gatype.input_subspaces != first.input_subspaces for gatype in gatypes):
        raise ValueError("a collection requires equal input axes")
    if any(gatype.fields != first.fields for gatype in gatypes):
        raise ValueError("a collection requires equal fields")
    output = reduce(SubSpace.union, (gatype.output_subspace for gatype in gatypes))
    return rules.STACK.result(gatypes, (output,) + first.input_subspaces).derive.with_fields(first.fields)


def _collection_values(extensors: Iterable[Extensor]) -> tuple[Extensor, ...]:
    values = tuple(extensors)
    first = values[0]
    for value in values[1:]:
        if not first.context.is_compatible_with(value.context):
            raise ValueError("Extensor contexts are incompatible")
    return values


def _shape_arguments(
    arguments: tuple[SupportsIndex | Sequence[SupportsIndex], ...],
) -> tuple[int, ...]:
    if len(arguments) == 1 and isinstance(arguments[0], (tuple, list)):
        arguments = tuple(arguments[0])
    return tuple(integer_index(size) for size in arguments)


def _batch_axes(
    axis: int | tuple[int, ...] | None,
    ndim: int,
) -> tuple[int, ...]:
    if axis is None:
        return tuple(range(ndim))
    raw_axes = axis if isinstance(axis, tuple) else (axis,)
    return tuple(_existing_axis(item, ndim) for item in raw_axes)


def _existing_axis(axis: SupportsIndex, ndim: int) -> int:
    normalized = integer_index(axis)
    if normalized < 0:
        normalized += ndim
    if normalized < 0 or normalized >= ndim:
        raise IndexError(f"batch axis {axis} is out of range for ndim {ndim}")
    return normalized


def _insertion_axis(axis: SupportsIndex, ndim: int) -> int:
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
) -> tuple[object, ...]:
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
