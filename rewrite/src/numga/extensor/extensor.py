"""The single immutable Extensor value at every stage and arity."""

from __future__ import annotations

from numbers import Number
from operator import index as integer_index
from types import NotImplementedType
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping, Sequence, SupportsIndex

from numga.backend.context import binding_context
from numga.binding import AxisTransform, BindingPlan, TypeRules, normalize_bind_arguments, nullary_identity_groups
from numga.extension import ExtensionMethod
from numga.gatype import GAType, Trait
from numga.subspace import SubSpace

from .namespaces import RestrictNamespace, SelectNamespace
from .application import application, unary_application

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
    norm_squared = ExtensionMethod("norm_squared")
    norm = ExtensionMethod("norm")
    normalized = ExtensionMethod("normalized")
    square_root = ExtensionMethod("square_root")
    inverse_square_root = ExtensionMethod("inverse_square_root")
    study_norm_squared = ExtensionMethod("study_norm_squared")
    study_norm = ExtensionMethod("study_norm")
    exp = ExtensionMethod("exp")
    log = ExtensionMethod("log")
    exp_linear = ExtensionMethod("exp_linear")
    exp_linear_normalized = ExtensionMethod("exp_linear_normalized")
    exp_quadratic = ExtensionMethod("exp_quadratic")
    exp_cayley = ExtensionMethod("exp_cayley")
    exp_bisect = ExtensionMethod("exp_bisect")
    log_linear = ExtensionMethod("log_linear")
    log_linear_normalized = ExtensionMethod("log_linear_normalized")
    log_quadratic = ExtensionMethod("log_quadratic")
    log_pade = ExtensionMethod("log_pade")
    square_root_denman_beavers = ExtensionMethod("square_root_denman_beavers")
    geometric_mean = ExtensionMethod("geometric_mean")
    decompose_polar = ExtensionMethod("decompose_polar")
    decompose_invariant = ExtensionMethod("decompose_invariant")
    motor_translator = ExtensionMethod("motor_translator")
    motor_rotor = ExtensionMethod("motor_rotor")
    motor_split = ExtensionMethod("motor_split")
    inverse_shirokov = ExtensionMethod("inverse_shirokov")
    inverse_factor = ExtensionMethod("inverse_factor")
    inverse_hitzer = ExtensionMethod("inverse_hitzer")
    trace = ExtensionMethod("trace")
    transpose = ExtensionMethod("transpose")
    det = ExtensionMethod("det")
    solve = ExtensionMethod("solve")
    lstsq = ExtensionMethod("lstsq")
    pinv = ExtensionMethod("pinv")
    cholesky = ExtensionMethod("cholesky")
    eig = ExtensionMethod("eig", operand_counts=(1, 2))
    eigvals = ExtensionMethod("eigvals", operand_counts=(1, 2))
    eigh = ExtensionMethod("eigh", operand_counts=(1, 2))
    eigvalsh = ExtensionMethod("eigvalsh", operand_counts=(1, 2))
    svd = ExtensionMethod("svd")
    svdvals = ExtensionMethod("svdvals")
    sin = ExtensionMethod("sin")
    cos = ExtensionMethod("cos")
    tan = ExtensionMethod("tan")
    arcsin = ExtensionMethod("arcsin")
    arccos = ExtensionMethod("arccos")
    arctan = ExtensionMethod("arctan")
    sinh = ExtensionMethod("sinh")
    cosh = ExtensionMethod("cosh")
    tanh = ExtensionMethod("tanh")
    arcsinh = ExtensionMethod("arcsinh")
    arccosh = ExtensionMethod("arccosh")
    arctanh = ExtensionMethod("arctanh")
    clip = ExtensionMethod("clip")
    isnan = ExtensionMethod("isnan")
    isfinite = ExtensionMethod("isfinite")
    isinf = ExtensionMethod("isinf")
    argsort = ExtensionMethod("argsort")
    argmax = ExtensionMethod("argmax")
    less = ExtensionMethod("less")
    less_equal = ExtensionMethod("less_equal")
    greater = ExtensionMethod("greater")
    greater_equal = ExtensionMethod("greater_equal")
    to_array = ExtensionMethod("to_array")

    def __init__(self, context: Context, gatype: GAType, kernel: Any) -> None:
        kernel = context.prepare_kernel(kernel)
        if gatype.algebra is not context.algebra:
            raise ValueError("GAType and Context belong to different algebras")
        if kernel.shape[-len(gatype.subspaces):] != gatype.structural_shape:
            raise ValueError(
                f"kernel trailing shape does not match GAType structural shape {gatype.structural_shape}"
            )
        if context.is_exact and kernel.shape != gatype.structural_shape:
            raise ValueError("exact Extensors cannot carry batch dimensions")
        object.__setattr__(self, "_context", context)
        object.__setattr__(self, "_gatype", gatype)
        object.__setattr__(self, "_kernel", kernel)

    @classmethod
    def _from_prepared_kernel(
        cls,
        context: "Context",
        gatype: GAType,
        kernel: Any,
    ) -> "Extensor":
        """Wrap an internal result without coercion, copies, or validation."""

        self = cls.__new__(cls)
        object.__setattr__(self, "_context", context)
        object.__setattr__(self, "_gatype", gatype)
        object.__setattr__(self, "_kernel", context.freeze_kernel(kernel))
        return self

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("Extensor instances are immutable")

    @property
    def context(self) -> "Context":
        return self._context

    @property
    def gatype(self) -> GAType:
        return self._gatype

    @property
    def kernel(self) -> Any:
        return self.context.expose_kernel(self._kernel)

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
        return self._kernel.shape[: -len(self.axes)]

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def dtype(self) -> np.dtype:
        return self.context.dtype

    def __getitem__(self, index: object) -> "Extensor":
        kernel_index = _batch_kernel_index(index, self.ndim, len(self.axes))
        return type(self)._from_prepared_kernel(self.context, self.gatype, self._kernel[kernel_index])

    def __iter__(self):
        return (self[index] for index in range(self.shape[0]))

    def __invert__(self) -> Extensor:
        return self.reverse()

    def formula(self) -> str:
        """Expand an exact extensor into equations for its blade coefficients."""
        from numga.operator.format import formula
        return formula(self)

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
        gatype = self.gatype if preserve_traits else self.gatype.structural
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
        return type(self)._from_prepared_kernel(self.context, self.gatype.structural, kernel)

    def mean(
        self,
        axis: int | tuple[int, ...] | None = None,
        *,
        keepdims: bool = False,
    ) -> "Extensor":
        axes = _batch_axes(axis, self.ndim)
        kernel = self.context.xp.mean(self._kernel, axis=axes, keepdims=keepdims)
        return type(self)._from_prepared_kernel(self.context, self.gatype.structural, kernel)

    @classmethod
    def stack(
        cls,
        extensors: Iterable[Extensor],
        axis: int = 0,
    ) -> "Extensor":
        values = _collection_values(extensors)
        first = values[0]
        gatype = TypeRules.collection("stack", axis, tuple(value.gatype for value in values))
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
        gatype = TypeRules.collection("concatenate", axis, tuple(value.gatype for value in values))
        logical_axis = _existing_axis(axis, first.ndim)
        kernel = first.context.xp.concatenate(
            tuple(_embed_kernel(first.context, value, gatype.subspaces) for value in values),
            axis=logical_axis,
        )
        return cls._from_prepared_kernel(first.context, gatype, kernel)

    def bind(self, *args: Extensor | Mapping[int, Extensor]) -> Extensor:
        raw_operands = normalize_bind_arguments(self.arity, args)
        if not raw_operands:
            return self

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

    def __call__(self, *operands: Extensor) -> Extensor:
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
        gatype = TypeRules.operation(
            "squeeze_output",
            (self.gatype,),
            (output,) + self.input_subspaces,
        )
        return type(self)._from_prepared_kernel(self.context, gatype, self.kernel.take(indices, axis=0))

    def materialize(self, context: "Context") -> "Extensor":
        return context.lower(self)

    def cast(self, target_subspace: SubSpace) -> "Extensor":
        """Explicitly project or embed every output into ``target_subspace``."""

        if self.output_subspace is target_subspace:
            return self
        return self.algebra.operator.cast(
            self.output_subspace, target_subspace,
        ).bind({0: self})

    @property
    def select(self) -> SelectNamespace:
        """Select exact output support by grade or SubSpace constructor name."""

        return SelectNamespace(self)

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
            self.output_subspace, subspace,
        ).bind({0: self})

    def select_grade(self, grade: int) -> "Extensor":
        return self.select_subspace(self.algebra.subspace.k_vector(grade))

    def restrict_grade(self, grade: int) -> "Extensor":
        return self.restrict_subspace(self.algebra.subspace.k_vector(grade))

    def with_traits(self, *traits: Trait) -> Extensor:
        """Trust additional immutable facts without changing coefficients."""

        gatype = self.gatype.with_traits(*traits)
        if gatype is self.gatype:
            return self
        return type(self)._from_prepared_kernel(self.context, gatype, self._kernel)

    def _symmetric_product(
        self, transform: str, *, scalar_only: bool = False,
    ) -> "Extensor":
        """Measure coefficients on the carrier proved by this whole-value type."""

        operator = self.algebra.operator._symmetric_product(
            self.gatype, transform, scalar_only=scalar_only,
        )
        return operator(self, self)

    def squared(self) -> "Extensor":
        """Evaluate this value's square with structural cancellations combined."""

        return self._symmetric_product("identity")

    def symmetric_reverse_product(self) -> "Extensor":
        """Measure ``x * reverse(x)``, never substituting one for coefficients."""

        return self._symmetric_product("reverse")

    def symmetric_conjugate_product(self) -> "Extensor":
        return self._symmetric_product("clifford_conjugate")

    def symmetric_scalar_negation_product(self) -> "Extensor":
        return self._symmetric_product("scalar_negation")

    def symmetric_pseudoscalar_negation_product(self) -> "Extensor":
        return self._symmetric_product("pseudoscalar_negation")

    def symmetric_involute_product(self) -> "Extensor":
        return self._symmetric_product("involute")

    def _grade_transform(self, transform: str) -> "Extensor":
        if self.gatype.grade_transform_is_identity(transform):
            return self
        result = self.algebra.operator._grade_transform(
            self.output_subspace, transform,
        ).bind({0: self})
        gatype = TypeRules.operation(
            transform,
            (self.gatype,),
            result.axes,
        )
        if result.gatype is gatype:
            return result
        return type(self)._from_prepared_kernel(result.context, gatype, result._kernel)

    def reverse(self) -> "Extensor":
        """Reverse every output multivector of this Extensor."""

        return self._grade_transform("reverse")

    def dual(self) -> Extensor:
        return self.algebra.operator.dual(self.subspace)(self)

    def dual_inverse(self) -> Extensor:
        return self.algebra.operator.dual_inverse(self.subspace)(self)

    def clifford_conjugate(self) -> "Extensor":
        """Clifford-conjugate outputs, leaving complex coefficients unchanged."""

        return self._grade_transform("clifford_conjugate")

    def involute(self) -> "Extensor":
        return self._grade_transform("involute")

    def scalar_negation(self) -> "Extensor":
        return self._grade_transform("scalar_negation")

    def pseudoscalar_negation(self) -> "Extensor":
        return self._grade_transform("pseudoscalar_negation")

    def __neg__(self) -> "Extensor":
        gatype = TypeRules.operation(
            "negative",
            (self.gatype,),
            self.gatype.subspaces,
        )
        return type(self)._from_prepared_kernel(self.context, gatype, -self._kernel)

    def __lt__(self, other: object) -> Any:
        return self.less(_scalar_operand(self.context, other))

    def __le__(self, other: object) -> Any:
        return self.less_equal(_scalar_operand(self.context, other))

    def __gt__(self, other: object) -> Any:
        return self.greater(_scalar_operand(self.context, other))

    def __ge__(self, other: object) -> Any:
        return self.greater_equal(_scalar_operand(self.context, other))

    def __add__(self, other: object) -> Extensor | NotImplementedType:
        other = _promote_identity(other)
        if isinstance(other, Number):
            if self.arity:
                raise TypeError("scalar addition requires a nullary Extensor")
            scalar = self.context.prepare_scalar(other)
            other = type(self)._from_prepared_kernel(
                self.context, self.algebra.subspace.scalar().canonical.gatype,
                self.context.scalar_kernel(scalar),
            )
        if _is_array(other):
            other = _batch_scalar(self.context, other)
        if not isinstance(other, Extensor):
            return NotImplemented
        if self.arity != other.arity:
            raise ValueError("Extensor addition requires equal arity")
        if self.algebra is not other.algebra:
            raise ValueError("Extensor addition requires one algebra")

        result_axes = tuple(
            left.union(right)
            for left, right in zip(self.axes, other.axes)
        )
        context = _common_context(self, other)
        left = context.lower(self)
        right = context.lower(other)
        left_kernel = _embed_kernel(context, left, result_axes)
        right_kernel = _embed_kernel(context, right, result_axes)
        gatype = TypeRules.operation(
            "add", (self.gatype, other.gatype), result_axes
        )
        return type(self)._from_prepared_kernel(context, gatype, left_kernel + right_kernel)

    def __radd__(self, other: object) -> Extensor | NotImplementedType:
        other = _promote_identity(other)
        if not isinstance(other, (Extensor, Number)) and not _is_array(other):
            return NotImplemented
        return self + other

    def __sub__(self, other: object) -> Extensor | NotImplementedType:
        other = _promote_identity(other)
        if not isinstance(other, (Extensor, Number)) and not _is_array(other):
            return NotImplemented
        return self + (-other)

    def __rsub__(self, other: object) -> Extensor | NotImplementedType:
        other = _promote_identity(other)
        if not isinstance(other, (Extensor, Number)) and not _is_array(other):
            return NotImplemented
        return -self + other

    def __mul__(self, scalar: object) -> Extensor:
        from numga.expression import geometric_product, is_expression_operand

        if is_expression_operand(scalar):
            return geometric_product(self, scalar)
        if _is_array(scalar):
            return geometric_product(self, _batch_scalar(self.context, scalar))
        scalar = self.context.prepare_scalar(scalar)
        gatype = TypeRules.operation("scale", (self.gatype,), self.gatype.subspaces)
        return type(self)._from_prepared_kernel(self.context, gatype, self._kernel * scalar)

    def __rmul__(self, scalar: object) -> Extensor:
        from numga.expression import geometric_product, is_expression_operand

        if is_expression_operand(scalar):
            return geometric_product(scalar, self)
        return self * scalar

    def __truediv__(self, other: object) -> Extensor | NotImplementedType:
        if isinstance(other, Number):
            # Prepare first: 1 / Fraction stays exact, unlike 1 / an int.
            scalar = self.context.prepare_scalar(other)
            return self * (1 / scalar)
        if isinstance(other, Extensor):
            if self.arity or other.arity:
                raise TypeError("geometric-product division requires nullary Extensors")
            if self.algebra is not other.algebra:
                raise ValueError("Extensor division requires one algebra")
            return self * other.inverse()
        if _is_array(other):
            return self * _batch_scalar(self.context, 1 / other)
        return NotImplemented

    def __rtruediv__(self, other: object) -> Extensor | NotImplementedType:
        if _is_array(other):
            other = _batch_scalar(self.context, other)
        if not isinstance(other, (Number, Extensor)):
            return NotImplemented
        if self.arity:
            raise TypeError("geometric-product division requires a nullary Extensor")
        return self.inverse() * other

    def wedge(self, other: Extensor | GAType | SubSpace) -> Extensor:
        from numga.expression import wedge

        return wedge(self, other)

    def scalar_product(self, other: Extensor | GAType | SubSpace) -> Extensor:
        from numga.expression import scalar_product

        return scalar_product(self, other)

    def inner(self, other: Extensor | GAType | SubSpace) -> Extensor:
        from numga.expression import inner

        return inner(self, other)

    def bivector_product(self, other: Extensor | GAType | SubSpace) -> Extensor:
        from numga.expression import bivector_product

        return bivector_product(self, other)

    def trivector_product(self, other: Extensor | GAType | SubSpace) -> Extensor:
        from numga.expression import trivector_product

        return trivector_product(self, other)

    def commutator(self, other: Extensor | GAType | SubSpace) -> Extensor:
        from numga.expression import commutator

        return commutator(self, other)

    def regressive(self, other: Extensor | GAType | SubSpace) -> Extensor:
        from numga.expression import regressive

        return regressive(self, other)

    def sandwich(self, passenger: Extensor | GAType | SubSpace) -> Extensor:
        from numga.expression import sandwich

        return sandwich(self, passenger)

    def reverse_sandwich(self, passenger: Extensor | GAType | SubSpace) -> Extensor:
        return self.reverse().sandwich(passenger)

    def inverse_sandwich(self, passenger: Extensor | GAType | SubSpace) -> Extensor:
        return self.inverse().sandwich(passenger)

    def __rshift__(self, other: Extensor | GAType | SubSpace) -> Extensor:
        return self.sandwich(other)

    def __lshift__(self, other: Extensor | GAType | SubSpace) -> Extensor:
        return self.inverse_sandwich(other)

    def __xor__(self, other: Extensor | GAType | SubSpace) -> Extensor:
        return self.wedge(other)

    def __rxor__(self, other: Extensor | GAType | SubSpace) -> Extensor:
        from numga.expression import wedge

        return wedge(other, self)

    def __and__(self, other: Extensor | GAType | SubSpace) -> Extensor:
        return self.regressive(other)

    def __rand__(self, other: Extensor | GAType | SubSpace) -> Extensor:
        from numga.expression import regressive

        return regressive(other, self)

    def __or__(self, other: Extensor | GAType | SubSpace) -> Extensor:
        return self.inner(other)

    def __ror__(self, other: Extensor | GAType | SubSpace) -> Extensor:
        from numga.expression import inner

        return inner(other, self)

    def _require_compatible(self, other: "Extensor") -> None:
        if not self.context.is_compatible_with(other.context):
            raise ValueError("Extensor contexts are incompatible")
        if self.gatype != other.gatype:
            raise ValueError("Extensor GATypes differ")

    def __repr__(self) -> str:
        return (
            f"Extensor(gatype={self.gatype!r}, shape={self.shape}, "
            f"context={type(self.context).__name__})"
        )


class _AtIndexer:
    __slots__ = ("_extensor",)

    def __init__(self, extensor: Extensor) -> None:
        self._extensor = extensor

    def __getitem__(self, index: object) -> "_AtUpdate":
        return _AtUpdate(self._extensor, index)


def _scalar_operand(context, value) -> Extensor:
    return value if isinstance(value, Extensor) else _batch_scalar(context, context.xp.asarray(value))


def _batch_scalar(context, values) -> Extensor:
    """A raw array in arithmetic is a batch of scalars of exactly the array's shape: the structural
    axis is always appended, so a trailing axis of length one is batch, never structural."""
    return context.multivector.scalar(values[..., None])


def _is_array(value: object) -> bool:
    """An array (or array-like with a shape) of any rank in linear arithmetic is a batch of scalars."""

    return hasattr(value, "shape") and not isinstance(value, Extensor) and len(value.shape) > 0


def _promote_identity(value: object) -> object:
    """A bare SubSpace or nullary GAType in linear arithmetic is the identity map on it.

    This is the same reading an open slot has in an expression: the slot left alone maps
    every element to itself.
    """
    from numga.gatype import GAType
    from numga.subspace import SubSpace

    if isinstance(value, SubSpace):
        return value.algebra.operator.identity(value)
    if isinstance(value, GAType):
        if value.arity:
            raise TypeError("only a nullary GAType promotes to an identity map")
        return value.algebra.operator.identity(value.output_subspace)
    return value


def _common_context(left: Extensor, right: Extensor) -> "Context":
    if left.context.is_compatible_with(right.context):
        return left.context
    if left.context.is_exact:
        return right.context
    if right.context.is_exact:
        return left.context
    raise ValueError("Extensor contexts are incompatible")


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
        target = self._extensor
        kernel_index = _batch_kernel_index(
            self._index,
            target.ndim,
            len(target.axes),
        )
        if isinstance(value, Extensor):
            target._require_compatible(value)
            replacement = value._kernel
            operand_gatypes = (target.gatype, value.gatype)
        else:
            replacement = value
            operand_gatypes = (target.gatype,)
        gatype = TypeRules.collection("set", self._index, operand_gatypes)
        kernel = target.context.functional_set(
            target._kernel,
            kernel_index,
            replacement,
        )
        return type(target)._from_prepared_kernel(target.context, gatype, kernel)


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
