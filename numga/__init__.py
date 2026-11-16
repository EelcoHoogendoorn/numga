"""The top-level `numga` namespace exposes geometric algebras (`Algebra`), blade subspaces
(`SubSpace`), extensor types and traits (`GAType`, `Orientation`, `Trait`), multilinear values and
maps (`Extensor`, `MultivectorFactory`, `OperatorFactory`), and backend execution contexts
(`ExactContext`, `NumpyContext`, `JaxContext`, `TorchContext`). Optional JAX and PyTorch contexts
load lazily on attribute access so importing `numga` requires only NumPy.

Algebraic conventions and product rules are introduced in `docs/conventions.md`, extensor map and
binding syntax in `docs/extensor_syntax.md`, intrinsic and extrinsic orientation in
`docs/orientation.md`, discrete fields over sites in `docs/fields.md`, simplicial complexes in
`docs/topology.md`, WebGPU shaders and rendering in `docs/rendering.md`, and type-dispatch and
backend internals in `docs/internals.md`."""

from __future__ import annotations

from .algebra import Algebra, AlgebraDescription
from .backend import Context, ExactContext, NumpyContext
from .binding import AxisTransform, AxisTransformKind, BindingPlan
from .extensor import Extensor, concatenate, stack, where
from .extension import ExtensionMethod
from . import expression
from .gatype import (
    EMPTY_TRAITS,
    ROTOR_TRAITS,
    AmbiguousGATypeDispatchWarning,
    Blade,
    BladeProduct,
    CoefficientOrthogonal,
    GAType,
    GATypeDispatch,
    GATypeError,
    GATypeFactory,
    GATypePattern,
    CliffordConjugateProduct,
    CliffordConjugateProductNonzero,
    CliffordConjugateProductOne,
    CliffordConjugateProductScalar,
    CliffordConjugateProductZero,
    GradeInvolutionProduct,
    Orientation,
    ProductFact,
    ProductRelation,
    ProductResult,
    SelfProduct,
    ReverseProduct,
    ReverseProductNonzero,
    ReverseProductOne,
    ReverseProductScalar,
    ReverseProductZero,
    Trait,
    TraitSet,
    Versor,
    VersorProduct,
)
from .multivector import MultivectorFactory
from .operator import OperatorFactory, SymbolicKernel
from .subspace import SubSpace, SubSpaceFactory, SupportKey

# Default extension overloads register once core classes exist.
from . import extensions as _extensions

__version__ = "2.0.0"


def __getattr__(name: str) -> type[Context]:
    if name == "JaxContext":
        from .backend.jax import JaxContext
        return JaxContext
    if name == "TorchContext":
        from .backend.torch import TorchContext
        return TorchContext
    raise AttributeError(f"module 'numga' has no attribute {name!r}")
