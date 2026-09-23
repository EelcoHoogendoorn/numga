"""numga: geometric algebra and extensors in NumPy and JAX."""

from .algebra import Algebra, AlgebraDescription
from .backend import Context, ExactContext, NumpyContext
from .binding import AxisTransform, AxisTransformKind, BindingPlan
from .extensor import Extensor, concatenate, stack
from .extension import ExtensionMethod
from .gatype import (
    EMPTY_TRAITS,
    ROTOR_TRAITS,
    AmbiguousGATypeDispatchWarning,
    CoefficientOrthogonal,
    GAType,
    GATypeDispatch,
    GATypeFactory,
    GATypePattern,
    CliffordConjugateProduct,
    CliffordConjugateProductNonzero,
    CliffordConjugateProductOne,
    CliffordConjugateProductScalar,
    CliffordConjugateProductZero,
    GradeInvolutionProduct,
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

# Core classes are now available; register defaults without changing the
# descriptor mechanism used by end-user overloads.
from . import extensions as _extensions

__version__ = "2.0.0a0"
