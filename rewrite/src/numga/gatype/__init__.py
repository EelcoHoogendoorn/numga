"""Whole-extensor GAType and trait metadata."""

from .dispatch import AmbiguousGATypeDispatchWarning, GATypeDispatch
from .factory import GATypeFactory
from .gatype import GAType
from .pattern import GATypePattern
from .traits import (
    CoefficientOrthogonal,
    EMPTY_TRAITS,
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
    ROTOR_TRAITS,
    ReverseProductNonzero,
    ReverseProductOne,
    ReverseProductScalar,
    ReverseProductZero,
    Trait,
    TraitSet,
    Versor,
    VersorProduct,
)
