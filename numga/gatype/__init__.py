"""Whole-extensor `GAType` metadata, orientation rules, trait inference, and signature dispatch."""

from numga.units import Unit, UnitGrid
from . import rules
from .dispatch import AmbiguousGATypeDispatchWarning, GATypeDispatch
from .errors import GATypeError, UnorientedSubspace
from .factory import GATypeFactory
from .gatype import GAType
from .metric import MetricKind
from .orientation import Orientation
from .pattern import GATypePattern
from .traits import (
    EMPTY_TRAITS,
    ROTOR_TRAITS,
    Blade,
    BladeProduct,
    CliffordConjugateProduct,
    CliffordConjugateProductNonzero,
    CliffordConjugateProductOne,
    CliffordConjugateProductScalar,
    CliffordConjugateProductZero,
    CoefficientOrthogonal,
    GradeInvolutionProduct,
    ProductFact,
    ProductRelation,
    ProductResult,
    ReverseProduct,
    ReverseProductNonzero,
    ReverseProductOne,
    ReverseProductScalar,
    ReverseProductZero,
    SelfProduct,
    Trait,
    TraitSet,
    Versor,
    VersorProduct,
)
