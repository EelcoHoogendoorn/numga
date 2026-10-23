"""What a type says about its axes, its forms, its sites and its self-products, each a namespace on
the type: `gatype.axis_roles`, `gatype.form`, `gatype.over_sites` and `gatype.self_products`. Like the
derivations, each is computed once per type."""

from __future__ import annotations

from functools import cached_property, lru_cache
from typing import TYPE_CHECKING

from numga.subspace import SubSpace

from .traits import ClosureRelation, ProductFact, ProductRelation, ProductResult, SelfProduct

if TYPE_CHECKING:
    from numga.extensor import Extensor

    from .gatype import GAType


class _Queries:
    __slots__ = ("_gatype", "__dict__")

    def __init__(self, gatype: GAType) -> None:
        self._gatype = gatype


class AxisRoles(_Queries):
    """The axes of a type, output 0 and inputs from 1: which carry an index, which pair, and what an
    operand given for an input binds there."""

    @cached_property
    def indexed(self) -> tuple[int, ...]:
        """The axes that carry an index: the output unless it is scalar, and every input."""
        gatype = self._gatype
        output = () if gatype.output_subspace.same_support(gatype.algebra.subspace.scalar()) else (0,)
        return output + tuple(range(1, len(gatype.input_subspaces) + 1))

    @property
    def pair(self) -> tuple[int, int]:
        """The only two axes that carry an index."""
        if len(self.indexed) != 2:
            raise TypeError(f"name the two axes to pair; {self._gatype} has {len(self.indexed)} axes that carry an index")
        return self.indexed

    @lru_cache(maxsize=None)
    def paired(self, first: int, second: int) -> tuple[int, int]:
        """Two axes to pair, in order, each one that carries an index."""
        first, second = sorted((first, second))
        if first not in self.indexed:
            raise TypeError(f"axis {first} carries no index in {self._gatype}: a scalar output pairs with nothing")
        return first, second

    @lru_cache(maxsize=None)
    def contracted(self, first: int, second: int) -> tuple[int, int]:
        """Two axes to pair through the inner product: two inputs must share one space."""
        first, second = self.paired(first, second)
        inputs = self._gatype.input_subspaces
        if first and not inputs[second - 1].same_support(inputs[first - 1]):
            raise TypeError(f"contract pairs inputs of one space; got {inputs[first - 1]} and {inputs[second - 1]}")
        return first, second

    @lru_cache(maxsize=None)
    def narrowing(self, axis: int, operand: GAType) -> Extensor | None:
        """The identity map restricting an argument's domain to the given subtype, or None when
        the type already matches and no restriction is needed."""
        gatype = self._gatype
        space = operand.output_subspace
        own = gatype.input_subspaces[axis]
        if space.same_support(own) and operand.orientation is gatype.input_orientations[axis]:
            return None
        if space not in own:
            raise TypeError(
                f"axis {axis} takes {own}; {space} is not inside it, so a cast must say what is projected away"
            )
        return gatype.algebra.operator.identity(operand)

    def same_as(self, other: GAType) -> bool:
        """The same spaces, orientations and sites, whatever facts either type carries."""
        gatype = self._gatype
        return (gatype.subspaces == other.subspaces and gatype.orientations == other.orientations
                and gatype.fields == other.fields)


class Form(_Queries):
    """A type read as a form, a scalar-valued map of two axes."""

    @cached_property
    def is_square(self) -> bool:
        """A form pairing one space with itself: the same blades and the same orientation in both axes."""
        gatype = self._gatype
        return (gatype.is_form and gatype.subspaces[1].same_support(gatype.subspaces[2])
                and gatype.orientations[1] is gatype.orientations[2])

    def matches(self, other: GAType) -> bool:
        """Both square forms on the same space, in the same orientations."""
        gatype = self._gatype
        return (self.is_square and other.form.is_square
                and gatype.subspaces[2].same_support(other.subspaces[2]) and gatype.orientations == other.orientations)

    def is_system(self, rhs: GAType) -> bool:
        """A form against a scalar-valued extensor whose last input matches the form's last input."""
        gatype = self._gatype
        return (
            gatype.is_form
            and rhs.arity >= 1
            and rhs.output_subspace.same_support(gatype.algebra.subspace.scalar())
            and rhs.subspaces[-1].same_support(gatype.subspaces[2])
        )


class OverSites(_Queries):
    """A type with fields read as one map or form over the sites of its axes."""

    @cached_property
    def is_square(self) -> bool:
        """A map, or a scalar form of two axes, between one space over the same sites, either side
        ranging over them."""
        gatype = self._gatype
        sites = dict(gatype.fields)
        if gatype.arity == 1:
            rows, columns = 0, 1
        elif gatype.arity == 2 and gatype.output_subspace.same_support(gatype.algebra.subspace.scalar()) and 0 not in sites:
            rows, columns = 1, 2
        else:
            return False
        return (gatype.subspaces[rows] == gatype.subspaces[columns] and sites.get(rows) == sites.get(columns)
                and gatype.orientations[rows] is gatype.orientations[columns])

    @cached_property
    def is_endomorphism(self) -> bool:
        return self._gatype.arity == 1 and self.is_square

    def is_map_system(self, rhs: GAType) -> bool:
        """A map whose output takes the right-hand side's output, either side ranging over sites."""
        return self._gatype.arity == 1 and rhs.output_subspace.support_is_subset_of(self._gatype.output_subspace)

    def is_form_system(self, rhs: GAType) -> bool:
        """A scalar form of two axes against a scalar-valued extensor ending in the form's last axis."""
        gatype = self._gatype
        scalar = gatype.algebra.subspace.scalar()
        return (
            gatype.arity == 2
            and gatype.output_subspace.same_support(scalar)
            and rhs.arity >= 1
            and rhs.output_subspace.same_support(scalar)
            and 0 not in dict(rhs.fields)
            and rhs.subspaces[-1].same_support(gatype.subspaces[2])
            and dict(rhs.fields).get(rhs.arity) == dict(gatype.fields).get(2)
        )


class SelfProducts(_Queries):
    """What a type's facts say about the products of its values with their own transforms."""

    def result(self, family: SelfProduct) -> ProductResult | None:
        """The strongest result this type entails in exactly this self-product family."""
        for result in (ProductResult.ONE, ProductResult.NONZERO, ProductResult.ZERO, ProductResult.SCALAR):
            if self._gatype.entails(ProductFact(family, result)):
                return result
        return None

    def relation(self, family: SelfProduct) -> ProductRelation | None:
        """This type's relation in one self-product family; different families never substitute."""
        return next((
            trait for trait in self._gatype.traits
            if isinstance(trait, ProductRelation) and trait.self_product == family
        ), None)

    @lru_cache(maxsize=None)
    def closure(self, kind: type[ClosureRelation]) -> ClosureRelation | None:
        """This type's relation carrying a family closed under its operation, versors or blades."""
        return next((trait for trait in self._gatype.traits if isinstance(trait, kind)), None)
