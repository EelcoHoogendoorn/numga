"""Cached query namespaces on `GAType` group structural and semantic inspections by domain so the
core `GAType` class stays focused on representation and algebra. Each property (`gatype.axis_roles`,
`gatype.form`, `gatype.over_sites`, and `gatype.self_products`) constructs its lightweight wrapper
once per `GAType` instance and caches repeated queries.

`AxisRoles` inspects output (`0`) and input (`1..arity`) axes for index contractions, pairings, and
subtype domain restrictions when binding operands. `Form` tests whether a binary scalar-valued map
acts symmetrically on a single space or forms a compatible linear system with a right-hand side.
`OverSites` extends endomorphism, correlation, and linear-system checks to extensors whose axes
carry coupled field sites, distinguishing coupled site operators from site-by-site systems.
`SelfProducts` extracts certified nullary `ProductFact` categories and positive-arity
`ProductRelation` or `ClosureRelation` laws per self-product family.
"""

from __future__ import annotations

from functools import cached_property, lru_cache
from typing import TYPE_CHECKING

from .traits import ClosureRelation, ProductFact, ProductRelation, ProductResult, SelfProduct

if TYPE_CHECKING:
    from numga.extensor import Extensor

    from .gatype import GAType


class _Queries:
    __slots__ = ("_gatype", "__dict__")

    def __init__(self, gatype: GAType) -> None:
        self._gatype = gatype


class AxisRoles(_Queries):
    """Axis-index queries on `gatype.axis_roles` number the output `0` and inputs `1..arity`."""

    @cached_property
    def indexed(self) -> tuple[int, ...]:
        """Axes available for contraction include every input axis plus any non-scalar output axis."""
        gatype = self._gatype
        output = () if gatype.output_subspace.same_support(gatype.algebra.subspace.scalar()) else (0,)
        return output + tuple(range(1, len(gatype.input_subspaces) + 1))

    @property
    def pair(self) -> tuple[int, int]:
        """When exactly two axes carry indices, they form the unique default contraction pair."""
        if len(self.indexed) != 2:
            raise TypeError(f"name the two axes to pair; {self._gatype} has {len(self.indexed)} axes that carry an index")
        return self.indexed

    @lru_cache(maxsize=None)
    def paired(self, first_axis: int, second_axis: int) -> tuple[int, int]:
        """Two indexed axes sort in ascending order for regressive or trace pairing, rejecting
        scalar outputs."""
        first_axis, second_axis = sorted((first_axis, second_axis))
        if first_axis not in self.indexed:
            raise TypeError(f"axis {first_axis} carries no index in {self._gatype}: a scalar output pairs with nothing")
        return first_axis, second_axis

    @lru_cache(maxsize=None)
    def contracted(self, first_axis: int, second_axis: int) -> tuple[int, int]:
        """Metric inner-product contraction requires two input axes to share the same blade support."""
        first_axis, second_axis = self.paired(first_axis, second_axis)
        inputs = self._gatype.input_subspaces
        if first_axis and not inputs[second_axis - 1].same_support(inputs[first_axis - 1]):
            raise TypeError(f"contract pairs inputs of one space; got {inputs[first_axis - 1]} and {inputs[second_axis - 1]}")
        return first_axis, second_axis

    @lru_cache(maxsize=None)
    def narrowing(self, input_index: int, operand: GAType) -> Extensor | None:
        """Passing a narrower `GAType` as a map argument restricts that input domain via its identity
        map; exact matches return `None` and unsupported blades raise `TypeError`."""
        gatype = self._gatype
        space = operand.output_subspace
        own = gatype.input_subspaces[input_index]
        if space.same_support(own) and operand.orientation is gatype.input_orientations[input_index]:
            return None
        if not space.support_is_subset_of(own):
            raise TypeError(
                f"axis {input_index} takes {own}; {space} is not inside it, so a cast must say what is projected away"
            )
        return gatype.algebra.operator.identity(operand)

    def same_as(self, other: GAType) -> bool:
        """Two types have identical axis roles when their subspaces, orientations, and field sites
        match regardless of value traits."""
        gatype = self._gatype
        return (gatype.subspaces == other.subspaces and gatype.orientations == other.orientations
                and gatype.fields == other.fields)


class Form(_Queries):
    """Queries on `gatype.form` inspect scalar-valued bilinear forms (`arity == 2`)."""

    @cached_property
    def is_square(self) -> bool:
        """Square forms pair a single blade support and orientation with itself across both inputs."""
        gatype = self._gatype
        return (gatype.is_form and gatype.subspaces[1].same_support(gatype.subspaces[2])
                and gatype.orientations[1] is gatype.orientations[2])

    def matches(self, other: GAType) -> bool:
        """Two square forms match when they act on the same blade support with identical orientations."""
        gatype = self._gatype
        return (self.is_square and other.form.is_square
                and gatype.subspaces[2].same_support(other.subspaces[2]) and gatype.orientations == other.orientations)

    def is_system(self, rhs: GAType) -> bool:
        """A bilinear form solves against a scalar-valued `rhs` whose final input matches the form's
        test-input space."""
        gatype = self._gatype
        return (
            gatype.is_form
            and rhs.arity >= 1
            and rhs.output_subspace.same_support(gatype.algebra.subspace.scalar())
            and rhs.subspaces[-1].same_support(gatype.subspaces[2])
        )


class OverSites(_Queries):
    """Queries on `gatype.over_sites` view field-bearing maps and forms as single block operators
    across coupled sites."""

    @cached_property
    def is_square(self) -> bool:
        """Unary maps or site-free-output bilinear forms are square over sites when both coupled
        axes share identical subspaces, orientations, and site counts."""
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
        """Site-coupled unary maps are endomorphisms when input and output share subspace,
        orientation, and site count."""
        return self._gatype.arity == 1 and self.is_square

    @cached_property
    def is_correlation(self) -> bool:
        """Site-coupled duality maps take a space to its oppositely oriented complement over equal
        site counts, requiring a second duality map as metric in eigensystems."""
        gatype = self._gatype
        sites = dict(gatype.fields)
        return gatype.is_correlation and sites.get(0) == sites.get(1)

    def is_map_system(self, rhs: GAType) -> bool:
        """Site-coupled unary maps solve against any `rhs` whose output support lies inside the map's
        output subspace."""
        gatype = self._gatype
        return gatype.arity == 1 and gatype.has_fields and rhs.output_subspace.support_is_subset_of(gatype.output_subspace)

    def is_form_system(self, rhs: GAType) -> bool:
        """Site-coupled bilinear forms solve against scalar-valued `rhs` extensors whose final input
        matches the form's test space in both blade support and site count."""
        gatype = self._gatype
        scalar = gatype.algebra.subspace.scalar()
        return (
            gatype.arity == 2
            and gatype.has_fields
            and gatype.output_subspace.same_support(scalar)
            and rhs.arity >= 1
            and rhs.output_subspace.same_support(scalar)
            and 0 not in dict(rhs.fields)
            and rhs.subspaces[-1].same_support(gatype.subspaces[2])
            and dict(rhs.fields).get(rhs.arity) == dict(gatype.fields).get(2)
        )

    def is_site_by_site_system(self, rhs: GAType) -> bool:
        """Site-free maps or forms applied to a site-bearing `rhs` solve the same system independently
        at each site, provided a form's `rhs` carries sites on either its output or its final input
        rather than both."""
        gatype = self._gatype
        sites = dict(rhs.fields)
        return not gatype.has_fields and rhs.has_fields and (
            gatype.arity == 1 or gatype.form.is_system(rhs) and not (0 in sites and rhs.arity in sites)
        )


class SelfProducts(_Queries):
    """Queries on `gatype.self_products` look up certified self-product facts and propagation
    relations per `SelfProduct` family."""

    def result(self, family: SelfProduct) -> ProductResult | None:
        """Entailment checks select the strongest `ProductResult` certified in the given family, or
        `None` when uncertified."""
        for result in (ProductResult.ONE, ProductResult.NONZERO, ProductResult.ZERO, ProductResult.SCALAR):
            if self._gatype.entails(ProductFact(family, result)):
                return result
        return None

    def relation(self, family: SelfProduct) -> ProductRelation | None:
        """Positive-arity types carry at most one `ProductRelation` per self-product family, and
        distinct families never substitute for one another."""
        return next((
            trait for trait in self._gatype.traits
            if isinstance(trait, ProductRelation) and trait.self_product == family
        ), None)

    @lru_cache(maxsize=None)
    def closure(self, relation_type: type[ClosureRelation]) -> ClosureRelation | None:
        """Positive-arity types carry at most one closure relation of a given `ClosureRelation`
        subclass (`VersorProduct` or `BladeProduct`)."""
        return next((trait for trait in self._gatype.traits if isinstance(trait, relation_type)), None)
