"""The `numga.algebra` package defines backend-independent geometric algebra descriptions,
canonical bit-mask blades, and Cayley product tables."""

from .algebra import Algebra, BladeProduct, OrientedBlade, ProductTable
from .description import AlgebraDescription, parse_description

