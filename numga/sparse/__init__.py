"""Field maps stored sparsely: `Sparse` maps of plain numbers, which belong to no algebra, and `SparseExtensor` maps of
typed cells."""

from .numeric import Sparse
from .extensor import SparseExtensor, spdiag

__all__ = ["Sparse", "SparseExtensor", "spdiag"]
