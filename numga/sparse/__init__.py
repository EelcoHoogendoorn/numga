"""Sparse field maps couple discrete sites either through algebra-free numbers in `Sparse` or
through typed algebraic cells in `SparseExtensor` and `spdiag`."""

from .numeric import Sparse
from .extensor import SparseExtensor, spdiag

__all__ = ["Sparse", "SparseExtensor", "spdiag"]
