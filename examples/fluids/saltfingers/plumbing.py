"""Periodic and seam-opened triangulations of a unit-depth tank and their vertex readings."""

from __future__ import annotations

import numpy as np

from numga.sparse import Sparse
from numga.topology import Topology


def tank(n_along: int, n_across: int, period: float) -> tuple[Topology, Topology, Sparse, Sparse, np.ndarray]:
    """Closed periodic and seam-opened triangulations of a unit-depth tank, the vertex readings between them, and the opened vertex coordinates."""
    closed = Topology.chains([n_across]).product(Topology.loops([n_along]))
    opened = Topology.chains([n_across]).product(Topology.chains([n_along + 1]))
    V, E, F = closed.n_elements
    V_opened, E_opened, F_opened = opened.n_elements
    places = np.stack(
        np.meshgrid(np.arange(n_along + 1) / n_along * period, np.linspace(0, 1, n_across), indexing="ij"),
        axis=-1,
    ).reshape(-1, 2)                                                          # [V_opened, 2]
    closing = opened.reading(np.arange(V), 0)                                 # Scalar+[V <- V_opened]
    opening = closed.reading(np.arange(V_opened) % V, 0)                      # Scalar+[V_opened <- V]
    return closed, opened, closing, opening, places
