"""Points of PGA3D from Cartesian coordinates: plumbing shared by the PGA3D examples.

A point is an antivector, the meet of three planes. Each coordinate sits on the blade that
lacks its own axis: x on yzw, y on zxw, z on xyw. The weight sits on zyx, the blade without
w, so a finite point has weight one and an ideal point (a direction) has weight zero. These
four blades, with these orientations, are the algebra's default antivector layout.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext
from numga.algebras import PGA3D

mv = NumpyContext(PGA3D).multivector
Point = PGA3D.gatype.antivector()


def point(coords: np.ndarray) -> Point:
    """Finite points of unit weight at (..., 3) coordinates."""
    return mv("yzw zxw xyw", coords) + mv.zyx


def direction(coords: np.ndarray) -> Point:
    """Ideal points: the directions (..., 3), with zero weight."""
    return mv("yzw zxw xyw", coords)
