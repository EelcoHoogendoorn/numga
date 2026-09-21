"""GATypes, algebra binding, and coordinate helpers for multiview reconstruction.

Isolates algebra selection and GAType definitions from the mathematical core,
allowing scenarios to select an algebra (e.g. PGA2D, PGA3D) and smoothly inject
it into multiview modules without polluting `core.py` with setup or conversion logic.
"""

from __future__ import annotations

import os
import sys

if __name__ == "types":
    _stdlib_types = os.path.join(os.path.dirname(os.__file__), "types.py")
    with open(_stdlib_types, "r") as _f:
        _code = compile(_f.read(), _stdlib_types, "exec")
    exec(_code, globals())
else:
    from typing import TYPE_CHECKING
    import numpy as np

    from numga import Context, NumpyContext, stack
    from numga.algebras import PGA2D

    if TYPE_CHECKING:
        from numga import Algebra, Extensor
        from numga.gatype import GAType

    # Module-level globals:
    ga: Algebra
    ctx: Context
    mv: object
    Point: GAType
    Plane: GAType
    Hyperplane: GAType
    Line: GAType
    Motor: GAType
    Twist: GAType
    Camera: GAType
    PointMap: GAType
    Projective: GAType
    Quadric: GAType
    TwistMap: GAType
    Scalar: GAType
    w: Extensor

    _SYNC_NAMES = (
        "ga", "ctx", "mv",
        "Point", "Plane", "Hyperplane", "Line", "Motor", "Twist", "Camera", "PointMap", "Projective", "Quadric", "TwistMap", "Scalar",
        "w", "stack",
    )

    _TARGET_MODULES = (
        "examples.geometry.multiview.core",
        "examples.geometry.multiview.render",
        "examples.geometry.multiview.scenarios",
    )


    def bind(algebra: Algebra | Context = PGA2D, context: Context | None = None) -> None:
        """Bind module-level algebra, context, GATypes, and plane at infinity."""
        global ga, ctx, mv
        global Point, Plane, Hyperplane, Line, Motor, Twist, Camera, PointMap, Projective, Quadric, TwistMap, Scalar
        global w

        if isinstance(algebra, Context):
            ctx = algebra
            ga = algebra.algebra
        else:
            ga = algebra
            ctx = context or NumpyContext(ga)
        mv = ctx.multivector

        Point = ga.gatype.antivector()
        Plane = ga.gatype.vector()
        Hyperplane = Plane
        Line = Plane
        Motor = ga.gatype.rotor()
        Twist = ga.gatype.bivector()
        Camera = ga.gatype((Point, Point))
        PointMap = Camera
        Projective = PointMap
        Quadric = ga.gatype((Hyperplane, Point))
        TwistMap = ga.gatype((Twist, Twist))
        Scalar = ga.gatype.scalar()
        w = mv.w

        # Sync dynamically into downstream multiview modules if already loaded:
        for mod_name in _TARGET_MODULES:
            if mod_name in sys.modules:
                mod = sys.modules[mod_name]
                this_mod = sys.modules[__name__]
                for name in _SYNC_NAMES:
                    if hasattr(mod, name) or hasattr(this_mod, name):
                        setattr(mod, name, getattr(this_mod, name))


    def point(coords: np.ndarray) -> Point:
        """Construct affine Point antivectors from Cartesian coordinates."""
        coords = np.asarray(coords)
        ones = np.ones((*coords.shape[:-1], 1), dtype=coords.dtype)
        return mv.antivector(np.concatenate([coords, ones], axis=-1))


    def coordinates(points: Point) -> np.ndarray:
        """Extract Cartesian coordinates from Point antivectors."""
        k = points.kernel
        return k[..., :-1] / k[..., -1:]


    # Default initialization with PGA2D:
    bind(PGA2D)
