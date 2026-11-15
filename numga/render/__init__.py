"""`numga.render` draws geometric algebra scenes on the GPU in two layers built on `numga.wgsl`: `drawing` rasterizes
and composites supersampled `Draw` passes without inspecting geometry, while `camera`, `simplices`, `contours`,
`quadrics`, `meshes`, `flow`, `solids`, and `scene` pair elements directly against camera planes in their own algebra.
"""

from __future__ import annotations

from . import camera, contours, drawing, flow, meshes, planar, quadrics, scene, simplices, solids
from .camera import Camera, preimage, read, reading, whole
from .drawing import Draw, Target, composited, drawn
from .scene import Scene
from .simplices import Cap
from .solids import Ink, View, aim, view

__all__ = [
    "Camera",
    "Cap",
    "Draw",
    "Ink",
    "Scene",
    "Target",
    "View",
    "aim",
    "camera",
    "composited",
    "contours",
    "drawing",
    "drawn",
    "flow",
    "meshes",
    "planar",
    "preimage",
    "quadrics",
    "read",
    "reading",
    "scene",
    "simplices",
    "solids",
    "view",
    "whole",
]
