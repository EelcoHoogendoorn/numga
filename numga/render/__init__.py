"""Rendering in geometric algebra on the GPU, on top of the dialect of `numga.wgsl`, in two layers.

`drawing` draws: a `Draw` is a shader's triangles over a picture, drawn into a `Target` that stays on the GPU, laid
over the draws before it or kept where it is nearest, averaged over each pixel's samples and read back, or kept in
slots for a pass that reads them; it reads no geometry. The marks make draws from geometry, read by a camera's planes:
strokes, tubes and markers of simplices, contours of fields over meshes, quadric curves and regions, lit meshes, flow
textures, and ray-traced quadric solids, composed with meshes into a scene.
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
