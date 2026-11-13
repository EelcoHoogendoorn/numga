"""Pure-GA GPU rendering on top of `numga.wgsl`: cameras, ray-traced quadric solids, lit triangle meshes, simplices
(strokes, tubes, markers), and planar fields (contours, quadric curves/regions, surface flow)."""

from __future__ import annotations

from . import camera, contours, flow, meshes, planar, quadrics, scene, simplices, solids
from .camera import Camera, preimage, read, reading, whole
from .scene import Scene
from .simplices import Cap, Layer, composited
from .solids import Ink, View, aim, view

__all__ = [
    "Camera",
    "Cap",
    "Ink",
    "Layer",
    "Scene",
    "View",
    "aim",
    "camera",
    "composited",
    "contours",
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
