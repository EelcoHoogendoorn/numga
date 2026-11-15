"""`Scene` caches static rasterized surface sample slots on the GPU across animation frames while
combining static and per-frame quadric solids, surface inks, translucent solids, and moving raster
draws in a single ray-tracing pass per frame."""

from __future__ import annotations

from typing import Any

import numpy as np

from .drawing import Draw, sampled, slots
from .solids import Ink, Quadric, View, scene, shader, trace


class Scene:
    """Reusable `PGA3D` animation scene holding a fixed `View`, RGB `background`, static quadric
    `parts`, static `inks`, static `translucent` parts, and static rasterized `meshes`.

    Static `meshes` rasterize once into leading GPU sample `Slots` (reallocated only if a later frame
    supplies more moving draws), while each call to `frame` or `draw` rasterizes only the frame's
    moving `meshes` into trailing slots before tracing all static and moving solids together."""

    def __init__(self, view: View, background: np.ndarray, parts: list[tuple[Quadric, np.ndarray]],
                 meshes: list[tuple[Draw, float]], inks: list[Ink],
                 translucent: list[tuple[Quadric, np.ndarray, float]]) -> None:
        self.view, self.background = view, np.asarray(background, dtype=float)
        self.parts, self.inks, self.translucent, self.meshes = list(parts), list(inks), list(translucent), list(meshes)
        self.tracer = shader()
        # The buffers holding the still meshes' samples, and the meshes they have room for, made by the first frame.
        self.buffers, self.n_slots = None, 0

    def made(self, name: str, size: int, usage: int) -> Any:
        """Allocate a dedicated GPU buffer of `size` bytes with `usage` on the scene's WebGPU device."""
        return self.tracer.device.create_buffer(size=size, usage=usage)

    def frame(self, parts: list[tuple[Quadric, np.ndarray]], inks: list[Ink],
              translucent: list[tuple[Quadric, np.ndarray, float]],
              meshes: list[tuple[Draw, float]]) -> tuple[np.ndarray, np.ndarray]:
        """Composite a frame's moving `parts`, `inks`, `translucent` solids, and `meshes` alongside
        the scene's static elements over `background`, returning coverage-weighted RGBA colour
        `[rows, columns, 4]` and view depth `[rows, columns]`."""
        encoder = self.tracer.device.create_command_encoder()
        n_still, n_slots = len(self.meshes), len(self.meshes) + len(meshes)
        # Buffers too small for the frame's meshes are made again, and the still meshes drawn into them once.
        if self.buffers is None or self.n_slots < n_slots:
            self.buffers, self.n_slots = slots(self.view.shape, n_slots, self.made), n_slots
            sampled(encoder, [mesh for mesh, _ in self.meshes], 0, self.buffers)
        sampled(encoder, [mesh for mesh, _ in meshes], n_still, self.buffers)
        solids = scene(self.parts + list(parts), self.translucent + list(translucent), self.inks + list(inks))
        return trace(self.tracer, encoder, self.view, solids, self.buffers,
                     [opacity for _, opacity in self.meshes + list(meshes)], self.background)

    def draw(self, parts: list[tuple[Quadric, np.ndarray]], inks: list[Ink],
             translucent: list[tuple[Quadric, np.ndarray, float]], meshes: list[tuple[Draw, float]]) -> np.ndarray:
        """Render a frame's moving and static elements over `background` as `uint8` RGB
        `[rows, columns, 3]`."""
        colour, _ = self.frame(parts, inks, translucent, meshes)
        return np.round(np.clip(colour[..., :3], 0, 1) * 255).astype(np.uint8)
