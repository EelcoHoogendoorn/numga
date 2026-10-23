"""A picture set up once and drawn again for every frame of an animation: the parts that stay put kept, on the GPU
where they can be, and each frame's moving parts added to them."""

from __future__ import annotations

from typing import Any

import numpy as np

from examples.util.wgsl.library import drawn
from examples.util.wgsl.meshes import Samples
from examples.util.wgsl.solids import Ink, Quadric, View, mesh_buffers, sampled, scene, shader, trace


class Scene:
    """A picture whose still parts are set up once, for frames that add what moves: the solids, lines inked on them
    and translucent solids that stay put, traced again with every frame's, and the meshes that stay put, drawn into
    their samples once, in buffers the scene keeps on the GPU, with every frame's meshes drawn into the slots after
    theirs."""

    def __init__(self, view: View, background: np.ndarray, parts: list[tuple[Quadric, np.ndarray]],
                 meshes: list[tuple[Samples, float]], inks: list[Ink] = (),
                 translucent: list[tuple[Quadric, np.ndarray, float]] = ()) -> None:
        self.view, self.background = view, np.asarray(background, dtype=float)
        self.parts, self.inks, self.translucent, self.meshes = list(parts), list(inks), list(translucent), list(meshes)
        # The view as it is traced: where the CPU draws, with fewer pixels, enlarged to the view's.
        self.seen = view._replace(pixels=drawn(view.pixels))
        self.tracer = shader()
        # The buffers holding the still meshes' samples, and the meshes they have room for, made by the first frame.
        self.buffers, self.n_slots = None, 0

    def made(self, name: str, size: int, usage: int) -> Any:
        """A buffer of the scene's own."""
        return self.tracer.device.create_buffer(size=size, usage=usage)

    def frame(self, parts: list[tuple[Quadric, np.ndarray]] = (), inks: list[Ink] = (),
              translucent: list[tuple[Quadric, np.ndarray, float]] = (),
              meshes: list[tuple[Samples, float]] = ()) -> tuple[np.ndarray, np.ndarray]:
        """The picture with a frame's moving parts among the still ones, over the background: its colour,
        `[rows, columns, 4]`, and its depth along the view, `[rows, columns]`, each weighted by coverage."""
        encoder = self.tracer.device.create_command_encoder()
        n_still, n_slots = len(self.meshes), len(self.meshes) + len(meshes)
        # Buffers too small for the frame's meshes are made again, and the still meshes drawn into them once.
        if self.buffers is None or self.n_slots < n_slots:
            self.buffers, self.n_slots = mesh_buffers(self.seen, n_slots, self.made), n_slots
            sampled(encoder, [mesh for mesh, _ in self.meshes], self.seen, 0, self.buffers)
        sampled(encoder, [mesh for mesh, _ in meshes], self.seen, n_still, self.buffers)
        solids = scene(self.parts + list(parts), self.translucent + list(translucent), self.inks + list(inks))
        return trace(self.tracer, encoder, self.view, self.seen, solids, self.buffers,
                     [opacity for _, opacity in self.meshes + list(meshes)], self.background)

    def draw(self, parts: list[tuple[Quadric, np.ndarray]] = (), inks: list[Ink] = (),
             translucent: list[tuple[Quadric, np.ndarray, float]] = (),
             meshes: list[tuple[Samples, float]] = ()) -> np.ndarray:
        """The picture of a frame on the background, `[rows, columns, 3]`, as `frame`."""
        colour, _ = self.frame(parts, inks, translucent, meshes)
        return np.round(np.clip(colour[..., :3], 0, 1) * 255).astype(np.uint8)
