"""Thin-walled cross-section topologies for the bending scenes represent each section's midline as a
one-dimensional edge complex with node positions (`Point-[V]`) and wall thicknesses (`Scalar+[E]`)."""

from __future__ import annotations

import numpy as np

from numga import Algebra, NumpyContext
from numga.algebras import PGA3D as ga
from numga.topology import Topology

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
Plane = ga.gatype.vector()
Point = ga.gatype.antivector()
SectionPlane = Plane.derive.within(Algebra("x+y+"))

section_names = ["I-section", "turned I-section", "Z-section", "slotted tube"]
# In-plane rotation angles of the I-sections and Z-section, in degrees.
section_turns = np.array([0.0, 30.0, 0.0])
# Web height between flange midlines, flange width, and wall thickness.
web_height, flange_width, wall_thickness = 0.92, 0.6, 0.08
# Slotted-tube midline radius, wall thickness, slot opening angle, and polygonal segment count.
tube_radius, tube_wall = 0.46, 0.1
slot_angle = np.deg2rad(60.0)
n_arc_segments = 48


def clamp_point(coordinates: np.ndarray) -> Point:
    """Two-dimensional `(x, y)` `coordinates` (`[..., 2]`) embed into unit-weight points (`[...] Point-`)
    in the clamp plane `z = 0`."""
    return (mv(SectionPlane, coordinates) + mv.w).dual()                    # [...] Point-


def slotted_tube(radius: float, opening: float, n_segments: int) -> tuple[Point, np.ndarray]:
    """Polygonal nodes (`[n_segments + 1] Point-`) and consecutive wall pairs (`[n_segments, 2]`) trace a
    circular arc whose slot faces `+x`."""
    angles = np.linspace(opening / 2, 2 * np.pi - opening / 2, n_segments + 1)  # [n_segments + 1]
    rotors = (mv.xy * (-angles / 2)).exp()                                  # [n_segments + 1] Rotor+
    nodes = rotors >> clamp_point([radius, 0.0])                            # [n_segments + 1] Point-
    return nodes, np.stack([np.arange(n_segments), np.arange(1, n_segments + 1)], axis=-1)


def section_complex(nodes: Point, wall_ends: np.ndarray, thickness: float) -> tuple[Topology, Point, Scalar]:
    """Node points (`[V] Point-`) and `wall_ends` (`[E, 2]`) assemble into a one-dimensional `Topology`
    together with vertex positions (`Point-[V]`) and wall thicknesses (`Scalar+[E]`)."""
    return (Topology(nodes.shape[0], wall_ends), nodes.field(),
            mv.scalar(np.full((len(wall_ends), 1), thickness)).field())


def section_walls() -> list[tuple[Topology, Point, Scalar]]:
    """Four thin-walled cross-sections (I-section, turned I-section, Z-section, and slotted tube)
    provide their `Topology`, node field (`Point-[V]`), and wall-thickness field (`Scalar+[E]`)."""
    rise, half_flange = web_height / 2, flange_width / 2
    # An I-section connects the web between flange midpoints and extends each flange to both sides.
    i_nodes = [[-half_flange, -rise], [0.0, -rise], [half_flange, -rise],
               [-half_flange, rise], [0.0, rise], [half_flange, rise]]
    i_ends = np.array([[1, 4], [1, 0], [1, 2], [4, 3], [4, 5]])
    # A Z-section extends the bottom flange left, runs up the web, and extends the top flange right.
    z_nodes = [[-flange_width, -rise], [0.0, -rise], [0.0, rise], [flange_width, rise]]
    z_ends = np.array([[0, 1], [1, 2], [2, 3]])
    turns = (mv.xy * (-np.deg2rad(section_turns) / 2)).exp()                # [3] Rotor+
    shapes = [(i_nodes, i_ends), (i_nodes, i_ends), (z_nodes, z_ends)]
    tube_nodes, tube_ends = slotted_tube(tube_radius, slot_angle, n_arc_segments)
    return [section_complex(turn >> clamp_point(np.array(coordinates)), ends, wall_thickness)
            for (coordinates, ends), turn in zip(shapes, turns)] + [section_complex(tube_nodes, tube_ends, tube_wall)]
