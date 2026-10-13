"""Polygon outlines and triangle connectivity for the bending scenes."""

import numpy as np
from numga import concatenate, stack

from examples.mechanics.bending.core import Point, mv, point

NAMES = ["I-section", "turned I-section", "Z-section", "slotted tube"]
TURNS = np.array([0.0, 30.0, 0.0])
HEIGHT, FLANGE, THICKNESS = 1.0, 0.6, 0.08
TUBE_RADIUS, TUBE_WALL = 0.5, 0.08
SLOT_ANGLE = np.deg2rad(60.0)
ARC_SEGMENTS = 48


def slotted_tube(radius: float, wall: float, opening: float, segments: int) -> tuple[Point, Point]:
    """An annular arc, triangulated between its outer and inner boundaries; the slot faces right."""
    angles = np.linspace(opening / 2, 2 * np.pi - opening / 2, segments + 1)
    turns = (mv.xy * (-angles / 2)).exp()
    outer = turns >> point([radius, 0.0])
    inner = turns >> point([radius - wall, 0.0])
    triangles = stack([
        stack([outer[:-1], outer[1:], inner[1:]], axis=-1),
        stack([outer[:-1], inner[1:], inner[:-1]], axis=-1),
    ]).reshape(-1, 3)
    return triangles, concatenate([outer, inner[::-1]], axis=0)


def triangulate(outline: np.ndarray) -> np.ndarray:
    """Triangle indices for a counter-clockwise simple polygon `[vertices, 2]`, by ear clipping."""
    remaining = np.arange(len(outline))
    faces = []
    while len(remaining) > 3:
        candidates = np.stack([np.roll(remaining, 1), remaining, np.roll(remaining, -1)], axis=-1)
        corners = outline[candidates]                                      # [candidates, 3, 2]
        edges = np.roll(corners, -1, axis=1) - corners
        offsets = outline[remaining][None, None, :, :] - corners[:, :, None, :]
        crosses = edges[:, :, None, 0] * offsets[..., 1] - edges[:, :, None, 1] * offsets[..., 0]
        inside = (crosses >= 0).all(axis=1)
        own_corner = (candidates[:, :, None] == remaining).any(axis=1)
        turns = edges[:, 0, 0] * edges[:, 1, 1] - edges[:, 0, 1] * edges[:, 1, 0]
        # An ear is convex and contains no other boundary vertex, including on its diagonal.
        ear = np.flatnonzero((turns > 0) & ~(inside & ~own_corner).any(axis=1))[0]
        faces.append(candidates[ear])
        remaining = np.delete(remaining, ear)
    return np.concatenate([np.array(faces), remaining[None, :]])


def section_meshes() -> list[tuple[Point, Point]]:
    """One pair of triangle corners and boundary vertices per section."""
    half_height, half_flange, half_web = HEIGHT / 2, FLANGE / 2, THICKNESS / 2
    shoulder = half_height - THICKNESS
    i_outline = [
        [-half_flange, -half_height], [half_flange, -half_height],
        [half_flange, -shoulder], [half_web, -shoulder],
        [half_web, shoulder], [half_flange, shoulder],
        [half_flange, half_height], [-half_flange, half_height],
        [-half_flange, shoulder], [-half_web, shoulder],
        [-half_web, -shoulder], [-half_flange, -shoulder],
    ]
    reach = FLANGE - half_web
    z_outline = [
        [-reach, -half_height], [half_web, -half_height],
        [half_web, shoulder], [reach, shoulder],
        [reach, half_height], [-half_web, half_height],
        [-half_web, -shoulder], [-reach, -shoulder],
    ]
    outlines = [np.array(i_outline), np.array(i_outline), np.array(z_outline)]
    turns = (mv.xy * (-np.deg2rad(TURNS) / 2)).exp()
    return [(turn >> point(outline[triangulate(outline)]), turn >> point(outline))
            for outline, turn in zip(outlines, turns)] + [slotted_tube(TUBE_RADIUS, TUBE_WALL, SLOT_ANGLE, ARC_SEGMENTS)]
