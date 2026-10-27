"""The thin-walled sections of the bending scenes: the nodes of each section's midline, the pairs of
nodes its walls run between, and the walls' thickness."""

import numpy as np

from examples.mechanics.bending.core import Point, mv, point

NAMES = ["I-section", "turned I-section", "Z-section", "slotted tube"]
# The turn of each I- and Z-section in its plane, in degrees.
TURNS = np.array([0.0, 30.0, 0.0])
# The height between the flanges' midlines, the flanges' width, and the walls' thickness.
HEIGHT, FLANGE, THICKNESS = 0.92, 0.6, 0.08
# The tube's midline radius, its wall's thickness, the opening of its slot, and the walls along it.
TUBE_RADIUS, TUBE_WALL = 0.46, 0.1
SLOT_ANGLE = np.deg2rad(60.0)
ARC_SEGMENTS = 48


def slotted_tube(radius: float, opening: float, segments: int) -> tuple[Point, np.ndarray]:
    """The nodes of an arc of a circle, and its walls between neighbours; the slot faces right."""
    angles = np.linspace(opening / 2, 2 * np.pi - opening / 2, segments + 1)
    nodes = (mv.xy * (-angles / 2)).exp() >> point([radius, 0.0])           # [segments + 1] Point-
    return nodes, np.stack([np.arange(segments), np.arange(1, segments + 1)], axis=-1)


def section_walls() -> list[tuple[Point, np.ndarray, np.ndarray]]:
    """The nodes, the walls as pairs of nodes, and the thickness of each wall, for every section."""
    rise, half = HEIGHT / 2, FLANGE / 2
    # An I-section: the web between the flanges' middles, and each flange out to both sides.
    i_nodes = [[-half, -rise], [0.0, -rise], [half, -rise], [-half, rise], [0.0, rise], [half, rise]]
    i_ends = np.array([[1, 4], [1, 0], [1, 2], [4, 3], [4, 5]])
    # A Z-section: the bottom flange out to the left, the web, and the top flange out to the right.
    z_nodes = [[-FLANGE, -rise], [0.0, -rise], [0.0, rise], [FLANGE, rise]]
    z_ends = np.array([[0, 1], [1, 2], [2, 3]])
    turns = (mv.xy * (-np.deg2rad(TURNS) / 2)).exp()
    shapes = [(i_nodes, i_ends), (i_nodes, i_ends), (z_nodes, z_ends)]
    nodes, ends = slotted_tube(TUBE_RADIUS, SLOT_ANGLE, ARC_SEGMENTS)
    return [(turn >> point(np.array(coordinates)), joined, np.full(len(joined), THICKNESS))
            for (coordinates, joined), turn in zip(shapes, turns)] + [(nodes, ends, np.full(len(ends), TUBE_WALL))]
