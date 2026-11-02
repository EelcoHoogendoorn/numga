"""The Meissner effect: a rod cooled through its transition in a field across it, then the field switched off, as a
superconductor and as a perfect conductor side by side; the field's decay into the cooled superconductor; and a ring
holding whole flux quanta, and the flux it keeps once cooled in a field and let go.

The superfluid's share of the electrons below the transition follows the two-fluid model's empirical law,
`1 - (temperature / transition) ** 4`, as in the Stockholm lecture notes on the two-fluid model and the London equations.
"""

from __future__ import annotations

import numpy as np

from examples.electromagnetism.superconductor import core
from examples.util import shapes

mv = core.mv

# The disk around the rod, in rod radii: its radius and its rings across it; the London penetration depth.
half, divisions, penetration = 2.0, 56, 0.12
# The steps: normal, cooling from the transition to a third of it, held cold in the field, the field switched off,
# and at rest; the time step, in the time a field takes to diffuse across the normal rod; the steps between frames, and
# the frames' duration in milliseconds.
n_normal, n_cooling, n_held, n_switching, n_rest = 10, 120, 20, 100, 20
coldest = 1 / 3
dt, stride, frame_duration_ms = 2e-3, 3, 40
# The half width of the band of triangles along the rod's upper radius read for the field's decay, in rod radii.
radial_width = 0.06
# The ring, in ring radii: its hole's radius and the rings of triangles across it; the fluxoids of the trapped fields
# drawn, the fluxoids weighed when cooling, and the applied fields it is cooled in, in flux quanta per squared radius.
hole_radius, ring_divisions = 0.5, 40
trapped_quanta = np.array([1.0, 2.0, 3.0])
candidate_quanta = np.arange(0.0, 6.0)
cooling_fields = np.linspace(0.0, 1.9, 77)


def schedule() -> tuple[np.ndarray, np.ndarray]:
    """The superfluid's share of the electrons, and the applied field, at each step `[steps]` each."""
    temperatures = np.concatenate([np.ones(n_normal), np.linspace(1, coldest, n_cooling),
                                   np.full(n_held + n_switching + n_rest, coldest)])
    applied = np.concatenate([np.ones(n_normal + n_cooling + n_held), np.linspace(1, 0, n_switching),
                              np.zeros(n_rest)])
    return 1 - np.minimum(temperatures, 1) ** 4, applied


def rod() -> tuple[core.Topology, np.ndarray, np.ndarray]:
    """The disk around the rod, in rings with one on the rod's edge: its topology, its vertices' places in the plane
    z = 0 `[vertices, 3]`, and which of them lie in the rod `[vertices]`."""
    topology, places = shapes.concentric_disk(np.array([1.0, half]), divisions)
    return topology, np.column_stack([places, 0 * places[:, 0]]), np.linalg.norm(places, axis=-1) <= 1.0 + 1e-9


def cooled():
    """The rod's potential and supercurrent over time, `[2] Scalar+[vertices]` each, as a superconductor and as a
    perfect conductor."""
    topology, places, in_rod = rod()
    vertices = mv(core.Plane, places).dual().field()                          # Point-[vertices]
    areas, gradients, cells = core.triangles(topology, vertices)
    inside = mv.scalar(in_rod[:, None] * 1.0).field()                         # Scalar+[vertices]
    heights = mv.scalar(places[:, 1:2]).field()                               # Scalar+[vertices]
    superfluid, applied = schedule()
    # The superconductor's superfluid responds to the potential itself; the perfect conductor's, to its change since the
    # rod cooled in the unit field.
    reference = mv.scalar(np.stack([0 * places[:, 1], places[:, 1]])[..., None]).field()   # [2] Scalar+[vertices]
    return topology, vertices, in_rod, core.cooling(topology, areas, gradients, cells, inside, topology.exterior[0],
                                                     heights, penetration, superfluid, applied, reference, dt, stride)


def ring() -> tuple[core.Topology, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The ring around its hole, in rings of triangles with one on the hole's edge: its topology, its vertices' places
    in the plane z = 0 `[vertices, 3]`, which triangles lie in the ring and which in the hole `[faces]` each, which
    vertices lie in the hole or on its edge `[vertices]`, and the rim's vertices."""
    topology, places = shapes.concentric_disk(np.array([hole_radius, 1.0]), ring_divisions)
    in_hole = np.linalg.norm(places, axis=-1) <= hole_radius + 1e-9
    hole_faces = in_hole[topology.elements[2]].all(axis=-1)
    return (topology, np.column_stack([places, 0 * places[:, 0]]), ~hole_faces, hole_faces, in_hole,
            topology.exterior[0])


def penetrating(topology: core.Topology, vertices: core.Point, potential: core.Scalar) -> tuple[core.Scalar, core.Scalar]:
    """The depth below the rod's surface `[triangles] Scalar+` and the field's strength `[triangles] Scalar+` in the
    triangles along the rod's upper radius, from the potential `Scalar+[vertices]`."""
    corners = topology.incidence[2, 0]                                        # [corners] Scalar+[faces <- vertices]
    _, gradients, _ = core.triangles(topology, vertices)
    # The magnetic field, the potential's gradient wedged with its direction, a bivector in planes through z.
    field = (gradients ^ ((corners * potential[..., None]) * mv.z)).sum(axis=-1)   # Field+[faces]
    # Each triangle's middle, its distance from the rod's axis, and the triangles along the upper radius inside the rod.
    centroids = corners.mean(axis=0) * vertices                               # Point-[faces]
    radii = centroids.squared().square_root()                                 # Scalar+[faces]
    along = ((mv.x & centroids).abs() < radial_width) & ((mv.y & centroids) > 0) & (radii < 1)   # [faces]
    return (1 - radii).batch()[along], field.norm().batch()[along]


def kept_quanta(topology: core.Topology, areas: core.Scalar, gradients: core.Plane, cells: core.Scalar,
                in_ring: np.ndarray, hole_faces: np.ndarray, in_hole: np.ndarray, rim: np.ndarray) -> np.ndarray:
    """The fluxoid a ring keeps when cooled in each of the cooling fields `[fields]`: of the candidates, the one whose
    field costs the least, its departure from the applied field and its current's kinetic energy together."""
    corners = topology.incidence[2, 0]                                        # [corners] Scalar+[faces <- vertices]
    fields = core.ring_fields(topology, areas, gradients, in_ring, hole_faces, in_hole, rim, penetration,
                              candidate_quanta[:, None], cooling_fields[None, :])   # [candidates, fields] Scalar+[vertices]
    # The supercurrent in each triangle: the field strength's gradient wedged with a unit axial field's excitation.
    slopes = (gradients * (corners * fields[..., None])).sum(axis=-1)         # [candidates, fields] Plane+[faces]
    currents = slopes ^ mv.xy.dual()                                          # [candidates, fields] Current-[faces]
    ring_areas = mv.scalar(in_ring[:, None] * 1.0).field() * areas            # Scalar+[faces]
    departure = fields - cooling_fields[None, :]                              # [candidates, fields] Scalar+[vertices]
    kinetic = penetration ** 2 * (ring_areas * (currents & core.inertia(currents))).sites.sum()   # [candidates, fields] Scalar+
    costs = (cells * departure * departure).sites.sum() + kinetic
    return candidate_quanta[costs.argmin(axis=0)]


def main() -> None:
    from examples.util.animation import save_figure
    from examples.electromagnetism.superconductor import render

    topology, vertices, in_rod, frames = cooled()
    save_figure(render.draw_rod(topology, vertices, in_rod), "superconductor_rod")
    frames = list(frames)
    # The superconductor at the end of its hold in the field.
    held = frames[(n_normal + n_cooling + n_held - 1) // stride][0][0]
    save_figure(render.draw_penetration(*penetrating(topology, vertices, held), penetration), "superconductor_depth")
    # The ring, with whole flux quanta trapped in it, and the quanta it keeps when cooled in a field and let go.
    topology, places, in_ring, hole_faces, in_hole, rim = ring()
    vertices = mv(core.Plane, places).dual().field()                          # Point-[vertices]
    areas, gradients, cells = core.triangles(topology, vertices)
    trapped = core.ring_fields(topology, areas, gradients, in_ring, hole_faces, in_hole, rim, penetration,
                               trapped_quanta, 0 * trapped_quanta)            # [quanta] Scalar+[vertices]
    save_figure(render.draw_trapped(topology, vertices, in_hole, trapped), "superconductor_ring")
    kept = kept_quanta(topology, areas, gradients, cells, in_ring, hole_faces, in_hole, rim)
    left = core.ring_fields(topology, areas, gradients, in_ring, hole_faces, in_hole, rim, penetration, kept,
                            0 * kept)                                         # [fields] Scalar+[vertices]
    save_figure(render.draw_staircase(cooling_fields, (left * cells).sites.sum()), "superconductor_staircase")


def pinned() -> None:
    """The pictures the repository tracks: the rod cooled in the field as a superconductor and as a perfect conductor side by side, for the gallery."""
    from examples.util.animation import pin_animation
    from examples.electromagnetism.superconductor import render

    topology, vertices, in_rod, frames = cooled()
    pin_animation(render.animate(topology, vertices, in_rod, penetration, frames), "superconductor", frame_duration_ms)


if __name__ == "__main__":
    main()
