"""The rod and the ring on coarse disks: the rod's first step, drawn, its field's decay below the surface, and the
ring's trapped fields and kept quanta."""

from examples.electromagnetism.superconductor import core, render, scenarios


def test_the_rod_and_the_ring_run():
    topology, vertices, in_rod, frames = scenarios.cooled(divisions=8)
    first = next(frames)
    next(render.animate(topology, vertices, in_rod, scenarios.penetration, [first]))
    scenarios.penetrating(topology, vertices, first[0][0])

    topology, places, in_ring, hole_faces, in_hole, rim = scenarios.ring(ring_divisions=8)
    vertices = core.mv(core.Plane, places).dual().field()
    areas, gradients, cells = core.triangles(topology, vertices)
    scenarios.kept_quanta(topology, areas, gradients, cells, in_ring, hole_faces, in_hole, rim)
