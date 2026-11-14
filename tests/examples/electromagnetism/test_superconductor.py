"""Coverage tests for Meissner field expulsion in a superconducting rod and flux quantization in a ring."""

import matplotlib.pyplot as plt
import numpy as np

from examples.electromagnetism.superconductor import core, render, scenarios


def test_rod_and_ring_scenarios() -> None:
    """Coarse disks verify rod cooling, radial decay, ring flux quantization, and all rendering entry points."""
    topology, vertices, in_rod, states = scenarios.cooling_run(8)
    corners = topology.incidence[2, 0]                                        # [3] Scalar+[F <- V]
    areas = core.triangle_areas(topology, vertices)                           # Scalar+[F]
    gradients = core.simplex_gradients(topology, vertices)                    # [3] Plane+[F]
    heights = core.mv.y & vertices                                            # Scalar+[V]
    assert (areas > 0).all()
    assert (((gradients * (corners * heights)).sum(axis=0) - core.mv.y).norm() < 1e-10).all()

    rod_fig = render.rod_figure(topology, vertices, in_rod)
    plt.close(rod_fig)

    first_state = next(states)
    frame = next(render.cooling_animation(topology, vertices, in_rod, scenarios.penetration_depth, [first_state]))
    assert frame.ndim == 3 and frame.shape[-1] == 3

    field = core.magnetic_field(topology, gradients, first_state[0][0])       # Field+[F]
    depths, strengths = core.radial_profile(topology, vertices, field, 0.25)
    assert (depths >= 0).all() and (strengths > 0).all()
    depth_fig = render.penetration_figure(depths, strengths, scenarios.penetration_depth)
    plt.close(depth_fig)

    ring_topology, ring_coordinates, in_hole = scenarios.ring_scene(8)
    ring_vertices = core.mv(core.Plane, ring_coordinates).dual().field()      # Point-[V]
    ring_areas = core.triangle_areas(ring_topology, ring_vertices)            # Scalar+[F]
    ring_gradients = core.simplex_gradients(ring_topology, ring_vertices)     # [3] Plane+[F]
    trapped_fields = core.ring_fields(
        ring_topology, ring_areas, ring_gradients, in_hole,
        scenarios.penetration_depth, scenarios.trapped_quanta, 0 * scenarios.trapped_quanta,
    )                                                                         # [3] Scalar+[V]
    trapped_fig = render.trapped_figure(ring_topology, ring_vertices, in_hole, trapped_fields)
    plt.close(trapped_fig)

    coarse_cooling = np.linspace(0.0, 1.5, 5)                                 # [5]
    retained_quanta = core.equilibrium_quanta(
        ring_topology, ring_areas, ring_gradients, in_hole,
        scenarios.penetration_depth, scenarios.candidate_quanta, coarse_cooling,
    )                                                                         # [5]
    ring_corners = ring_topology.incidence[2, 0]                              # [3] Scalar+[F <- V]
    ring_dual_areas = (ring_corners.adjugate() * (ring_areas / 3)).sum(axis=0)   # Scalar+[V]
    retained_fields = core.ring_fields(
        ring_topology, ring_areas, ring_gradients, in_hole,
        scenarios.penetration_depth, retained_quanta, np.zeros_like(retained_quanta, dtype=float),
    )                                                                         # [5] Scalar+[V]
    retained_flux = (retained_fields * ring_dual_areas).sites.sum()           # [5] Scalar+
    assert (retained_flux >= -1e-6).all()
    staircase_fig = render.staircase_figure(coarse_cooling, retained_flux)
    plt.close(staircase_fig)
