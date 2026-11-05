"""Discrete minimal surfaces: the math imports no plotting, and the Plateau relaxation, the harmonic conjugates
and Bonnet's isometric family run on coarse disks and a coarse catenoid, one step and one frame each, and draw."""

import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np

from examples.surfaces.minimal_surfaces import core, render, scenarios


def test_minimal_surfaces_math_does_not_import_plotting():
    probe = (
        "import sys, examples.surfaces.minimal_surfaces.core, examples.surfaces.minimal_surfaces.scenarios; "
        "print([m for m in sys.modules if m.split('.')[0] in ('matplotlib', 'PIL')])"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]", out.stdout


def test_the_scenes_run_and_draw():
    plateau_topology, flat_vertices, initial_vertices = scenarios.four_arch_dome(1.0, 4, 0.35, 0.55)
    states = list(core.relax_plateau(plateau_topology, initial_vertices, 0.3, 1))
    assert len(list(render.animate_plateau(plateau_topology, flat_vertices, states[:1]))) == 1

    minimal_vertices, _, _ = states[-1]
    plateau_conjugate, _ = core.conjugate_minimal_surface(plateau_topology, minimal_vertices)
    enneper_topology, enneper_flat, enneper_vertices, _ = scenarios.enneper_disk(0.8, 4)
    enneper_conjugate, _ = core.conjugate_minimal_surface(enneper_topology, enneper_vertices)
    plt.close(render.draw_conjugates(
        plateau_topology, flat_vertices, minimal_vertices, plateau_conjugate,
        enneper_topology, enneper_flat, enneper_vertices, enneper_conjugate,
    ))

    catenoid_topology, catenoid, exact_helicoid, rings, meridians = scenarios.slit_catenoid(1.0, 0.7 * np.pi, 8, 12)
    family = core.bonnet_family(catenoid_topology, catenoid, exact_helicoid, np.array([np.pi / 4]))
    assert len(list(render.animate_bonnet(catenoid_topology, rings, meridians, family))) == 1
