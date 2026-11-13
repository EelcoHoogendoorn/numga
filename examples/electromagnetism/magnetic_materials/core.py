"""A permanent magnet, soft iron and a conductor share one magnetic field.

Magnetization supplies a source in the magnet. Permeability describes the iron's
induced magnetic moment, and conductivity delays changes in the conductor.
"""
from __future__ import annotations

from functools import reduce
from operator import and_
import numpy as np
from numga import Algebra, NumpyContext, stack
from numga.sparse import spdiag

ga = Algebra("x+y+z+t-w0")
mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                       # Scalar+
Point = ga.gatype.antivector()                    # Antivector-
Field = ga.gatype.bivector()                      # Bivector+
Moment = ga.gatype.antibivector()                 # Antibivector-
origin = mv.w.dual()
time = -mv.t.dual()
unread = mv.z.dual() & time
patches = stack((mv.xz.dual(), mv.yz.dual()))
magnetic_response = (patches * (Field & patches)).sum(axis=0)
sweep = mv.z.dual() & time
electric_response = sweep * (Field & sweep)


def medium(topology, vertices: Point, permeability: Scalar, conductivity: Scalar,
           remanence: Moment):
    """Assemble magnetic storage, conduction and remanent magnetization on one mesh."""
    corners = topology.incidence[2, 0]
    mean = corners.mean(axis=0)
    spans = reduce(and_, corners * vertices, unread)
    areas = spans.norm() / 2
    cells = ~mean * areas
    facing = topology.off[2, 0] * vertices
    complements = facing[0] & facing[1] & unread
    hats = complements / (complements & (corners * vertices))
    gradients = hats - (hats & origin) * mv.w
    # A spatial potential along z gives the magnetic field within this cross-section.
    hat_fields = gradients ^ mv.z
    response = magnetic_response / (mean * permeability)
    local = areas * (hat_fields[:, None] & response(hat_fields[None, :]))
    stiffness = (~corners[:, None] * spdiag(local) * corners[None, :]).sum(axis=0).sum(axis=0)
    source = (~corners * (areas * (hat_fields & (mean * remanence)))).sum(axis=0)
    mass = cells * conductivity
    extension = ~topology.reading(topology.interior[0], 0)
    return areas, cells, gradients, response, stiffness, source, mass, extension


def evolution(stiffness, source: Scalar, mass: Scalar, extension, coil: Scalar,
              drive: np.ndarray, step: float):
    """Evolve one spatial potential under a coil pulse, with the magnet always present."""
    initial = (~extension * stiffness * extension) * Scalar
    potential = extension * initial.solve(~extension * (source + coil * drive[0]))
    yield potential
    storage = spdiag(mass / step)
    implicit = (~extension * (storage + stiffness / 2) * extension) * Scalar
    explicit = storage - stiffness / 2
    # An isolated conductor redistributes charge until its net current along z vanishes.
    # Its uniform electric reading is determined alongside the spatial potential.
    uniform = extension * implicit.solve(~extension * (mass / step))
    total_mass = mass.sites.sum()
    denominator = 1 - (mass * uniform).sites.sum() / total_mass
    for previous, following in zip(drive[:-1], drive[1:]):
        load = source + coil * ((previous + following) / 2)
        mean_potential = (mass * potential).sites.sum() / total_mass
        particular = extension * implicit.solve(~extension * (explicit * potential + load - mass * mean_potential / step))
        mean_following = (mass * particular).sites.sum() / total_mass / denominator
        potential = particular + uniform * mean_following
        yield potential
