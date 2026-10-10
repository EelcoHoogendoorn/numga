"""Optical layers from their constitutive maps in spacetime algebra.

The surface Maxwell residual packs the tangential field and excitation into separate
trivector and vector grades. Two remaining Maxwell constraints determine the longitudinal
fields. Solving them reconstructs the field bivector from four boundary amplitudes;
the temporal residual then gives their propagation generator. Both polarizations stay
coupled, including when the material axes turn or its axion response changes.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from functools import reduce
from itertools import accumulate

import numpy as np

from numga import NumpyContext, stack
from numga.algebras import STA as ga
from numga.sparse import SparseExtensor

mv = NumpyContext(ga, dtype=np.complex128).multivector
# Real fields along z at normal incidence, stepped in time: no phases, so no complex numbers.
real = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
Vector = ga.gatype.vector()
Bivector = ga.gatype.bivector()
Antibivector = ga.gatype.antibivector()
Polarization = ga.gatype(ga.subspace("x y"))
Boundary = ga.gatype(ga.subspace("x y txz tyz"))
Constitutive = ga.gatype((Antibivector, Bivector))
Inverse = ga.gatype((Bivector, Antibivector))
BoundaryMap = ga.gatype((Boundary, Bivector))
FieldMap = ga.gatype((Bivector, Boundary))
Propagation = ga.gatype((Boundary, Boundary))
PolarizationMap = ga.gatype((Polarization, Polarization))
ElectricMap = ga.gatype((Vector, Polarization))
Outgoing = ga.gatype((Boundary, Polarization))
Readout = ga.gatype((Polarization, Boundary))
axes = stack([mv.x, mv.y, mv.z])
field_boundary = mv.t ^ (mv.t | (mv.z ^ Bivector))
excitation_boundary = mv.t | (mv.t ^ (mv.z ^ Antibivector).dual())
# Undo the two wedges defining the electric part of the boundary state.
electric_trace = -mv.t | (mv.z | Boundary)                 # Polarization <- Boundary


# --- math -----------------------------------------------------------------------------
def dielectric(permittivity: np.ndarray, reluctivity: np.ndarray) -> Constitutive:
    """Principal electric responses and scalar inverse permeability, in vacuum units."""
    planes = axes ^ mv.t
    electric_response = (permittivity * planes * (planes | Bivector)).sum(axis=-1)
    magnetic_response = (Bivector + (mv.t >> Bivector)) / 2
    return (electric_response + reluctivity * magnetic_response).dual()


@dataclass(frozen=True)
class Medium:
    boundary: BoundaryMap
    reconstruct: FieldMap
    generator: Propagation

    @classmethod
    def from_chi(cls, chi: Constitutive, parallel: Vector) -> Medium:
        """Reduce Maxwell to a layer normal to z, at a fixed tangential wave covector.

        Parallel includes the unit temporal component t. The longitudinal material
        response must be invertible; no free surface charge or current is present.
        """
        # The four shared boundary amplitudes are continuous between different media.
        boundary = field_boundary + excitation_boundary(chi)
        # The remaining two equations contain no normal derivative: impose them as zero.
        constraint = (-(mv.z | (mv.z ^ (parallel ^ Bivector)))
                      - mv.z * (mv.z | (parallel ^ chi).dual()))
        reconstruct = (boundary + constraint).solve(Boundary)
        # Positive eigenvalues give phase propagation towards +z when temporal phase increases.
        generator = (field_boundary(mv.z | (parallel ^ reconstruct))
                     + excitation_boundary(mv.z | (parallel ^ chi(reconstruct))))
        return cls(boundary, reconstruct, generator)

    def interior(self, thicknesses: np.ndarray, wavelength: float,
                 exit_state: Boundary, fractions: np.ndarray) -> Bivector:
        """Physical fields inside a stack, `[cases, layers, samples]`.

        Material maps and thicknesses have shape [layers]. The exit state has shape
        [cases], and fractions [samples] measure distance from each layer's left face
        to its right face.
        """
        optical_distances = 2 * np.pi * thicknesses / wavelength                  # [layers]
        crossings = propagate(self.generator, optical_distances)                   # [layers] Boundary <- Boundary
        # Each layer's right face holds the exit state carried back through every layer after it.
        faces = accumulate(crossings[:0:-1], lambda state, crossing: crossing(state), initial=exit_state)
        right_faces = stack(tuple(faces)[::-1], axis=-1)                           # [cases, layers] Boundary
        # From each right face back to the samples inside its layer, all layers at once.
        inward = propagate(self.generator[:, None], optical_distances[:, None] * (1 - fractions))  # [layers, samples] Boundary <- Boundary
        return self.reconstruct[:, None](inward(right_faces[..., None]))         # [cases, layers, samples] Bivector


def propagate(generator: Propagation, optical_distance: np.ndarray) -> Propagation:
    """Carry boundary fields right to left through a diagonalizable homogeneous layer.

    Optical distance is vacuum wavenumber times thickness. Modes need not be
    orthogonal: solving through their summed dyads supplies their reciprocal readout.
    """
    indices, modes = generator.eig()                              # [..., modes] Scalar, Boundary
    frame = (modes * modes.scalar_product(Boundary)).sum(axis=-1)  # [...] Boundary <- Boundary
    # Each reciprocal reads the amplitude of its own mode and of no other.
    reciprocal = frame[..., None].solve(modes)                     # [..., modes] Boundary
    phases = (indices * (1j * optical_distance[..., None])).exp()  # [..., modes] Scalar
    return (phases * modes * reciprocal.scalar_product(Boundary)).sum(axis=-1)


def compose(layers: Propagation) -> Propagation:
    """Layers are ordered from the incident medium to the substrate, on the first batch axis."""
    return reduce(lambda propagation, layer: propagation(layer), layers)


@dataclass(frozen=True)
class Ports:
    outgoing: Outgoing
    incoming: Readout
    returning: Readout
    electric_field: ElectricMap
    normal_index: Scalar

    @classmethod
    def from_medium(cls, medium: Medium, index: float, parallel: Vector) -> Ports:
        """Waves in a lossless, nonmagnetic isotropic exterior, including a real constant axion term.

        The normal index is real and nonzero: the exterior waves must propagate.
        Polarization specifies the tangential electric field, including its phase.
        """
        normal_index = (index ** 2 + (parallel - mv.t).squared()).square_root()
        # Complete the tangential electric field to a field perpendicular to the ray.
        electric_field = Polarization + mv.z * (parallel | Polarization) / normal_index
        wavevector = parallel + mv.z * normal_index
        outgoing = medium.boundary(wavevector ^ electric_field)
        # Exterior evolution has just two eigenvalues, the signed normal indices.
        incoming = electric_trace((Boundary + medium.generator / normal_index) / 2)
        returning = electric_trace((Boundary - medium.generator / normal_index) / 2)
        return cls(outgoing, incoming, returning, electric_field, normal_index)

    @classmethod
    def isotropic(cls, index: float, parallel: Vector) -> Ports:
        """Waves in a lossless, nonmagnetic isotropic exterior of the given refractive index."""
        chi = dielectric(np.full(3, index ** 2), 1.0)
        return cls.from_medium(Medium.from_chi(chi, parallel), index, parallel)

    def power(self, polarization: Polarization) -> Scalar:
        """Normal power flux, omitting the common factor of half the vacuum admittance."""
        field = self.electric_field(polarization)
        # Complex coefficients carry time phase. Average the two real quadratures;
        # the spatial metric is negative, hence the minus sign in their squared lengths.
        magnitude = -(field.real().scalar_norm_squared()
                      + (-1j * field).real().scalar_norm_squared())
        return self.normal_index.real() * magnitude


@dataclass(frozen=True)
class Scattering:
    reflection: PolarizationMap
    transmission: PolarizationMap

    def powers(self, polarization: Polarization, incident: Ports,
               substrate: Ports) -> tuple[Scalar, Scalar]:
        normalization = incident.power(polarization)
        reflected = incident.power(self.reflection(polarization)) / normalization
        transmitted = substrate.power(self.transmission(polarization)) / normalization
        return reflected, transmitted


def scatter(entrance: Outgoing, incident: Ports) -> Scattering:
    """Solve the interface conditions with the incident polarization left open.

    The entrance state is the substrate's outgoing wave, carried back through any coating to the
    entrance face.
    """
    transmission = incident.incoming(entrance).inverse()
    reflection = incident.returning(entrance)(transmission)
    return Scattering(reflection, transmission)


@dataclass(frozen=True)
class Line:
    """A periodic line of cells along z. Electric planes sit on the nodes, magnetic planes halfway
    between each node and the next."""
    response: Constitutive                                         # [nodes] Antibivector <- Bivector
    inverse: Inverse                                               # [nodes] Bivector <- Antibivector
    between: Constitutive                                          # the medium between nodes
    difference: SparseExtensor                                     # [nodes, nodes] Scalar cells
    spacing: float

    @classmethod
    def from_chi(cls, nodes: Constitutive, between: Constitutive, spacing: float) -> Line:
        """The material at each node, and the one material between nodes, which only meets magnetic
        planes. The difference reads, at each node, the next node's value less its own: from the
        electric planes on either side of a magnetic plane."""
        count = len(nodes)
        sites = np.arange(count)
        difference = SparseExtensor(real.scalar(np.tile([[1.0], [-1.0]], (count, 1))), np.repeat(sites, 2),
                                    np.stack([(sites + 1) % count, sites], axis=-1).reshape(-1), (count, count))
        return cls(nodes, nodes.inverse(), between, difference, spacing)

    def energy(self, electric: Bivector, magnetic: Bivector) -> Scalar:
        """The field energy in each cell; magnetic planes pair with their excitation to minus their
        energy."""
        return ((electric & self.response(electric)) - (magnetic & self.between(magnetic))) * (self.spacing / 2)

    def evolve(self, electric: Bivector, magnetic: Bivector, interval: float,
               count: int) -> Iterator[tuple[Bivector, Bivector]]:
        """Both exterior Maxwell equations, `d ^ field == 0` and `d ^ excitation == 0`, as a leapfrog.

        The time part of `d ^ X` is `t ^ (dX / dt)`; contracting with the observer `t` undoes that
        wedge, so each update is minus `t | (z ^ dX / dz)`. Magnetic planes move with the electric
        ones on either side; the excitation's spatial planes move with the excitation of the magnetic
        planes on either side, and the inverse response turns them back into electric planes.
        """
        rate = interval / self.spacing
        for _ in range(count):
            magnetic = magnetic - (real.t | (real.z ^ (self.difference * electric.field()).batch())) * rate
            excitation = self.between(magnetic)
            # The reverse runs each difference back, from the magnetic planes to the electric planes
            # between them: the adjoint pair under which the leapfrog keeps its energy.
            displacement = (real.t | (real.z ^ (~self.difference * excitation.field()).batch())) * rate
            electric = electric + self.inverse(displacement)
            yield electric, magnetic
