"""Recover a principal frame and its motor with two eigenproblems.

Both constructions start from the supplied inertia: an arbitrary mass cloud,
and a moved diagonal tensor. The reference is just the canonical coordinate
planes, not a known body pose. The same solve covers spherical and Euclidean
PGA models in algebra dimensions four and five.

Run from the repository root with python -m examples.mechanics.inertia.
"""

import numpy as np

from numga import Algebra, Extensor, NumpyContext


# --- plumbing -------------------------------------------------------------------------
def cloud(context: NumpyContext, seed: int) -> tuple[Extensor, np.ndarray]:
    """Normalize random points and place the cloud with a random motor."""
    ga, mv = context.algebra, context.multivector
    rng = np.random.default_rng(seed)
    points = mv.antivector(rng.normal(size=(80, ga.dimension))).normalized()
    placement = mv.bivector(rng.normal(size=len(ga.subspace.bivector())) * 0.3).exp()
    return placement >> points, rng.uniform(0.5, 1.5, size=80)


# --- math -----------------------------------------------------------------------------
def second_moment(inertia: Extensor) -> Extensor:
    """Recover the point cloud's second-moment map from its inertia.

    The inertia is a single physical inertia map, AntiBivector <- Bivector, mapping
    rigid-body velocity to momentum in its current frame. The result is a Point <- Plane
    map: pairing its output with another plane gives the scalar form
    sum(m * (a & p) * (b & p)) over the mass points p.
    """
    ga = inertia.algebra
    Point, Plane, Bivector = ga.gatype.antivector(), ga.gatype.vector(), ga.gatype.bivector()
    # This construction has type (AntiBivector <- Point, Plane, Bivector).
    # Duality turns the second input into a point while keeping its plane slot.
    construction = Point & Plane.dual().commutator(Bivector)
    # The supplied inertia matches AntiBivector <- Bivector, leaving Point <- Plane
    # as the unknown second-moment map. The solve infers these axes from the types.
    return construction.lstsq(inertia)


def diagonalizing_motor(moment: Extensor, reference: Extensor) -> Extensor:
    """Fit a motor taking the moment's principal planes onto a reference frame.

    The moment is the Point <- Plane second-moment map recovered from inertia. The
    reference is a batch of orthogonal unit target planes, with PGA's null plane last;
    their batch order specifies the target correspondence. The result is a normalized
    motor mapping the principal planes onto the reference, up to orientation signs.
    Coordinate reference planes diagonalize inertia.
    """
    ga = moment.algebra
    Plane, Rotor = ga.gatype.vector(), ga.gatype.rotor()

    # Pairing with another plane gives the scalar second-moment form: Plane & moment.
    # For mass points p, this form is sum(m * (a & p) * (b & p)).
    # Recover principal planes from metric(v, .) = value * moment_form(v, .).
    # This order gives PGA's plane at infinity a zero eigenvalue, rather than an infinite one.
    _, planes = (Plane | Plane).eigh(Plane & moment)
    # Keep the full frame, with PGA's null plane last, matching the reference.
    source = planes[planes.norm().argsort()[::-1]]

    # Descending norms and powers of two give distinct eigenvalues for the nonnull sign choices.
    # The paired null planes contribute zero on the rotor space.
    weights = 2.0 ** np.arange(source.shape[0])[::-1]
    # Build a map from rotors to even grade elements
    # A matching motor is an eigenvector of each term: the product returns R
    # times the source plane's norm, with either sign for its orientation.
    # The plane maps commute, so this weighted sum finds their common
    # eigenvectors in one solve. The weights keep distinct sign choices apart.
    _, motors = (reference * Rotor * source * weights).sum(axis=0).eig()
    # Scalar overlap selects a motor; the pure ideal PGA candidates have none.
    choice = motors.select[0].norm().argmax()
    return motors[choice].normalized()


def main(signature: str, seed: int) -> tuple[Extensor, Extensor]:
    # --- plumbing: inputs for the two independent constructions.
    ga = Algebra(signature)
    ctx = NumpyContext(ga, dtype=np.complex128)
    mv = ctx.multivector
    Bivector = ga.gatype.bivector()
    points, masses = cloud(ctx, seed)
    reference = mv.vector(np.eye(ga.dimension))
    reference = reference[reference.norm().argsort()[::-1]]
    point_basis = mv.antivector(np.eye(ga.dimension))
    second_moments = 2.0 ** np.arange(ga.dimension)
    rng = np.random.default_rng(seed + 1)
    placement = mv.bivector(rng.normal(size=len(Bivector.output_subspace)) * 0.2).exp()

    # --- math: 1. Construct and diagonalize inertia from arbitrary mass points.
    cloud_inertia = (points & points.commutator(Bivector) * masses).sum(axis=0)
    cloud_moment = second_moment(cloud_inertia)
    cloud_motor = diagonalizing_motor(cloud_moment, reference)
    cloud_diagonal = cloud_motor >> cloud_inertia(cloud_motor << Bivector)

    # 2. Diagonal second moments induce a diagonal energy form on bivectors.
    # Move that inertia, then find a diagonalizing motor from the moved tensor.
    diagonal = (point_basis & point_basis.commutator(Bivector) * second_moments).sum(axis=0)
    moved = placement >> diagonal(placement << Bivector)
    moment = second_moment(moved)
    motor = diagonalizing_motor(moment, reference)
    recovered = motor >> moved(motor << Bivector)

    # --- checks
    # Both aligned inertias have a diagonal energy form on the bivector blades. The round
    # trip recovers the diagonal it started from, up to a permutation of the axes.
    blades = mv.bivector(np.eye(len(Bivector.output_subspace)))
    energies = [(blades[:, None] & aligned(blades[None, :])).to_array().real
                for aligned in (cloud_diagonal, recovered, diagonal)]
    for energy in energies[:2]:
        off_diagonal = energy - np.diag(np.diag(energy))
        assert np.abs(off_diagonal).max() < 1e-12 * np.abs(energy).max()
    np.testing.assert_allclose(np.sort(np.diag(energies[1])), np.sort(np.diag(energies[2])), rtol=1e-9)
    return cloud_motor, motor


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signature", default="x+y+z+w0")
    parser.add_argument("--seed", type=int, default=0)
    main(**vars(parser.parse_args()))
