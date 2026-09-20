"""Recover a principal frame and its motor with two eigenproblems.

Both constructions start from the supplied inertia: an arbitrary mass cloud,
and a moved diagonal tensor. The reference is just the canonical coordinate
planes, not a known body pose. The same solve covers spherical and Euclidean
PGA models in algebra dimensions four and five.

Run from rewrite/ with PYTHONPATH=src:. python -m examples.mechanics.inertia.
"""

import numpy as np

from numga import Algebra, Extensor, NumpyContext
from examples.mechanics.inertia_plumbing import cloud, report


def second_moment(inertia: Extensor) -> Extensor:
    """Recover the point cloud's second-moment map from its inertia.

    Args:
        inertia: A single physical inertia map, AntiBivector <- Bivector,
            mapping rigid-body velocity to momentum in its current frame.
    Returns:
        A Point <- Plane map. Pairing its output with another plane gives
        the scalar form sum(m * (a & p) * (b & p)) over the mass points p.
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

    Args:
        moment: The Point <- Plane second-moment map recovered from inertia.
        reference: A batch of orthogonal unit target planes, with PGA's null
            plane last. Their batch order specifies the target correspondence.
    Returns:
        A normalized motor mapping the principal planes onto reference,
        up to orientation signs. Coordinate reference planes diagonalize inertia.
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


def main(signature: str = "x+y+z+w0", seed: int = 0) -> tuple[Extensor, Extensor]:
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

    # --- plumbing: inspect both results; the recovered frame may permute or reverse axes.
    report(Bivector & cloud_inertia, Bivector & cloud_diagonal,
           Bivector & diagonal, Bivector & moved, Bivector & recovered)
    return cloud_motor, motor


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signature", default="x+y+z+w0")
    parser.add_argument("--seed", type=int, default=0)
    main(**vars(parser.parse_args()))
