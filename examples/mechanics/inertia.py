"""Principal frames and diagonalizing motors induce two eigenproblems on an inertia map.

Both constructions recover a principal frame from a supplied inertia map alone, first on an arbitrary mass cloud and then on a moved diagonal inertia. Canonical coordinate planes serve as the reference frame without a known initial pose, across spherical and Euclidean projective models in four and five dimensions.

Run from the repository root with `python -m examples.mechanics.inertia`.
"""

import numpy as np

from numga import Algebra, Extensor, NumpyContext


# --- plumbing -------------------------------------------------------------------------
def cloud(context: NumpyContext, seed: int) -> tuple[Extensor, np.ndarray]:
    """Random unit points placed by a random motor come paired with random positive masses (`[80] Point-`, `[80]`)."""
    ga, mv = context.algebra, context.multivector
    rng = np.random.default_rng(seed)
    n_points = 80
    rest_points = mv.vector(rng.normal(size=(n_points, ga.dimension))).dual().normalized()   # [n_points] Point-
    placement = mv.bivector(rng.normal(size=len(ga.subspace.bivector())) * 0.3).exp()        # [] Rotor+
    points = placement >> rest_points                                                        # [n_points] Point-
    masses = rng.uniform(0.5, 1.5, size=n_points)                                            # [n_points]
    return points, masses


# --- math -----------------------------------------------------------------------------
def second_moment(inertia: Extensor) -> Extensor:
    """Linear least squares recovers the exact `Point- <- Plane+` second-moment map from an `Antibivector- <- Bivector+` inertia map.

    Pairing the recovered map's output with a second plane reproduces the scalar second-moment form `(masses * (first_plane & points) * (second_plane & points)).sum()`.
    """
    ga = inertia.algebra
    Point = ga.gatype.antivector()
    Plane = ga.gatype.vector()
    Rate = ga.gatype.bivector()
    # Duality turns the plane input into a point so the linear map matches `Antibivector- <- (Point-, Plane+, Bivector+)`.
    moment_to_inertia = Point.regressive(Plane.dual().commutator(Rate))              # [] Antibivector- <- (Point-, Plane+, Bivector+)
    # Unbinding the matched `Antibivector- <- Bivector+` inertia solves an overdetermined consistent system for `Point- <- Plane+`.
    return moment_to_inertia.lstsq(inertia)                                          # [] Point- <- Plane+


def diagonalizing_motor(moment: Extensor, reference: Extensor) -> Extensor:
    """Two eigenproblems fit a normalized motor (`Rotor+`) mapping the moment's principal planes onto `reference`.

    The `moment` map has type `Point- <- Plane+`, and `reference` holds orthogonal unit target planes with any null plane last. Coordinate reference planes bring the inertia map to diagonal form.
    """
    ga = moment.algebra
    Plane = ga.gatype.vector()
    Rotor = ga.gatype.rotor()

    # Generalized eigenvectors of the metric `Plane | Plane` against `Plane & moment` give any null plane at infinity eigenvalue zero.
    _, principal_planes = (Plane | Plane).eigh(Plane & moment)                       # [...] Plane+
    # Sorting by descending norm places any null plane last, matching `reference`.
    source_planes = principal_planes[principal_planes.norm().argsort()[::-1]]        # [...] Plane+

    # Powers of two separate the eigenvalues of distinct nonnull orientation-sign choices while paired null planes contribute zero on rotors.
    n_planes = source_planes.shape[0]
    plane_weights = 2 ** np.arange(n_planes)[::-1]                                   # [n_planes]
    # Because the orthogonal plane reflections commute, one eigensolve of their weighted sum finds the common motor eigenvectors.
    alignment_map = (reference * Rotor * source_planes * plane_weights).sum(axis=0)  # [] Even+ <- Rotor+
    _, candidate_motors = alignment_map.eig()                                        # [...] Rotor+
    # Selecting the largest scalar part picks a proper motor over pure ideal candidates.
    best_index = candidate_motors.select[0].norm().argmax()
    return candidate_motors[best_index].normalized()                                 # [] Rotor+


def main(signature: str, seed: int) -> tuple[Extensor, Extensor]:
    """Both a random mass cloud and a moved diagonal inertia diagonalize under their recovered motors."""
    # --- plumbing: inputs for the two independent constructions.
    ga = Algebra(signature)
    context = NumpyContext(ga, dtype=np.complex128)
    mv = context.multivector
    Rate = ga.gatype.bivector()
    n_planes = ga.dimension
    n_rates = len(Rate.output_subspace)

    points, masses = cloud(context, seed)                                            # [80] Point-, [80]
    basis_planes = mv.vector(np.eye(n_planes, dtype=int))                            # [n_planes] Plane+
    reference = basis_planes[basis_planes.norm().argsort()[::-1]]                    # [n_planes] Plane+
    point_basis = basis_planes.dual()                                                # [n_planes] Point-
    second_moments = 2 ** np.arange(n_planes)                                        # [n_planes]
    rng = np.random.default_rng(seed + 1)
    placement = mv.bivector(rng.normal(size=n_rates) * 0.2).exp()                    # [] Rotor+

    # --- math: 1. Construct and diagonalize inertia from arbitrary mass points.
    cloud_inertia = (masses * points).regressive(points.commutator(Rate)).sum(axis=0)  # [] Antibivector- <- Bivector+
    cloud_moment = second_moment(cloud_inertia)                                      # [] Point- <- Plane+
    cloud_motor = diagonalizing_motor(cloud_moment, reference)                       # [] Rotor+
    cloud_diagonal = cloud_motor >> cloud_inertia(cloud_motor << Rate)               # [] Antibivector- <- Bivector+

    # 2. Diagonal second moments induce a diagonal energy form on bivectors; move it and recover a diagonalizing motor.
    diagonal_inertia = (second_moments * point_basis).regressive(point_basis.commutator(Rate)).sum(axis=0)  # [] Antibivector- <- Bivector+
    placed_inertia = placement >> diagonal_inertia(placement << Rate)                # [] Antibivector- <- Bivector+
    moment = second_moment(placed_inertia)                                           # [] Point- <- Plane+
    motor = diagonalizing_motor(moment, reference)                                   # [] Rotor+
    recovered_inertia = motor >> placed_inertia(motor << Rate)                       # [] Antibivector- <- Bivector+

    # checks
    # Both aligned inertias have a diagonal energy form on the bivector blades, and the round trip preserves the spectrum.
    rate_blades = mv.bivector(np.eye(n_rates, dtype=int))                            # [n_rates] Bivector+
    off_diagonal_mask = ~np.eye(n_rates, dtype=bool)                                 # [n_rates, n_rates]
    for aligned_inertia in (cloud_diagonal, recovered_inertia):
        energy_form = rate_blades[:, None] & aligned_inertia(rate_blades[None, :])   # [n_rates, n_rates] Scalar+
        scale = energy_form.abs().max()                                              # [] Scalar+
        assert (energy_form[off_diagonal_mask].abs() < 1e-10 * scale).all()

    recovered_diagonal = rate_blades & recovered_inertia(rate_blades)                # [n_rates] Scalar+
    target_diagonal = rate_blades & diagonal_inertia(rate_blades)                    # [n_rates] Scalar+
    recovered_sorted = recovered_diagonal.abs()[recovered_diagonal.abs().argsort()]  # [n_rates] Scalar+
    target_sorted = target_diagonal.abs()[target_diagonal.abs().argsort()]           # [n_rates] Scalar+
    assert ((recovered_sorted - target_sorted).abs() < 1e-8 * target_sorted.max()).all()
    return cloud_motor, motor


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signature", default="x+y+z+w0")
    parser.add_argument("--seed", type=int, default=0)
    main(**vars(parser.parse_args()))
