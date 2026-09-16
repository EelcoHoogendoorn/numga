"""What does a symmetric object's stiffness, conductivity or inertia look like? Average and see.

Take a crystal, a turbine blade, a cube. Some rotations leave it looking the same. Any
physical map on it, say the stiffness taking a force vector to a displacement vector, must
then look the same after those rotations too. The map that survives is found without any
theory: rotate the map by every symmetry of the object, using the rotor sandwich with the
input left open, and take the mean. Whatever the rotations disagree about averages away, and
what is left is the part every symmetry allows. Here that is done for a random tensor under
three symmetries, then for the inertia of a point cloud, which the cube's symmetries make
isotropic. The list of rotations is the same handful of lines for any object.
"""

from __future__ import annotations

import numpy as np

from numga import Algebra, Extensor, NumpyContext

ga = Algebra("x+y+z+")
ctx = NumpyContext(ga)
mv = ctx.multivector
V = ga.subspace.vector()
B = ga.subspace.bivector()
Rotor = ga.gatype.rotor()
Tensor = ga.gatype((V, V))


# --- plumbing -------------------------------------------------------------------------
def closure(generators: list[Rotor]) -> Rotor:
    """All rotations reachable by composing the given ones, each rotor counted once (g and -g are the same rotation)."""
    def key(rotor: Rotor) -> tuple[float, ...]:
        k = np.round(rotor.kernel, 6) + 0.0
        return tuple(k if k[np.argmax(np.abs(k) > 1e-9)] > 0 else -k)

    identity = mv.bivector(np.zeros(3)).exp()
    elements, seen, frontier = [identity], {key(identity)}, [identity]
    while frontier:
        fresh = [g * h for g in frontier for h in generators]
        frontier = []
        for element in fresh:
            if key(element) not in seen:
                seen.add(key(element)); elements.append(element); frontier.append(element)
    return Extensor.stack(elements)


def symmetric_map(rng: np.random.Generator) -> Tensor:
    m = rng.normal(size=(3, 3))
    return Extensor(ctx, Tensor, m + m.T)


# --- math -----------------------------------------------------------------------------
def main() -> None:
    rng = np.random.default_rng(0)
    groups = {
        "quarter turns about z": closure([(mv.xy * (np.pi / 4)).exp()]),
        "half turns about z and about x": closure([(mv.xy * (np.pi / 2)).exp(), (mv.yz * (np.pi / 2)).exp()]),
        "every rotation of a cube": closure([(mv.xy * (np.pi / 4)).exp(), (mv.yz * (np.pi / 4)).exp()]),
    }

    # 1. A map on vectors: a random symmetric tensor, averaged over each set of rotations.
    #    Rotate the map by g on both sides (g >> V is the rotation as a map, g << V its inverse).
    stiffness = symmetric_map(rng)
    for name, group in groups.items():
        invariant = (group >> V)(stiffness(group << V)).mean(axis=0)
        print(f"{name} ({len(group.kernel)} elements) allows:\n{np.round(invariant.kernel, 3)}")

    # 2. A map on bivectors: the inertia of a point cloud, I(Ω) = Σ x ∧ (x × Ω). The commutator
    #    of a point with a bivector is the direction the point moves under the action of that
    #    bivector; wedging it with the position gives the point's angular momentum. Averaged over
    #    the cube's rotations the inertia becomes the same in every direction; the inertia of the
    #    cube's own corners already is, so it is left alone.
    cloud = mv.vector(rng.normal(size=(50, 3)))
    inertia = (cloud ^ cloud.commutator(B)).sum(axis=0)
    cube = groups["every rotation of a cube"]
    isotropic = (cube >> B)(inertia(cube << B)).mean(axis=0)
    corners = mv.vector(np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float))
    cube_inertia = (corners ^ corners.commutator(B)).sum(axis=0)
    print("cloud inertia averaged over the cube's rotations:\n", np.round(isotropic.kernel, 3))

    # 3. A plain vector has no part that every rotation of a cube agrees on: it averages to zero.
    print("a vector averaged over the cube's rotations:", np.round((cube >> mv.vector(np.array([1.0, 2.0, 3.0]))).mean(axis=0).kernel, 12))

    # --- checks: kernel-level assertions, deliberately outside the demonstration ----------
    np.testing.assert_allclose(isotropic.kernel, np.eye(3) * np.trace(inertia.kernel) / 3, atol=1e-8)
    np.testing.assert_allclose((cube >> B)(cube_inertia(cube << B)).mean(axis=0).kernel, cube_inertia.kernel, atol=1e-8)


if __name__ == "__main__":
    main()
