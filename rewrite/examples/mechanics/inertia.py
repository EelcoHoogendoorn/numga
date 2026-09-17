"""Maximally compact Extensor demonstration: inertia, summing, and operator rotation in PGA3D.

Demonstrates:
1. Defining rigid body inertia as an arity-1 operator via bivector hole and summation:
       I_body = points.regressive(points.commutator(B)).sum(axis=0)
2. Forming the 6x6 adjoint rotation matrix on bivectors via sandwich: R = rotor >> B
3. Rotating the inertia operator via operator composition (I_world = R(I_body(R.inverse()))) or inline sandwich (rotor >> I_body(rotor << B)).
4. Exact equivalence with rotating the point cloud first and recomputing inertia.
5. Linear action (momentum = I(rate)), exact inverse (I.inverse()), and kinetic energy.
"""

import numpy as np

from numga import NumpyContext
from numga.algebras import PGA3D

ctx = NumpyContext(PGA3D)
mv = ctx.multivector
B = PGA3D.subspace.bivector()

def main() -> None:
    # 1. Point cloud (random antivectors normalized to unit projective weight: P ~P == 1)
    np.random.seed(0)
    points = mv.antivector(np.random.normal(size=(10, 4))).normalized()

    # 2. Body-frame inertia operator via bivector hole & reduction:
    # maps bivector rates -> antibivector momenta
    I_body = points.regressive(points.commutator(B)).sum(axis=0)

    # 3. Rotation motor (90° in the xy-plane)
    rotor = (mv.xy * (-np.pi / 4.0)).exp()

    # 4. Form the 6x6 adjoint rotation matrix on bivectors via sandwich:
    R = rotor >> B

    # 5. Rotate inertia operator via operator composition:
    # (In 4D, bivectors and antibivectors coincide, so R acts on both rates and momenta)
    I_world = R(I_body(R.inverse()))

    # Alternatively, written directly as an inline expression:
    I_world_inline = rotor >> I_body(rotor << B)

    # 6. Verify equivalence with rotating the points first:
    rpoints = rotor >> points
    I_direct = rpoints.regressive(rpoints.commutator(B)).sum(axis=0)

    # 7. Evaluate linear map, exact inverse, and kinetic energy:
    rate = mv.bivector(np.random.normal(size=6))
    momentum = I_world(rate)
    energy = 0.5 * rate.regressive(momentum)
    recovered_rate = I_world.solve(momentum)

    print("Body-frame Inertia (6x6):\n", np.around(I_body.kernel, 2))
    print("\nWorld-frame Inertia via Operator Rotation (6x6):\n", np.around(I_world.kernel, 2))
    print("\nAngular rate:", rate.kernel)
    print("Angular momentum:", np.around(momentum.kernel, 2))
    print(f"Kinetic energy: {float(energy.kernel.item()):.4f}")

    # --- checks -------------------------------------------------------------
    assert I_body.arity == 1
    assert R.arity == 1
    np.testing.assert_allclose(I_world.kernel, I_world_inline.kernel, atol=1e-12)
    np.testing.assert_allclose(I_world.kernel, I_direct.kernel, atol=1e-12)
    np.testing.assert_allclose(recovered_rate.kernel, rate.kernel, atol=1e-12)



if __name__ == "__main__":
    main()
