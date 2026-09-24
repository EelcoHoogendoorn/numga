"""The library surface as functions of a context, shared by the array-backend tests."""

import numpy as np

from numga import Algebra, Extensor


def operations(ga: Algebra, rng: np.random.Generator) -> dict:
    """The library surface, each as a function of a context."""
    bivectors = rng.normal(size=(5, len(ga.subspace.bivector()))) * 0.4
    evens = rng.normal(size=(5, len(ga.subspace.even())))
    vectors = rng.normal(size=(5, len(ga.subspace.vector())))
    Vector = ga.gatype.vector()
    Map = ga.gatype((Vector, Vector))
    matrices = rng.normal(size=(5, len(Vector.output_subspace), len(Vector.output_subspace)))
    symmetric = matrices @ np.swapaxes(matrices, -1, -2) + np.eye(matrices.shape[-1])
    return {
        "product": lambda c: c.multivector.even(evens) * c.multivector.vector(vectors),
        "wedge": lambda c: c.multivector.vector(vectors) ^ c.multivector.even(evens),
        "regressive": lambda c: c.multivector.vector(vectors) & c.multivector.even(evens),
        "inner": lambda c: c.multivector.vector(vectors) | c.multivector.even(evens),
        "sum": lambda c: c.multivector.vector(vectors) + c.multivector.even(evens) * 2.0,
        "reverse dual": lambda c: (~c.multivector.even(evens)).dual(),
        "exp": lambda c: c.multivector.bivector(bivectors).exp(),
        "log": lambda c: c.multivector.bivector(bivectors).exp().log(),
        "normalized": lambda c: c.multivector.even(evens).normalized(),
        "inverse": lambda c: c.multivector.even(evens).inverse(),
        "square root": lambda c: c.multivector.bivector(bivectors).exp().square_root(),
        "decompose": lambda c: c.multivector.bivector(bivectors).decompose_invariant()[0],
        "sandwich": lambda c: c.multivector.bivector(bivectors).exp() >> c.multivector.vector(vectors),
        "sandwich map": lambda c: (c.multivector.bivector(bivectors).exp() >> Vector)(c.multivector.vector(vectors)),
        "stack reduce": lambda c: Extensor.stack([c.multivector.vector(vectors)] * 2).sum(axis=0),
        "reduce unbatched": lambda c: c.multivector.vector(vectors)[0].sum(),
        "index reshape": lambda c: c.multivector.vector(vectors)[1:3].reshape(2, 1),
        "det": lambda c: c.extensor(Map, matrices).det(),
        "trace": lambda c: c.extensor(Map, matrices).trace(),
        "map inverse": lambda c: c.extensor(Map, matrices).inverse(),
        "solve": lambda c: c.extensor(Map, matrices).solve(c.multivector.vector(vectors)),
        "pinv": lambda c: c.extensor(Map, matrices).pinv(),
        "eigh": lambda c: c.extensor(Map, symmetric).eigh()[0],
        "svdvals": lambda c: c.extensor(Map, matrices).svdvals(),
        "compose": lambda c: c.extensor(Map, matrices) * c.extensor(Map, matrices),
        "outermorphism": lambda c: c.extensor(Map, matrices).outermorphism(ga.gatype.bivector()),
    }
