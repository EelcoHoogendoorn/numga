"""Opt-in warm-call timings: python -m pytest benchmarks/test_runtime.py -q -s.

No speed thresholds or additional benchmark dependency. Setup, compilation and
cache warming are excluded. Report the best of three calibrated runs; JAX calls
include dispatch and completion, not just asynchronous enqueue time.
"""

from timeit import Timer
from typing import Callable

import numpy as np
import pytest

from numga import Algebra, NumpyContext
from numga.algebras import PGA3D


@pytest.fixture(params=("numpy", "jax"))
def backend(request):
    if request.param == "numpy":
        return NumpyContext, lambda function: function, lambda result: result
    jax = pytest.importorskip("jax")
    from numga.backend.jax import JaxContext

    return JaxContext, jax.jit, lambda result: result._kernel.block_until_ready()


@pytest.fixture(params=("dense", "sparse"))
def execution(request):
    return request.param


@pytest.fixture(params=(1, 1024))
def batch_size(request):
    return request.param


def microseconds(call: Callable[[], object]) -> float:
    call()
    timer = Timer(call)
    number, _ = timer.autorange()
    return min(timer.repeat(repeat=3, number=number)) / number * 1e6


def report(name, context, batch_size, elapsed, record_property):
    record_property("microseconds_per_call", elapsed)
    print(
        f"\n{name:16s} {context.key[0]:5s} {context.execution:6s} "
        f"{str(context.dtype):7s} batch={batch_size:4d}: {elapsed:9.2f} us/call"
    )


@pytest.mark.parametrize("batch_size", (1, 10, 100, 1024))
def test_matrix_application(backend, execution, batch_size, record_property):
    context_type, compile, finish = backend
    algebra = Algebra("x+y+z+")
    context = context_type(algebra, execution=execution)
    vector = algebra.subspace.vector()
    rng = np.random.default_rng(0)
    matrix = context.extensor(algebra.gatype((vector, vector)), rng.normal(size=(3, 3)))
    shape = (3,) if batch_size == 1 else (batch_size, 3)
    vectors = context.multivector.vector(rng.normal(size=shape))
    apply = compile(lambda matrix, vectors: matrix(vectors))

    expected = np.matmul(np.asarray(matrix.kernel), np.asarray(vectors.kernel)[..., None])[..., 0]
    np.testing.assert_allclose(apply(matrix, vectors).kernel, expected, rtol=2e-5, atol=2e-6)
    elapsed = microseconds(lambda: finish(apply(matrix, vectors)))
    report("3x3 application", context, batch_size, elapsed, record_property)

    if context_type is NumpyContext:
        a, x = matrix._kernel, vectors._kernel
        raw = microseconds(lambda: np.matmul(a, x[..., None])[..., 0])
        record_property("raw_numpy_microseconds_per_call", raw)
        overhead = (elapsed / raw - 1) * 100
        record_property("overhead_percent", overhead)
        print(
            f"  raw NumPy: {raw:.2f} us/call; "
            f"extensor overhead: {elapsed - raw:.2f} us/call ({overhead:.1f}%)"
        )


def test_rotor_product(backend, execution, batch_size, record_property):
    context_type, compile, finish = backend
    context = context_type(Algebra("x+y+z+"), execution=execution)
    rng = np.random.default_rng(1)
    shape = (4,) if batch_size == 1 else (batch_size, 4)
    left, right = (rng.normal(size=shape) for _ in range(2))
    a = context.multivector.rotor(left / np.linalg.norm(left, axis=-1, keepdims=True))
    b = context.multivector.rotor(right / np.linalg.norm(right, axis=-1, keepdims=True))
    multiply = compile(lambda a, b: a * b)

    result = multiply(a, b)
    assert result.gatype <= context.algebra.gatype.rotor()
    assert result.shape == a.shape
    elapsed = microseconds(lambda: finish(multiply(a, b)))
    report("rotor product", context, batch_size, elapsed, record_property)


def test_inertia_construction(backend, execution, batch_size, record_property):
    context_type, compile, finish = backend
    context = context_type(PGA3D, execution=execution)
    coordinates = np.random.default_rng(2).normal(size=(batch_size, 4))
    coordinates[:, -1] = 1
    points = context.multivector.antivector(coordinates)
    rate = PGA3D.subspace.bivector()

    def build(points):
        return points.regressive(points.commutator(rate)).sum(axis=0)

    build = compile(build)
    result = build(points)
    assert result.axes == (rate, rate)
    assert result.shape == ()
    elapsed = microseconds(lambda: finish(build(points)))
    report("PGA3 inertia", context, batch_size, elapsed, record_property)
