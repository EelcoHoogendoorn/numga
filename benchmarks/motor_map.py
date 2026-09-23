"""Motor sandwich benchmarks. Run a scenario below; no command-line configuration."""

from collections.abc import Callable
from functools import partial
from itertools import product
from math import ceil
import platform
from statistics import median
from time import perf_counter
from timeit import Timer

import numpy as np

from numga import Extensor, GAType, NumpyContext


def jax_map_amortization() -> None:
    """Does building a shared motor map pay for itself in one application?

    Compare one CGA trivector with 16,384 using compiled dense execution.
    Include construction in the total: comparing application alone hides the
    work moved out of the passenger loop. The single-item case exposes the
    cost of splitting a compiled expression into two dispatched calls.

    M4 Pro, JAX 0.6.0: one item took 7.9 us direct versus 14.6 us
    build+map; 16,384 took 524 versus 223 us (2.35x faster including build).
    Precomputation wins by amortizing motor work across passengers, not by
    making the single-passenger expression intrinsically cheaper.
    """
    print_environment("jax", 0)
    print_map_header("jax")
    for batch in (1, 16384):
        compare("x+y+z+p+n-", 3, "jax", batch, "dense", 0)


def jax_unrolling_crossover() -> None:
    """Does unrolling help equally for 4D vectors and CGA trivectors?

    Hold the batch at 16,384 and compare dense contractions with stacked
    nonzero expressions. The larger sandwich has many more scalar terms;
    unrolling trades contraction structure for explicit arithmetic. Report
    compilation as well as execution to expose both costs of that expansion.
    Computed map application is dense under either policy.

    M4 Pro, JAX 0.6.0: stacked sparse beat dense for 4D vectors, 75 versus
    265 us. For CGA trivectors it lost, 794 versus 648 us, and compilation
    grew to 905 versus 23 ms. Unrolling is useful for the smaller expression;
    it is not a universal optimization. In the 4D case the direct unrolled
    sandwich even beat the precomputed map's application (146 us).
    """
    print_environment("jax", 0)
    print_map_header("jax")
    for (signature, grade), execution in product(
        (("x+y+z+w+", 1), ("x+y+z+p+n-", 3)),
        ("dense", "sparse"),
    ):
        compare(signature, grade, "jax", 16384, execution, 0)


def numpy_unrolling_cost() -> None:
    """Show why skipping zeros is insufficient in an eager array backend.

    Use 16,384 CGA trivectors: NumPy executes each unrolled term as a separate
    array operation. Dense contractions and a precomputed map avoid those
    repeated traversals. Unlike JAX, there is no compiler to fuse the terms.
    Include map construction to measure the cost of the complete alternative.

    M4 Pro, NumPy 2.2.5: sparse direct took 97.3 ms versus 1.69 ms dense.
    Building and applying the map took 2.55 ms sparse versus 0.48 ms dense;
    both applied the computed map in 0.44 ms. Unrolling hurts construction
    too, but paying that once is far cheaper than carrying it over the batch.
    """
    print_environment("numpy", 0)
    print_map_header("numpy")
    for execution in ("dense", "sparse"):
        compare("x+y+z+p+n-", 3, "numpy", 16384, execution, 0)


def sandwich(motor: Extensor, values: Extensor) -> Extensor:
    return motor >> values


def build(motor: Extensor, Passenger: GAType) -> Extensor:
    return motor >> Passenger


def apply(transform: Extensor, values: Extensor) -> Extensor:
    return transform(values)


def inputs(signature, grade, backend, batch, execution, seed):
    dtype = np.float64
    if backend == "numpy":
        context = NumpyContext(signature, dtype=dtype, execution=execution)
    else:
        import jax
        from numga.backend.jax import JaxContext

        jax.config.update("jax_enable_x64", True)
        context = JaxContext(signature, dtype=dtype, execution=execution)

    rng = np.random.default_rng(seed)
    generator = context.multivector.bivector(
        rng.normal(size=len(context.subspace.bivector())) * 0.15,
    )
    motor = generator.exp()
    Passenger = context.gatype.k_vector(grade)
    values = context.multivector(
        Passenger, rng.normal(size=(batch, len(Passenger.output_subspace))),
    )
    return motor, values


def measure(operation: Callable, arguments: tuple, backend: str) -> tuple[float, float]:
    """Return XLA compilation and synchronized warm-call times in seconds."""
    compilation = 0.0
    if backend == "jax":
        import jax

        jax.config.update("jax_enable_compilation_cache", False)
        jax.clear_caches()
        lowered = jax.jit(operation).lower(*arguments)
        start = perf_counter()
        operation = lowered.compile()
        compilation = perf_counter() - start

    def invoke():
        result = operation(*arguments)
        if backend == "jax":
            result.kernel.block_until_ready()
        return result

    timer = Timer(invoke)
    timer.timeit(number=1)
    elapsed = timer.timeit(number=1)
    number = max(1, ceil(0.02 / elapsed))
    return compilation, median(timer.repeat(repeat=3, number=number)) / number


def compare(signature, grade, backend, batch, execution, seed) -> None:
    motor, values = inputs(signature, grade, backend, batch, execution, seed)
    build_for_grade = partial(build, Passenger=values.gatype)
    transform = build_for_grade(motor)
    _, construction = measure(build_for_grade, (motor,), backend)
    direct_jit, direct = measure(sandwich, (motor, values), backend)
    _, mapped = measure(apply, (transform, values), backend)
    total = construction + mapped
    print(f"{signature:12} {grade:5} {batch:7} {execution:>6} "
          f"{direct * 1e6:10.1f} {construction * 1e6:9.1f} {mapped * 1e6:9.1f} "
          f"{total * 1e6:12.1f} {direct / total:7.2f}x"
          + (f" {direct_jit * 1e3:13.1f}" if backend == "jax" else ""), flush=True)


def print_environment(backend: str, seed: int) -> None:
    print(f"Python {platform.python_version()}, {platform.system()} {platform.machine()}, NumPy {np.__version__}.")
    if backend == "jax":
        import jax
        import jaxlib

        print(f"JAX {jax.__version__}, jaxlib {jaxlib.__version__}, {jax.devices()}.")
    print(f"Backend: {backend}. Seed {seed}; float64; one shared motor/map; (batch, coefficients) storage.")
    print("Sparse = unrolled nonzero terms; dense = staged contractions; computed maps use dense application.")
    print("Warm timings: median of 3 runs targeting 20 ms, excluding setup and compilation.")
    if backend == "jax":
        print("JAX calls synchronize; compilation excludes tracing and uses no compilation cache.")
    print("Build+map sums two warm calls; gain compares direct against that total.")


def print_map_header(backend: str) -> None:
    print(f"{'algebra':12} {'grade':>5} {'batch':>7} {'policy':>6} "
          f"{'direct us':>10} {'build us':>9} {'map us':>9} "
          f"{'build+map us':>12} {'gain':>8}"
          + (f" {'direct JIT ms':>13}" if backend == "jax" else ""), flush=True)


if __name__ == "__main__":
    jax_map_amortization()
