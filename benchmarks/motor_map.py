"""Motor sandwich benchmarks. Run a scenario below; no command-line configuration."""

from collections.abc import Callable
from itertools import product
from math import ceil
import platform
from statistics import median
from time import perf_counter
from timeit import Timer

import numpy as np

from numga import Extensor, NumpyContext


n_passengers = 16384
seed = 0


def jax_map_amortization() -> None:
    """Does building a shared motor map pay for itself in one application?

    One CGA trivector is compared with a batch of `n_passengers` under compiled
    dense execution. The total includes construction: comparing application
    alone hides the work moved out of the passenger loop. The single-item case
    exposes the cost of splitting a compiled expression into two dispatched calls.
    """
    print_environment("jax", seed)
    print_map_header("jax")
    for n_batch in (1, n_passengers):
        compare("x+y+z+p+n-", 3, "jax", n_batch, "dense", seed)


def jax_unrolling_crossover() -> None:
    """Does unrolling help equally for 4D vectors and CGA trivectors?

    With the batch held at `n_passengers`, dense contractions are compared with
    stacked nonzero expressions. The larger sandwich has many more scalar terms;
    unrolling trades contraction structure for explicit arithmetic. Compilation
    is reported as well as execution, exposing both costs of that expansion.
    Computed map application is dense under either policy.
    """
    print_environment("jax", seed)
    print_map_header("jax")
    for (signature, grade), execution in product(
        (("x+y+z+w+", 1), ("x+y+z+p+n-", 3)),
        ("dense", "sparse"),
    ):
        compare(signature, grade, "jax", n_passengers, execution, seed)


def numpy_unrolling_cost() -> None:
    """Does skipping zeros pay in an eager array backend?

    For a batch of `n_passengers` CGA trivectors, NumPy executes each unrolled
    term as a separate array operation. Dense contractions and a precomputed
    map avoid those repeated traversals. Unlike JAX, there is no compiler to
    fuse the terms. Map construction is included, measuring the cost of the
    complete alternative.
    """
    print_environment("numpy", seed)
    print_map_header("numpy")
    for execution in ("dense", "sparse"):
        compare("x+y+z+p+n-", 3, "numpy", n_passengers, execution, seed)


def inputs(signature: str, grade: int, backend: str, n_batch: int, execution: str, seed: int) -> tuple[Extensor, Extensor]:
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
    )                                                                       # [] Bivector+
    motor = generator.exp()                                                 # [] Motor+
    Passenger = context.gatype.k_vector(grade)
    passengers = context.multivector(
        Passenger, rng.normal(size=(n_batch, len(Passenger.output_subspace))),
    )                                                                       # [passengers] Passenger+
    return motor, passengers


def measure(operation: Callable, arguments: tuple, backend: str) -> tuple[float, float]:
    """The XLA compilation time and the synchronized warm-call time of an operation, in seconds."""
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


def compare(signature: str, grade: int, backend: str, n_batch: int, execution: str, seed: int) -> None:
    motor, passengers = inputs(signature, grade, backend, n_batch, execution, seed)
    Passenger = passengers.gatype
    motor_map = motor >> Passenger                                          # [] Passenger+ <- Passenger+
    # Each timed operation takes its extensors as arguments, so JAX traces them as inputs.
    _, construction = measure(lambda motor: motor >> Passenger, (motor,), backend)
    direct_compilation, direct = measure(lambda motor, passengers: motor >> passengers, (motor, passengers), backend)
    _, application = measure(lambda motor_map, passengers: motor_map(passengers), (motor_map, passengers), backend)
    total = construction + application
    print(f"{signature:12} {grade:5} {n_batch:7} {execution:>6} "
          f"{direct * 1e6:10.1f} {construction * 1e6:9.1f} {application * 1e6:9.1f} "
          f"{total * 1e6:12.1f} {direct / total:7.2f}x"
          + (f" {direct_compilation * 1e3:13.1f}" if backend == "jax" else ""), flush=True)


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
