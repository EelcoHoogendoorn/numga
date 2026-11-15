"""Every `Extensor` pairs a static `GAType` with an immutable `Context` governing coefficient storage
and contraction execution, with `ExactContext` and `NumpyContext` available directly and optional
`JaxContext` and `TorchContext` in `numga.backend.jax` and `numga.backend.torch`."""

from numga.backend.context import Context
from numga.backend.exact import ExactContext
from numga.backend.numpy import NumpyContext
