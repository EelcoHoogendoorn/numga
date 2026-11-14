"""The `numga.subspace` package provides immutable signed blade-layout axes (`SubSpace`),
layout-independent support keys (`SupportKey`), and per-algebra flyweight constructors
(`SubSpaceFactory`)."""

from .factory import SubSpaceFactory
from .subspace import SubSpace, SupportKey

