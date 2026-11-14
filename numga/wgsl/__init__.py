"""Shaders in the `numga` WGSL dialect write geometric algebra expressions with `numga` operators and
types, lowering algebraic operations to polynomial WGSL functions at compile time and running on WebGPU."""

from .dialect import Binding, Program, compile, template
from .library import level_lines, sampled, samples_per_side, software, supersampled
from .runner import Layout, Resident, Shader, coefficients, device

__all__ = [
    "Binding",
    "Layout",
    "Program",
    "Resident",
    "Shader",
    "coefficients",
    "compile",
    "device",
    "level_lines",
    "sampled",
    "samples_per_side",
    "software",
    "supersampled",
    "template",
]

