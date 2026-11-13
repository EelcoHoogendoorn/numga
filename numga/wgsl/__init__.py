"""numga in WGSL: shaders whose expressions are numga's, compiled to WGSL and run on WebGPU."""

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

