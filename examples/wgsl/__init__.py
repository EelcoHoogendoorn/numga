"""numga in WGSL: shaders whose expressions are numga's, run headless over images."""

from examples.wgsl.dialect import compile
from examples.wgsl.library import compose, image_figure, labels, show, supersampled, vec3
from examples.wgsl.runner import Shader

__all__ = ["Shader", "compile", "compose", "image_figure", "labels", "show", "supersampled", "vec3"]
