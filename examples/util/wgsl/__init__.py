"""numga in WGSL: shaders whose expressions are numga's, run headless over images."""

from examples.util.wgsl.dialect import compile
from examples.util.wgsl.library import compose, fitted, image_figure, labels, pinned, room, sampled, show, show_pixels, supersampled
from examples.util.wgsl.runner import Shader

__all__ = ["Shader", "compile", "compose", "fitted", "image_figure", "labels", "pinned", "room", "sampled", "show", "show_pixels", "supersampled"]
