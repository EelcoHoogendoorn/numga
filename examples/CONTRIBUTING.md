# Contributing an example

`numga`'s [examples](README.md) showcase geometric algebra and [extensors](../docs/extensors.md) across geometry, physics, engineering and mathematics. Contributions are welcome, whether a pull request adding a worked example or an [issue](https://github.com/EelcoHoogendoorn/numga/issues) challenging the approach with a problem where it might fall short.

A good example pairs a concrete phenomenon with a clear visual or animation, and writes the geometric-algebra calculation that drives it directly in front of the reader.

Both the calculation and the rendering stay in geometric algebra from start to finish. Coordinates enter once when constructing elements, quantities carry their geometric grade and [orientation](../docs/orientation.md), and neither the calculation nor the renderer unpacks raw coordinate arrays for geometry. Scene drawing uses [`numga.render` and `numga.wgsl`](../docs/rendering.md) directly on typed elements in the example's own algebra.

In the notebook, scene construction and drawing code stay in helper modules while the algebra, types and mathematical steps are written out directly in the cells. Where a scene has non-trivial setup, drawing an early static plot lets the setup explain itself visually without construction boilerplate. Short, flat cells then develop the main calculation step by step, handing results to a render call or streamed animation and collecting invariant assertions in a final `# checks` cell.

On disk, a visual example typically lives in `examples/<topic>/<name>/` with `core.py` for the importable model, `scenarios.py` for scene data and command-line runs, `render.py` for drawing, and `<name>.ipynb` for the notebook, as in [Projection](geometry/projection/), [Scenegraph](geometry/scenegraph/), [Rattleback](mechanics/rattleback/) and [Snowflake](thermodynamics/snowflake/).
