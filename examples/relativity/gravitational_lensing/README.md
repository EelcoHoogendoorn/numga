# Gravitational lensing

Open the notebook from the repository root:

```sh
jupyter lab examples/relativity/gravitational_lensing/gravitational_lensing.ipynb
```

The notebook defines the binary lens deflection, its local derivative map, and the moving source brightness in its own cells, while `render.py` draws the critical curve, caustic, and sky images. Run `scenarios.py` to write the figure and animation to `plots/` directly:

```sh
MPLBACKEND=Agg python -m examples.relativity.gravitational_lensing.scenarios
```
