# One star, several images

Open the notebook from the repository root:

```sh
jupyter lab examples/relativity/gravitational_lensing/gravitational_lensing.ipynb
```

The notebook lays out the lens, its local map and the source's light in its own cells; drawing lives
in `render.py`. To write the figures and the animation to `plots/` without opening the notebook:

```sh
MPLBACKEND=Agg python -m examples.relativity.gravitational_lensing.scenarios
```
