# Contributing to numga

Bug reports, questions and pull requests are welcome through the [GitHub issue tracker](https://github.com/EelcoHoogendoorn/numga/issues) and pull requests.

## Reporting issues

Include a short, self-contained script that reproduces the problem, the Python and backend versions (`numpy`, and `jax`, `torch` or `wgpu` if involved), and the expected and actual behaviour.

## Development setup

Clone the repository and install it in editable mode with Python 3.11 or newer:

```sh
git clone https://github.com/EelcoHoogendoorn/numga.git
cd numga
python -m pip install -e ".[linalg,jax,examples,test]"
```

The core library requires only NumPy, so a minimal test setup needs only `python -m pip install -e ".[test]"`. On headless Linux systems without a physical GPU, install `mesa-vulkan-drivers` if running WebGPU rendering tests.

## Running tests

Run targeted test files directly with `pytest` and a headless Matplotlib backend:

```sh
MPLBACKEND=Agg pytest tests/test_extensor.py
MPLBACKEND=Agg pytest tests/gatype/test_gatype.py
```

Whole-suite collection is guarded in `tests/conftest.py` so accidental broad runs do not stall local work. To run the complete library and example coverage suite as CI does, pass `--full-sweep=push-to-main`:

```sh
MPLBACKEND=Agg pytest tests --full-sweep=push-to-main
```

To run the NumPy-only library suite without optional backends or example dependencies:

```sh
pytest --full-sweep=push-to-main tests --ignore=tests/examples
```

## Pull requests

Keep pull requests focused on one change at a time. Cover new library functionality or bug fixes with fast unit tests under `tests/`. In example modules and notebooks, compute geometric quantities on `Extensor` objects throughout and keep coordinate array extraction inside `render.py`.
