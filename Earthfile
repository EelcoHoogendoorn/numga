VERSION 0.6

base-python:
    FROM python:3.8

    # Install Poetry
    ENV PIP_CACHE_DIR /pip-cache
    RUN --mount=type=cache,target=$PIP_CACHE_DIR \
        pip install poetry==1.6.1
    RUN --mount=type=cache,target=$PIP_CACHE_DIR \
        poetry config virtualenvs.create false

build:
    FROM +base-python

    WORKDIR /app
    ENV PYTHONPATH ${PYTHONPATH}:/app

    # Copy poetry files
    COPY pyproject.toml poetry.lock readme.md .

    # We only want to install the dependencies once, so if we copied
    # our code here now, we'd reinstall the dependencies ever ytime
    # the code changes. Instead, comment out the line making us depend
    # on our code, install, then copy our code.
    RUN sed -e '/packages/ s/^#*/#/' -i pyproject.toml

    # Install dependencies
    RUN --mount=type=cache,target=$PIP_CACHE_DIR \
        poetry install --all-extras
    # Install CPU version of torch
    RUN --mount=type=cache,target=$PIP_CACHE_DIR \
        pip install torch --index-url https://download.pytorch.org/whl/cpu

    # Copy the code and expose it to python
    COPY --dir numga .

test:
    FROM +build
    RUN poetry run pytest -n auto

test-examples:
    FROM +build

    # Run examples, not checking their results yet but they
    # should not produce any errors
    RUN poetry run python -m numga.examples.conformal
    RUN poetry run python -m numga.examples.ga_sparse
    RUN poetry run python -m numga.examples.integrators
    RUN poetry run python -m numga.examples.tennis_racket_theorem
    RUN poetry run python -m numga.examples.test_conformal
    RUN poetry run python -m numga.examples.test_simplex
    RUN poetry run python -m numga.examples.physics.run_chain_jax
    RUN poetry run python -m numga.examples.physics.run_chain_numpy
    RUN poetry run python -m numga.examples.physics.run_chain_torch
