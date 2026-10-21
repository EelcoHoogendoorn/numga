"""Scalar readouts for the three triality actions."""

import numpy as np

from examples.math.triality.core import Scalar


def return_table(factors: Scalar, names: tuple[str, ...], turns: tuple[str, ...]) -> str:
    """The scalar multiples of the identity reached after whole turns."""
    values = factors.to_array()
    label_width = max(map(len, names))
    header = " " * label_width + "  " + "  ".join(f"{turn:>8}" for turn in turns)
    rows = [f"{name:<{label_width}}  " + "  ".join(f"{value:>8.0f}" for value in row)
            for name, row in zip(names, np.round(values))]
    return "\n".join([header, *rows])
