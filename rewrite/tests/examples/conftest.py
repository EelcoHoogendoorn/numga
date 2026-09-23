"""Example tests build figures and frames but never save them; plots/ holds only real runs."""

import matplotlib.pyplot as plt
import pytest


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")
