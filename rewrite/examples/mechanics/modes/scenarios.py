"""Scenes for the normal modes example: two spring arrangements of the same plate."""

from __future__ import annotations

from examples.mechanics.modes import core
from examples.mechanics.modes.core import ModeCase


def suspensions() -> list[ModeCase]:
    """Two vertical springs, then the same with an off-centre angled spring added."""
    return [core.normal_modes(core.suspension(2)), core.normal_modes(core.suspension(3))]


if __name__ == "__main__":
    import argparse
    from examples.animation import save_animation, save_figure
    from examples.mechanics.modes import render

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--animate", action="store_true", help="Also save the normal modes as a GIF.")
    args = parser.parse_args()
    cases = suspensions()
    save_figure(render.draw_modes(cases, "Normal Modes: Baseline (top) vs Coupled (bottom)"), "modes")
    if args.animate:
        save_animation(render.animate_modes(cases, 120), "modes", 50)
