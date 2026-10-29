"""Scenes for the normal modes example: two spring arrangements of the same plate."""

from __future__ import annotations

from examples.mechanics.modes import core
from examples.mechanics.modes.core import ModeCase

# The frames of the animation, and the time per frame.
N_FRAMES = 150
DURATION_MS = 40


def suspensions() -> list[ModeCase]:
    """Two vertical springs, then the same with an off-centre angled spring added."""
    return [core.normal_modes(core.suspension(2)), core.normal_modes(core.suspension(3))]


def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.mechanics.modes import render

    coupled = suspensions()[1:]
    save_figure(render.draw_modes(coupled), "modes")
    save_animation(render.animate_modes(coupled, N_FRAMES), "modes", DURATION_MS)


if __name__ == "__main__":
    main()
