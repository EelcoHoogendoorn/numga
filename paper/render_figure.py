"""Composes the four-panel static overview figure for the JOSS paper from native-resolution example renders."""

from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageDraw

PAPER_DIR = Path(__file__).resolve().parent
REPO_ROOT = PAPER_DIR.parent
WORKSPACE_PLOTS = REPO_ROOT.parent / "plots"
REPO_PLOTS = REPO_ROOT / "plots"

GUTTER_PX = 20
PAD_PX = 4


def find_plot(filename: str) -> Path:
    """Resolves `filename` from the shared workspace `plots/` folder or the checkout's `plots/`."""
    candidate = WORKSPACE_PLOTS / filename
    return candidate if candidate.exists() else REPO_PLOTS / filename


def load_rgba_frame(filename: str, frame_index: int = 0, border: bool = False) -> Image.Image:
    """Loads a static PNG or a single frame of an animated GIF as an unresampled RGBA image."""
    with Image.open(find_plot(filename)) as source:
        if frame_index:
            source.seek(frame_index)
        panel = Image.new("RGBA", source.size, (255, 255, 255, 255))
        rgba = source.convert("RGBA")
        panel.alpha_composite(rgba)
    if border:
        draw = ImageDraw.Draw(panel)
        width, height = panel.size
        draw.rectangle((0, 0, width - 1, height - 1), outline=(210, 218, 226, 255), width=1)
    return panel


def main() -> None:
    panels = [
        load_rgba_frame("multiview_covariance.png"),
        load_rgba_frame("liquid_crystals.gif", frame_index=9, border=True),
        load_rgba_frame("kerr_camera.png"),
        load_rgba_frame("snowflake_grown.png"),
    ]
    row_height = max(panel.height for panel in panels)
    total_width = sum(panel.width for panel in panels) + GUTTER_PX * (len(panels) - 1) + 2 * PAD_PX
    total_height = row_height + 2 * PAD_PX

    canvas = Image.new("RGB", (total_width, total_height), (255, 255, 255))
    cursor_x = PAD_PX
    for panel in panels:
        offset_y = PAD_PX + (row_height - panel.height) // 2
        canvas.paste(panel.convert("RGB"), (cursor_x, offset_y))
        cursor_x += panel.width + GUTTER_PX

    paper_target = PAPER_DIR / "joss_overview.png"
    canvas.save(paper_target, optimize=True)
    if WORKSPACE_PLOTS.is_dir():
        canvas.save(WORKSPACE_PLOTS / "joss_overview.png", optimize=True)


if __name__ == "__main__":
    main()
