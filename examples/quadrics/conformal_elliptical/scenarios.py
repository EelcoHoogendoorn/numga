"""The cube-plane scenario renders the thirteen great circles on the front hemisphere and checks rotational closure."""

from __future__ import annotations

from examples.quadrics.conformal_elliptical.core import alignments, planes


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    """Saves the front-hemisphere great-circle figure and verifies rotational closure of the thirteen planes."""
    from examples.quadrics.conformal_elliptical import render
    from examples.util.animation import save_figure

    save_figure(render.planes_figure(planes), "cube_axis_planes")

    # checks
    # All thirteen planes have unit norm and remain pairwise distinct up to orientation.
    assert ((planes.scalar_norm_squared() - 1).abs() < 1e-6).all()
    assert ((planes[:, None] | planes).abs() > 1 - 1e-6).sum() == 13
    # Each generating rotation carries every plane onto one of the thirteen planes up to orientation.
    assert ((alignments.max(axis=-1) - 1).abs() < 1e-6).all()


if __name__ == "__main__":
    main()
