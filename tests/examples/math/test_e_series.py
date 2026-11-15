"""The nested E6, E7 and E8 root systems, and their covariance under a mirror."""

from examples.math.e_series import core, render, scenarios
from examples.math.e_series.core import Plane, mv


def frobenius_squared(operator: core.PlaneMap) -> core.Scalar:
    """Composing a plane endomorphism with its adjoint has trace equal to its squared Frobenius norm (`[] Scalar+`)."""
    return operator.adjoint()(operator).trace()


def test_nested_root_systems_and_mirror_covariance():
    cases = scenarios.family()
    mirror = (mv.a + 2 * mv.b - 3 * mv.c + mv.h).normalized()                 # [] Plane+
    spans = tuple((roots * (roots | Plane)).sum(axis=(0, 1)) / (2 * n_steps)
                  for (_, roots, _, _, _), n_steps in zip(cases, scenarios.periods))
    tolerance = 1e-10

    # checks
    for inner, outer, inner_span, outer_span in zip(cases[:-1], cases[1:], spans[:-1], spans[1:]):
        inner_roots = inner[1].reshape((-1,))                                 # [...] Plane+
        outer_roots = outer[1].reshape((-1,))                                 # [...] Plane+
        distances = (inner_roots[:, None] - outer_roots[None, :]).scalar_norm_squared()
        assert (distances.min(axis=-1) < tolerance).all()
        assert frobenius_squared(outer_span(inner_span) - inner_span) < tolerance
        assert frobenius_squared(inner_span(outer_span) - inner_span) < tolerance

    for n_rank, n_steps, case, span in zip(scenarios.ranks, scenarios.periods, cases, spans):
        action, roots, _, _, _ = case
        reflected_action, reflected_roots = core.root_system(mirror >> core.simple_roots()[:n_rank], n_steps)
        reflected_span = (reflected_roots * (reflected_roots | Plane)).sum(axis=(0, 1)) / (2 * n_steps)
        assert ((reflected_roots - (mirror >> roots)).scalar_norm_squared() < tolerance).all()
        assert frobenius_squared(reflected_action - (mirror >> action(mirror << Plane))) < tolerance
        assert frobenius_squared(reflected_span - (mirror >> span(mirror << Plane))) < tolerance


def test_the_diagram_and_the_nested_roots_draw():
    cases = scenarios.family()
    _, _, plane_x, plane_y, _ = cases[-1]
    render.diagram_figure(core.simple_roots(), scenarios.ranks, scenarios.names)
    render.nested_roots_figure(tuple(roots for _, roots, _, _, _ in cases), plane_x[0], plane_y[0], scenarios.names)
