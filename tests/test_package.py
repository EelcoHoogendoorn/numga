"""PyPI distributions stage `numga` without `examples` and resolve relative README links against the release ref."""

import pytest

pytest.importorskip("setuptools")

import numga
from tools.package import build_distributions, resolve_readme_links


def test_build_distributions_excludes_examples_and_checks_tag(tmp_path):
    wheel_path, sdist_path = build_distributions(tmp_path / "dist", tag=f"v{numga.__version__}")
    assert wheel_path.is_file()
    assert sdist_path.is_file()

    with pytest.raises(RuntimeError):
        build_distributions(tmp_path / "mismatch", tag="v0.0.0")


def test_resolve_readme_links_targets_release_ref():
    sample = (
        '<a href="examples/mechanics/modes/"><img src="plots/modes.gif"></a> '
        '[Guide](docs/extensors.md) [Live](https://example.com/page.html)'
    )
    resolved = resolve_readme_links(sample, "v2.0.0")
    assert 'href="https://github.com/EelcoHoogendoorn/numga/tree/v2.0.0/examples/mechanics/modes/"' in resolved
    assert 'src="https://raw.githubusercontent.com/EelcoHoogendoorn/numga/v2.0.0/plots/modes.gif"' in resolved
    assert '](https://github.com/EelcoHoogendoorn/numga/blob/v2.0.0/docs/extensors.md)' in resolved
    assert '](https://example.com/page.html)' in resolved
