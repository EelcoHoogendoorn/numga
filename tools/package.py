"""Builds the PyPI wheel and source distribution containing `numga` without `examples`.

GitHub archive installs (`pip install "numga[examples] @ .../archive/HEAD.tar.gz"`, used by the Colab notebooks) read
the repository's root `pyproject.toml` and install `numga` and `examples` together from the same commit while
`.gitattributes` omits `plots/` from the tarball. PyPI releases run this module to stage `numga`, `LICENSE`, a
link-resolved `README.md` and a `numga`-only `pyproject.toml`, then build and verify both archives in `dist/`.

    python -m tools.package
    python -m tools.package --tag v2.0.0 --outdir dist
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import tarfile
import tomllib
import zipfile
from pathlib import Path

from setuptools import build_meta

import numga

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "EelcoHoogendoorn/numga"


def resolve_readme_links(readme: str, ref: str) -> str:
    """Rewrites relative image and document links in `readme` so they resolve on PyPI against GitHub `ref`."""
    raw_base = f"https://raw.githubusercontent.com/{REPOSITORY}/{ref}/"
    github_base = f"https://github.com/{REPOSITORY}"

    def github_target(path: str) -> str:
        kind = "tree" if path.endswith("/") else "blob"
        return f"{github_base}/{kind}/{ref}/{path}"

    readme = re.sub(
        r'(<img\s+[^>]*?src=")(?!https?://|data:)([^"]+)(")',
        lambda match: f"{match.group(1)}{raw_base}{match.group(2)}{match.group(3)}",
        readme,
    )
    readme = re.sub(
        r'(<a\s+[^>]*?href=")(?!https?://|#|mailto:)([^"]+)(")',
        lambda match: f"{match.group(1)}{github_target(match.group(2))}{match.group(3)}",
        readme,
    )
    readme = re.sub(
        r'(\]\()(?!https?://|#|mailto:)([^)\s]+)(\))',
        lambda match: f"{match.group(1)}{github_target(match.group(2))}{match.group(3)}",
        readme,
    )
    return readme


def pypi_pyproject(source: str) -> str:
    """Narrows `pyproject.toml` to package `numga*` only and drops `examples`-specific packaging metadata."""
    narrowed = source.replace('include = ["numga*", "examples*"]', 'include = ["numga*"]', 1)
    narrowed = re.sub(r"\nexamples = \[\n(?:    .*\n)*\]\n", "\n", narrowed, count=1)
    narrowed = re.sub(r'\n\[tool\.setuptools\.package-data\]\n"examples\.util" = \["models/\*\.obj"\]\n', "\n", narrowed, count=1)
    parsed = tomllib.loads(narrowed)
    if parsed["tool"]["setuptools"]["packages"]["find"]["include"] != ["numga*"]:
        raise RuntimeError("Failed to narrow setuptools package discovery to ['numga*'].")
    if "examples" in parsed["project"].get("optional-dependencies", {}):
        raise RuntimeError("Failed to strip the [examples] optional-dependency group for PyPI.")
    if "package-data" in parsed["tool"]["setuptools"]:
        raise RuntimeError("Failed to strip examples package-data for PyPI.")
    return narrowed


def verify_archives(wheel_path: Path, sdist_path: Path) -> None:
    """Checks that both built archives contain `numga/__init__.py` and no `examples/` files."""
    with zipfile.ZipFile(wheel_path) as wheel:
        wheel_names = wheel.namelist()
    if "numga/__init__.py" not in wheel_names:
        raise RuntimeError(f"{wheel_path.name} is missing numga/__init__.py.")
    leaked_wheel = [name for name in wheel_names if name == "examples" or name.startswith("examples/")]
    if leaked_wheel:
        raise RuntimeError(f"{wheel_path.name} unexpectedly includes examples: {leaked_wheel[:5]}")

    with tarfile.open(sdist_path, "r:gz") as sdist:
        sdist_names = [member.name.split("/", 1)[1] for member in sdist.getmembers() if "/" in member.name]
    if "numga/__init__.py" not in sdist_names:
        raise RuntimeError(f"{sdist_path.name} is missing numga/__init__.py.")
    leaked_sdist = [name for name in sdist_names if name == "examples" or name.startswith("examples/")]
    if leaked_sdist:
        raise RuntimeError(f"{sdist_path.name} unexpectedly includes examples: {leaked_sdist[:5]}")


def build_distributions(outdir: Path, tag: str | None = None) -> tuple[Path, Path]:
    """Stages a clean `numga`-only tree, builds wheel and sdist into `outdir`, and verifies both archives."""
    pyproject_source = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    version = tomllib.loads(pyproject_source)["project"]["version"]
    if numga.__version__ != version:
        raise RuntimeError(f"Version mismatch: pyproject.toml has {version!r} while numga.__version__ is {numga.__version__!r}.")
    if tag is not None:
        expected_version = tag.removeprefix("v")
        if expected_version != version:
            raise RuntimeError(f"Release tag {tag!r} does not match package version {version!r}.")

    ref = tag if tag is not None else "main"
    outdir = outdir.resolve()
    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True)

    stage = outdir / ".stage"
    stage.mkdir()
    previous_cwd = Path.cwd()
    try:
        shutil.copytree(
            ROOT / "numga",
            stage / "numga",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
        )
        shutil.copy2(ROOT / "LICENSE", stage / "LICENSE")
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        (stage / "README.md").write_text(resolve_readme_links(readme, ref), encoding="utf-8")
        (stage / "pyproject.toml").write_text(pypi_pyproject(pyproject_source), encoding="utf-8")

        os.chdir(stage)
        wheel_name = build_meta.build_wheel(str(outdir))
        sdist_name = build_meta.build_sdist(str(outdir))
    finally:
        os.chdir(previous_cwd)
        shutil.rmtree(stage, ignore_errors=True)

    wheel_path = outdir / wheel_name
    sdist_path = outdir / sdist_name
    verify_archives(wheel_path, sdist_path)
    return wheel_path, sdist_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build numga PyPI distributions without examples.")
    parser.add_argument("--outdir", type=Path, default=ROOT / "dist", help="Output directory for built archives.")
    parser.add_argument("--tag", type=str, default=None, help="Release git tag (e.g. v2.0.0) to verify and link against.")
    args = parser.parse_args()

    wheel_path, sdist_path = build_distributions(args.outdir, args.tag)
    print(f"Built {wheel_path.name} ({wheel_path.stat().st_size / 1024:.1f} KB)")
    print(f"Built {sdist_path.name} ({sdist_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
