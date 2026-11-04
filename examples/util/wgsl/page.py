"""Shaders in numga's dialect run live in a web page, on the reader's own GPU: a shader's uniforms packed as the runner
packs them, bytes as the page holds them, and the page written from its template.

A page is one HTML file beside its example, its template with the compiled shaders and everything they read in place
of `/*CONTENTS*/null`. GitHub Pages serves it from the repository's tracked files, so an example writes it only when
asked, in its `web.page()`.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

from examples.util.wgsl.runner import Shader


def packed(shader: Shader, values: dict) -> tuple[bytes, dict]:
    """The uniform block as the runner packs it, and where each uniform begins in it."""
    data, offsets = bytearray(), {}
    for name, kind in shader.program.uniforms:
        alignment, _ = shader.layout.size(kind)
        data += bytes(-len(data) % alignment)
        offsets[name] = len(data)
        data += shader.layout.pack(kind, values[name])
    return bytes(data) + bytes(-len(data) % 16), offsets


def encoded(data: bytes) -> str:
    """Bytes as the page holds them, in base64."""
    return base64.b64encode(data).decode("ascii")


def write_page(template: Path, path: Path, contents: dict) -> None:
    """Write the page at `path`: the template with `contents` in place of `/*CONTENTS*/null`."""
    path.write_text(template.read_text().replace("/*CONTENTS*/null", json.dumps(contents)))
    print(f"Wrote {path}, {path.stat().st_size / 1e3:.0f} kB")
