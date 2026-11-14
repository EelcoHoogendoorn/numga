"""Standalone HTML pages run shaders compiled from the `numga` WGSL dialect directly on a browser's
WebGPU device. Each page embeds its compiled WGSL sources, base64-encoded storage tables, and aligned
initial uniform buffer in place of the `/*CONTENTS*/null` placeholder in an HTML template.

Because interactive JavaScript updates individual uniforms in place inside that single buffer, `packed`
returns each uniform's byte offset alongside the padded initial bytes laid out by `Shader.layout`.
Static pages are served directly from tracked files, so examples regenerate them on demand from their
`web` modules.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

from .runner import Shader


def packed(shader: Shader, values: Mapping[str, Any]) -> tuple[bytes, dict]:
    """Pack `values` into a 16-byte-aligned uniform buffer in declaration order, paired with a map
    of each uniform's starting byte offset."""
    data, offsets = bytearray(), {}
    for name, kind in shader.program.uniforms:
        alignment, _ = shader.layout.size(kind)
        data += bytes(-len(data) % alignment)
        offsets[name] = len(data)
        data += shader.layout.pack(kind, values[name])
    return bytes(data) + bytes(-len(data) % 16), offsets


def encoded(data: bytes) -> str:
    """Encode raw buffer bytes as an ASCII base64 string for embedding in page JSON."""
    return base64.b64encode(data).decode("ascii")


def write_page(template: Path, path: Path, contents: Mapping[str, Any]) -> None:
    """Substituting JSON-serialized `contents` for the `/*CONTENTS*/null` marker in `template` writes
    a self-contained HTML page to `path`."""
    path.write_text(template.read_text().replace("/*CONTENTS*/null", json.dumps(contents)))
    print(f"Wrote {path}, {path.stat().st_size / 1e3:.0f} kB")
