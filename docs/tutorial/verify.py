"""Run the code blocks of each tutorial page in order and compare what they print with the text
blocks that follow them.

    python docs/tutorial/verify.py             # list the pages whose printed blocks are out of date
    python docs/tutorial/verify.py --refresh   # rewrite the printed blocks from what the code prints

A page is a Markdown file. Its `python` blocks run in one namespace, top to bottom; a block that
prints is followed by a `text` block holding exactly what it prints.
"""

from __future__ import annotations

import contextlib
import io
import re
import sys
from pathlib import Path

TUTORIALS = Path(__file__).parent
BLOCK = re.compile(r"```python\n(?P<code>.*?)```\n(?:\n```text\n(?P<printed>.*?)```\n)?", re.S)


def refreshed(page: Path) -> str:
    """The page with each code block's printed block replaced by what the code prints."""
    text = page.read_text()
    namespace: dict = {}
    pieces, position = [], 0
    for match in BLOCK.finditer(text):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(compile(match["code"], str(page), "exec"), namespace)
        printed = output.getvalue()
        pieces += [text[position:match.start()], f"```python\n{match['code']}```\n"]
        if printed:
            pieces.append(f"\n```text\n{printed}```\n")
        position = match.end()
    pieces.append(text[position:])
    return "".join(pieces)


def pages() -> list[Path]:
    return sorted(TUTORIALS.rglob("*.md"))


def main() -> None:
    refresh = "--refresh" in sys.argv[1:]
    stale = []
    for page in pages():
        text = refreshed(page)
        if text != page.read_text():
            stale.append(page)
            if refresh:
                page.write_text(text)
    for page in stale:
        print(f"{'refreshed' if refresh else 'out of date'}: {page.relative_to(TUTORIALS)}")
    sys.exit(1 if stale and not refresh else 0)


if __name__ == "__main__":
    main()
