# JIT Array Extraction in Render Path Only

- **Zero array extraction in notebooks or core**: Absolutely ZERO extraction of arrays (`.kernel`, `.values`, `.cast().kernel`, coordinate slicing, etc.) in notebooks or `core.py`.
- **Array extraction is JIT; in the render path; just before going into plotting code**:
  - Notebooks and mathematical modules operate purely on algebraic Extensors and Multivectors from start to finish.
  - Pass geometric objects directly to `render.py`.
  - All coordinate unpacking, `.kernel` extraction, and conversions for matplotlib belong strictly inside `render.py` right before handing values to plotting routines.
