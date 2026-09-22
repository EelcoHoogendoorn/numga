# Working in the numga rewrite

Rules for code in this tree, especially `examples/`. They are the owner's, stated during review;
follow them exactly.

> [!CRITICAL]
> **RULE #1: ZERO ARRAY EXTRACTION IN NOTEBOOKS OR CORE**
> Absolutely ZERO extraction of arrays (`.kernel`, `.values`, `.cast().kernel`, coordinate slicing, etc.) in notebooks or `core.py`!
> **Array extraction is JIT; in the render path; just before going into plotting code.**
> Notebooks and core modules operate purely on algebraic Extensors from start to finish.
> All array extraction, coordinate readouts, and conversions for matplotlib belong strictly inside `render.py` at the visualization boundary.

## Code first, not blog posts

- **Max ~8 lines of text before code**: Nobody reads a blog without anchoring it in something that is actually happening. Get to executable code immediately.
- **Let the code speak for itself**: Use generous, clear inline comments to explain concepts right where they are computed.
- **Lead with plain English, then mathematics**: Explain the physical intuition first, then provide the geometric/algebraic terminology.
- **No jargon for jargon's sake**: Avoid terms like "bundle adjustment" or academic filler; use clear descriptive phrasing ("scene reconstruction and camera alignment").

## Algebra first

- **ZERO ARRAY EXTRACTION IN NOTEBOOKS OR CORE**: Numbers exist only at construction. Never extract arrays (`.kernel`, `.values`, coordinate unpacking) in notebooks or `core.py`. **Array extraction is JIT; in the render path; just before going into plotting code.** If a function or notebook cell needs to plot or visualize something, pass the geometric objects (Extensors/Multivectors) directly to `render.py`. Coordinate readout happens strictly inside `render.py`.
- Numbers exist only at construction. Encode every quantity as an algebraic element as soon as it
  exists: a joint state is `axis * angle`, a rate is a bivector, a placement is a point, a shape
  is a quadric. Raw `np.ndarray` is allowed for scalars (masses, angles, margins), indices, and
  image data; never for coordinates of something the algebra can hold.
- Excursions outside the algebra (eigh, svd, solve, lstsq, scatter-add, sampling on `.kernel`)
  go in minimal, contained helper functions that take Extensors and return typed Extensors,
  documented for what they compute. Never spread a numeric escape through a function that also
  does geometry. One escape, one helper.
- Do not optimize around the library. If an expression is awkward, find the algebraic spelling
  (a form with both points open is its Gram matrix: `Point & form(mv.rotor() >> Point)`); do not
  hand-roll pairing matrices, basis permutations or kernel slicing.
- Prefer basis blades and operators: `mv.xw`, `a | b`, `>>`/`<<`, `^`, `&`. Identity maps are
  `mv.rotor() >> Point`. Use `inverse()` where an inverse is meant.
- Sums over axes are written in sum form over a batch, not as unrolled terms.

## Generality

- Any special case for a shape, a signature, a wall, or a dimension is a bug to fix, not to
  paper over. Rendering, collision detection and collision handling must be one code path for
  every quadric. Setting up a large object must be a single line, like any other body.
- Populations are batched state (`Bodies` with batched Extensor fields), stepped in one call;
  never lists of per-body objects stacked and unstacked per step.
- Any `if`, sentinel value, flag or fallback that can be removed, must be removed.
- No dimension polymorphism and certainly no `squeeze`. Functions accept clean array inputs
  (`coords: np.ndarray` of shape `(..., d)`), never varargs (`*coords`) or dimension-branching
  logic (`if len(coords) ...`). Never call `np.squeeze` or drop axes unpredictably; preserve
  batch and coordinate axes uniformly.


## Structure

- Math converges visibly in one scope; plumbing (constructors, samplers, rendering, readout)
  lives in helpers under a `# --- plumbing` header, math under `# --- math`.
- No nullable arguments, no type unions. Annotate with the narrowest named GAType.
- Avoid cryptic single-letter variable names for domain quantities (e.g., `spring_constants`
  instead of `k`, `stiffness` instead of `s`). Standard loop/iteration variables (`i`, `j`) are
  fine—use sound engineering judgement and loop back when in doubt.
- No interleaved plotting and math; a generator yields geometry, a draw helper consumes it.
- **ZERO ARRAY EXTRACTION IN NOTEBOOKS OR CORE (NO VIEWPORT OR COORDINATE EXTRACTION IN MATH/NOTEBOOKS)**: Absolutely zero array extraction (`.kernel`, `.values`, `.cast().kernel`) or coordinate unpacking helpers (such as `coordinates(points: Point) -> np.ndarray`, `euclidean()`, or `.kernel` coordinate slicing for matplotlib) in math or notebooks. **Array extraction is JIT; in the render path; just before going into plotting code.** Mathematical modules and notebook cells must operate purely on algebraic Extensors from start to finish. They must never extract arrays or define/import coordinate unpacking shims. Viewport conversion and array extraction happen strictly inside `render.py` JIT when passing points/vectors to matplotlib artists.
- Checks are kernel-level assertions in one labelled block at the end of `main`.
- Do not touch comments or docstrings unasked. Group functions on the type they belong to
  (`Bodies.join`), do not pollute the module namespace.
- Public-facing code and documentation must stand on their own. Explain the mathematics,
  behavior, and relevant design decisions; never narrate the editing process, refer to the
  conversation, or advertise compliance with instructions. Such commentary makes readers
  interpret our workflow instead of helping them understand the subject. Keep it in chat
  or review discussion. For example, label a block `checks`, not `kernel-level assertions,
  deliberately outside the demonstration`.

## Running & Git

- Never interrupt a running job and never delete files unless explicitly told to.
- Deliverables (plots, GIFs) go to `rewrite/plots/`; no previews or collages in their place.
- Vectorize; JIT is not the answer.
- **Never run full test suites**: Always run targeted tests against only the specific file or test function being worked on (e.g. `pytest tests/examples/electromagnetism/test_constitutive.py`). NEVER run `pytest` across the entire repo. Full suite runs are strictly blocked in `conftest.py` and require `--i-am-a-dunce-for-ignoring-instructions`.
- **Strictly headless execution (NEVER steal user focus)**: Pytest runs headlessly by default via `conftest.py` (which intercepts `plt.show()` to close figures and sets `Agg` unless `--show-plot` is passed). Agents running terminal commands outside pytest must run with `MPLBACKEND=Agg` so commands never spawn GUI windows or yank OS window focus away from the editor under any circumstances.
- **NEVER commit or push without explicit user command**: Never execute `git commit` or `git push` autonomously.
  Always leave modifications in the working tree for user review. Only commit or push when the user explicitly commands it.


