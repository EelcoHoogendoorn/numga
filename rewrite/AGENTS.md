# Working in the numga rewrite

Rules for code in this tree, especially `examples/`. They are the owner's, stated during review;
follow them exactly.

## Algebra first

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
- No interleaved plotting and math; a generator yields geometry, a draw helper consumes it.
- Checks are kernel-level assertions in one labelled block at the end of `main`.
- Do not touch comments or docstrings unasked. Group functions on the type they belong to
  (`Bodies.join`), do not pollute the module namespace.
- Public-facing code and documentation must stand on their own. Explain the mathematics,
  behavior, and relevant design decisions; never narrate the editing process, refer to the
  conversation, or advertise compliance with instructions. Such commentary makes readers
  interpret our workflow instead of helping them understand the subject. Keep it in chat
  or review discussion. For example, label a block `checks`, not `kernel-level assertions,
  deliberately outside the demonstration`.

## Running

- Never interrupt a running job and never delete files unless explicitly told to.
- Deliverables (plots, GIFs) go to `rewrite/plots/`; no previews or collages in their place.
- Vectorize; JIT is not the answer.
- Always run targeted tests (`pytest path/to/test_file.py`). Only run the full test suite when editing test infrastructure across the entire suite or during explicit pre-commit checks.
