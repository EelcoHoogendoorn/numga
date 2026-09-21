# Dirty Plot Auto-Regeneration Rule

Whenever any code in an example, rendering module, core mathematical script, or scenario is touched or modified:
1. Identify all plots, figures, and animations (GIFs) that are affected ("dirty").
2. Automatically re-execute the corresponding script to regenerate ALL dirty plots and animations immediately. Do not wait for the user to ask.
3. Canonical plots and animations must be written directly to the workspace's `examples.PLOT_DIR` (`rewrite/plots/`), where the user can see, inspect, and track them.
4. Repository code and tests must NEVER reference or write to internal agent cache directories (`~/.gemini/antigravity-ide/brain`).

## Test Suite Invariant: Illegal to Test Without Regenerating Plots
- Example unit test modules must include an autouse fixture (e.g. `enforce_regenerate_multiview_plots`) that executes the scenario and convergence animation, asserting that canonical plot and GIF files in `PLOT_DIR` are generated fresh during the test run.
- It is strictly illegal for tests to pass against stale or absent plots. Tests fail if plots are not actively produced and verified.
- The assistant MUST embed all regenerated plots and GIFs in its response whenever tests are run.

## Transparency: Expose Private Agent Work
- Never hide scratch scripts, debug plots, or trial media in private IDE cache directories where the user cannot see them.
- All temporary experiments, debug figures, and exploratory work should be saved in the workspace (e.g. `rewrite/plots/` or a symlinked workspace scratch folder) so the user has complete visibility into what the assistant is doing.
