# Headless Matplotlib & Focus Preservation Rules

- NEVER spawn interactive GUI plot windows or allow commands to steal desktop focus from the user.
- ALWAYS run matplotlib headlessly (`MPLBACKEND=Agg` or `matplotlib.use('Agg')`) in all background commands, scripts, tests, and runners.
- Never execute scripts or notebook code that calls `plt.show()` with a GUI backend (`MacOSX`, `TkAgg`, `Qt`).
