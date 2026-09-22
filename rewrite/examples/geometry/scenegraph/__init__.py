"""Scenegraph, kinematics, and compound multi-lens camera optics in PGA3D."""

def __getattr__(name: str):
    if name == "main":
        from .scenarios import main
        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
