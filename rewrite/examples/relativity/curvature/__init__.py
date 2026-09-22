"""Plane gravitational wave curvature and tidal response in Spacetime Algebra."""


def __getattr__(name: str):
    if name == "main":
        from .scenarios import main
        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["main"]
