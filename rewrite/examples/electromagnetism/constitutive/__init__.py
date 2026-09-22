"""Electromagnetic constitutive extensor: birefringence, ferrites, axions, and Fresnel drag."""


def __getattr__(name: str):
    if name == "main":
        from examples.electromagnetism.constitutive.scenarios import main
        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["main"]
