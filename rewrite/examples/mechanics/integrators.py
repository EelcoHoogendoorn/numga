"""Standard explicit Runge-Kutta numerical integrators."""

from __future__ import annotations

from typing import Any, Callable


def RK1(f: Callable[[Any], Any], y: Any, h: float) -> Any:
    """Euler forward step."""
    return y + f(y) * h


def RK4(f: Callable[[Any], Any], y: Any, h: float) -> Any:
    """Classical 4th-order Runge-Kutta step."""
    k1 = f(y)
    k2 = f(y + 0.5 * h * k1)
    k3 = f(y + 0.5 * h * k2)
    k4 = f(y + h * k3)
    return y + (h / 3.0) * (k2 + k3 + (k1 + k4) * 0.5)


def RK8(f: Callable[[Any], Any], y: Any, h: float) -> Any:
    """8th-order Runge-Kutta step."""
    k1 = f(y)
    k2 = f(y + (h * 4.0 / 27.0) * k1)
    k3 = f(y + (h / 18.0) * (k1 + 3.0 * k2))
    k4 = f(y + (h / 12.0) * (k1 + 3.0 * k3))
    k5 = f(y + (h / 8.0) * (k1 + 3.0 * k4))
    k6 = f(y + (h / 54.0) * (13.0 * k1 - 27.0 * k3 + 42.0 * k4 + 8.0 * k5))
    k7 = f(y + (h / 4320.0) * (389.0 * k1 - 54.0 * k3 + 966.0 * k4 - 824.0 * k5 + 243.0 * k6))
    k8 = f(y + (h / 20.0) * (-234.0 * k1 + 81.0 * k3 - 1164.0 * k4 + 656.0 * k5 - 122.0 * k6 + 800.0 * k7))
    k9 = f(y + (h / 288.0) * (-127.0 * k1 + 18.0 * k3 - 678.0 * k4 + 456.0 * k5 - 9.0 * k6 + 576.0 * k7 + 4.0 * k8))
    k10 = f(y + (h / 820.0) * (
        1481.0 * k1 - 81.0 * k3 + 7104.0 * k4 - 3376.0 * k5 + 72.0 * k6 - 5040.0 * k7 - 60.0 * k8 + 720.0 * k9
    ))
    return y + (h / 840.0) * (
        41.0 * k1 + 27.0 * k4 + 272.0 * k5 + 27.0 * k6 + 216.0 * k7 + 216.0 * k9 + 41.0 * k10
    )
