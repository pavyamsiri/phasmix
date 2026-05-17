"""Components that compose to form a model to generate mock data from."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, override, runtime_checkable, Any
from scipy import special
import numpy as np

if TYPE_CHECKING:
    from optype import numpy as onp


@runtime_checkable
class Component(Protocol):
    def __call__[ShapeT: tuple[Any, ...]](
        self, x: onp.ArrayND[np.float64, ShapeT], y: onp.ArrayND[np.float64, ShapeT]
    ) -> onp.ArrayND[np.float64, ShapeT]: ...


@dataclass
class GaussianComponent(Component):
    x_scale: float
    y_scale: float
    amplitude: float
    variance: float
    x_offset: float = 0
    y_offset: float = 0

    @override
    def __call__[ShapeT: tuple[Any, ...]](
        self, x: onp.ArrayND[np.float64, ShapeT], y: onp.ArrayND[np.float64, ShapeT]
    ) -> onp.ArrayND[np.float64, ShapeT]:
        rxy: onp.ArrayND[np.float64, ShapeT] = np.hypot(
            x / self.x_scale, y / self.y_scale
        )
        result: onp.ArrayND[np.float64, ShapeT] = self.amplitude * np.exp(
            -np.square(rxy) / self.variance
        )
        return result


@dataclass
class AlinderComponent(Component):
    alpha: float
    b: float
    c: float
    theta0: float
    scale_factor: float
    rho: float

    @override
    def __call__[ShapeT: tuple[Any, ...]](
        self, x: onp.ArrayND[np.float64], y: onp.ArrayND[np.float64]
    ) -> onp.ArrayND[np.float64]:
        rxy: onp.ArrayND[np.float64, ShapeT] = np.hypot(x, y / self.scale_factor)
        theta: onp.ArrayND[np.float64, ShapeT] = np.arctan2(y, self.scale_factor * x)

        phase: onp.ArrayND[np.float64]
        if self.c != 0.0:
            phase = -0.5 * self.b / self.c + np.sqrt(
                np.square(0.5 * self.b / self.c) + rxy / self.c
            )
        else:
            phase = rxy / self.b
        flattening = special.expit((rxy - self.rho) / 0.1)
        return 1.0 + self.alpha * flattening * np.cos(theta - phase - self.theta0)
