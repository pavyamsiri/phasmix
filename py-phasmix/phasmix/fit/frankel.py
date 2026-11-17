"""Implementation of Frankel et al. 2023's "zebra stripe" phase spiral method."""

from __future__ import annotations

import functools
import logging
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
from scipy import interpolate, ndimage, optimize

if TYPE_CHECKING:
    from typing import Any, Final

    from optype import numpy as onp

__all__: Final[list[str]] = ["FrankelResult", "compute_frankel_parameters"]

AMP_TOLERANCE: Final[float] = 1e-9
log: Final[logging.Logger] = logging.getLogger(__name__)


@runtime_checkable
class BackgroundInterpolator(Protocol):
    """A protocol for a background interpolator."""

    def __call__[S: tuple[Any, ...]](
        self,
        x: onp.ArrayND[np.float64, S],
    ) -> onp.ArrayND[np.float64, S]: ...


@dataclass(frozen=True)
class FrankelResult:
    """The fitted parameters of the Frankel et al. 2023 zebra stripe model.

    Attributes
    ----------
    amplitude : float
        The perturbation amplitude.
    thetaz0 : float
        The initial phase.
    t : float
        The winding time.
    background : BackgroundInterpolator
        The background stored as an interpolator object.

    """

    amplitude: float
    thetaz0: float
    t: float
    background: BackgroundInterpolator

    def __post_init__(self) -> None:
        """Validate the parameters are constrained."""
        if self.t < 0.0:
            msg = f"Winding time `t` must be non-negative: t = {self.t}."
            raise ValueError(msg)
        if self.amplitude < 0.0 or self.amplitude > 1.0:
            msg = f"Amplitude `A` must be in the range [0, 1]: A = {self.amplitude}."
            raise ValueError(msg)

    def reconstruct(self, thetaz_mesh: onp.Array2D[np.float64], omegaz_mesh: onp.Array2D[np.float64]) -> onp.Array2D[np.float64]:
        """Reconstruct the number count map in the (Omegaz, thetaz) plane.

        Parameters
        ----------
        thetaz_mesh : Array2D[f64]
            The thetaz grid in radians.
        omegaz_mesh : Array2D[f64]
            The omegaz grid.

        Returns
        -------
        recons : Array2D[f64]
            The reconstructed map.

        """
        num_thetaz_bins: int = thetaz_mesh.shape[1]
        phase = thetaz_mesh - omegaz_mesh * self.t - self.thetaz0
        background = self.background(omegaz_mesh) / num_thetaz_bins
        background[np.isnan(background)] = 0.0
        result: onp.Array2D[np.float64] = background * (1.0 + self.amplitude * np.cos(phase))
        return result


def compute_frankel_parameters(
    omegaz: onp.Array1D[np.float64],
    thetaz: onp.Array1D[np.float64],
    *,
    max_time: float = 2.0,
    omegaz_bins: onp.Array1D[np.float64] | None = None,
) -> FrankelResult:
    """Fit the three free parameters in Frankel et al. 2023's zebra stripe model of the (1-arm) phase spiral.

    Parameters
    ----------
    omegaz : Array1D[f64]
        The vertical frequencies of the particles.
    thetaz : Array1D[f64]
        The vertical conjugate angle of the particles, in radians.
    max_time : float
        The maximum winding time allowed. Used to determine the upper bound when optimising for maximum likelihood.
    omegaz_bins : Array1D[f64] | None
        The omegaz binning used to construct the background interpolator for later use when doing reconstructions.

    Returns
    -------
    FrankelResult
        The result of fitting the model parameters and background estimation.

    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="invalid value encountered in subtract",
            category=RuntimeWarning,
            module=r"scipy\.optimize\._numdiff",
        )
        res = optimize.differential_evolution(
            functools.partial(_objective, omegaz=omegaz, thetaz=thetaz),
            bounds=[
                (-1.0, 1.0),
                (-1.0, 1.0),
                (0.0, max_time),
            ],
            polish=True,
        )

    if not res.success:
        log.warning("Optimization failed: %s", res.message)

    a, b, t = res.x

    amplitude: float = np.hypot(a, b)  # pyright: ignore[reportAny]
    thetaz0: float = np.arctan2(b, a)  # pyright: ignore[reportAny]

    if omegaz_bins is None:
        min_omegaz: float
        max_omegaz: float
        min_omegaz, max_omegaz = np.nanpercentile(omegaz, (1, 99))  # pyright: ignore[reportAny]
        num_omegaz_bins: int = 100
        omegaz_bins = np.linspace(min_omegaz, max_omegaz, num_omegaz_bins + 1)

    omegaz_centres = 0.5 * (omegaz_bins[:-1] + omegaz_bins[1:])
    marginal_density, _ = np.histogram(
        omegaz,
        bins=omegaz_bins,
    )
    smoothed_density = ndimage.gaussian_filter1d(marginal_density, sigma=1)

    bg_interp: BackgroundInterpolator = interpolate.PchipInterpolator(omegaz_centres, smoothed_density, extrapolate=False)  # pyright: ignore[reportAssignmentType]
    return FrankelResult(amplitude=amplitude, thetaz0=thetaz0, t=t, background=bg_interp)


def _objective(parameters: onp.Array1D[np.float64], *, omegaz: onp.Array1D[np.float64], thetaz: onp.Array1D[np.float64]) -> float:
    a, b, t = parameters
    amp = np.hypot(a, b)  # pyright: ignore[reportAny]

    if amp > 1.0:
        return np.inf

    phase = thetaz - omegaz * t
    cos_term = a * np.cos(phase)
    sin_term = b * np.sin(phase)
    log_arg = cos_term + sin_term
    return -np.sum(np.log1p(log_arg))  # pyright: ignore[reportAny]
