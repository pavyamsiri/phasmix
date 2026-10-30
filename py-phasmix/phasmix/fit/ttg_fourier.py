"""Implementation of Tepper-Garcia et al. 2025's Fourier based phase spiral finder."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy import stats

if TYPE_CHECKING:
    from typing import Final

    from optype import numpy as onp

__all__: Final[list[str]] = ["TTGCoeffs", "TTGResult", "compute_ttg_coefficients", "detect_ttg_phase_spiral"]


@dataclass(frozen=True)
class TTGCoeffs:
    """The coefficients fitted in the style of Tepper-Garcia et al. 2025's.

    Attributes
    ----------
    coefficients : Array2D[complex128]
        The Fourier coefficients in the shape (num_r_bins, num_modes).
    r_edges : Array1D[f64]
        The bin edges in phase radius with shape (num_r_bins + 1,).

    """

    coefficients: onp.Array2D[np.complex128]
    r_edges: onp.Array1D[np.float64]

    def __post_init__(self) -> None:
        num_r_bins = len(self.r_edges) - 1
        if self.coefficients.shape[0] != num_r_bins:
            msg = f"Expected {num_r_bins} for the coefficients array but {self.coefficients.shape}"
            raise ValueError(msg)


@dataclass(frozen=True)
class TTGResult:
    """The result of a Tepper-Garcia et al. 2025's phase spiral finder.

    Attributes
    ----------
    strength : float
        The phase spiral strength equal to median(A1/A0) over the bins in phase radius.
    phase : Array1D[f64]
        The phase angle of the m=1 mode over the phase radius bins with shape (num_r_bins,).
    r_edges : Array1D[f64]
        The bin edges in phase radius with shape (num_r_bins + 1,).
    is_spiral : bool
        Whether a phase spiral was detected meaning it has passed all four criteria.
    amp_criterion : bool
        Whether the radial mean of A1/A0 exceeds that of each mode m >= 2.
    slope_criterion : bool
        Whether the slope criterion was passed, that is the fitted slope's magnitude is greater than some threshold.
    scatter_criterion : bool
        Whether sqrt(SSR) of the phase residuals is strictly below the scatter threshold.
    slope_error_criterion : bool
        Whether the relative slope error is strictly below its threshold.
    amps : Array2D[f64]
        Amplitudes normalised by A0, with shape (num_r_bins, num_modes).
    fit_mask : Array1D[bool]
        Bins where mode 1 dominates every higher mode, used for the phase fit.
    slope, intercept, slope_stderr, slope_relative_error, scatter : float
        Masked linear fit diagnostics; scatter is sqrt(SSR) in radians.
        All are NaN if fewer than three bins pass the mask.
    predicted : Array1D[f64]
        Fitted phase evaluated at all radial bin centres, or NaN when no fit is available.
    slope_threshold, slope_err_threshold, scatter_threshold : float
        Thresholds used for this detection.
    winding : -1 or +1
        Sign of the fitted slope; defaults to -1 when no fit is available.

    """

    strength: float
    phase: onp.Array1D[np.float64]
    r_edges: onp.Array1D[np.float64]
    is_spiral: bool
    amp_criterion: bool
    slope_criterion: bool
    scatter_criterion: bool
    slope_error_criterion: bool
    winding: Literal[-1, 1]
    amps: onp.Array2D[np.float64]
    fit_mask: onp.Array1D[np.bool_]
    slope: float
    intercept: float
    slope_stderr: float
    slope_relative_error: float
    scatter: float
    predicted: onp.Array1D[np.float64]
    slope_threshold: float
    slope_err_threshold: float
    scatter_threshold: float


def compute_ttg_coefficients(
    r: onp.Array1D[np.float64], thetaz: onp.Array1D[np.float64], *, r_edges: onp.Array1D[np.float64], max_mode: int
) -> TTGCoeffs:
    """Compute the Fourier coefficients in a series of rings in phase radius.

    Step 1 of the Tepper-Garcia et al. 2025 phase spiral finding method.

    Parameters
    ----------
    r : Array1D[f64]
        The phase radius of the stars in the sample, r = sqrt(Jz).
    thetaz : Array1D[f64]
        The conjugate angle of the stars in the sample in units of radians.
    r_edges : Array1D[f64]
        The bin edges in phase radius.

    Returns
    -------
    TTGCoeffs
        The computed Fourier coefficients.

    """
    num_r_bins = len(r_edges) - 1
    num_modes = max_mode + 1
    coefficients = np.full((num_r_bins, num_modes), np.nan, dtype=np.complex128)

    for r_index, (r_min, r_max) in enumerate(itertools.pairwise(r_edges)):
        for m in range(num_modes):
            mask = np.logical_and(r >= r_min, r < r_max)

            current_theta = thetaz[mask]
            coefficients[r_index, m] = np.sum(np.exp(1j * m * current_theta))

    return TTGCoeffs(coefficients=coefficients, r_edges=r_edges)


def detect_ttg_phase_spiral(
    coeffs: TTGCoeffs, *, slope_threshold: float = 0.667, slope_err_threshold: float = 0.2, scatter_threshold: float = 1.25
) -> TTGResult:
    """Post-process the Fourier coefficients and use four criteria to detect phase spirals.

    Parameters
    ----------
    coeffs : TTGCoeffs
        The computed coefficients.
    slope_threshold : float
        The absolute slope threshold. The phase must vary linearly with phase radius with a slope at least this large.
        Default is 0.667.
    slope_err_threshold : float
        The relative slope error threshold. The fitted slope must have a relative error no greater than the threshold.
        Default is 0.2 (20%).
    scatter_threshold : float
        Strict upper limit on sqrt(SSR) = sqrt(sum((phase - fitted_phase)**2)), in radians.
        Default is 1.25, equivalent to toy method 2's 1/sqrt(SSR) > 0.8.
        This statistic is not divided by the number of radial bins.

    Returns
    -------
    TTGResult
        The result of the phase spiral detection and post-processing.

    """
    r_centres = 0.5 * (coeffs.r_edges[:-1] + coeffs.r_edges[1:])

    amps = np.abs(coeffs.coefficients) / np.abs(coeffs.coefficients[:, 0])[:, None]
    phase = np.unwrap(np.angle(coeffs.coefficients[:, 1]))

    strength = np.median(amps[:, 1])

    avg_amps = np.mean(amps, axis=0)
    amp_criterion = all(avg_amps[1, None] > avg_amps[2:])

    mask = amps[:, 1] > avg_amps[1]
    slope_criterion: bool = False
    slope_error_criterion: bool = False
    scatter_criterion: bool = False
    winding: Literal[-1, 1] = -1
    slope = intercept = slope_stderr = slope_relative_error = scatter = float("nan")
    predicted = np.full_like(r_centres, np.nan)
    if np.count_nonzero(mask) > 2:
        phase_fit = stats.linregress(r_centres[mask], phase[mask])
        slope = float(phase_fit.slope)
        intercept = float(phase_fit.intercept)
        slope_stderr = float(phase_fit.stderr)
        slope_relative_error = abs(slope_stderr / slope) if slope != 0.0 else float("inf")
        predicted = slope * r_centres + intercept
        scatter = float(np.sqrt(np.sum(np.square(phase[mask] - predicted[mask]))))

        slope_criterion = abs(slope) >= slope_threshold
        slope_error_criterion = slope_relative_error < slope_err_threshold
        scatter_criterion = bool(scatter < scatter_threshold)

        winding = 1 if slope >= 0 else -1

    return TTGResult(
        strength=strength,
        phase=phase,
        r_edges=coeffs.r_edges,
        is_spiral=amp_criterion and slope_criterion and scatter_criterion and slope_error_criterion,
        amp_criterion=amp_criterion,
        slope_criterion=slope_criterion,
        scatter_criterion=scatter_criterion,
        slope_error_criterion=slope_error_criterion,
        winding=winding,
        amps=amps,
        fit_mask=mask,
        slope=slope,
        intercept=intercept,
        slope_stderr=slope_stderr,
        slope_relative_error=slope_relative_error,
        scatter=scatter,
        predicted=predicted,
        slope_threshold=slope_threshold,
        slope_err_threshold=slope_err_threshold,
        scatter_threshold=scatter_threshold,
    )
