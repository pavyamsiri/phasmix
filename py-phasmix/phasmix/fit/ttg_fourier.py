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
        Whether a phase spiral was detected meaning it has passed the three criteria.
    amp_criterion : bool
        Whether the amplitude criterion was passed, that is A1/A0 > Am/A0 for m >= 2 in all phase radius bins.
    slope_criterion : bool
        Whether the slope criterion was passed, that is the fitted slope's magnitude is greater than some threshold.
    scatter_criterion : bool
        Whether the scatter criterion was passed, that is the scatter around the fitted slope is smaller than some threshold.

    """

    strength: float
    phase: onp.Array1D[np.float64]
    r_edges: onp.Array1D[np.float64]
    is_spiral: bool
    amp_criterion: bool
    slope_criterion: bool
    scatter_criterion: bool
    winding: Literal[-1, 1]


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


def detect_ttg_phase_spiral(coeffs: TTGCoeffs, *, slope_threshold: float = 0.1, scatter_threshold: float = 0.3) -> TTGResult:
    """Post-process the Fourier coefficients and use three criteria to detect phase spirals.

    Parameters
    ----------
    coeffs : TTGCoeffs
        The computed coefficients.
    slope_threshold : float
        The absolute slope threshold. The phase must vary linearly with phase radius with a slope at least this large.
        Default is 0.1.
    scatter_threshold : float
        The scatter threshold. The raw phase and the linear fit of phase to phase radius must not scatter more than this.
        Default is 0.3.

    Returns
    -------
    TTGResult
        The result of the phase spiral detection and post-processing.

    """
    r_centres = 0.5 * (coeffs.r_edges[:-1] + coeffs.r_edges[1:])

    amps = np.abs(coeffs.coefficients) / np.abs(coeffs.coefficients[:, 0])[:, None]
    phase = np.unwrap(np.angle(coeffs.coefficients[:, 1]))

    phase_fit = stats.linregress(r_centres, phase)

    predicted = phase_fit.slope * r_centres + phase_fit.intercept
    strength = np.median(amps[:, 1])
    scatter = np.float64(np.sqrt(np.mean(np.square(phase - predicted))))  # pyright: ignore[reportAny]

    amp_criterion = all(np.all(amps[:, 1, None] > amps[:, 2:], axis=1))
    slope_criterion = bool(np.abs(phase_fit.slope) >= slope_threshold)  # pyright: ignore[reportAny]
    scatter_criterion = bool(scatter <= scatter_threshold)

    return TTGResult(
        strength=strength,
        phase=phase,
        r_edges=coeffs.r_edges,
        is_spiral=amp_criterion and slope_criterion and scatter_criterion,
        amp_criterion=amp_criterion,
        slope_criterion=slope_criterion,
        scatter_criterion=scatter_criterion,
        winding=1 if np.sign(phase_fit.slope) >= 0 else -1,
    )
