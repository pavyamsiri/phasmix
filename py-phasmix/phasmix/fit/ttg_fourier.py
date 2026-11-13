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

__all__: Final[list[str]] = [
    "TTGCoeffs",
    "TTGResult",
    "compute_ttg_coefficients",
    "compute_ttg_coefficients_grid",
    "detect_ttg_phase_spiral",
]


@dataclass(frozen=True)
class TTGCoeffs:
    """The coefficients fitted in the style of Tepper-Garcia et al. 2025's.

    Attributes
    ----------
    coefficients : Array2D[complex128]
        The Fourier coefficients in the shape (num_r_bins, num_modes).
    r_edges : Array1D[f64]
        The bin edges in phase radius with shape (num_r_bins + 1,).
    orientation : float
        Phase of the m=1 mode in the orientation annulus, in [-pi, pi] radians.
        The annulus is centred on 0.5 kpc by default.

    """

    coefficients: onp.Array2D[np.complex128]
    r_edges: onp.Array1D[np.float64]
    orientation: float

    def __post_init__(self) -> None:
        """Validate that each radial bin has a row of coefficients."""
        num_r_bins = len(self.r_edges) - 1
        if self.coefficients.shape[0] != num_r_bins:
            msg = f"Expected {num_r_bins} for the coefficients array but {self.coefficients.shape}"
            raise ValueError(msg)

    def reconstruct(self, r_mesh: onp.Array2D[np.float64], theta_mesh: onp.Array2D[np.float64]) -> onp.Array2D[np.float64]:
        """Reconstruct counts per pixel on a uniform phase-space mesh.

        Parameters
        ----------
        r_mesh : Array2D[f64]
            Radius of each pixel, in the same units as ``r_edges``.
        theta_mesh : Array2D[f64]
            Angle of each pixel in radians, with the same shape as ``r_mesh``.

        Returns
        -------
        Array2D[f64]
            Truncated Fourier reconstruction with the input mesh shape.
            Pixels outside the radial bins are zero. Complete annuli are
            required for the usual Fourier normalization.

        """
        r_edges = self.r_edges
        binnumber = np.digitize(r_mesh.flatten(), r_edges).reshape(r_mesh.shape)

        recons = np.zeros_like(r_mesh)

        for i in range(len(r_edges) - 1):
            mask = binnumber == (i + 1)
            num_pixels = np.count_nonzero(mask)
            if num_pixels == 0:
                continue

            c = self.coefficients[i] / num_pixels
            recons[mask] += np.real(c[0])
            for m in range(1, len(c)):
                recons[mask] += 2 * np.real(c[m] * np.exp(-1j * m * theta_mesh[mask]))
        return recons


@dataclass(frozen=True)
class TTGResult:
    """The result of a Tepper-Garcia et al. 2025's phase spiral finder.

    Attributes
    ----------
    strength : float
        The phase spiral strength equal to median(A1/A0) over the bins in phase radius.
    strength_m2 : float
        Median A2/A0 over the same selected bins as ``strength``.
        NaN if mode 2 was not computed.
    phase : Array1D[f64]
        The phase angle of the m=1 mode over the phase radius bins with shape (num_r_bins,).
    orientation : float
        Phase of the separate orientation annulus, in [-pi, pi] radians.
        NaN when that annulus has no measurable m=1 coefficient.
    r_edges : Array1D[f64]
        The bin edges in phase radius with shape (num_r_bins + 1,).
    is_spiral : bool
        Whether a phase spiral was detected meaning it has passed all four criteria.
    amp_criterion : bool
        Whether the radial mean of A1/Am is at least one for every mode m >= 2.
    slope_criterion : bool
        Whether the slope criterion was passed, that is the fitted slope's magnitude is greater than some threshold.
    slope_error_criterion : bool
        Whether the fitted slope's standard error is at most its threshold.
    monotonicity_criterion : bool
        Whether the selected unwrapped phases are nondecreasing or nonincreasing.
    amps : Array2D[f64]
        Amplitudes normalised by A0, with shape (num_r_bins, num_modes).
    fit_mask : Array1D[bool]
        Interior radial bins used for strength, amplitude ratios, and fitting.
    slope, slope_err : float
        Linear phase slope and its standard error in radians per radial unit.
    intercept, intercept_err : float
        Linear phase intercept and its standard error in radians.
    slope_threshold, slope_err_threshold : float
        Minimum absolute slope and maximum slope standard error.
    winding : -1 or +1
        Sign of the fitted slope, with zero assigned +1.

    """

    strength: float
    strength_m2: float
    amps: onp.Array2D[np.float64]
    phase: onp.Array1D[np.float64]
    orientation: float
    fit_mask: onp.Array1D[np.bool_]
    r_edges: onp.Array1D[np.float64]
    winding: Literal[-1, 1]
    slope_threshold: float
    slope_err_threshold: float

    # Fit
    slope: float
    slope_err: float
    intercept: float
    intercept_err: float

    # Criteria
    is_spiral: bool
    amp_criterion: bool
    slope_criterion: bool
    slope_error_criterion: bool
    monotonicity_criterion: bool


def compute_ttg_coefficients(
    r: onp.Array1D[np.float64],
    thetaz: onp.Array1D[np.float64],
    *,
    r_edges: onp.Array1D[np.float64],
    max_mode: int,
    orientation_centre: float = 0.5,
    orientation_width: float = 0.1,
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
    max_mode : int
        Highest Fourier mode to compute, including all modes from zero.
    orientation_centre : float
        The radius at which to calculate the orientation at in kpc.
    orientation_width : float
        Full width in kpc of the separate annulus centred on `orientation_centre` kpc.

    Returns
    -------
    TTGCoeffs
        The computed Fourier coefficients.

    """
    return _compute_annular_coefficients(
        r,
        thetaz,
        np.ones_like(r),
        r_edges=r_edges,
        max_mode=max_mode,
        orientation_centre=orientation_centre,
        orientation_width=orientation_width,
    )


def compute_ttg_coefficients_grid(
    counts: onp.Array2D[np.float64],
    r_mesh: onp.Array2D[np.float64],
    phi_mesh: onp.Array2D[np.float64],
    *,
    r_extent: tuple[float, float],
    max_mode: int,
    num_bins: int,
    orientation_centre: float = 0.5,
    orientation_width: float = 0.1,
) -> TTGCoeffs:
    """Compute the Fourier coefficients in a series of rings in phase radius.

    Step 1 of the Tepper-Garcia et al. 2025 phase spiral finding method.

    Parameters
    ----------
    counts : Array2D[f64]
        Star counts per pixel on a uniform phase-space grid.
    r_mesh : Array2D[f64]
        Radius of each pixel in kpc, with the same shape as ``counts``.
    phi_mesh : Array2D[f64]
        Angle of each pixel in radians, with the same shape as ``counts``.
    r_extent : tuple[float, float]
        Lower and upper radial limits in kpc.
    max_mode : int
        Highest Fourier mode to compute, including all modes from zero.
    num_bins : int
        Number of equally spaced radial bins within ``r_extent``.
    orientation_centre : float
        The radius at which to calculate the orientation at in kpc.
    orientation_width : float
        Full width in kpc of the separate annulus centred on `orientation_centre` kpc.

    Returns
    -------
    TTGCoeffs
        The computed Fourier coefficients.

    """
    radial_bins = np.linspace(r_extent[0], r_extent[1], num_bins + 1)
    return _compute_annular_coefficients(
        r_mesh.ravel(),
        phi_mesh.ravel(),
        counts.ravel(),
        r_edges=radial_bins,
        max_mode=max_mode,
        orientation_centre=orientation_centre,
        orientation_width=orientation_width,
    )


def _compute_annular_coefficients(
    r: onp.Array1D[np.float64],
    theta: onp.Array1D[np.float64],
    weights: onp.Array1D[np.float64],
    *,
    r_edges: onp.Array1D[np.float64],
    max_mode: int,
    orientation_centre: float,
    orientation_width: float,
) -> TTGCoeffs:
    """Process the radial bins and extra orientation annulus in one Fourier pass."""
    if not np.isfinite(orientation_width) or not 0 < orientation_width <= 1.0:
        msg = "`orientation_width` must be finite and in (0, 1] kpc"
        raise ValueError(msg)
    annuli: list[tuple[float, float]] = list(itertools.pairwise(r_edges))
    annuli.append((orientation_centre - orientation_width / 2, orientation_centre + orientation_width / 2))
    values_list: list[onp.Array1D[np.float64]] = []
    phis_list: list[onp.Array1D[np.float64]] = []
    for r_in, r_out in annuli:
        mask = (r >= r_in) & (r < r_out)
        values_list.append(weights[mask])
        phis_list.append(theta[mask])
    coefficients = fourier_custom(values=values_list, phis=phis_list, max_mode=max(max_mode, 1))
    orientation_coefficient = complex(coefficients[-1, 1])  # pyright: ignore[reportAny]
    orientation = float(np.angle(orientation_coefficient)) if abs(orientation_coefficient) > 0 else float("nan")
    return TTGCoeffs(
        coefficients=coefficients[:-1, : max_mode + 1],
        r_edges=r_edges,
        orientation=orientation,
    )


def detect_ttg_phase_spiral(
    coeffs: TTGCoeffs, *, slope_threshold: float = 4.5, slope_err_threshold: float = 4.5, exclude: int = 2
) -> TTGResult:
    """Post-process the Fourier coefficients and use four criteria to detect phase spirals.

    Parameters
    ----------
    coeffs : TTGCoeffs
        The computed coefficients.
    slope_threshold : float
        The absolute slope threshold. The phase must vary linearly with phase radius with a slope at least this large.
        Default is 4.5.
    slope_err_threshold : float
        Maximum standard error of the fitted linear slope, in radians per radial unit.
        Default is 4.5.
    exclude : int
        Exclude the given number of bins on the inside and outside.

    Returns
    -------
    TTGResult
        The result of the phase spiral detection and post-processing.

    """
    r_centres = 0.5 * (coeffs.r_edges[:-1] + coeffs.r_edges[1:])
    fit_mask = np.zeros_like(r_centres, np.bool_)
    fit_mask[exclude:-exclude] = True

    coefficients = coeffs.coefficients
    amps = np.abs(coefficients) / np.abs(coefficients[:, 0])[:, None]
    phase = np.unwrap(np.angle(coefficients[:, 1]))

    strength = np.nanmedian(amps[:, 1][fit_mask])
    strength_m2 = float(np.nanmedian(amps[fit_mask, 2])) if amps.shape[1] > 2 else float("nan")

    fit_amps = np.abs(coefficients)[fit_mask]

    with np.errstate(divide="ignore", invalid="ignore"):
        mode_ratios = fit_amps[:, 1, None] / fit_amps[:, 2:]

    mean_mode_ratios = np.nanmean(mode_ratios, axis=0)
    amp_criterion = bool(np.all(mean_mode_ratios >= 1.0))

    # Slope criterion
    fit_x = r_centres[fit_mask]
    fit_y = np.unwrap(phase[fit_mask])
    gradient_signs = np.sign(np.diff(fit_y))
    monotonicity = bool(np.all(gradient_signs >= 0) or np.all(gradient_signs <= 0))

    res = stats.linregress(fit_x, fit_y)

    slope_criterion: bool = bool(np.abs(res.slope) >= slope_threshold)  # pyright: ignore[reportAny]
    slope_error_criterion: bool = bool(res.stderr <= slope_err_threshold)
    winding = 1 if res.slope >= 0 else -1

    return TTGResult(
        strength=strength,
        strength_m2=strength_m2,
        orientation=coeffs.orientation,
        amps=amps,
        phase=phase,
        winding=winding,
        # Binning
        r_edges=coeffs.r_edges,
        fit_mask=fit_mask,
        # Thresholds
        slope_threshold=slope_threshold,
        slope_err_threshold=slope_err_threshold,
        # Fit
        slope=res.slope,
        slope_err=res.stderr,
        intercept=res.intercept,
        intercept_err=res.intercept_stderr,
        # Criteria
        is_spiral=amp_criterion and slope_criterion and slope_error_criterion and monotonicity,
        amp_criterion=amp_criterion,
        slope_criterion=slope_criterion,
        slope_error_criterion=slope_error_criterion,
        monotonicity_criterion=monotonicity,
    )


def azimuthal_bin(
    matrix: onp.Array2D[np.float64],
    r_mesh: onp.Array2D[np.float64],
    phi_mesh: onp.Array2D[np.float64],
    r_in: float,
    r_out: float,
) -> tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]]:
    """Return row/column indices, total, and values in [r_in, r_out).

    Radii are measured in pixels about the same centre used by the Fourier
    analysis below. Adjacent annuli do not share pixels.
    """
    mask: onp.Array2D[np.bool_] = np.logical_and(r_mesh >= r_in, r_mesh < r_out)
    values: onp.Array1D[np.float64] = matrix[mask]
    return values, phi_mesh[mask]


def fourier_custom(
    values: list[onp.Array1D[np.float64]],
    phis: list[onp.Array1D[np.float64]],
    max_mode: int = 6,
) -> onp.Array2D[np.complex128]:
    """Compute summed complex Fourier coefficients for each annulus.

    ``indices`` contains (row, column) arrays and ``mat_centre`` is (x, y).
    The positive exponential convention gives an increasing m=1 phase for
    the default toy spiral cos(r - theta). Results have shape
    (number_of_annuli, max_mode + 1); empty annuli remain NaN.
    """
    coefficients = np.full((len(values), max_mode + 1), np.nan, dtype=np.complex128)
    modes = np.arange(max_mode + 1)

    for bin_index, (bin_values, bin_phis) in enumerate(zip(values, phis, strict=True)):
        valid = np.isfinite(bin_values) & np.isfinite(bin_phis)
        if not np.any(valid):
            continue
        coefficients[bin_index, :] = np.sum(
            bin_values[valid] * np.exp(1j * modes[:, None] * bin_phis[valid]),
            axis=1,
        )
    return coefficients
