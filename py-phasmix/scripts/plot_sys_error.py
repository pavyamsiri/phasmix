"""Plot fitted alpha, angle, and scale factor against their mock values."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from matplotlib import pyplot as plt
from phasmock.component import AlinderComponent, GaussianComponent
from phasmock.mock import MockModel
from rich.console import Console
from rich.logging import RichHandler
from scipy import ndimage, stats

from phasmix import fit
from phasmix._likelihood_utils import ln_likelihood
from phasmix.component import PSpiralComponent
from phasmix.fit import FitInput, FitSuccess, GaussianSmoothConfig, PSpiralFitter

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Final

    from optype import numpy as onp


FORMAT: Final[str] = "%(message)s"
log: Final[logging.Logger] = logging.getLogger(__name__)


def setup_logging() -> Iterable[logging.Handler]:
    """Set up logging.

    Returns
    -------
    handlers : Iterable[logging.Handler]
        The logging handlers.

    """
    console = Console()
    console_handler = RichHandler(console=console, show_time=False, markup=True)
    handlers: list[logging.Handler] = [console_handler]
    logging.basicConfig(
        level="NOTSET",
        format=FORMAT,
        datefmt="[%X]",
        handlers=handlers,
        encoding="utf-8",
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    return handlers


@dataclass
class MockData:
    """Mock observations and their generating parameters."""

    density: onp.Array2D[np.float64]
    background: onp.Array2D[np.float64]
    mask: onp.Array2D[np.float64]
    x_edges: onp.Array1D[np.float64]
    y_edges: onp.Array1D[np.float64]
    x_mesh: onp.Array2D[np.float64]
    y_mesh: onp.Array2D[np.float64]
    x_centres: onp.Array1D[np.float64]
    y_centres: onp.Array1D[np.float64]
    num_stars: int
    z: onp.Array1D[np.float64]
    vz: onp.Array1D[np.float64]
    phase: float
    scale_factor: float
    amp_bg: float


def _generate_mock(alpha: float) -> MockData:
    rng = np.random.default_rng()
    b = rng.uniform(0.005, 0.1, size=1)[0]
    c = rng.uniform(0.0, 0.004, size=1)[0]
    theta = rng.uniform(-np.pi, np.pi, size=1)[0]
    scale_factor_sig = rng.uniform(30.0, 70.0, size=1)[0]
    scale_factor_bg = rng.uniform(30.0, 70.0, size=1)[0]
    amp_bg = rng.uniform(0.01, 10.0, size=1)[0]
    sigma_bg = rng.uniform(0.01, 5.0, size=1)[0]
    rho = rng.uniform(0.0, 0.18, size=1)[0]
    signal1 = AlinderComponent(
        alpha=alpha,
        b=b,
        c=c,
        theta0=theta,
        scale_factor=scale_factor_sig,
        rho=rho,
        winding=1,
    )
    background_comp = GaussianComponent(x_scale=1, y_scale=scale_factor_bg, amplitude=amp_bg, variance=sigma_bg)

    mock_model = MockModel(
        (signal1,),
        (background_comp,),
    )

    num_x_bins = 100
    num_y_bins = 100
    x_edges = np.linspace(-1.2, 1.2, num_x_bins + 1)
    y_edges = np.linspace(-60.0, 60.0, num_y_bins + 1)

    x_centres = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centres = 0.5 * (y_edges[:-1] + y_edges[1:])
    x_mesh, y_mesh = np.meshgrid(x_centres, y_centres)

    num_particles: int = 100_000
    particles = mock_model.mock_particles(num_particles, x_edges, y_edges)
    z_samples = particles.x
    vz_samples = particles.y

    # density, _, _ = np.histogram2d(z_samples, vz_samples, bins=(x_edges, y_edges))
    # density = density.T

    log.info("Generating initial background estimate via KDE...")
    # initial_background = ndimage.gaussian_filter(density, sigma=2)
    # initial_background = 0.5 * (initial_background + np.flipud(initial_background))
    density = particles.density
    initial_background = particles.background
    mask = fit.create_sigmoid_mask(1.0, 40.0)(x_mesh, y_mesh)

    phase = PSpiralComponent(
        alpha=alpha, b=b, c=c, theta0=theta, scale_factor=scale_factor_sig, rho=rho, winding=1, flattening_strength=0.1
    ).model_phase()

    log.info("ln likelihood (null) = %f", ln_likelihood(density, initial_background, mask))
    return MockData(
        density=density,
        background=initial_background,
        mask=mask,
        x_edges=x_edges,
        x_centres=x_centres,
        x_mesh=x_mesh,
        y_edges=y_edges,
        y_centres=y_centres,
        y_mesh=y_mesh,
        num_stars=num_particles,
        z=z_samples,
        vz=vz_samples,
        phase=phase,
        scale_factor=scale_factor_sig,
        amp_bg=amp_bg,
    )


def main() -> None:
    """Generate mocks, fit them, and plot parameter recovery."""

    x_edges = np.linspace(-1.2, 1.2, 100 + 1)
    y_edges = np.linspace(-60.0, 60.0, 100 + 1)
    dz: float = np.mean(np.diff(x_edges))
    dvz: float = np.mean(np.diff(y_edges))
    sigma_z = 0.1 / dz
    sigma_vz = 5.0 / dvz

    fitter = PSpiralFitter(
        backend="rust", max_iterations=10, smoothing_func=GaussianSmoothConfig(z_scale=sigma_z, vz_scale=sigma_vz)
    )

    true_alphas = np.linspace(0.0, 1.0, 25)
    actual_alphas = np.tile(true_alphas, 10)
    actual_thetas = np.zeros_like(actual_alphas)
    actual_scales = np.zeros_like(actual_alphas)
    measured_alphas = np.zeros_like(actual_alphas)
    measured_thetas = np.zeros_like(actual_thetas)
    measured_scales = np.zeros_like(actual_scales)
    actual_bg_amp = np.zeros_like(actual_scales)
    requests: list[FitInput] = []
    for idx, alpha in enumerate(actual_alphas):
        data = _generate_mock(alpha)
        actual_thetas[idx] = data.phase
        actual_scales[idx] = data.scale_factor
        actual_bg_amp[idx] = data.amp_bg
        requests.append(
            FitInput(
                density=data.density,
                background=data.background,
                z_mesh=data.x_mesh,
                vz_mesh=data.y_mesh,
                winding=1,
                num_components=1,
                improve_background=True,
            )
        )
    log.info("Starting fit...")
    results = fitter.fit_batch(requests, workers=6, progress=True)
    log.info("Fits complete...")
    for idx, result in enumerate(results):
        assert isinstance(result, FitSuccess)
        measured_alphas[idx] = result.result.final_model.components[0].alpha
        measured_thetas[idx] = result.result.final_model.components[0].model_phase()
        measured_scales[idx] = result.result.final_model.components[0].scale_factor

    corr_fit = stats.linregress(actual_alphas, measured_alphas)
    slope, slope_error = round_value_with_error(corr_fit.slope, corr_fit.stderr)
    intercept, intercept_error = round_value_with_error(corr_fit.intercept, corr_fit.intercept_stderr)
    log.info("fit: %s alpha + %s", corr_fit.slope, corr_fit.intercept)
    log.info("slope = %s +/- %s", slope, slope_error)
    log.info("intercept = %s +/- %s", intercept, intercept_error)

    actual_angles = (np.rad2deg(actual_thetas) + 180.0) % 360.0 - 180.0
    measured_angles = (np.rad2deg(measured_thetas) + 180.0) % 360.0 - 180.0

    fig, (alpha_axes, theta_axes, scale_axes) = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")

    for axes, values in (
        (alpha_axes, actual_alphas),
        (theta_axes, actual_angles),
        (scale_axes, actual_scales),
    ):
        limits = (values.min(), values.max())
        axes.plot(limits, limits, linestyle="--", color="k", label="Ideal")
        axes.grid(alpha=0.2)

    alpha_points = alpha_axes.scatter(actual_alphas, measured_alphas, marker="x", cmap="magma", c=actual_scales)
    alpha_axes.plot(true_alphas, corr_fit.slope * true_alphas + corr_fit.intercept, label="Linear fit")
    alpha_axes.set(xlabel=r"True $\alpha$", ylabel=r"Fitted $\alpha$")
    alpha_axes.legend()
    fig.colorbar(alpha_points, ax=alpha_axes, label=r"True $S$")

    theta_points = theta_axes.scatter(actual_angles, measured_angles, marker="x", cmap="viridis", c=actual_alphas)
    theta_axes.set(
        xlabel="True phase (degrees)",
        ylabel=r"Fitted $\theta_0$ (degrees)",
        xlim=(-180, 180),
        ylim=(-180, 180),
        xticks=(-180, -90, 0, 90, 180),
        yticks=(-180, -90, 0, 90, 180),
    )
    fig.colorbar(theta_points, ax=theta_axes, label=r"True $\alpha$")

    scale_points = scale_axes.scatter(actual_scales, measured_scales, c=actual_alphas, cmap="viridis", marker="x")
    scale_axes.set(xlabel="True scale factor", ylabel="Fitted scale factor")
    fig.colorbar(scale_points, ax=scale_axes, label=r"True $\alpha$")

    plt.show()
    plt.close(fig)


def round_value_with_error(value: float, error: float, *, num_sig_figs: int = 1) -> tuple[str, str]:
    """Round a value with an associated error to the errors significant figures.

    Parameters
    ----------
    value : float
        The value to round.
    error : float
        The value's error to also round.
    num_sig_figs : int
        The number of significant figures to round the error to.

    Returns
    -------
    rounded_value : float
        The rounded value.
    rounded_error : float
        The rounded error.

    """
    if not np.isfinite(value) or not np.isfinite(error) or error <= 0:
        return f"{value:.6g}", f"{error:.6g}"
    order = np.floor(np.log10(error))
    places = max(int(num_sig_figs - order - 1), 0)
    return (np.format_float_positional(value, precision=places), np.format_float_positional(error, precision=places))


if __name__ == "__main__":
    _ = setup_logging()
    main()
