"""Plot fitted alpha, angle, and scale factor against their mock values."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from itertools import combinations
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


# Column order is shared by the Sobol design and mock generation.
PARAMETER_BOUNDS: Final = np.array(
    [(0.0, 1.0), (0.005, 0.1), (0.0, 0.004), (-np.pi, np.pi), (30.0, 70.0), (30.0, 70.0), (0.01, 5.0), (0.0, 0.18)]
)


def sample_parameters(exponent: int, seed: int) -> onp.Array2D[np.float64]:
    """Draw a complete scrambled Sobol design for all eight parameters."""
    unit = stats.qmc.Sobol(d=len(PARAMETER_BOUNDS), scramble=True, rng=np.random.default_rng(seed)).random_base2(exponent)
    return stats.qmc.scale(unit, PARAMETER_BOUNDS[:, 0], PARAMETER_BOUNDS[:, 1])


def estimate_design(seed: int, batch_size: int, exponent: int = 10) -> None:
    """Report pairwise coverage and array payloads without generating particles.

    Coverage uses eight equal-width bins per axis in normalized coordinates.
    Array sizes exclude Python overhead, native copies and optimizer workspace;
    they are payload estimates, not peak RSS predictions.
    """
    selected_points = 2**exponent
    log.info(
        "Selected run: %d points, batch size %d, %d batches (last batch: %d points).",
        selected_points,
        batch_size,
        (selected_points + batch_size - 1) // batch_size,
        (selected_points - 1) % batch_size + 1,
    )
    log.info("8 parameters; pairwise coverage on 8 x 8 grids (28 pairs).")
    log.info("Points | batches | worst pair occupied | minimum cell count | inputs MiB | batched inputs MiB")
    for candidate_exponent in range(8, 13):
        parameters = sample_parameters(candidate_exponent, seed)
        unit = (parameters - PARAMETER_BOUNDS[:, 0]) / np.diff(PARAMETER_BOUNDS, axis=1).ravel()
        counts = [np.histogram2d(unit[:, i], unit[:, j], bins=8, range=((0, 1), (0, 1)))[0] for i, j in combinations(range(8), 2)]
        points = len(parameters)
        # Each FitInput retains density, background and two meshes: 4 float64 grids.
        per_input = 4 * 100 * 100 * np.dtype(np.float64).itemsize
        log.info(
            "%6d | %7d | %6.1f%% | %5d | %8.2f | %8.2f",
            points,
            (points + batch_size - 1) // batch_size,
            100 * min(np.count_nonzero(c) / c.size for c in counts),
            int(min(c.min() for c in counts)),
            points * per_input / 2**20,
            min(points, batch_size) * per_input / 2**20,
        )
    log.info("One mock's retained particle coordinates: 1.53 MiB (100,000 stars).")
    log.info("Parameters + seven recovery arrays: %.2f MiB per 1,024 points.", 1024 * 15 * 8 / 2**20)
    log.info(
        "Rust result array payload: ~%.2f MiB per batch (data + initial/final backgrounds).", batch_size * 3 * 10000 * 8 / 2**20
    )
    log.info("Native copies, temporary arrays and optimizer workspace add to these payloads; peak RSS will be higher.")
    log.info("An 8-bin grid in all eight dimensions has 8**8 = 16,777,216 cells.")
    log.info("Pairwise coverage does not guarantee 8D coverage or convergence of recovery statistics.")


def _generate_mock(parameters: onp.Array1D[np.float64]) -> MockData:
    alpha, b, c, theta, scale_factor_sig, scale_factor_bg, sigma_bg, rho = parameters
    # num_particles: int = int(rng.integers(1_000, 500_000, size=1)[0])  # noqa: ERA001
    num_particles = 100_000
    signal1 = AlinderComponent(
        alpha=alpha,
        b=b,
        c=c,
        theta0=theta,
        scale_factor=scale_factor_sig,
        rho=rho,
        winding=1,
    )
    background_comp = GaussianComponent(x_scale=1, y_scale=scale_factor_bg, amplitude=1.0, variance=sigma_bg)

    mock_model = MockModel(
        (signal1,),
        (background_comp,),
    )

    num_x_bins = 100
    num_y_bins = 100
    x_edges = np.linspace(-1.2, 1.2, num_x_bins + 1)
    y_edges = np.linspace(-60.0, 60.0, num_y_bins + 1)

    dz: float = np.mean(np.diff(x_edges))
    dvz: float = np.mean(np.diff(y_edges))

    x_centres = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centres = 0.5 * (y_edges[:-1] + y_edges[1:])
    x_mesh, y_mesh = np.meshgrid(x_centres, y_centres)

    particles = mock_model.mock_particles(num_particles, x_edges, y_edges)
    z_samples = particles.x
    vz_samples = particles.y

    density, _, _ = np.histogram2d(z_samples, vz_samples, bins=(x_edges, y_edges))
    density = density.T

    log.info("Generating initial background estimate via KDE...")
    factor = (2 * num_particles) ** (-1 / 6)
    std_z_mirrored = np.sqrt(2 * np.sum((z_samples - np.mean(z_samples)) ** 2) / (2 * num_particles - 1))
    std_vz_mirrored = np.sqrt(2 * np.sum(vz_samples**2) / (2 * num_particles - 1))

    hz = factor * std_z_mirrored
    hvz = factor * std_vz_mirrored

    initial_background = ndimage.gaussian_filter(density, sigma=(hvz / dvz, hz / dz))
    initial_background = 0.5 * (initial_background + np.flipud(initial_background))
    # initial_background = generate_initial_background(z_samples, vz_samples, x_mesh, y_mesh)  # noqa: ERA001
    initial_background = initial_background / np.sum(initial_background) * np.sum(density)
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
    )


def main(*, exponent: int = 10, seed: int = 42, batch_size: int = 64, workers: int = 6) -> None:
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

    parameters = sample_parameters(exponent, seed)
    true_alphas = np.linspace(0.0, 1.0, 25)
    actual_alphas = parameters[:, 0]
    actual_thetas = np.zeros_like(actual_alphas)
    actual_scales = np.zeros_like(actual_alphas)
    initial_bg_sums = np.zeros_like(actual_alphas)
    measured_alphas = np.zeros_like(actual_alphas)
    measured_thetas = np.zeros_like(actual_thetas)
    measured_scales = np.zeros_like(actual_scales)
    ratios: list[float] = []
    for start in range(0, len(parameters), batch_size):
        requests: list[FitInput] = []
        stop = min(start + batch_size, len(parameters))
        for idx in range(start, stop):
            data = _generate_mock(parameters[idx])
            actual_thetas[idx] = data.phase
            actual_scales[idx] = data.scale_factor
            initial_bg_sums[idx] = np.sum(data.background)
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
            del data
        log.info("Fitting mocks %d-%d of %d...", start + 1, stop, len(parameters))
        results = fitter.fit_batch(requests, workers=workers, progress=True)
        for idx, result in enumerate(results, start=start):
            if not isinstance(result, FitSuccess):
                msg = f"Fit failed for Sobol point {idx}: {result}"
                raise RuntimeError(msg)  # noqa: TRY004 - FitFailure is an expected outcome type.
            measured_alphas[idx] = result.result.final_model.components[0].alpha
            measured_thetas[idx] = result.result.final_model.components[0].model_phase()
            measured_scales[idx] = result.result.final_model.components[0].scale_factor
            ratio = np.sum(result.result.final_model.background) / initial_bg_sums[idx]
            log.debug("Ratio = %f", ratio)
            ratios.append(ratio)
        del result, results, requests
    ratios = np.asarray(ratios)
    log.debug("Mean BG sum ratio = %f", np.mean(ratios))
    log.debug("Median BG sum ratio = %f", np.median(ratios))

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
        axes.plot(limits, limits, linestyle="--", color="k")
        axes.grid(alpha=0.2)

    alpha_points = alpha_axes.scatter(actual_alphas, measured_alphas, marker="x", cmap="magma", c=actual_scales)
    alpha_axes.plot(true_alphas, corr_fit.slope * true_alphas + corr_fit.intercept, color="tab:blue", ls="-.")
    alpha_axes.set(xlabel=r"$\alpha_{\mathrm{true}}$", ylabel=r"$\hat{\alpha}$")
    fig.colorbar(alpha_points, ax=alpha_axes, label=r"True $S$")

    theta_points = theta_axes.scatter(actual_angles, measured_angles, marker="x", cmap="viridis", c=actual_alphas)
    theta_axes.set(
        xlabel=r"$\theta_{\mathrm{model}, true} \,\, (\mathrm{deg})$",
        ylabel=r"$\hat{\theta}_{\mathrm{model}} \,\, (\mathrm{deg})$",
        xlim=(-180, 180),
        ylim=(-180, 180),
        xticks=(-180, -90, 0, 90, 180),
        yticks=(-180, -90, 0, 90, 180),
    )
    fig.colorbar(theta_points, ax=theta_axes, label=r"True $\alpha$")

    scale_points = scale_axes.scatter(actual_scales, measured_scales, c=actual_alphas, cmap="viridis", marker="x")
    scale_axes.set(
        xlabel=r"$S_{\mathrm{true}} \,\, (\mathrm{km}/\mathrm{s}/\mathrm{kpc})$",
        ylabel=r"$\hat{S} \,\, (\mathrm{km}/\mathrm{s}/\mathrm{kpc})$",
    )
    fig.colorbar(scale_points, ax=scale_axes, label=r"True $\alpha$")

    plt.savefig("sys_error.png")
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exponent", type=int, default=10, help="Sobol point count is 2**exponent (default: 1024).")
    parser.add_argument("--seed", type=int, default=42, help="Seed for parameter-design scrambling only.")
    parser.add_argument("--batch-size", type=int, default=64, help="Maximum mocks retained per fitting batch.")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--estimate-only", action="store_true", help="Report coverage and memory without generating or fitting mocks."
    )
    args = parser.parse_args()
    if not 1 <= args.exponent <= 20 or args.batch_size < 1 or args.workers < 1 or args.seed < 0:
        parser.error("Require exponent in [1, 20], positive batch size/workers, and nonnegative seed.")
    estimate_design(args.seed, args.batch_size, args.exponent)
    if not args.estimate_only:
        main(exponent=args.exponent, seed=args.seed, batch_size=args.batch_size, workers=args.workers)
