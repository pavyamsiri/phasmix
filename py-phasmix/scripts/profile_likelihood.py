"""Sample script to test parity between python and rust."""

from __future__ import annotations

import functools
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from phasmix import bootstrap_uncertainty, fit
from phasmix._background_utils import generate_initial_background
from phasmix._likelihood_utils import ln_likelihood
from phasmix.bounds import Fixed, Interval, ParameterBounds
from phasmix.fit import FitSuccess, PSpiralFitter
from phasmock.component import AlinderComponent, GaussianComponent
from phasmock.mock import MockModel
from rich.console import Console
from rich.logging import RichHandler
from scipy import optimize

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
    density: onp.Array2D[np.float64]
    background: onp.Array2D[np.float64]
    mask: onp.Array2D[np.float64]
    x_edges: onp.Array1D[np.float64]
    y_edges: onp.Array1D[np.float64]
    x_mesh: onp.Array2D[np.float64]
    y_mesh: onp.Array2D[np.float64]
    x_centres: onp.Array1D[np.float64]
    y_centres: onp.Array1D[np.float64]
    signal: AlinderComponent


def _generate_mock() -> MockData:
    signal1 = AlinderComponent(
        alpha=0.5,
        b=0.05,
        c=0.002,
        theta0=-np.pi / 2,
        scale_factor=40.00,
        rho=0.09,
        winding=1,
    )
    background_comp = GaussianComponent(x_scale=1, y_scale=40.0, amplitude=1, variance=0.25)

    mock_model = MockModel(
        (signal1,),
        (background_comp,),
    )

    log.info("%d-arm model", len(mock_model._signal))

    num_x_bins = 100
    num_y_bins = 100
    x_edges = np.linspace(-1.2, 1.2, num_x_bins + 1)
    y_edges = np.linspace(-60.0, 60.0, num_y_bins + 1)

    x_centres = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centres = 0.5 * (y_edges[:-1] + y_edges[1:])
    x_mesh, y_mesh = np.meshgrid(x_centres, y_centres)

    num_particles: int = 100_000
    log.info("Sampling %d particles...", num_particles)
    particles = mock_model.mock_particles(num_particles, x_edges, y_edges, seed=1)
    z_samples = particles.x
    vz_samples = particles.y

    density, _, _ = np.histogram2d(z_samples, vz_samples, bins=(x_edges, y_edges))
    density = density.T

    log.info("Generating initial background estimate via KDE...")
    initial_background = generate_initial_background(z_samples, vz_samples, x_mesh, y_mesh)
    # Normalize initial background
    initial_background = initial_background / np.sum(initial_background) * np.sum(density)

    mask = fit.create_sigmoid_mask(1.0, 40.0)(x_mesh, y_mesh)

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
        signal=signal1,
    )


def _objective(
    value: float,
    *,
    parameter: str = "alpha",
    density: onp.Array2D[np.float64],
    background: onp.Array2D[np.float64],
    x_mesh: onp.Array2D[np.float64],
    y_mesh: onp.Array2D[np.float64],
    ll_max: float,
) -> float:
    current_fitter = PSpiralFitter(
        backend="rust",
        max_iterations=10,
        bounds=ParameterBounds(**{parameter: Fixed(value)}),
    )
    current_outcome = current_fitter.fit_spiral_with_background(
        density,
        background,
        x_mesh,
        y_mesh,
        num_components=1,
        improve_background=False,
        winding=1,
    )
    if isinstance(current_outcome, FitSuccess):
        return 2.0 * (ll_max - current_outcome.result.lnl) - 3.84146
    msg = f"Profile fit failed for {parameter}={value}: {current_outcome.reason}: {current_outcome.message}"
    raise RuntimeError(msg)


def _main() -> None:
    mock_data = _generate_mock()
    density = mock_data.density
    initial_background = mock_data.background
    x_mesh = mock_data.x_mesh
    y_mesh = mock_data.y_mesh

    log.info("--- Rust Version ---")

    fitter_rust = PSpiralFitter(backend="rust", max_iterations=10)
    start_time = time.perf_counter()
    outcome_rust = fitter_rust.fit_spiral_with_background(
        density,
        initial_background,
        x_mesh,
        y_mesh,
        num_components=1,
        improve_background=True,
        winding=1,
    )
    elapsed_rust = time.perf_counter() - start_time
    if isinstance(outcome_rust, fit.FitFailure):
        log.info("Rust fit failed: %s: %s", outcome_rust.reason, outcome_rust.message)
        return
    res_rust = outcome_rust.result

    log.info("Took %.3f seconds to fit", elapsed_rust)

    assert res_rust.final_model.num_components == 1, "# of components was fixed to 1."

    bounds = ParameterBounds()
    component = res_rust.final_model.components[0]
    for parameter in ("alpha", "b", "c", "theta0", "scale_factor", "rho"):
        interval = getattr(bounds, parameter)
        assert isinstance(interval, Interval)
        estimate = float(getattr(component, parameter))
        truth = float(getattr(mock_data.signal, parameter))
        objective = functools.partial(
            _objective,
            parameter=parameter,
            density=density,
            background=res_rust.final_model.background,
            x_mesh=x_mesh,
            y_mesh=y_mesh,
            ll_max=res_rust.lnl,
        )
        try:
            if objective(estimate) > 0.0:
                msg = "Profile likelihood at the fitted value is outside the 95% cutoff."
                raise ValueError(msg)
            limits: list[float] = []
            for endpoint in (interval.lower, interval.upper):
                if objective(endpoint) <= 0.0:
                    log.warning("%s: 95%% CI reaches search limit %g", parameter, endpoint)
                    limits.append(endpoint)
                else:
                    lower, upper = sorted((endpoint, estimate))
                    limits.append(float(optimize.brentq(objective, lower, upper)))
        except (ValueError, RuntimeError) as exc:
            log.warning("%s: could not determine 95%% CI: %s", parameter, exc)
            continue
        log.info(
            "%s 95%% CI = [%g, %g, %g] (lower, estimate, upper) vs %g (truth)", parameter, limits[0], estimate, limits[1], truth
        )

    log.info("--- Python Bootstrap Errors (fixed background, 200 local refits) ---")
    start_time = time.perf_counter()
    uncertainty = bootstrap_uncertainty(
        PSpiralFitter(backend="python", max_iterations=10), res_rust, n_resamples=200, seed=2, workers=4
    )
    elapsed_bootstrap = time.perf_counter() - start_time
    log.info("Bootstrap alone took %.3f seconds", elapsed_bootstrap)
    log.info("Successful local refits: %d/%d", uncertainty.n_successful, len(uncertainty.replicates))
    log.info("Parameter medians +/- bootstrap standard errors; 95% percentile intervals:")
    parameter_names = ("alpha", "b", "c", "theta0", "scale_factor", "rho")
    for index, (estimate, error, interval) in enumerate(
        zip(uncertainty.median, uncertainty.standard_errors, uncertainty.intervals, strict=True)
    ):
        component, parameter = divmod(index, len(parameter_names))
        log.info("\tComponent %d %s", component + 1, parameter_names[parameter])
        log.info("\t%.6g +/- %.6g [%.6g, %.6g]", estimate, error, interval[0], interval[2])
    for component, (error, interval) in enumerate(
        zip(uncertainty.model_phase_standard_errors, uncertainty.model_phase_intervals, strict=True), start=1
    ):
        log.info(
            "Component %d model_phase(r=%g): %.6g +/- %.6g [%.6g, %.6g] rad",
            component,
            uncertainty.model_phase_radius,
            interval[1],
            error,
            interval[0],
            interval[2],
        )
    for warning in uncertainty.warnings:
        log.info("Bootstrap note: %s", warning)


if __name__ == "__main__":
    _ = setup_logging()
    _main()
