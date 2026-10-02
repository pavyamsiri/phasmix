"""Sample script to test parity between python and rust."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from matplotlib import pyplot as plt
from phasmix import bootstrap_uncertainty, fit
from phasmix._likelihood_utils import ln_likelihood
from phasmix.fit import PSpiralFitter
from phasmock.mock import MockModel
from phasmock.recipe import MockRecipe
from rich.console import Console
from rich.logging import RichHandler

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
    """Count map, background, and coordinate grids for the benchmark."""

    density: onp.Array2D[np.float64]
    background: onp.Array2D[np.float64]
    mask: onp.Array2D[np.float64]
    x_edges: onp.Array1D[np.float64]
    y_edges: onp.Array1D[np.float64]
    x_mesh: onp.Array2D[np.float64]
    y_mesh: onp.Array2D[np.float64]
    x_centres: onp.Array1D[np.float64]
    y_centres: onp.Array1D[np.float64]


def _generate_mock() -> MockData:
    smoke_recipe = MockRecipe.from_yaml(Path(__name__).parent / "recipes/smoke_recipe.yaml")
    particles = smoke_recipe.generate()
    z_samples = particles.x
    vz_samples = particles.y

    density, _, _ = np.histogram2d(z_samples, vz_samples, bins=(smoke_recipe.x_edges, smoke_recipe.y_edges))
    density = density.T

    log.info("Generating initial background estimate via KDE...")
    background_model = MockModel(signal=[], background=smoke_recipe.model.background)
    particles = background_model.mock_particles(
        smoke_recipe.num_samples, x_edges=smoke_recipe.x_edges, y_edges=smoke_recipe.y_edges, rng=None
    )
    z_samples = particles.x
    vz_samples = particles.y
    initial_background, _, _ = np.histogram2d(z_samples, vz_samples, bins=(smoke_recipe.x_edges, smoke_recipe.y_edges))
    initial_background = initial_background.T

    mask = fit.create_sigmoid_mask(1.0, 40.0)(smoke_recipe.x_mesh, smoke_recipe.y_mesh)

    log.info("ln likelihood (null) = %f", ln_likelihood(density, initial_background, mask))
    return MockData(
        density=density,
        background=initial_background,
        mask=mask,
        x_edges=smoke_recipe.x_edges,
        x_centres=smoke_recipe.x_centres,
        x_mesh=smoke_recipe.x_mesh,
        y_edges=smoke_recipe.y_edges,
        y_centres=smoke_recipe.y_centres,
        y_mesh=smoke_recipe.y_mesh,
    )


def _main() -> None:
    mock_data = _generate_mock()
    density = mock_data.density
    initial_background = mock_data.background
    x_mesh = mock_data.x_mesh
    y_mesh = mock_data.y_mesh
    mask = mock_data.mask

    log.info("--- Rust Version ---")

    fitter_rust = PSpiralFitter(
        backend="rust",
        optimizer="differential_evolution",
        max_iterations=10,
    )
    start_time = time.perf_counter()
    outcome_rust = fitter_rust.fit_spiral_with_background(
        density, initial_background, x_mesh, y_mesh, num_components=None, improve_background=True, rng=np.random.default_rng(1)
    )
    elapsed_rust = time.perf_counter() - start_time
    if isinstance(outcome_rust, fit.FitFailure):
        log.info("Rust fit failed: %s: %s", outcome_rust.reason, outcome_rust.message)
        return
    res_rust = outcome_rust.result
    log.info("Rust took %.3f seconds", elapsed_rust)
    log.info("Rust refinement attempts: %d", res_rust.num_iterations)
    log.info("Rust nfev: %d", outcome_rust.diagnostics.nfev)
    log.info("Rust termination: %s", res_rust.reason)
    log.info("Rust final model: %s", res_rust.final_model)
    log.info("Rust final lnl: %.2f", res_rust.lnl)
    log.info("Rust pvalue: %f", res_rust.final_model.pvalue(density, mask))

    log.info("--- Python Version ---")
    fitter_py = PSpiralFitter(
        backend="python",
        max_iterations=10,
    )
    start_time = time.perf_counter()
    outcome_py = fitter_py.fit_spiral_with_background(
        density, initial_background, x_mesh, y_mesh, num_components=None, improve_background=True, rng=np.random.default_rng(1)
    )
    elapsed_py = time.perf_counter() - start_time
    if isinstance(outcome_py, fit.FitFailure):
        log.info("Python fit failed: %s: %s", outcome_py.reason, outcome_py.message)
        return
    res_py = outcome_py.result
    log.info("Python took %.3f seconds", elapsed_py)
    log.info("Python refinement attempts: %d", res_py.num_iterations)
    log.info("Python nfev: %d", outcome_py.diagnostics.nfev)
    log.info("Python termination: %s", res_py.reason)
    log.info("Python final model: %s", res_py.final_model)
    log.info("Python final lnl: %.2f", res_py.lnl)
    log.info("Python pvalue: %f", res_py.final_model.pvalue(density, mask))

    log.info("--- Bootstrap comparison: same Python reference, seed, counts, and bounds ---")
    parameter_names = ("alpha", "b", "c", "theta0", "scale_factor", "rho")
    for name, bootstrap_fitter in (("Python", fitter_py), ("Rust", fitter_rust)):
        for workers in (1, 4):
            start_time = time.perf_counter()
            uncertainty = bootstrap_uncertainty(bootstrap_fitter, res_py, n_resamples=200, seed=2, workers=workers)
            elapsed_bootstrap = time.perf_counter() - start_time
            evaluations = sum(item.diagnostics.nfev or 0 for item in uncertainty.replicates)
            iterations = sum(item.diagnostics.nit or 0 for item in uncertainty.replicates)
            log.info(
                "%s bootstrap: %.3fs, workers=%d, successful=%d/%d, nfev=%d, nit=%d",
                name,
                elapsed_bootstrap,
                workers,
                uncertainty.n_successful,
                len(uncertainty.replicates),
                evaluations,
                iterations,
            )
            if workers == 4:
                for index, (estimate, error, interval) in enumerate(
                    zip(uncertainty.reference, uncertainty.standard_errors, uncertainty.intervals, strict=True)
                ):
                    component, parameter = divmod(index, len(parameter_names))
                    log.info(
                        "%s component %d %s: %.6g +/- %.6g [%.6g, %.6g]",
                        name,
                        component + 1,
                        parameter_names[parameter],
                        interval[1],
                        error,
                        interval[0],
                        interval[2],
                    )
                for warning in uncertainty.warnings:
                    log.info("%s bootstrap note: %s", name, warning)

    rs_background = res_rust.final_model.background.reshape(x_mesh.shape)
    rs_density = res_rust.final_model.prediction()

    fig = plt.figure(figsize=(12, 8))  # pyright: ignore[reportUnknownMemberType]
    # [true density, python density, rust density]
    # [true background, python background, rust background]
    true_density_axes = fig.add_subplot(231)
    py_density_axes = fig.add_subplot(232)
    rs_density_axes = fig.add_subplot(233)
    true_background_axes = fig.add_subplot(234)
    py_background_axes = fig.add_subplot(235)
    rs_background_axes = fig.add_subplot(236)

    _ = true_density_axes.set_title("True density")  # pyright: ignore[reportUnknownMemberType]
    _ = py_density_axes.set_title(f"Python density: lnl = {res_py.lnl}")  # pyright: ignore[reportUnknownMemberType]
    _ = rs_density_axes.set_title(f"Rust density: lnl = {res_rust.lnl}")  # pyright: ignore[reportUnknownMemberType]
    _ = true_background_axes.set_title("True background")  # pyright: ignore[reportUnknownMemberType]
    _ = py_background_axes.set_title("Python background")  # pyright: ignore[reportUnknownMemberType]
    _ = rs_background_axes.set_title("Rust background")  # pyright: ignore[reportUnknownMemberType]

    _ = true_density_axes.pcolormesh(x_mesh, y_mesh, density)  # pyright: ignore[reportUnknownMemberType]
    _ = py_density_axes.pcolormesh(x_mesh, y_mesh, res_py.final_model.prediction())  # pyright: ignore[reportUnknownMemberType]
    _ = rs_density_axes.pcolormesh(x_mesh, y_mesh, rs_density)  # pyright: ignore[reportUnknownMemberType]

    _ = true_background_axes.pcolormesh(x_mesh, y_mesh, initial_background)  # pyright: ignore[reportUnknownMemberType]
    _ = py_background_axes.pcolormesh(x_mesh, y_mesh, res_py.final_model.background)  # pyright: ignore[reportUnknownMemberType]
    _ = rs_background_axes.pcolormesh(x_mesh, y_mesh, rs_background)  # pyright: ignore[reportUnknownMemberType]

    fig.tight_layout()
    fig.savefig("./out.png")  # pyright: ignore[reportUnknownMemberType]
    plt.close(fig)


if __name__ == "__main__":
    _ = setup_logging()
    _main()
