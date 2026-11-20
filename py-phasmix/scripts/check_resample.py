from __future__ import annotations

import datetime
import logging
import multiprocessing as mp
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from matplotlib import pyplot as plt
from phasmock.component import AlinderComponent, GaussianComponent
from phasmock.mock import MockModel
from rich.console import Console
from rich.logging import RichHandler

from phasmix.background_estimators import estimate_gaussian_smooth_background_axisymmetric
from phasmix.fit import FitInput, FitOutcome, FitSuccess, PSpiralFitter

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


def create_model() -> tuple[AlinderComponent, MockModel]:
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
    return (signal1, mock_model)


def generate_mock(mock_model: MockModel, seed: np.uint64) -> FitInput:
    num_x_bins: int = 100
    num_y_bins: int = 120
    x_edges = np.linspace(-1.0, 1.0, num_x_bins + 1)
    y_edges = np.linspace(-60.0, 60.0, num_y_bins + 1)

    x_centres = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centres = 0.5 * (y_edges[:-1] + y_edges[1:])
    x_mesh, y_mesh = np.meshgrid(x_centres, y_centres)

    num_particles: int = 100_000
    particles = mock_model.mock_particles(num_particles, x_edges, y_edges, rng=seed)
    z_samples = particles.x
    vz_samples = particles.y

    density, _, _ = np.histogram2d(z_samples, vz_samples, bins=(x_edges, y_edges))
    density = density.T

    background = estimate_gaussian_smooth_background_axisymmetric(z_samples, vz_samples, x_edges, y_edges)

    fit_gen = np.random.default_rng(seed=seed)
    return FitInput(
        density=density,
        background=background,
        z_mesh=x_mesh,
        vz_mesh=y_mesh,
        winding=1,
        num_components=1,
        improve_background=True,
        rng=fit_gen,
    )


def split_list[T](items: Sequence[T], n: int) -> Sequence[Sequence[T]]:
    return [items[i : i + n] for i in range(0, len(items), n)]


def wrap_angle(angle_deg: onp.Array1D[np.float64]) -> onp.Array1D[np.float64]:
    return (angle_deg + 180) % 360 - 180


def main() -> None:
    signal1, model = create_model()
    true_alpha = signal1.alpha
    true_phase_deg = wrap_angle(np.rad2deg(signal1.model_phase()))
    true_scale = signal1.scale_factor
    log.info("True alpha = %f", true_alpha)
    log.info("True phase = %f deg", true_phase_deg)
    log.info("True scale = %f km/s/kpc", true_scale)

    num_workers: int = mp.cpu_count() // 2

    rng = np.random.default_rng(seed=8734)
    min_u64 = np.iinfo(np.uint64).min
    max_u64 = np.iinfo(np.uint64).max
    num_seeds: int = 8 * num_workers
    seeds: onp.Array1D[np.uint64] = rng.integers(low=min_u64, high=max_u64, size=num_seeds, dtype=np.uint64)

    log.info("Number of workers: %d", num_workers)
    batches: Sequence[Sequence[FitInput]] = split_list([generate_mock(model, seed) for seed in seeds], num_workers)

    fitter = PSpiralFitter(backend="rust", optimizer="differential_evolution")

    local_tz = datetime.datetime.now().astimezone().tzinfo

    start_time = time.perf_counter()
    num_processed: int = 0
    num_remaining: int = num_seeds

    outcomes: list[FitOutcome] = []
    for idx, batch in enumerate(batches):
        log.info("Batch %d/%d", idx + 1, len(batches))

        current_outcomes = fitter.fit_batch(batch, progress=True, workers=num_workers)
        num_processed += len(current_outcomes)
        outcomes.extend(current_outcomes)

        elapsed = time.perf_counter() - start_time
        remaining = num_remaining - num_processed

        seconds_per_cell = elapsed / num_processed
        eta_seconds = remaining * seconds_per_cell
        completion_time = datetime.datetime.now(tz=local_tz) + datetime.timedelta(seconds=eta_seconds)

        log.info(
            "Progress: %d/%d | Elapsed: %s | Remaining: %s | ETA: %s",
            num_processed,
            num_remaining,
            str(datetime.timedelta(seconds=round(elapsed))),
            str(datetime.timedelta(seconds=round(eta_seconds))),
            completion_time.strftime("%Y-%m-%d %H:%M:%S"),
        )

    results: list[FitSuccess] = []
    for outcome in outcomes:
        if not isinstance(outcome, FitSuccess):
            msg = f"Fit failed: {outcome}"
            raise TypeError(msg)
        results.append(outcome)

    alpha = np.array([result.result.final_model.components[0].alpha for result in results])
    phase = wrap_angle(np.rad2deg(np.array([result.result.final_model.components[0].model_phase() for result in results])))
    scale = np.array([result.result.final_model.components[0].scale_factor for result in results])

    alpha_std = np.std(alpha)
    phase_std = np.std(phase)
    scale_std = np.std(scale)

    alpha_lim = (-10 * alpha_std, 10 * alpha_std)
    phase_lim = (-10 * phase_std, 10 * phase_std)
    scale_lim = (-10 * scale_std, 10 * scale_std)
    alpha_counts, alpha_edges = np.histogram(alpha - true_alpha, range=alpha_lim, density=True)
    phase_counts, phase_edges = np.histogram(phase - true_phase_deg, range=phase_lim, density=True)
    scale_counts, scale_edges = np.histogram(scale - true_scale, range=scale_lim, density=True)

    alpha_centres = 0.5 * (alpha_edges[:-1] + alpha_edges[1:])
    phase_centres = 0.5 * (phase_edges[:-1] + phase_edges[1:])
    scale_centres = 0.5 * (scale_edges[:-1] + scale_edges[1:])

    alpha_width = alpha_edges[1] - alpha_edges[0]
    phase_width = phase_edges[1] - phase_edges[0]
    scale_width = scale_edges[1] - scale_edges[0]

    fig = plt.figure()
    alpha_axes = fig.add_subplot(131)
    phase_axes = fig.add_subplot(132)
    scale_axes = fig.add_subplot(133)

    alpha_axes.set_title(r"$\alpha$")
    phase_axes.set_title(r"$\theta_{0,\mathrm{model}}$")
    scale_axes.set_title(r"$S$")

    alpha_axes.set_xlabel(r"$\hat{\alpha} - \alpha$")
    phase_axes.set_xlabel(r"$\hat{\theta}_{0,\mathrm{model}} - \theta_{0,\mathrm{model}}$")
    scale_axes.set_xlabel(r"$\hat{S} - S$")

    alpha_axes.set_xlim(*alpha_lim)
    phase_axes.set_xlim(*phase_lim)
    scale_axes.set_xlim(*scale_lim)

    alpha_axes.bar(alpha_centres, alpha_counts, width=alpha_width, edgecolor="black", alpha=0.3)
    phase_axes.bar(phase_centres, phase_counts, width=phase_width, edgecolor="black", alpha=0.3)
    scale_axes.bar(scale_centres, scale_counts, width=scale_width, edgecolor="black", alpha=0.3)

    alpha_axes.axvline(0.0, color="tab:orange", ls="--", label="True value")
    phase_axes.axvline(0.0, color="tab:orange", ls="--", label="True value")
    scale_axes.axvline(0.0, color="tab:orange", ls="--", label="True value")

    fig.tight_layout()
    fig.savefig("out.png")
    plt.close()


if __name__ == "__main__":
    _ = setup_logging()
    main()
