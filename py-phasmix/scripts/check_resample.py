from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from matplotlib import colors as mplcolors
from matplotlib import pyplot as plt
from matplotlib import widgets
from phasmock.component import AlinderComponent, GaussianComponent
from phasmock.mock import MockModel
from rich.console import Console
from rich.logging import RichHandler

from phasmix import fit
from phasmix._background_utils import generate_initial_background
from phasmix._likelihood_utils import ln_likelihood
from phasmix.fit import FitFailure, FitSuccess, PSpiralFitter

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
    num_stars: int
    z: onp.Array1D[np.float64]
    vz: onp.Array1D[np.float64]


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
    signal2 = AlinderComponent(
        alpha=0.5,
        b=0.05,
        c=0.002,
        theta0=np.pi / 2,
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

    num_particles: int = 1_000
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
        num_stars=num_particles,
        z=z_samples,
        vz=vz_samples,
    )


def main() -> None:
    mock_data = _generate_mock()

    data = mock_data.density

    fitter = PSpiralFitter(backend="rust", max_iterations=10)

    res = fitter.fit_spiral_with_background(
        mock_data.density,
        mock_data.background,
        mock_data.x_mesh,
        mock_data.y_mesh,
        winding=1,
        num_components=1,
    )
    if isinstance(res, FitFailure):
        log.error("Fit failed: %s", res.reason)
        return
    log.debug("Initial fit = %s", res.result.final_model)
    prediction = res.result.final_model.prediction()

    total = int(np.sum(data))
    probabilities = prediction.ravel() / np.sum(prediction)

    rng = np.random.default_rng()
    counts = rng.multinomial(total, probabilities).reshape(data.shape).astype(np.float64)
    norm = mplcolors.Normalize(vmin=0, vmax=np.nanmax(mock_data.density))

    fig = plt.figure()
    gs = fig.add_gridspec(nrows=2, ncols=10, height_ratios=(1.0, 0.1))
    true_axes = fig.add_subplot(gs[0, :5])
    resampled_axes = fig.add_subplot(gs[0, 5:])
    button_axes = fig.add_subplot(gs[1, 0])

    true_axes.pcolormesh(mock_data.x_mesh, mock_data.y_mesh, mock_data.density, norm=norm)
    resample_img = resampled_axes.pcolormesh(mock_data.x_mesh, mock_data.y_mesh, counts, norm=norm)

    btn = widgets.Button(button_axes, label="Regenerate")

    # def _update(_val: object) -> None:
    #     nonlocal resample_img, total, probabilities, data
    #     new_counts = rng.multinomial(total, probabilities).reshape(data.shape).astype(np.float64)
    #     resample_img.set_array(new_counts)
    #     fig.canvas.draw()

    def _update(_val: object) -> None:
        nonlocal resample_img, mock_data

        indices = rng.integers(mock_data.num_stars, size=mock_data.num_stars)
        z, vz = mock_data.z[indices], mock_data.vz[indices]
        new_counts, _, _ = np.histogram2d(z, vz, bins=(mock_data.x_edges, mock_data.y_edges))
        resample_img.set_array(new_counts.T)
        fig.canvas.draw()

        res = fitter.fit_spiral_with_background(
            new_counts,
            mock_data.background,
            mock_data.x_mesh,
            mock_data.y_mesh,
            winding=1,
            num_components=1,
        )
        if isinstance(res, FitSuccess):
            log.debug("Resample fit = %s", res.result.final_model)
        else:
            log.debug("Res = %s", res)

    btn.on_clicked(_update)

    plt.show(block=True)
    plt.close()


if __name__ == "__main__":
    _ = setup_logging()
    main()
