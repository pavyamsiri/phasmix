"""Test the Tavangar BFE phase spiral method."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from matplotlib import colors as mplcolors
from matplotlib import pyplot as plt
from phasmix.fit.tavangar_bfe import compute_orthogonal_bfe_coefficients, compute_tavbfe_coefficients
from phasmock.component import AlinderComponent, GaussianComponent
from phasmock.mock import MockModel, MockParticles
from phasmock.recipe import MockRecipe
from rich import print as rich_print
from rich.console import Console
from rich.logging import RichHandler
from scipy import ndimage

from phasmix.fit.ttg_fourier import compute_ttg_coefficients, compute_ttg_coefficients_grid, detect_ttg_phase_spiral

if TYPE_CHECKING:
    from typing import Final

    from optype import numpy as onp

FORMAT: Final[str] = "%(message)s"
log: Final[logging.Logger] = logging.getLogger(__name__)


def setup_logging() -> None:
    """Set up logging."""
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


def _generate_mock() -> tuple[MockParticles, onp.Array1D[np.float64], onp.Array1D[np.float64]]:
    signal1 = AlinderComponent(
        alpha=0.5,
        b=0.4,
        c=0.0,
        theta0=-np.pi / 2,
        scale_factor=1.0,
        rho=0.09,
        winding=-1,
    )
    background_comp = GaussianComponent(x_scale=1, y_scale=1.0, amplitude=1, variance=1.0)

    mock_model = MockModel(
        (signal1,),
        (background_comp,),
    )

    num_x_bins = 100
    num_y_bins = 100
    x_edges = np.linspace(-4.0, 4.0, num_x_bins + 1)
    y_edges = np.linspace(-4.0, 4.0, num_y_bins + 1)

    num_particles: int = 100_000
    log.info("Sampling %d particles...", num_particles)
    return (mock_model.mock_particles(num_particles, x_edges, y_edges), x_edges, y_edges)


def main() -> None:
    particles, x_edges, y_edges = _generate_mock()

    x_centres = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centres = 0.5 * (y_edges[:-1] + y_edges[1:])

    x_mesh, y_mesh = np.meshgrid(x_centres, y_centres)

    jz = np.square(np.hypot(particles.x, particles.y))
    thetaz = np.arctan2(particles.y, particles.x)
    coeffs = compute_orthogonal_bfe_coefficients(jz, thetaz)
    print(coeffs.strength(1))
    print(coeffs.strength(0))
    print(coeffs.strength(1) / coeffs.strength(0))

    density = particles.density
    # Convert probability per unit area to expected particle counts per bin.
    bin_areas = np.diff(y_edges)[:, None] * np.diff(x_edges)[None, :]
    reconstructed = coeffs.reconstruct_cartesian(x_mesh, y_mesh) * jz.size * bin_areas

    norm = mplcolors.Normalize()
    fig = plt.figure()
    axes = fig.add_subplot(121)
    recon_axes = fig.add_subplot(122)

    axes.pcolormesh(x_mesh, y_mesh, density, norm=norm)
    recon_axes.pcolormesh(x_mesh, y_mesh, reconstructed, norm=norm)

    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
