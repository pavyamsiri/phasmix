"""Test the Tavangar BFE phase spiral method."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from matplotlib import colors as mplcolors
from matplotlib import pyplot as plt
from phasmock.component import AlinderComponent, GaussianComponent
from phasmock.mock import MockModel, MockParticles
from rich.console import Console
from rich.logging import RichHandler
from scipy import special

from phasmix.fit.tavangar_bfe import compute_orthogonal_bfe_coefficients, compute_tavbfe_coefficients

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
        b=0.2,
        c=0.0,
        theta0=np.pi / 2,
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
    x_edges = np.linspace(-2.0, 2.0, num_x_bins + 1)
    y_edges = np.linspace(-2.0, 2.0, num_y_bins + 1)

    num_particles: int = 250_000
    log.info("Sampling %d particles...", num_particles)
    return (mock_model.mock_particles(num_particles, x_edges, y_edges), x_edges, y_edges)


def main() -> None:
    particles, x_edges, y_edges = _generate_mock()

    density = particles.density

    x_centres = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centres = 0.5 * (y_edges[:-1] + y_edges[1:])

    x_mesh, y_mesh = np.meshgrid(x_centres, y_centres)

    jz = np.square(np.hypot(particles.x, particles.y))
    thetaz = np.arctan2(particles.y, particles.x)

    orth_coeffs = compute_orthogonal_bfe_coefficients(jz, thetaz)
    orth_reconstructed = orth_coeffs.reconstruct_cartesian(x_mesh, y_mesh, number_counts=True)

    tav_coeffs = compute_tavbfe_coefficients(jz, thetaz)
    tav_reconstructed = tav_coeffs.reconstruct_cartesian(x_mesh, y_mesh, number_counts=True)

    print(f"Tavangar basis (alpha = 1): C1/C0 = {tav_coeffs.strength(1) / tav_coeffs.strength(0)}")
    print(f"Orthogonal basis (alpha = 0): C1/C0 = {orth_coeffs.strength(1) / orth_coeffs.strength(0)}")

    norm = mplcolors.Normalize(
        vmin=min(density.min(), orth_reconstructed.min()),
        vmax=max(density.max(), orth_reconstructed.max()),
    )
    fig = plt.figure(figsize=(5, 10))
    gs = fig.add_gridspec(nrows=4, ncols=2, width_ratios=(1.0, 0.1))
    observed_axes = fig.add_subplot(gs[0, 0])
    tav_axes = fig.add_subplot(gs[1, 0])
    orth_axes = fig.add_subplot(gs[2, 0])
    phase_axes = fig.add_subplot(gs[3, 0])
    cbar_axes = fig.add_subplot(gs[:, 1])

    for ax in (observed_axes, tav_axes, orth_axes):
        ax.set_aspect("equal")
        ax.set_xlim(x_edges[0], x_edges[-1])
        ax.set_ylim(y_edges[0], y_edges[-1])
        ax.set_xticks(np.arange(x_edges[0], x_edges[-1] + 1.0, 1.0))
        ax.set_yticks(np.arange(y_edges[0], y_edges[-1] + 1.0, 1.0))
        ax.set_xlabel(r"$X_{z} \, (\mathrm{a.u.})$")
        ax.set_ylabel(r"$Y_{z} \, (\mathrm{a.u.})$")

    observed = observed_axes.pcolormesh(x_edges, y_edges, density, norm=norm, cmap="viridis")
    tav_axes.pcolormesh(x_edges, y_edges, tav_reconstructed, norm=norm, cmap="viridis")
    orth_axes.pcolormesh(x_edges, y_edges, orth_reconstructed, norm=norm, cmap="viridis")

    observed_axes.set_title("Observed distribution")
    tav_axes.set_title(r"Reconstruction (Original)")
    orth_axes.set_title(r"Reconstruction (Modified)")

    fig.colorbar(observed, cax=cbar_axes, label="Number density")

    phase_axes.scatter(
        np.arange(orth_coeffs.coefficients.shape[0]), np.rad2deg(np.unwrap(np.angle(orth_coeffs.coefficients[:, 1])))
    )
    phase_axes.set_ylim(-360, 360)
    orientation = np.rad2deg(np.angle(np.sum(orth_coeffs.coefficients[:, 1])))
    phase_axes.axhline(orientation, ls="--", color="tab:orange")
    print(orientation)

    fig.tight_layout()
    fig.savefig("out.png")
    plt.close()


if __name__ == "__main__":
    main()
