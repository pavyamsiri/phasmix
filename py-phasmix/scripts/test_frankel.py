"""Test the Frankel et al. 2023 zebra stripes phase spiral method."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from matplotlib import cm
from matplotlib import colors as mplcolors
from matplotlib import pyplot as plt
from rich.console import Console
from rich.logging import RichHandler

from phasmix.fit.frankel import compute_frankel_parameters

if TYPE_CHECKING:
    from typing import Final

    from optype import numpy as onp

FORMAT: Final[str] = "%(message)s"
GYR_PER_KPC_KMS: Final[float] = 0.9777922217
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


def main() -> None:
    path = Path(__file__).parent / "../../test_data/frankel_2020_data.npz"

    data = np.load(path)
    assert data["type"] == "frankel", f"Data file has wrong type: {data['type']}"

    omegaz: onp.Array1D[np.float64] = data["omegaz"]
    thetaz: onp.Array1D[np.float64] = data["thetaz"]

    omegaz_bins = np.linspace(20.0, 100.0, 101)
    omegaz_centres = 0.5 * (omegaz_bins[:-1] + omegaz_bins[1:])

    thetaz_bins = np.linspace(0.0, 2.0 * np.pi, 181)
    thetaz_centres = 0.5 * (thetaz_bins[:-1] + thetaz_bins[1:])

    thetaz_mesh, omegaz_mesh = np.meshgrid(thetaz_centres, omegaz_centres)

    density, _, _ = np.histogram2d(thetaz, omegaz, bins=(thetaz_bins, omegaz_bins))

    result = compute_frankel_parameters(omegaz, thetaz)
    log.info("Amplitude = %f", result.amplitude)
    log.info("thetaz0 = %f deg", np.rad2deg(result.thetaz0))
    log.info("t = %f Gyr", result.t * GYR_PER_KPC_KMS)

    sm = cm.ScalarMappable(cmap="viridis", norm=mplcolors.Normalize())
    fig = plt.figure()
    gs = fig.add_gridspec(nrows=1, ncols=3, width_ratios=(1.0, 1.0, 0.05))
    obs_axes = fig.add_subplot(gs[0])
    model_axes = fig.add_subplot(gs[1])
    cbar_axes = fig.add_subplot(gs[2])

    _ = obs_axes.set_title("Observed")
    _ = model_axes.set_title("Reconstruction")

    for ax in (obs_axes, model_axes):
        _ = ax.set_xlabel(r"$\theta_{z} \,\, (\mathrm{rad})$")
        _ = ax.set_ylabel(r"$\Omega_{z} \,\, (\mathrm{km}/\mathrm{s}/\mathrm{kpc})$")

    _ = obs_axes.pcolormesh(thetaz_mesh, omegaz_mesh, density.T, norm=sm.norm, cmap=sm.get_cmap())
    _ = model_axes.pcolormesh(
        thetaz_mesh, omegaz_mesh, result.reconstruct(thetaz_mesh, omegaz_mesh), norm=sm.norm, cmap=sm.get_cmap()
    )
    _ = fig.colorbar(sm, cax=cbar_axes, label="Count")

    fig.tight_layout()
    _ = plt.show()
    plt.close(fig)


if __name__ == "__main__":
    setup_logging()
    main()
