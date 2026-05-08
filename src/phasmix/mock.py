from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from matplotlib import colors as mplcolors
from matplotlib import pyplot as plt
from scipy import ndimage, special

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from optype import numpy as onp


@dataclass
class AlinderModel:
    alpha: float
    b: float
    c: float
    theta0: float
    rho: float
    scale_factor: float

    def evaluate[ShapeT: tuple[Any, ...]](
        self, r: onp.ArrayND[np.float64, ShapeT], theta: onp.ArrayND[np.float64, ShapeT]
    ) -> onp.ArrayND[np.float64, ShapeT]:
        abs_b = abs(self.b)
        abs_c = abs(self.c)
        sign: float = float(np.sign(self.b)) if self.b != 0.0 else 1.0
        phase: onp.ArrayND[np.float64, ShapeT]
        if abs_c != 0.0:
            half_b_over_c = 0.5 * abs_b / abs_c
            phase = sign * (
                -half_b_over_c + np.sqrt(np.square(half_b_over_c) + r / abs_c)
            )
        else:
            phase = sign * (r / abs_b)

        return 1.0 + self.alpha * special.expit((r - self.rho) / 0.1) * np.cos(
            theta - phase - self.theta0
        )


@dataclass
class GaussianBackground:
    sigma: float

    def evaluate[ShapeT: tuple[Any, ...]](
        self, r: onp.ArrayND[np.float64, ShapeT]
    ) -> onp.ArrayND[np.float64, ShapeT]:
        return np.exp(-0.5 * np.square(r) / np.square(self.sigma))


def main(raw_args: Sequence[str]) -> None:
    args = _parse_args(raw_args)
    output_path: Path | None = (
        Path(args.output_path) if args.output_path is not None else None
    )

    model = AlinderModel(
        alpha=0.3, b=0.005, c=0.002, theta0=0.0, scale_factor=40.0, rho=0.09
    )

    gaussian = GaussianBackground(sigma=0.25)

    x_bins = np.linspace(-1, 1, 100)
    y_bins = np.linspace(-1, 1, 200)

    x_centres = 0.5 * (x_bins[:-1] + x_bins[1:])
    y_centres = 0.5 * (y_bins[:-1] + y_bins[1:])

    dx = np.mean(np.diff(x_centres))
    dy = np.mean(np.diff(y_centres))

    x_mesh, y_mesh = np.meshgrid(x_centres, y_centres)
    r_mesh = np.hypot(x_mesh, y_mesh)
    theta_mesh = np.arctan2(y_mesh, x_mesh)
    pert = model.evaluate(r_mesh, theta_mesh)
    bg = gaussian.evaluate(r_mesh)

    density = pert * bg

    density_flat = density.ravel()
    cdf_unorm = np.cumsum(density_flat)
    cdf_norm = cdf_unorm / np.sum(cdf_unorm)

    rng = np.random.default_rng()
    num_samples: int = 100_000
    sample_indices = rng.choice(
        np.arange(len(cdf_norm)),
        size=num_samples,
        p=density_flat / np.sum(density_flat),
    )

    sample_indices_y, sample_indices_x = np.unravel_index(sample_indices, density.shape)

    jitter_x = rng.uniform(-dx / 2, dx / 2, size=num_samples)
    jitter_y = rng.uniform(-dy / 2, dy / 2, size=num_samples)
    # sample_x = x_bins[sample_indices_x] + jitter_x
    # sample_y = y_bins[sample_indices_y] + jitter_y
    sample_x = x_bins[sample_indices_x]
    sample_y = y_bins[sample_indices_y]

    sample_density, _, _ = np.histogram2d(sample_y, sample_x, bins=(y_bins, x_bins))

    density_norm = density / density.sum()
    sample_density_norm = sample_density / sample_density.sum()
    residuals = ndimage.gaussian_filter(
        density_norm, sigma=2
    ) - ndimage.gaussian_filter(sample_density_norm, sigma=2)
    max_residual = np.abs(np.max(residuals))

    if output_path is None:
        norm = mplcolors.Normalize()
        fig = plt.figure()
        input_axes = fig.add_subplot(131)
        output_axes = fig.add_subplot(132)
        residual_axes = fig.add_subplot(133)

        _ = input_axes.pcolormesh(
            x_mesh, y_mesh, density / density.sum(), cmap="viridis", norm=norm
        )
        _ = output_axes.pcolormesh(
            x_mesh,
            y_mesh,
            sample_density / sample_density.sum(),
            cmap="viridis",
            norm=norm,
        )
        _ = residual_axes.pcolormesh(
            x_mesh,
            y_mesh,
            residuals,
            cmap="seismic",
            norm=mplcolors.Normalize(vmin=-max_residual, vmax=max_residual),
        )

        plt.show()
        plt.close(fig)
    else:
        np.savez(output_path, x=sample_x, y=sample_y)
        print("Saved data.")


def _parse_args(raw_args: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="mock", description="Mock data generator.")
    _ = parser.add_argument(
        "-o", "--o", type=str, dest="output_path", help="Path to output to."
    )
    return parser.parse_args(raw_args)


if __name__ == "__main__":
    main(sys.argv[1:])
