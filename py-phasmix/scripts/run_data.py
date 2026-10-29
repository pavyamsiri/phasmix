"""Utility to run fitter on test data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from matplotlib import colors as mplcolors
from matplotlib import pyplot as plt

from phasmix.data import GridData, ParticleData, load_data
from phasmix.fit import FitFailure, FitOutcome, FitSuccess, GaussianSmoothConfig, PSpiralFitter

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Final

DEFAULT_MIN_Z: Final[float] = -1.0
DEFAULT_MAX_Z: Final[float] = 1.0
DEFAULT_MIN_VZ: Final[float] = -60.0
DEFAULT_MAX_VZ: Final[float] = 60.0
DEFAULT_NUM_Z_BINS: Final[int] = 100
DEFAULT_NUM_VZ_BINS: Final[int] = 120


def main(raw_args: Sequence[str]) -> None:
    args = _parse_args().parse_args(raw_args)
    input_path = Path(args.input_path)
    data = load_data(input_path)

    fitter = PSpiralFitter(
        backend="rust",
        smoothing_func=GaussianSmoothConfig(z_scale=3, vz_scale=3),
    )
    z_bins = np.linspace(DEFAULT_MIN_Z, DEFAULT_MAX_Z, DEFAULT_NUM_Z_BINS + 1)
    vz_bins = np.linspace(DEFAULT_MIN_VZ, DEFAULT_MAX_VZ, DEFAULT_NUM_VZ_BINS + 1)

    outcome: FitOutcome
    if isinstance(data, ParticleData):
        outcome = fitter.fit_spiral(data.z, data.vz, z_bins, vz_bins, winding=1, num_components=1, improve_background=True)
        msg = "No support for ParticleData inputs right now."
    else:
        outcome = fitter.fit_spiral_with_background(
            data.density,
            data.background,
            data.z_mesh,
            data.vz_mesh,
            winding=1,
            num_components=1,
            improve_background=True,
        )

    if isinstance(outcome, FitFailure):
        msg = f"Failed fit: {outcome}"
        raise RuntimeError(msg)  # noqa: TRY004

    num_components = outcome.result.final_model.num_components

    parameters = np.zeros((num_components, 8))

    for component in range(num_components):
        parameters[component, 0] = outcome.result.final_model.components[component].alpha
        parameters[component, 1] = outcome.result.final_model.components[component].b
        parameters[component, 2] = outcome.result.final_model.components[component].c
        parameters[component, 3] = outcome.result.final_model.components[component].theta0
        parameters[component, 4] = outcome.result.final_model.components[component].scale_factor
        parameters[component, 5] = outcome.result.final_model.components[component].rho
        parameters[component, 6] = outcome.result.final_model.components[component].model_phase()
        parameters[component, 7] = outcome.result.final_model.components[component].winding

    names: tuple[str, ...] = (
        "alpha",
        "b",
        "c",
        "theta0",
        "scale_factor",
        "rho",
        "model_phase",
        "winding",
    )

    all_lines: list[list[str]] = []
    for component in range(num_components):
        lines: list[str] = []
        lines.append(f"Component {component + 1}")

        for name, value in zip(
            names,
            parameters[component, :],
            strict=True,
        ):
            if name in ("theta0", "model_phase"):
                value = (np.rad2deg(value) + 180) % 360 - 180
                lines.append(f"  {name:13s} {value:9.2f}°")
            else:
                lines.append(f"  {name:13s} {value:9.4f}")

        lines.append("")
        all_lines.append(lines)

    norm = mplcolors.LogNorm(vmin=1)
    residual_norm = mplcolors.CenteredNorm()
    fig = plt.figure()
    gs = fig.add_gridspec(nrows=3, ncols=4, height_ratios=(1.0, 1.0, 0.2))

    true_axes = fig.add_subplot(gs[0, 0])
    init_bg_axes = fig.add_subplot(gs[0, 1])
    best_fit_axes = fig.add_subplot(gs[1, 0])
    final_bg_axes = fig.add_subplot(gs[1, 1])

    true_minus_fit_axes = fig.add_subplot(gs[0, 2])
    true_minus_bg_axes = fig.add_subplot(gs[1, 2])
    spiral_axes = fig.add_subplot(gs[0, 3])
    residual_spiral_axes = fig.add_subplot(gs[1, 3])

    info_axes = fig.add_subplot(gs[2, :2])

    img = true_axes.pcolormesh(
        outcome.result.initial_model.z_mesh, outcome.result.initial_model.vz_mesh, outcome.result.data, norm=norm
    )
    fig.colorbar(img)
    true_axes.set_title("Data")
    img = init_bg_axes.pcolormesh(
        outcome.result.initial_model.z_mesh,
        outcome.result.initial_model.vz_mesh,
        outcome.result.initial_model.background,
        norm=norm,
    )
    fig.colorbar(img)
    init_bg_axes.set_title("Initial background")

    img = best_fit_axes.pcolormesh(
        outcome.result.final_model.z_mesh, outcome.result.final_model.vz_mesh, outcome.result.final_model.prediction(), norm=norm
    )
    fig.colorbar(img)
    best_fit_axes.set_title("Best fit")
    img = final_bg_axes.pcolormesh(
        outcome.result.final_model.z_mesh, outcome.result.final_model.vz_mesh, outcome.result.final_model.background, norm=norm
    )
    fig.colorbar(img)
    final_bg_axes.set_title("Final background")

    img = true_minus_fit_axes.pcolormesh(
        outcome.result.final_model.z_mesh,
        outcome.result.final_model.vz_mesh,
        outcome.result.data - outcome.result.final_model.prediction(),
        cmap="seismic",
        norm=residual_norm,
    )
    fig.colorbar(img)
    true_minus_fit_axes.set_title("Data - Best fit")
    img = true_minus_bg_axes.pcolormesh(
        outcome.result.final_model.z_mesh,
        outcome.result.final_model.vz_mesh,
        outcome.result.data - outcome.result.final_model.background,
        cmap="seismic",
        norm=residual_norm,
    )
    fig.colorbar(img)
    true_minus_bg_axes.set_title("Data - Final background")
    img = spiral_axes.pcolormesh(
        outcome.result.final_model.z_mesh,
        outcome.result.final_model.vz_mesh,
        outcome.result.final_model.prediction() / outcome.result.final_model.background,
        cmap="seismic",
        norm=mplcolors.Normalize(vmin=-1, vmax=1),
    )
    fig.colorbar(img)
    spiral_axes.set_title("Model perturbation")
    img = residual_spiral_axes.pcolormesh(
        outcome.result.final_model.z_mesh,
        outcome.result.final_model.vz_mesh,
        outcome.result.data / outcome.result.final_model.background,
        cmap="seismic",
        norm=mplcolors.Normalize(vmin=-1, vmax=1),
    )
    fig.colorbar(img)
    residual_spiral_axes.set_title("Data / Final background")

    info_axes.set_axis_off()
    dx: float = 0.9
    for idx, lines in enumerate(all_lines):
        x = 0.05 + idx * dx
        details_text = info_axes.text(
            x,
            0.95,
            "",
            transform=info_axes.transAxes,
            va="top",
            family="monospace",
            fontsize=10,
        )
        details_text.set_text("\n".join(lines))

    plt.show()
    plt.close()


def _parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="test_methods", description="Test various phase spiral measurement methods.")
    _ = parser.add_argument("input_path", type=str, help="Path to the input data.")
    return parser


if __name__ == "__main__":
    main(sys.argv[1:])
