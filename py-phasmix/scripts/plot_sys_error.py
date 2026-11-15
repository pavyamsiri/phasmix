"""Plot the parameter recovery plots."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING, Literal

import numpy as np
from matplotlib import colors as mplcolors
from matplotlib import pyplot as plt
from phasmock.component import AlinderComponent, GaussianComponent
from phasmock.mock import MockModel
from scipy import stats

from phasmix.background_estimators import estimate_gaussian_smooth_background_axisymmetric
from phasmix.bounds import Interval
from phasmix.fit import FitFailure, FitInput, ParameterBounds, PSpiralFitter

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from typing import Final

    from optype import numpy as onp


ALPHA_BOUNDS: Final[Interval] = Interval(lower=0.0, upper=1.0)
B_BOUNDS: Final[Interval] = Interval(lower=0.005, upper=0.1)
C_BOUNDS: Final[Interval] = Interval(lower=0.0, upper=0.004)
THETA0_BOUNDS: Final[Interval] = Interval(lower=-np.pi, upper=np.pi)
SCALE_FACTOR_BOUNDS: Final[Interval] = Interval(lower=20.0, upper=120.0)
RHO_BOUNDS: Final[Interval] = Interval(lower=0.0, upper=0.18)
BG_SCALE_FACTOR_BOUNDS: Final[Interval] = Interval(lower=20.0, upper=120.0)
VARIANCE_BOUNDS: Final[Interval] = Interval(lower=0.01, upper=5.0)
type BackgroundMode = Literal["refine", "fixed-estimate", "true"]


@dataclass(frozen=True)
class BatchElement:
    """A fitting input and the model components that generated it."""

    input: FitInput
    signal: AlinderComponent
    background: GaussianComponent


def _generate_model(
    num_particles: int,
    *,
    num_inputs_per_batch: int,
    power: int,
    background_mode: BackgroundMode = "refine",
) -> Iterator[Sequence[BatchElement]]:
    sampler = stats.qmc.Sobol(d=8, scramble=True, rng=1)
    samples = sampler.random_base2(m=power)
    num_z_bins: int = 100
    num_vz_bins: int = 120
    z_edges = np.linspace(-1.0, 1.0, num_z_bins + 1)
    # The scale factor has units of vz / z.  A [-1, 1] velocity range only
    # samples a thin horizontal slice through models whose scales are 20--120,
    # making the scale factor effectively unidentifiable.
    vz_edges = np.linspace(-60.0, 60.0, num_vz_bins + 1)

    z_centres = 0.5 * (z_edges[:-1] + z_edges[1:])
    vz_centres = 0.5 * (vz_edges[:-1] + vz_edges[1:])

    z_mesh, vz_mesh = np.meshgrid(z_centres, vz_centres)

    inputs: list[BatchElement] = []
    for sample_index, point in enumerate(samples):
        alpha: float = ALPHA_BOUNDS.lower + point[0] * (ALPHA_BOUNDS.upper - ALPHA_BOUNDS.lower)
        b: float = B_BOUNDS.lower + point[1] * (B_BOUNDS.upper - B_BOUNDS.lower)
        c: float = C_BOUNDS.lower + point[2] * (C_BOUNDS.upper - C_BOUNDS.lower)
        theta0: float = THETA0_BOUNDS.lower + point[3] * (THETA0_BOUNDS.upper - THETA0_BOUNDS.lower)
        scale_factor: float = SCALE_FACTOR_BOUNDS.lower + point[4] * (SCALE_FACTOR_BOUNDS.upper - SCALE_FACTOR_BOUNDS.lower)
        rho: float = RHO_BOUNDS.lower + point[5] * (RHO_BOUNDS.upper - RHO_BOUNDS.lower)
        bg_scale_factor: float = BG_SCALE_FACTOR_BOUNDS.lower + point[6] * (
            BG_SCALE_FACTOR_BOUNDS.upper - BG_SCALE_FACTOR_BOUNDS.lower
        )
        variance: float = VARIANCE_BOUNDS.lower + point[7] * (VARIANCE_BOUNDS.upper - VARIANCE_BOUNDS.lower)
        signal = AlinderComponent(alpha=alpha, b=b, c=c, theta0=theta0, scale_factor=scale_factor, rho=rho, winding=1)
        bg_comp = GaussianComponent(x_scale=1, y_scale=bg_scale_factor, amplitude=1, variance=variance)
        model = MockModel(
            signal=(signal,),
            background=(bg_comp,),
        )
        # phasmock currently does not use an integer argument as a seed.  Pass
        # a Generator explicitly to get reproducible, independent mocks.
        particles = model.mock_particles(num_particles, z_edges, vz_edges, rng=np.random.default_rng(sample_index + 1))
        density, _, _ = np.histogram2d(particles.x, particles.y, bins=(z_edges, vz_edges))

        if background_mode == "true":
            background = particles.background
        else:
            background = estimate_gaussian_smooth_background_axisymmetric(particles.x, particles.y, z_edges, vz_edges)

        inputs.append(
            BatchElement(
                input=FitInput(
                    density=density.T,
                    background=background,
                    z_mesh=z_mesh,
                    vz_mesh=vz_mesh,
                    improve_background=background_mode == "refine",
                    num_components=1,
                ),
                signal=signal,
                background=bg_comp,
            )
        )

        if len(inputs) >= num_inputs_per_batch:
            yield inputs
            inputs.clear()

    if len(inputs) > 0:
        yield inputs
    inputs.clear()


def main(raw_args: Sequence[str]) -> None:
    """Generate mocks, fit them, and plot recovery."""
    args = _parse_args(raw_args)
    output_path: Path | None = Path(args.output_path) if args.output_path is not None else None  # pyright: ignore[reportAny]
    cache_path = Path(args.cache_path)  # pyright: ignore[reportAny]

    fitter = PSpiralFitter(
        backend="rust",
        bounds=ParameterBounds(
            alpha=ALPHA_BOUNDS,
            b=B_BOUNDS,
            c=C_BOUNDS,
            theta0=THETA0_BOUNDS,
            scale_factor=SCALE_FACTOR_BOUNDS,
            rho=RHO_BOUNDS,
        ),
        optimizer="differential_evolution",
    )
    num_workers = mp.cpu_count() // 2
    num_inputs_per_batch = 2 * num_workers
    power: int = args.power  # pyright: ignore[reportAny]
    background_mode: BackgroundMode = args.background_mode  # pyright: ignore[reportAny]
    num_total_points: int = 2**power

    num_particle_array = (1_000, 10_000, 100_000, 500_000)
    num_particle_tests: int = len(num_particle_array)
    markers = ("o", "s", "^", "X")

    result_shape = (num_particle_tests, num_total_points)
    true_alphas: onp.Array2D[np.float64] = np.full(result_shape, np.nan)
    measured_alphas: onp.Array2D[np.float64] = np.full(result_shape, np.nan)

    true_phases: onp.Array2D[np.float64] = np.full(result_shape, np.nan)
    measured_phases: onp.Array2D[np.float64] = np.full(result_shape, np.nan)

    true_scale_factors: onp.Array2D[np.float64] = np.full(result_shape, np.nan)
    measured_scale_factors: onp.Array2D[np.float64] = np.full(result_shape, np.nan)

    if cache_path.exists() and not args.recompute:  # pyright: ignore[reportAny]
        (
            true_alphas,
            measured_alphas,
            true_phases,
            measured_phases,
            true_scale_factors,
            measured_scale_factors,
        ) = _load_cache(
            cache_path,
            power=power,
            num_particle_array=num_particle_array,
            background_mode=background_mode,
        )
        print(f"Loaded cached results from {cache_path}")  # noqa: T201

    num_total_batches = int(np.ceil(num_total_points / num_inputs_per_batch))  # pyright: ignore[reportAny]
    num_total_fits = num_particle_tests * num_total_points
    completed = np.isfinite(measured_alphas) & np.isfinite(measured_phases) & np.isfinite(measured_scale_factors)
    num_processed = int(np.count_nonzero(completed))
    num_previously_processed = num_processed
    start_time = monotonic()

    for particle_index, num_particles in enumerate(num_particle_array):
        if np.all(completed[particle_index]):
            print(f"Particles {num_particles:,}: using cached results")  # noqa: T201
            continue

        for batch_index, inputs in enumerate(
            _generate_model(
                num_particles,
                num_inputs_per_batch=num_inputs_per_batch,
                power=power,
                background_mode=background_mode,
            )
        ):
            batch_offset = batch_index * num_inputs_per_batch
            batch_end = batch_offset + len(inputs)
            if np.all(completed[particle_index, batch_offset:batch_end]):
                print(  # noqa: T201
                    f"Particles {particle_index + 1}/{num_particle_tests} ({num_particles:,}); "
                    f"batch {batch_index + 1}/{num_total_batches} cached"
                )
                continue

            print(  # noqa: T201
                f"Particles {particle_index + 1}/{num_particle_tests} ({num_particles:,}); "
                f"batch {batch_index + 1}/{num_total_batches}"
            )
            fit_inputs = [data.input for data in inputs]
            results = fitter.fit_batch(fit_inputs, progress=True, workers=num_workers)

            for sub_index, (current_input, res) in enumerate(zip(inputs, results, strict=True)):
                if isinstance(res, FitFailure):
                    msg = "Failed to fit?"
                    raise TypeError(msg)
                current_index = batch_offset + sub_index
                fitted_component = res.result.final_model.components[0]

                true_alphas[particle_index, current_index] = current_input.signal.alpha
                measured_alphas[particle_index, current_index] = fitted_component.alpha

                true_phases[particle_index, current_index] = _wrap_angle(current_input.signal.model_phase())
                measured_phases[particle_index, current_index] = _wrap_angle(fitted_component.model_phase())

                true_scale_factors[particle_index, current_index] = current_input.signal.scale_factor
                measured_scale_factors[particle_index, current_index] = fitted_component.scale_factor

            completed[particle_index, batch_offset:batch_end] = True
            num_processed += len(results)
            num_remaining = num_total_fits - num_processed
            elapsed = monotonic() - start_time
            num_processed_this_run = num_processed - num_previously_processed
            eta_seconds = elapsed / num_processed_this_run * num_remaining
            completion_time = datetime.now().astimezone() + timedelta(seconds=eta_seconds)

            _save_cache(
                cache_path,
                power=power,
                num_particle_array=num_particle_array,
                background_mode=background_mode,
                true_alphas=true_alphas,
                measured_alphas=measured_alphas,
                true_phases=true_phases,
                measured_phases=measured_phases,
                true_scale_factors=true_scale_factors,
                measured_scale_factors=measured_scale_factors,
            )

            print(  # noqa: T201
                f"Progress: {num_processed}/{num_total_fits} | "
                f"Elapsed: {timedelta(seconds=round(elapsed))} | "
                f"Remaining: {timedelta(seconds=round(eta_seconds))} | "
                f"ETA: {completion_time:%Y-%m-%d %H:%M:%S %Z}"
            )

    num_plot_points: int = 1000
    alpha_x = np.linspace(0.0, 1.0, num_plot_points)
    phase_x = np.linspace(-180.0, 180.0, num_plot_points)
    scale_factor_x = np.linspace(20.0, 120.0, num_plot_points)

    fig, (alpha_axes, phase_axes, scale_factor_axes) = plt.subplots(1, 3, figsize=(17, 5.5), layout="constrained")

    alpha_norm = mplcolors.Normalize(vmin=0.0, vmax=1.0)
    particle_colors = plt.get_cmap("tab10").colors
    for particle_index, (num_particles, marker) in enumerate(zip(num_particle_array, markers, strict=True)):
        point_opacity = np.clip(measured_alphas[particle_index] / 0.3, 0.05, 1.0)
        alpha_axes.scatter(
            true_alphas[particle_index],
            measured_alphas[particle_index],
            color=particle_colors[particle_index],
            edgecolors="black",
            linewidths=0.35,
            marker=marker,
            label=f"{num_particles:,}",
        )
        recovery_scatter_options = {
            "c": measured_alphas[particle_index],
            "norm": alpha_norm,
            "cmap": "plasma",
            "marker": marker,
            "alpha": point_opacity,
            "edgecolors": "black",
            "linewidths": 0.25,
        }
        phase_axes.scatter(true_phases[particle_index], measured_phases[particle_index], **recovery_scatter_options)
        scale_factor_axes.scatter(
            true_scale_factors[particle_index], measured_scale_factors[particle_index], **recovery_scatter_options
        )

    _print_alpha_regressions(true_alphas, measured_alphas, num_particle_array)

    alpha_axes.plot(alpha_x, alpha_x, ls="--", color="black")
    phase_axes.plot(phase_x, phase_x, ls="--", color="black")
    scale_factor_axes.plot(scale_factor_x, scale_factor_x, ls="--", color="black")

    alpha_axes.set(xlabel=r"True $\alpha$", ylabel=r"Measured $\alpha$")
    phase_ticks = np.arange(-180.0, 181.0, 45.0)
    phase_axes.set(
        xlabel=r"True $\theta_{0,\mathrm{model}}$ (deg)",
        ylabel=r"Measured $\theta_{0,\mathrm{model}}$ (deg)",
        xlim=(-180.0, 180.0),
        ylim=(-180.0, 180.0),
        xticks=phase_ticks,
        yticks=phase_ticks,
    )
    scale_factor_axes.set(xlabel=r"True $S$", ylabel=r"Measured $S$")
    alpha_axes.legend(title="Particles")
    for axes in (alpha_axes, phase_axes, scale_factor_axes):
        axes.grid(alpha=0.2)

    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=alpha_norm, cmap="plasma"),
        ax=(phase_axes, scale_factor_axes),
        pad=0.02,
    )
    colorbar.set_label(r"Measured $\alpha$")

    if output_path is not None:
        fig.savefig(output_path)
    else:
        plt.show()
    plt.close()


def _wrap_angle(angles_rad: float) -> float:
    angles_deg = np.rad2deg(angles_rad)  # pyright: ignore[reportAny]
    return (angles_deg + 180) % 360 - 180  # pyright: ignore[reportAny]


def _print_alpha_regressions(
    true_alphas: onp.Array2D[np.float64],
    measured_alphas: onp.Array2D[np.float64],
    num_particle_array: tuple[int, ...],
) -> None:
    """Print alpha attenuation diagnostics separately by particle count."""
    for particle_index, num_particles in enumerate(num_particle_array):
        valid = np.isfinite(true_alphas[particle_index]) & np.isfinite(measured_alphas[particle_index])
        if np.count_nonzero(valid) < 2 or np.ptp(true_alphas[particle_index, valid]) == 0:
            continue
        regression = stats.linregress(true_alphas[particle_index, valid], measured_alphas[particle_index, valid])
        print(  # noqa: T201
            f"Alpha recovery ({num_particles:,} particles): measured = {regression.slope:.6f} * true "
            f"{regression.intercept:+.6f}; R^2 = {regression.rvalue**2:.6f}; "
            f"slope stderr = {regression.stderr:.6f}"
        )


def _load_cache(
    cache_path: Path,
    *,
    power: int,
    num_particle_array: tuple[int, ...],
    background_mode: BackgroundMode,
) -> tuple[onp.Array2D[np.float64], ...]:
    """Load recovery arrays after verifying the experiment configuration."""
    with np.load(cache_path) as cache:
        if "background_mode" in cache.files:
            cached_background_mode = str(cache["background_mode"])
        elif "improve_background" in cache.files:
            cached_background_mode = "refine" if bool(cache["improve_background"]) else "fixed-estimate"
        else:
            msg = "Cache predates the background-mode setting. Use --recompute or a different --cache path."
            raise ValueError(msg)
        cached_power = int(cache["power"])
        cached_particle_array = tuple(int(value) for value in cache["num_particle_array"])
        if cached_power != power or cached_particle_array != num_particle_array or cached_background_mode != background_mode:
            msg = (
                "Cache configuration does not match: "
                f"power={cached_power}, particles={cached_particle_array}, "
                f"background_mode={cached_background_mode!r}; expected power={power}, "
                f"particles={num_particle_array}, background_mode={background_mode!r}. "
                "Use --recompute or a different --cache path."
            )
            raise ValueError(msg)
        return (
            cache["true_alphas"],
            cache["measured_alphas"],
            cache["true_phases"],
            cache["measured_phases"],
            cache["true_scale_factors"],
            cache["measured_scale_factors"],
        )


def _save_cache(
    cache_path: Path,
    *,
    power: int,
    num_particle_array: tuple[int, ...],
    background_mode: BackgroundMode,
    true_alphas: onp.Array2D[np.float64],
    measured_alphas: onp.Array2D[np.float64],
    true_phases: onp.Array2D[np.float64],
    measured_phases: onp.Array2D[np.float64],
    true_scale_factors: onp.Array2D[np.float64],
    measured_scale_factors: onp.Array2D[np.float64],
) -> None:
    """Atomically checkpoint recovery results after a completed batch."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_suffix(f"{cache_path.suffix}.tmp")
    with temporary_path.open("wb") as cache_file:
        np.savez_compressed(
            cache_file,
            power=power,
            num_particle_array=np.asarray(num_particle_array),
            background_mode=background_mode,
            true_alphas=true_alphas,
            measured_alphas=measured_alphas,
            true_phases=true_phases,
            measured_phases=measured_phases,
            true_scale_factors=true_scale_factors,
            measured_scale_factors=measured_scale_factors,
        )
    temporary_path.replace(cache_path)


def _parse_args(raw_args: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="plot_sys_error", description="Plot parameter recovery plots.")
    _ = parser.add_argument(
        "-o",
        "--output",
        dest="output_path",
        help="Path to plot to.",
    )
    _ = parser.add_argument(
        "--power",
        type=_nonnegative_int,
        default=8,
        help="Sobol exponent m, producing 2**m points per particle count (default: %(default)s).",
    )
    background_group = parser.add_mutually_exclusive_group()
    _ = background_group.add_argument(
        "--improve-background",
        dest="background_mode",
        action="store_const",
        const="refine",
        help="Estimate and iteratively refine the background (default).",
    )
    _ = background_group.add_argument(
        "--no-improve-background",
        dest="background_mode",
        action="store_const",
        const="fixed-estimate",
        help="Use the axisymmetric background estimate without refinement.",
    )
    _ = background_group.add_argument(
        "--true-background",
        dest="background_mode",
        action="store_const",
        const="true",
        help="Use the exact mock background without refinement.",
    )
    parser.set_defaults(background_mode="refine")
    _ = parser.add_argument(
        "--cache",
        dest="cache_path",
        default="plot_sys_error_cache.npz",
        help="Result cache path (default: %(default)s).",
    )
    _ = parser.add_argument(
        "--recompute",
        action="store_true",
        help="Ignore an existing cache and recompute every fit.",
    )
    return parser.parse_args(raw_args)


def _nonnegative_int(value: str) -> int:
    """Parse a nonnegative integer for argparse."""
    parsed = int(value)
    if parsed < 0:
        msg = "must be nonnegative"
        raise argparse.ArgumentTypeError(msg)
    return parsed


if __name__ == "__main__":
    main(sys.argv[1:])
